# -*- coding: utf-8 -*-
import functools
import json
import logging
import traceback
from typing import Any, Callable, Generator, List, Tuple

import pandas as pd
import tqdm

from tsfb.base.evaluation.evaluator import Evaluator
from tsfb.base.evaluation.strategy import STRATEGY
from tsfb.base.evaluation.strategy.constants import FieldNames
from tsfb.base.evaluation.strategy.strategy import Strategy
from tsfb.base.utils.parallel import ParallelBackend, TaskResult

logger = logging.getLogger(__name__)


def _safe_execute(fn: Callable, args: Tuple, get_default_result: Callable):
    """
    Safely execute a function with error handling.

    Args:
        fn (Callable): Function to execute
        args (Tuple): Arguments to pass to the function
        get_default_result (Callable): Function to generate default result on failure

    Returns:
        Any: Result of fn(*args) or default result if exception occurs
    """
    try:
        return fn(*args)
    except Exception as e:
        log = f"{traceback.format_exc()}\n{e}"
        return get_default_result(**{FieldNames.LOG_INFO: log})


class EvalResult:
    """
    Container for evaluation results and collection functionality.

    This class handles the collection and processing of evaluation results
    for a forecasting approach across multiple time series.
    """

    def __init__(
        self,
        strategy: Strategy,
        result_list: List[TaskResult],
        approach: Any,
        series_list: List[str],
    ):
        """
        Initialize EvalResult instance.

        Args:
            strategy (Strategy): Evaluation strategy instance
            result_list (List[TaskResult]): List of evaluation task results
            approach (Any): The forecasting approach being evaluated
            series_list (List[str]): List of time series names/identifiers
        """
        self.strategy = strategy
        self.result_list = result_list
        self.approach = approach
        self.series_list = series_list

    def collect(self) -> Generator[pd.DataFrame, None, None]:
        """
        Collect and process evaluation results.

        Yields:
            pd.DataFrame: Batches of processed evaluation results as DataFrames.
            Generated when collector size exceeds
            threshold or all results are processed.
        """
        collector = self.strategy.get_collector()
        min_interval = 0 if len(self.result_list) < 100 else 0.1

        for i, result in enumerate(
            tqdm.tqdm(
                self.result_list,
                desc=f"collecting {self._desc_name()}",
                mininterval=min_interval,
            )
        ):
            collector.add(
                _safe_execute(
                    result.result,
                    (),
                    functools.partial(
                        self.strategy.get_default_result,
                        **{FieldNames.FILE_NAME: self.series_list[i]},
                    ),
                )
            )
            if collector.get_size() > 100000:
                yield self._flush(collector)

        if collector.get_size() > 0:
            yield self._flush(collector)

    def _desc_name(self):
        """
        Get descriptive name for progress display.

        Returns:
            str: Name of the approach or strategy class
        """
        return getattr(self.approach, "name", self.strategy.__class__.__name__)

    def _flush(self, collector):
        """
        Process and return collected results, then reset collector.

        Args:
            collector: Result collector instance

        Returns:
            pd.DataFrame: Processed results as DataFrame
        """
        df = build_result_df(collector.collect(), self.approach, self.strategy)
        collector.reset()
        return df


def eval_model(
    series_list: List[str],
    evaluation_config: dict,
    approach: Any,
) -> EvalResult:
    """
    Evaluate a forecasting approach on multiple time series.

    Args:
        series_list (List[str]): List of time series names/identifiers to evaluate
        evaluation_config (dict): Configuration for evaluation
            including metrics and strategy
        approach (Any): The forecasting approach to evaluate

    Returns:
        EvalResult: Container with evaluation results and collection functionality

    Raises:
        RuntimeError: If strategy_class is None
    """
    strat_conf = evaluation_config["strategy_args"].copy()
    evaluator = Evaluator(evaluation_config["metrics"])
    strategy_class = STRATEGY.get(evaluation_config["strategy_args"]["strategy_name"])

    if strategy_class is None:
        raise RuntimeError("strategy_class is none")
    strategy = strategy_class(strat_conf, evaluator)  # type: ignore

    backend = ParallelBackend()
    result_list: List[TaskResult] = []
    for name in tqdm.tqdm(
        series_list, desc=f"scheduling {getattr(approach, 'name', '')}"
    ):
        result_list.append(backend.schedule(strategy.execute, (name, approach)))

    return EvalResult(strategy, result_list, approach, series_list)


def build_result_df(
    result_list: List, approach: Any, strategy: Strategy
) -> pd.DataFrame:
    """
    Build a DataFrame from evaluation results with metadata.

    Args:
        result_list (List): List of evaluation results
        approach (Any): The forecasting approach that was evaluated
        strategy (Strategy): The evaluation strategy used

    Returns:
        pd.DataFrame: DataFrame containing evaluation results and metadata

    Raises:
        ValueError: If any required fields are missing from the result
    """
    result_df = pd.DataFrame(result_list, columns=strategy.field_names)

    meta = getattr(approach, "config", None) or {}
    params_json = json.dumps(meta, sort_keys=True)

    result_df.insert(0, FieldNames.MODEL_PARAMS, params_json)

    result_df.insert(0, FieldNames.STRATEGY_ARGS, strategy.get_config_str())
    result_df.insert(0, FieldNames.APPROACH_NAME, getattr(approach, "name", ""))

    missing = set(FieldNames.all_fields()) - set(result_df.columns)
    if missing:
        raise ValueError(f"Missing field: {missing}")
    return result_df
