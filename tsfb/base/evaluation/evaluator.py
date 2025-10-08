import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from tsfb.base.evaluation.metrics.metrics import (
    BaseMetric,
    BaseNormMetric,
    get_instantiated_metric_dict,
)


class Evaluator:
    """
    Evaluator for computing metrics on predictions.
    """

    def __init__(self, metric: Union[List[Union[str, dict]], str]):
        """
        :param metric: str or list specifying metrics.
                       - 'all' to include every available metric.
                       - single name or list of names/dicts.
        """
        # Normalize input to list
        if isinstance(metric, str):
            metric_list: List[Union[str, Dict[str, Any]]] = [metric]
        elif isinstance(metric, list):
            metric_list = metric
        else:
            raise TypeError(f"'metric' must be str or list, got {type(metric)}")
        self.metric: List[Union[str, Dict[str, Any]]] = metric_list
        self.metric_objs: List[Tuple[BaseMetric, dict]] = []
        self.metric_names: List[str] = []

        # pool: Dict[str, BaseMetric] = get_instantiated_metric_dict()
        pool: Dict[str, Any] = get_instantiated_metric_dict()

        for m in self.metric:
            if isinstance(m, str) and m == "all":
                # add all metrics
                for name, obj in pool.items():
                    self.metric_names.append(name)
                    self.metric_objs.append((obj, {}))
                break
            if isinstance(m, str):
                info: Dict[str, Any] = {"name": m}
            elif isinstance(m, dict):
                info = m.copy()
            else:
                raise TypeError(f"Metric spec must be str or dict, got {type(m)}")

            name = info.pop("name")
            if name not in pool:
                raise ValueError(f"Unknown metric: {name}")
            obj = pool[name]
            self.metric_names.append(name)
            self.metric_objs.append((obj, info))

    # pylint: disable=too-many-arguments
    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        label_col: str = "label",
        predicted_col: str = "predicted",
        # **kwargs,
    ) -> List[float]:
        """
        Compute and return list of metric scores.
        """
        df = pd.DataFrame(
            {label_col: actual.flatten(), predicted_col: predicted.flatten()}
        )
        results: List[float] = []
        for obj, params in self.metric_objs:
            try:
                if isinstance(obj, BaseNormMetric):
                    score = obj.compute_scores(
                        df, label_col, predicted_col, scaler=scaler, **params
                    )
                else:
                    score = obj.compute_scores(df, label_col, predicted_col, **params)
            except Exception as e:
                raise RuntimeError(f"Error computing {obj.name}: {e}") from e
            results.append(score)
        return results

    # pylint: disable=too-many-arguments
    def evaluate_with_log(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        label_col: str = "label",
        predicted_col: str = "predicted",
        # **kwargs,
    ) -> Tuple[List[float], str]:
        """
        Compute metrics and return (scores list, error log).
        """
        df = pd.DataFrame(
            {label_col: actual.flatten(), predicted_col: predicted.flatten()}
        )
        results: List[float] = []
        log: str = ""
        for obj, params in self.metric_objs:
            try:
                if isinstance(obj, BaseNormMetric):
                    score = obj.compute_scores(
                        df, label_col, predicted_col, scaler=scaler, **params
                    )
                else:
                    score = obj.compute_scores(df, label_col, predicted_col, **params)
            except Exception as e:
                score = np.nan
                log += f"[{obj.name}] error: {traceback.format_exc()}\n{e}\n"
            results.append(score)
        return results, log

    def default_result(self) -> List[float]:
        """
        Default NaN list matching metric count.
        """
        return [np.nan] * len(self.metric_names)
