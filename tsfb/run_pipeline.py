from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from tsfb.base.approaches.base import ForecastingApproach
from tsfb.base.approaches.default import DefaultApproach
from tsfb.base.approaches.hybrid import HybridApproach
from tsfb.base.approaches.univariate import UnivariateToMultivariate
from tsfb.base.evaluation.evaluate_model import eval_model
from tsfb.base.models.model_loader import get_models
from tsfb.base.schema.backend_config import ParallelBackendConfig
from tsfb.base.utils.parallel import ParallelBackend
from tsfb.run_and_select_best import select_best_model


class PipelineExecutor:
    """
    Orchestrates the rolling-forecast evaluation flow from a YAML config.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    results_path : str, optional
        CSV path to save combined results. Default: "results_benchmark.csv".
    """

    def __init__(
        self, config_path: str, results_path: str = "results_benchmark.csv"
    ) -> None:
        self.config_path = config_path
        self.results_path = results_path

        self.cfg: dict | None = None
        self._backend_initialized: bool = False
        self.logger: logging.Logger = self._setup_logger(default_level="INFO")

    def run(self) -> None:
        """Execute the full pipeline end-to-end."""
        try:
            self._load_config()
            self._init_backend()
            self._setup_logger()
            assert self.logger is not None
            self.logger.info("Building approaches…")
            approaches = self._build_approaches()

            self._run_evaluation_and_save_results(approaches)
        except Exception as exc:
            if self.logger:
                self.logger.exception("Pipeline failed: %s", exc)
            else:
                self.logger.error("[FATAL] Pipeline failed: %s", exc)
            raise
        finally:
            self._shutdown_backend()

    def _load_config(self) -> None:
        """Load YAML config into self.cfg."""
        self.logger.info("\n>>> Loading config from %s", self.config_path)
        with open(self.config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        if not isinstance(self.cfg, dict):
            raise ValueError("Config must be a dict loaded from YAML.")
        # Early sanity checks
        if "evaluation" not in self.cfg:
            raise KeyError("Missing 'evaluation' section in config.")
        if "strategy_args" not in self.cfg["evaluation"]:
            raise KeyError("Missing 'evaluation.strategy_args' in config.")

    def _init_backend(self) -> None:
        """Initialize the ParallelBackend from config."""
        assert self.cfg is not None, "Config must be loaded before init backend."
        self.logger.info(">>> Initializing ParallelBackend")
        backend_cfg = self.cfg.get("backend", {}) or {}

        backend_type = backend_cfg.get("type") or "local"
        backend_type = str(backend_type)

        default_timeout = backend_cfg.get("default_timeout")
        if default_timeout is None:
            default_timeout = 0.0
        else:
            default_timeout = float(default_timeout)

        parallel_config = ParallelBackendConfig(
            backend=backend_type,
            n_workers=int(backend_cfg.get("n_workers", 1)),
            n_cpus=int(backend_cfg.get("n_cpus", 1)),
            gpu_devices=backend_cfg.get("gpu_devices"),
            max_tasks_per_child=backend_cfg.get("max_tasks_per_child"),
            worker_initializers=backend_cfg.get("worker_initializers"),
            default_timeout=default_timeout,
        )
        ParallelBackend().init(parallel_config)
        self._backend_initialized = True

    def _setup_logger(self, default_level: str = "INFO") -> logging.Logger:
        """
        Create or reconfigure the class logger and return it.
        """
        # Resolve level from config or default
        cfg_level = None
        if self.cfg is not None:
            cfg_level = (self.cfg.get("logging") or {}).get("level")
        log_level = (cfg_level or default_level or "INFO").upper()
        level = getattr(logging, log_level, logging.INFO)

        logger = (
            self.logger
            if getattr(self, "logger", None)
            else logging.getLogger("run_pipeline")
        )
        logger.setLevel(level)

        has_stdout_handler = any(
            isinstance(h, logging.StreamHandler)
            and getattr(h, "stream", None) is sys.stdout
            for h in logger.handlers
        )
        if not has_stdout_handler:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(
                logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            )
            logger.addHandler(ch)

        # Avoid double propagation to root
        logger.propagate = False

        logger.info("Logging initialized at %s", log_level)
        return logger

    def _build_approaches(self) -> Dict[str, ForecastingApproach]:
        """
        Build forecasting approaches from the 'model' section of config.

        Returns
        -------
        Dict[str, ForecastingApproach]
        """
        assert self.cfg is not None and self.logger is not None

        strategy_args = self.cfg["evaluation"]["strategy_args"]
        model_cfg = self.cfg.get("model", {}) or {}
        approaches: Dict[str, ForecastingApproach] = {}

        for name, conf in model_cfg.items():
            atype = conf.get("approach") or ("hybrid" if "group" in conf else "default")

            if atype == "default":
                factory = get_models({"models": [conf]})[0]
                app: ForecastingApproach = DefaultApproach(model_factory=factory)

            elif atype == "univariate_per_series":
                factory = get_models({"models": [conf]})[0]
                app = UnivariateToMultivariate(model_factory=factory)

            elif atype == "hybrid":
                # Merge strategy data loader config into the hybrid approach
                hybrid_full_conf = {
                    **conf,
                    "data_loader_config": strategy_args["data_loader_config"],
                }
                app = HybridApproach(config=hybrid_full_conf)

            else:
                raise ValueError(f"Unknown approach type '{atype}' for '{name}'")

            app.config = conf
            app.name = name
            approaches[name] = app
            self.logger.info("  • Built %s (%s)", app.__class__.__name__, name)

        if not approaches:
            raise ValueError(
                "No approaches defined. Please check 'model' section in config."
            )
        return approaches

    def _run_evaluation_and_save_results(
        self, approaches: Dict[str, ForecastingApproach]
    ) -> None:
        """Evaluate each approach and save combined results to CSV,
        plus print summary."""
        assert self.cfg is not None and self.logger is not None

        eval_cfg = {
            "strategy_args": self.cfg["evaluation"]["strategy_args"],
            "metrics": self.cfg["evaluation"]["metrics"],
        }
        series_list = self.cfg["evaluation"]["series_list"]

        all_results: List[pd.DataFrame] = []
        for name, approach in approaches.items():
            self.logger.info("=== Running eval for %s ===", name)
            result = eval_model(series_list, eval_cfg, approach)

            self.logger.info("Collecting batches for %s…", name)
            batches = list(result.collect())
            if not batches:
                self.logger.warning("No results for %s", name)
                continue

            df = pd.concat(batches, ignore_index=True)
            all_results.append(df)

        if not all_results:
            self.logger.error("No results collected for any approach.")
            raise RuntimeError("Evaluation produced no results.")

        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(self.results_path, index=False)
        self.logger.info(
            "Saved combined results to %s (%d rows)", self.results_path, len(final_df)
        )

        metric_cols = [
            col
            for col in final_df.columns
            if col
            not in ("strategy_args", "model_params", "actual_data", "inference_data")
            and not any(key in col.lower() for key in ["config", "param"])
            and pd.api.types.is_numeric_dtype(final_df[col])
        ]
        display_df = final_df[["approach_name"] + metric_cols].copy()

        self.logger.info("================================================")
        self.logger.info("Evaluation Summary:\n%s", display_df.to_string(index=False))
        self.logger.info("================================================")
        self.logger.info("For details, check: %s", self.results_path)

    def _shutdown_backend(self) -> None:
        """Close the parallel backend if it was initialized."""
        if self._backend_initialized:
            try:
                ParallelBackend().close(force=True)
                assert self.logger is not None
                self.logger.info("Backend closed.")
            except Exception as e:
                self.logger.warning("Failed to close backend cleanly: %s", e)
        if self.logger:
            self.logger.info("Done!")

    def run_and_select_best(
        self,
        metric: str,
        config_path: Optional[str] = None,
        results_path: str = "results_benchmark.csv",
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Run the forecasting pipeline (write to results_path) and
        select the best model by metric.
        """
        effective_config = config_path or self.config_path
        if not effective_config:
            raise ValueError(
                "config_path is required (argument or instance attribute)."
            )

        effective_results = os.path.abspath(
            results_path or self.results_path or "results_benchmark.csv"
        )
        os.makedirs(os.path.dirname(effective_results) or ".", exist_ok=True)

        self.config_path = effective_config
        self.results_path = effective_results

        self.logger.info("Running pipeline with config: %s", self.config_path)
        self.run()

        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        df = pd.read_csv(self.results_path)

        if "approach_name" not in df.columns:
            cands = [c for c in df.columns if re.search(r"^approach.*name$", c, re.I)]
            if cands:
                df = df.rename(columns={cands[0]: "approach_name"})
            else:
                raise ValueError("Column 'approach_name' not found in results file.")

        return select_best_model(
            df=df, config_path=self.config_path, metric=metric, mode=mode
        )
