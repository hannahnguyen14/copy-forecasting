import functools
import logging
from typing import Callable, Dict, Tuple

import darts.models as darts_models

from tsfb.base.models.darts.darts_models import DartsModelAdapter
from tsfb.base.schema.darts_config import FactoryConfig
from tsfb.conf.darts_conf import (
    DEEP_MODEL_ARGS,
    DEEP_MODEL_REQUIRED_ARGS,
    REGRESSION_MODEL_REQUIRED_ARGS,
    STAT_MODEL_REQUIRED_ARGS,
)

logger = logging.getLogger(__name__)


def _generate_model_factory(cfg: FactoryConfig) -> Dict:
    return {
        "model_factory": functools.partial(
            DartsModelAdapter,
            model_class=cfg.model_class,
            model_args=cfg.model_args,
            model_name=cfg.model_name,
            allow_fit_on_eval=cfg.allow_fit_on_eval,
            supports_validation=cfg.supports_validation,
        ),
        "required_hyper_params": cfg.required_args,
    }


def _get_model_info(model_name: str, required_args: Dict, model_args: Dict) -> Tuple:
    """
    Helper function to retrieve darts model information by name.
    :param model_name: name of the model.
    :param required_args: arguments that the model requires from the pipeline.
    :param model_args: specified model arguments.
    :return: a tuple including model name, model_class, required args and model args.
    """
    model_class = getattr(darts_models, model_name, None)
    # if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
    #     model_class = None
    return model_name, model_class, required_args, model_args


# deep models implemented by darts
DARTS_DEEP_MODELS = [
    _get_model_info("TCNModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info(
        "TFTModel",
        DEEP_MODEL_REQUIRED_ARGS,
        DEEP_MODEL_ARGS,
    ),
    _get_model_info("TransformerModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("NHiTSModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("TiDEModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("BlockRNNModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("RNNModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("DLinearModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("NBEATSModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("NLinearModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
]

# regression models implemented by darts
DARTS_REGRESSION_MODELS = [
    _get_model_info("RandomForest", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("XGBModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("CatBoostModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("LightGBMModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("LinearRegressionModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("RegressionModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
]

# statistical models implemented by darts,
# these models are specially allowed to retrain during inference
DARTS_STAT_MODELS = [
    _get_model_info("KalmanForecaster", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("ARIMA", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("VARIMA", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("AutoARIMA", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("AutoCES", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("AutoTheta", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("AutoETS", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("ExponentialSmoothing", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("AutoTBATS", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("FFT", STAT_MODEL_REQUIRED_ARGS, {}),
    # _get_model_info("FourTheta", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("Croston", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveDrift", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveMean", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveSeasonal", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveMovingAverage", STAT_MODEL_REQUIRED_ARGS, {}),
]

# Generate model factories for each model class and r
# equired parameters in DARTS_DEEP_MODELS
# and add them to global variables
for _model_name, _model_class, _required_args, _model_args in DARTS_DEEP_MODELS:
    if _model_class is None:
        logger.warning(
            "Model %s is not available, skipping model registration", _model_name
        )
        globals()[_model_name] = None
        continue
    globals()[_model_name] = _generate_model_factory(
        FactoryConfig(
            model_class=_model_class,
            model_args=_model_args,
            model_name=_model_name,
            required_args=_required_args,
            allow_fit_on_eval=False,
            supports_validation=True,
        )
    )

# Generate model factories for each model class and
# required parameters in DARTS_REGRESSION_MODELS
# and add them to global variables
for _model_name, _model_class, _required_args, _model_args in DARTS_REGRESSION_MODELS:
    if _model_class is None:
        logger.warning(
            "Model %s is not available, skipping model registration", _model_name
        )
        globals()[_model_name] = None
        continue
    globals()[_model_name] = _generate_model_factory(
        FactoryConfig(
            model_class=_model_class,
            model_args=_model_args,
            model_name=_model_name,
            required_args=_required_args,
            allow_fit_on_eval=False,
            supports_validation=False,
        )
    )

# Generate model factories for each model class and
# required parameters in DARTS_STAT_MODELS
# and add them to global variables
for _model_name, _model_class, _required_args, _model_args in DARTS_STAT_MODELS:
    if _model_class is None:
        logger.warning(
            "Model %s is not available, skipping model registration", _model_name
        )
        globals()[_model_name] = None
        continue
    globals()[_model_name] = _generate_model_factory(
        FactoryConfig(
            model_class=_model_class,
            model_args=_model_args,
            model_name=_model_class.__name__,
            required_args=_required_args,
            allow_fit_on_eval=True,
            supports_validation=False,
        )
    )


# Adapters for general darts models


def darts_deep_model_adapter(model_class: type) -> Dict:
    """
    Adapts a Darts deep model class to OTB protocol.
    :param model_class: a class of deep forecasting model from Darts library.
    :return: model factory that follows the OTB protocol.
    """
    return _generate_model_factory(
        FactoryConfig(
            model_class,
            DEEP_MODEL_ARGS,
            model_class.__name__,
            DEEP_MODEL_REQUIRED_ARGS,
            allow_fit_on_eval=False,
            supports_validation=True,
        )
    )


def darts_statistical_model_adapter(model_class: type) -> Dict:
    """
    Adapts a Darts statistical model class to OTB protocol.
    :param model_class: a class of statistical forecasting model from Darts library.
    :return: model factory that follows the OTB protocol.
    """
    return _generate_model_factory(
        FactoryConfig(
            model_class,
            {},
            model_class.__name__,
            STAT_MODEL_REQUIRED_ARGS,
            allow_fit_on_eval=True,
            supports_validation=False,
        )
    )


def darts_regression_model_adapter(model_class: type) -> Dict:
    """
    Adapts a Darts regression model class to OTB protocol.
    :param model_class: a class of regression forecasting model from Darts library.
    :return: model factory that follows the OTB protocol.
    """
    return _generate_model_factory(
        FactoryConfig(
            model_class,
            {},
            model_class.__name__,
            REGRESSION_MODEL_REQUIRED_ARGS,
            allow_fit_on_eval=True,
            supports_validation=False,
        )
    )


def get_adapter_mapping() -> Dict[str, Callable]:
    """
    Get a mapping from model names to their adapter factory functions.
    :return: Dictionary mapping model names to adapter functions.
    """
    adapter_map = {}

    for name, _, _, _ in DARTS_DEEP_MODELS:
        if name:
            adapter_map[name] = darts_deep_model_adapter

    for name, _, _, _ in DARTS_REGRESSION_MODELS:
        if name:
            adapter_map[name] = darts_regression_model_adapter

    for name, _, _, _ in DARTS_STAT_MODELS:
        if name:
            adapter_map[name] = darts_statistical_model_adapter

    return adapter_map
