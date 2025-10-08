DARTS_STAT_MODELS_NO_SERIES_ARG = {
    "KalmanForecaster",
    "ARIMA",
    "VARIMA",
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
    "StatsForecastAutoETS",
    "ExponentialSmoothing",
    "StatsForecastAutoARIMA",
    "FFT",
    # "FourTheta",
    "Croston",
    "NaiveDrift",
    "NaiveMean",
    "NaiveSeasonal",
    "NaiveMovingAverage",
}

# predefined model_args and required_args for darts models
DEEP_MODEL_REQUIRED_ARGS = {
    "input_chunk_length": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
}
REGRESSION_MODEL_REQUIRED_ARGS = {
    "lags": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
}
STAT_MODEL_REQUIRED_ARGS = {}  # type: ignore

DEEP_MODEL_ARGS = {
    "pl_trainer_kwargs": {
        "enable_progress_bar": False,
        "accelerator": "cuda",
        "devices": 1,
    }
}
