from __future__ import absolute_import

from typing import List


class FieldNames:
    """
    Constants for field names used in evaluation results.

    This class defines standard field names used throughout the evaluation process
    for storing model information, timing data, and results. These constants ensure
    consistency in field naming across different parts of the evaluation pipeline.
    """

    APPROACH_NAME = "approach_name"
    FILE_NAME = "file_name"
    MODEL_PARAMS = "model_params"
    STRATEGY_ARGS = "strategy_args"
    FIT_TIME = "fit_time"
    INFERENCE_TIME = "inference_time"
    ACTUAL_DATA = "actual_data"
    INFERENCE_DATA = "inference_data"
    LOG_INFO = "log_info"

    @classmethod
    def all_fields(cls) -> List[str]:
        """
        Get list of all available field names.

        Returns:
            List[str]: List of all field name constants defined in this class.
        """
        return [
            cls.APPROACH_NAME,
            cls.FILE_NAME,
            cls.MODEL_PARAMS,
            cls.STRATEGY_ARGS,
            cls.FIT_TIME,
            cls.INFERENCE_TIME,
            cls.ACTUAL_DATA,
            cls.INFERENCE_DATA,
            cls.LOG_INFO,
        ]
