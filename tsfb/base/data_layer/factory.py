from typing import Any, Dict, Type

from tsfb.base.data_layer.dataloader import BaseTimeSeriesDataLoader

# from tsfb.base.data_layer.dataloader import CustomTimeSeriesDataLoader
# from tsfb.base.data_layer.dataloader import SparkTimeSeriesDataLoader


class DataLoaderFactory:
    """
    Factory class for instantiating time series data loaders.

    This class supports dynamic selection and registration of different
    `BaseTimeSeriesDataLoader` subclasses based on
        a string key provided in the configuration.
    """

    LOADERS: Dict[str, Type[BaseTimeSeriesDataLoader]] = {
        "base": BaseTimeSeriesDataLoader,
        # "custom": CustomTimeSeriesDataLoader,
        # "spark": SparkTimeSeriesDataLoader,
    }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseTimeSeriesDataLoader:
        """
        Instantiate a data loader based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing
                the 'loader' key.
            spark (Optional[Any]): Optional SparkSession,
                used if the selected loader supports Spark.

        Returns:
            BaseTimeSeriesDataLoader: An instance of the requested data loader.

        Raises:
            KeyError: If 'loader' key is missing from the configuration.
            ValueError: If the loader key is not registered in the factory.
        """
        loader_key = config.get("loader")
        if loader_key is None:
            raise KeyError("Missing key 'loader' in data_config.")
        loader_cls = cls.LOADERS.get(loader_key)
        if loader_cls is None:
            valid = ", ".join(cls.LOADERS.keys())
            raise ValueError(f"Unknown loader '{loader_key}'. Valid options: {valid}")

        return loader_cls(config)

    @classmethod
    def register_loader(
        cls, key: str, loader_cls: Type[BaseTimeSeriesDataLoader]
    ) -> None:
        """
        Register a new loader class with the factory.

        Args:
            key (str): The string identifier for the loader.
            loader_cls (Type[BaseTimeSeriesDataLoader]): The loader class to register.

        Raises:
            KeyError: If the key is already registered.
        """
        if key in cls.LOADERS:
            raise KeyError(f"Loader '{key}' already registered.")
        cls.LOADERS[key] = loader_cls
