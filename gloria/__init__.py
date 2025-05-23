# Standard Library
from importlib.metadata import PackageNotFoundError, version

# Gloria
from gloria.api import (
    BoxCar,
    CalendricData,
    Cauchy,
    DataConfig,
    Exponential,
    ExternalRegressor,
    Gaussian,
    Gloria,
    GloriaConfig,
    Holiday,
    IntermittentEvent,
    MetricConfig,
    PeriodicEvent,
    RunConfig,
    Seasonality,
    SingleEvent,
    cast_series_to_kind,
    generate_cutoffs,
    get_holidays,
    gloria_copy,
    infer_sampling_period,
    log_config,
    make_holiday_dataframe,
    model_from_dict,
    model_from_json,
    model_to_dict,
    model_to_json,
    time_to_integer,
)

__all__ = [
    "Gloria",
    "ExternalRegressor",
    "Seasonality",
    "SingleEvent",
    "IntermittentEvent",
    "PeriodicEvent",
    "BoxCar",
    "Gaussian",
    "Cauchy",
    "Exponential",
    "get_holidays",
    "make_holiday_dataframe",
    "Holiday",
    "CalendricData",
    "RunConfig",
    "GloriaConfig",
    "MetricConfig",
    "DataConfig",
    "model_to_dict",
    "model_from_dict",
    "model_to_json",
    "model_from_json",
    "time_to_integer",
    "infer_sampling_period",
    "cast_series_to_kind",
    "log_config",
    "generate_cutoffs",
    "gloria_copy",
]

# Read the version dynamically from pyproject.toml
try:
    __version__ = version("gloria")
except PackageNotFoundError:
    __version__ = "unknown"
