"""
Define public API by import all functions and classes exposed to the end-user
"""

# Gloria
# Configuration
from gloria.configuration import (
    DataConfig,
    GloriaConfig,
    MetricConfig,
    RunConfig,
)

# Events
from gloria.events import BoxCar, Gaussian, SuperGaussian

# Gloria forecaster
from gloria.interface import Gloria

# Protocols: Calendric Data
from gloria.protocols.calendric import (
    CalendricData,
    Holiday,
    get_holidays,
    make_holiday_dataframe,
)

# Regressors
from gloria.regressors import (
    ExternalRegressor,
    IntermittentEvent,
    PeriodicEvent,
    Seasonality,
    SingleEvent,
)

# Serialization
from gloria.serialize import (
    model_from_dict,
    model_from_json,
    model_to_dict,
    model_to_json,
)

# Utilities
from gloria.utilities import (
    cast_series_to_kind,
    infer_sampling_period,
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
    "SuperGaussian",
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
]
