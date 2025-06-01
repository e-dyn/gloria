"""
Define public API by import all functions and classes exposed to the end-user
"""

# Gloria
# Events
from gloria.events import BoxCar, Cauchy, Exponential, Gaussian

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

# Configuration
from gloria.utilities.configuration import model_from_toml
from gloria.utilities.diagnostics import (
    cross_validation,
    generate_cutoffs,
    gloria_copy,
    performance_metrics,
    single_cutoff_forecast,
)
from gloria.utilities.logging import log_config
from gloria.utilities.misc import (
    cast_series_to_kind,
    infer_sampling_period,
    time_to_integer,
)

# Utilities
from gloria.utilities.serialize import (
    model_from_dict,
    model_from_json,
    model_to_dict,
    model_to_json,
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
    "single_cutoff_forecast",
    "cross_validation",
    "performance_metrics",
    "model_from_toml",
]
