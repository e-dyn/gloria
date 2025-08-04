# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Define public API by import all functions and classes exposed to the end-user
"""

# Gloria
# Gloria forecaster
from gloria.interface import Gloria

# Events
from gloria.profiles import BoxCar, Cauchy, Exponential, Gaussian

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
    performance_metrics,
)
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
    "cross_validation",
    "performance_metrics",
    "model_from_toml",
]
