# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Standard Library
from importlib.metadata import PackageNotFoundError, version

# Gloria
from gloria.api import (
    BoxCar,
    CalendricData,
    Cauchy,
    Exponential,
    ExternalRegressor,
    Gaussian,
    Gloria,
    Holiday,
    IntermittentEvent,
    PeriodicEvent,
    Seasonality,
    SingleEvent,
    cast_series_to_kind,
    cross_validation,
    get_holidays,
    infer_sampling_period,
    make_holiday_dataframe,
    model_from_dict,
    model_from_json,
    model_from_toml,
    model_to_dict,
    model_to_json,
    performance_metrics,
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

# Read the version dynamically from pyproject.toml
try:
    __version__ = version("gloria")
except PackageNotFoundError:
    __version__ = "unknown"
