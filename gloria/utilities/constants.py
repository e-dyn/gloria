# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Constant definitions used throughout the Gloria code
"""

# Standard Library
from pathlib import Path
from typing import Literal, Optional, TypedDict

# Third Party
import pandas as pd

# Gloria
from gloria.utilities.types import DTypeKind

# Local path of the gloria package
_GLORIA_PATH = Path(__file__).parent.parent

# The timestamp this module was loaded. Serves as unique ID for a single
# python main-script run.
_RUN_TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

### --- Gloria Default Settings --- ###
_GLORIA_DEFAULTS = dict(
    model="normal",
    sampling_period=pd.Timedelta("1d"),
    timestamp_name="ds",
    metric_name="y",
    population_name="",
    changepoints=None,
    n_changepoints=25,
    changepoint_range=0.8,
    seasonality_prior_scale=3,
    event_prior_scale=3,
    changepoint_prior_scale=3,
    dispersion_prior_scale=3,
    interval_width=0.8,
    uncertainty_samples=1000,
)


class FitDefaults(TypedDict):
    optimize_mode: Literal["MAP", "MLE"]
    sample: bool
    capacity: Optional[int]
    capacity_mode: Optional[str]
    capacity_value: Optional[float]


_FIT_DEFAULTS: FitDefaults = {
    "optimize_mode": "MAP",
    "sample": False,
    "capacity": None,
    "capacity_mode": None,
    "capacity_value": None,
}


class PredictDefaults(TypedDict):
    periods: int
    include_history: bool


_PREDICT_DEFAULTS: PredictDefaults = {"periods": 1, "include_history": True}


class LoadDataDefaults(TypedDict):
    source: str
    dtype_kind: DTypeKind


_LOAD_DATA_DEFAULTS: LoadDataDefaults = {"source": "", "dtype_kind": "f"}

### --- Column Name Construction --- ##

# The delimiter is mainly used to construct feature matrix column names
_DELIM = "__delim__"
# Column name for the timestamp column converted to integer values
_T_INT = "ds_int"

# Column name for holidays within the self generated holiday dataframes
_HOLIDAY = "holiday"


### --- Serialization --- ###

# Key to be used for pandas series dtype.kind while serializing Gloria models
_DTYPE_KIND = "dtype_kind"


### --- Miscellaneous --- ###

# Cmdstan Version to use for the Gloria model backend
_CMDSTAN_VERSION = "2.36.0"
