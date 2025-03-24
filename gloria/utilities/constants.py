"""
Constant definitions used throughout the Gloria code
"""

# Standard Library
from pathlib import Path
from typing import Literal, TypedDict

# Third Party
import pandas as pd

# Local path of the gloria package
_GLORIA_PATH = Path(__file__).parent.parent.parent

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
    seasonality_mode="additive",
    seasonality_prior_scale=10,
    event_mode="additive",
    event_prior_scale=10,
    changepoint_prior_scale=0.05,
    interval_width=0.8,
    uncertainty_samples=1000,
)


class BackendDefaults(TypedDict):
    optimize_mode: Literal["MAP", "MLE"]
    sample: bool


_BACKEND_DEFAULTS: BackendDefaults = {"optimize_mode": "MAP", "sample": True}

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
