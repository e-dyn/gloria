"""
Constant definitions used throughout the Gloria code
"""

### --- Gloria Default Settings --- ###

_EVENT_PRIOR_SCALE = 10
_EVENT_MODE = "additive"
_SEASONALITY_PRIOR_SCALE = 10
_SEASONALITY_MODE = "additive"


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


# Turned off for the time-being as it seems to be not in use
if False:
    # Conversion factors to seconds. The keys correspond to pd.Timedelta units
    _T_CONVERSION = {
        "s": 1,
        "min": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "W": 7 * 24 * 60 * 60,
    }
