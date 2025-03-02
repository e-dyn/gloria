"""
TODO
 - Docstring for the module
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path
from typing import Union, Literal

# Third Party
import pandas as pd
import numpy as np

# Inhouse Packages



### --- Global Constants Definitions --- ###



### --- Class and Function Definitions --- ###

def time_to_integer(
        time: Union[pd.Series, pd.Timestamp],
        t0: pd.Timestamp,
        sampling_delta: pd.Timedelta
    ) -> pd.Series:
    """
    Converts a timestamp or series of timestamps to integers with respect to
    a given reference date.
    
    (!) Note: If the input timestamp contains does not lie on the grid 
    specified by input parameters t0 and sampling_delta, the output integer
    times correspond to different dates and hence are not convertible.

    Parameters
    ----------
    time : Union[pd.Series, pd.Timestamp]
        Input Timestamp or series of timestamps to be converted
    t0 : pd.Timestamp
        The reference timestamp
    sampling_delta : pd.Timedelta
        The timedelta tat is used for the conversion, i.e. the integer time
        will be expressed in multiples of sampling_delta

    Returns
    -------
    time_as_int : TYPE
        The timestamps converted to integer values

    """
    # Convert to a float
    time_as_float = (time - t0) / sampling_delta
    
    # Cast the float to an int.
    # !! NOTE !! If time_as_float contains real fractional values, ie. the
    # input time does not lie on the grid specified by t0 and sampling_delta,
    # the cast operation will lead to information loss and not be invertible 
    if isinstance(time, pd.Series):
        time_as_int = (time_as_float).astype(np.int16)
    elif isinstance(time, pd.Timestamp):
        time_as_int = int(time_as_float)
    
    return time_as_int


def infer_sampling_period(timestamps: pd.Series, q = 0.5) -> pd.Timedelta:
    return timestamps.diff().quantile(q)
    

def cast_series_to_kind(series: pd.Series, kind: Literal[tuple('biuf')]) -> pd.Series:
    """
    Casts a pandas Series to a dtype based on the given dtype kind.
    
    Parameters:
    ----------
    series : pd.Series
        The pandas Series to be cast.
    kind : str
        A dtype kind string. Supported kinds are:
        - 'u': unsigned integer (default to uint64)
        - 'i': signed integer (default to int64)
        - 'f': floating-point (default to float64)
        - 'b': boolean (default to bool)
    
    Returns:
    -------
    pd.Series
        The pandas Series cast to the corresponding dtype.

    Raises:
    ------
    ValueError
        If an unsupported dtype kind is provided.
    """
    kind_to_dtype = {
        'u': np.uint64,  # Default to 64-bit unsigned integer
        'i': np.int64,   # Default to 64-bit signed integer
        'f': np.float64, # Default to 64-bit floating-point
        'b': np.bool_,   # Boolean
    }
    
    dtype = kind_to_dtype.get(kind)
    if dtype is None:
        raise ValueError(f"Unsupported dtype kind: {kind}")
    
    try:
        return series.astype(dtype)
    except ValueError as e:
        raise ValueError(f"Failed to cast Series to {dtype}: {e}")


### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent