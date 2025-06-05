"""
A collection of helper functions used througout the gloria code
"""

### --- Module Imports --- ###
# Standard Library
from typing import TYPE_CHECKING, Union, cast

# Third Party
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Gloria
    from gloria.models import ModelInputData
    from gloria.utilities.types import DTypeKind


### --- Class and Function Definitions --- ###


def time_to_integer(
    time: Union[pd.Series, pd.Timestamp],
    t0: pd.Timestamp,
    sampling_delta: pd.Timedelta,
) -> Union[pd.Series, int]:
    """
    Converts a timestamp or series of timestamps to integers with respect to
    a given reference date.

    Note: If the input timestamp contains does not lie on the grid
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
    time_as_int : Union[pd.Series, int]
        The timestamps converted to integer values

    """
    if not (isinstance(time, pd.Series) or isinstance(time, pd.Timestamp)):
        raise TypeError("Input time is neither a series nor a timestamp.")

    # Convert to a float
    time_as_float = (time - t0) / sampling_delta

    # Cast the float to an int.
    # !! NOTE !! If time_as_float contains real fractional values, ie. the
    # input time does not lie on the grid specified by t0 and sampling_delta,
    # the cast operation will lead to information loss and not be invertible
    if isinstance(time, pd.Series):
        # Signal to type checker, that we are sure it's a series
        time_as_float = cast(pd.Series, time_as_float)
        return (time_as_float).astype(np.int16)
    else:
        # Signal to type checker, that we are sure it's a float
        time_as_float = cast(float, time_as_float)
        return int(time_as_float)


def infer_sampling_period(timestamps: pd.Series, q=0.5) -> pd.Timedelta:
    """
    Tries to infer a sampling period of given timestamps.

    The function evaluates the q-quantile of differences between subsequent
    timestamps. Hence, it does not necessarily return the most frequent
    timestamp. Instead it confirms that the q'th fraction of data has periods
    below or equal to the inferred one, which is sufficient for its main
    purpose: checking whether the Nyquist sampling condition is fulfilled.

    Parameters
    ----------
    timestamps : pd.Series
        Input pandas series of timestamps
    q : TYPE, optional
        The level of the quantile The default is 0.5.

    Returns
    -------
    pd.Timestamp
        The inferred sampling period

    """
    # Calculate differences between subsequent timestamps and take their
    # q-quantile
    return timestamps.diff().quantile(q)


def cast_series_to_kind(series: pd.Series, kind: "DTypeKind") -> pd.Series:
    """
    Casts a pandas Series to a dtype based on the given dtype kind.

    Parameters
    ----------

    series : pd.Series
        The pandas Series to be cast.
    kind : str
        A dtype kind string. Supported kinds are:
        - 'u': unsigned integer (default to uint64)
        - 'i': signed integer (default to int64)
        - 'f': floating-point (default to float64)
        - 'b': boolean (default to bool)

    Returns
    -------

    pd.Series
        The pandas Series cast to the corresponding dtype.

    Raises
    ------

    ValueError
        If an unsupported dtype kind is provided.
    """
    # Lookup for standard dtypes for the requested dtype-kind
    kind_to_dtype = {
        "u": np.uint64,  # Default to 64-bit unsigned integer
        "i": np.int64,  # Default to 64-bit signed integer
        "f": np.float64,  # Default to 64-bit floating-point
        "b": np.bool_,  # Boolean
    }

    # Get the dtype
    dtype = kind_to_dtype.get(kind)
    if dtype is None:
        raise ValueError(f"Unsupported dtype kind: {kind}")

    try:
        return series.astype(dtype)
    except ValueError as e:
        raise ValueError(f"Failed to cast Series to {dtype}.") from e


def simple_poisson_model(stan_data: "ModelInputData") -> pd.Series:
    """
    Fits a simple poisson model that can be used for further estimations.

    Parameters
    ----------
    stan_data : "ModelInputData"
        Model agnostic input data provided by the Gloria interface

    Returns
    -------
    pd.Series
        Predicted values for the poisson model.
    """
    # Gloria
    from gloria.models import Poisson

    m_poisson = Poisson(model_name="poisson")
    m_poisson.fit(
        stan_data=stan_data,
        optimize_mode="MLE",
        sample=False,
        augmentation_config=None,
    )
    result = m_poisson.predict(
        t=stan_data.t,
        X=stan_data.X,
        interval_width=0.8,
        n_samples=0,
    )
    return result.yhat


def calculate_dispersion(
    y_obs: Union[np.ndarray, pd.Series],
    y_model: Union[np.ndarray, pd.Series],
    dof: int,
) -> tuple[float, float]:
    """
    Calculates the dispersion factor with respect to poisson distributed
    data given observations, modeled data, and degrees of freedom.

    It can be used to pick an appropriate model:

    alpha approx. 1 => Poisson
    alpha < 1       => Binomial
    alpha >         => negative Binomial

    Parameters
    ----------
    y_obs : Union[np.ndarray, pd.Series]
        Observed data
    y_model : Union[np.ndarray, pd.Series]
        Modeled data
    dof : int
        Degrees of freedom of the model

    Returns
    -------
    alpha : float
        Dispersion factor with respect to Poisson model
    phi: float
        Dispersion factor for Stan's negative Binomial model (Note: negative
        for underdispersed data)
    """
    # Get number of observations
    n = len(y_obs)
    # Calculate dispersion factor using chi square
    alpha = ((y_obs - y_model) ** 2 / y_model).sum() / (n - dof)
    # Calculate Stan's dispersion factor
    phi = (y_model / (alpha - 1)).mean()
    return alpha, phi


def convert_to_timedelta(timedelta: Union[pd.Timedelta, str]) -> pd.Timedelta:
    """
    Takes Timedelta or Timedelta like string and converts it to a Timedelta.
    If any errors occur, they will be logged and raised as ValueError so the
    function can be used as field validator for pydantic models.

    Parameters
    ----------
    timedelta : Union[pd.Timedelta, str]
        The input timedelta

    Raises
    ------
    ValueError
        Raised if the input was a string that could not be converted to a
        Timedelta.

    Returns
    -------
    pd.Timedelta
        Converted Timedelta

    """
    # Third Party
    from pandas._libs.tslibs.parsing import DateParseError

    # Gloria
    from gloria.utilities.logging import get_logger

    try:
        return pd.Timedelta(timedelta)
    except (DateParseError, ValueError) as e:
        msg = f"Could not parse input sampling period: {e}"
        get_logger().error(msg)
        raise ValueError(msg) from e
