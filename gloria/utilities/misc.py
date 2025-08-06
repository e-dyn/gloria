# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


def infer_sampling_period(
    timestamps: pd.Series, q: float = 0.5
) -> pd.Timedelta:
    """
    Estimates an upper bound for the sampling period of a time series.

    This function computes the q-th quantile of the time differences between
    consecutive timestamps in the input series. The returned time delta
    represents the value below which a fraction ``q`` of the sampling intervals
    fall. It can be used, for example, to verify whether the time series
    satisfies the `Nyquist sampling criterion
    <https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem>`_.


    Parameters
    ----------
    timestamps : pd.Series
        A pandas Series of timestamp values.
    q : float, optional
        The quantile level to compute, between 0 and 1. Defaults to 0.5.

    Returns
    -------
    pd.Timestamp
        The estimated sampling period based on the specified quantile.

    Raises
    ------

    ValueError
        If the quantile ``q`` is not between 0 and 1 (inclusive).

    """
    if q < 0 or q > 1:
        raise ValueError("Parameter q must be between 0 and 1.")
    # Calculate differences between subsequent timestamps and take their
    # q-quantile
    return timestamps.diff().quantile(q)


def cast_series_to_kind(series: pd.Series, kind: "DTypeKind") -> pd.Series:
    """
    Cast a :class:`pandas.Series` to a canonical NumPy dtype chosen by its
    one-letter *dtype kind* code.

    This utility is useful for preparing data before calling
    :meth:`Gloria.fit`, as each model supports different data types.

    Parameters
    ----------
    series : pd.Series
        The data to cast.
    kind : {"u", "i", "f", "b"}
        One-letter code defined by `numpy.dtype.kind
        <https://tinyurl.com/3czjfr3u>`_. The mapping used here is:

        * ``"u"``: unsigned integer  → ``uint64``
        * ``"i"``: signed integer    → ``int64``
        * ``"f"``: floating point    → ``float64``
        * ``"b"``: boolean           → ``bool``

    Returns
    -------
    :class:`pandas.Series`
        The input series, cast to the corresponding dtype.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the supported codes **or** the cast fails
        (e.g. due to incompatible values).

    Examples
    --------
    >>> cast_series_to_kind(pd.Series([1, 2, 3]), "f").dtype
    dtype('float64')
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
    m_poisson.fit(stan_data=stan_data, optimize_mode="MLE", use_laplace=False)
    result = m_poisson.predict(
        t=stan_data.t,
        X=stan_data.X,
        interval_width=0.8,
        trend_samples=0,
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


def convert_to_timestamp(timestamp: Union[pd.Timestamp, str]) -> pd.Timestamp:
    """
    Takes Timestamp or Timestamp like string and converts it to a Timestamp.
    If any errors occur, they will be logged and raised as ValueError so the
    function can be used as field validator for pydantic models.

    Parameters
    ----------
    timestamp : Union[pd.Timestamp, str]
        The input timestamp

    Raises
    ------
    ValueError
        Raised if the input was a string that could not be converted to a
        Timestamp.

    Returns
    -------
    pd.Timedelta
        Converted Timestamp

    """
    # Third Party
    from pandas._libs.tslibs.parsing import DateParseError

    # Gloria
    from gloria.utilities.logging import get_logger

    try:
        return pd.Timestamp(timestamp)
    except (DateParseError, ValueError) as e:
        msg = f"Could not parse input timestamp: {e}"
        get_logger().error(msg)
        raise ValueError(msg) from e
