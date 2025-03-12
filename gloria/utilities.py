"""
A collection of helper functions used througout the gloria code

TODO:
    - include flags whether user wishes to log to file and stream.
"""

### --- Module Imports --- ###
# Standard Library
import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Union, cast

# Third Party
import numpy as np
import pandas as pd

# Gloria
from gloria.constants import (
    _FILE_LEVEL,
    _GLORIA_PATH,
    _RUN_TIMESTAMP,
    _STREAM_LEVEL,
)
from gloria.types import DTypeKind

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


def cast_series_to_kind(series: pd.Series, kind: DTypeKind) -> pd.Series:
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


def error_with_traceback(func: Callable[[str], Any]) -> Callable[[str], Any]:
    """
    A decorator for the logger.error function, adding a traceback from the
    invocation point to the latest call.

    Parameters
    ----------
    func : Callable[str, Any]
        The logging.error function to be decorated

    Returns
    -------
    Callable[str, Any]
        The decorated error function

    """

    # Define the wrapper for the error function
    def wrapper(
        msg: str, *args: tuple[Any, ...], **kwargs: dict[Any, Any]
    ) -> None:
        # Get the path of the main script
        # Third Party
        import __main__

        main_script = str(Path(__main__.__file__)).lower()

        # Walk through the entire traceback and extract the filepaths
        traceback_files = [
            frame.f_code.co_filename.lower()
            for frame, _ in traceback.walk_stack(None)
        ]
        traceback_files = traceback_files[::-1]

        # Find the index of the invocation point, ie. the first time the main
        # script was executed
        main_script_index = min(
            [
                idx
                for idx, file in enumerate(traceback_files)
                if main_script == file
            ]
        )

        # Use the index to filter the traceback and append it to the original
        # error message. The slice runs only until -1 to remove the call of
        # the wrapper itself
        msg = f"{msg}\n" + "".join(
            traceback.format_stack()[main_script_index:-1]
        )
        return func(msg, *args, **kwargs)

    return wrapper


@functools.lru_cache(maxsize=None)
def get_logger(
    log_path: Path = _GLORIA_PATH / "logfiles",
    timestamp: str = _RUN_TIMESTAMP,
) -> logging.Logger:
    """
    Set up the gloria logger for sending log entries both to the stream and
    a log file.

    The logger has a static name except for a timestamp when the main script
    was executed. Hence, the logger will be unique for a single session across
    all gloria modules.

    Parameters
    ----------
    log_path : Path, optional
        The path the log file will be saved to.
        The default is _GLORIA_PATH / 'logfiles'.
    timestamp : pd.Timestampstr, optional
        A timestamp to integrate into the logger name. The default is
        _RUN_TIMESTAMP, which is the time he main script was executed.

    Returns
    -------
    logging.Logger
        The configured Logger

    """

    def stream_filter(record):
        """
        Keep only logs of level WARNING or below
        """
        return record.levelno <= logging.WARNING

    # Get the logger. Note that the timestamp is part of the logger name. If
    # timestamp uses the default _RUN_TIMESTAMP the logger will be unique
    # for a single session and all modules will use the same logger.
    logger = logging.getLogger(f"gloria_{timestamp}")
    # Set the level of the root logger to debug so nothing gets lost
    logger.setLevel("DEBUG")

    # Configure the stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(_STREAM_LEVEL)
    # Don't show errors in the stream, as python will take care of it
    stream_handler.addFilter(stream_filter)

    # Configue the file handler
    # Create the log-path if it doesn't exist
    Path(log_path).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_path / f"{timestamp}.log", mode="a", encoding="utf-8"
    )
    file_handler.setLevel(_FILE_LEVEL)

    # A common format for all log entries
    formatter = logging.Formatter(
        "{asctime} - gloria - {levelname} - {message}",
        style="{",
        datefmt="%H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the configured handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Decorate the loggers error method so it will include error tracebacks
    # in the log-file.
    # Note ruff doesn't like the setattr, hence the noqa. With direct
    # assignment mypy complains. No way to make everyone happy.
    setattr(logger, "error", error_with_traceback(logger.error))  # noqa: B010
    return logger


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
