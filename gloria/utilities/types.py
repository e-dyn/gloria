"""
Package-wide used type aliases
"""

# Standard Library
from typing import Annotated, Iterable, Literal, Mapping, Union

# Third Party
import pandas as pd
from pydantic import BeforeValidator
from typing_extensions import TypeAlias


# Field validation functions
def is_timedelta(timedelta: Union[pd.Timedelta, str]) -> pd.Timedelta:
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


# The strings representing implemented backend models
Distribution: TypeAlias = Literal[
    "binomial constant n",
    "binomial vectorized n",
    "normal",
    "poisson",
    "negative binomial",
    "gamma",
]

# Allowed dtype kinds
DTypeKind: TypeAlias = Literal["b", "i", "u", "f"]

# All log levels
LogLevel: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Series type for changepoints
SeriesData: TypeAlias = Union[
    pd.Series,
    Mapping,  # includes dict
    Iterable,  # includes list, tuple, np.array, range, etc.
    int,
    float,
    str,
    bool,
    None,  # scalar types
]

Timedelta = Annotated[pd.Timedelta, BeforeValidator(is_timedelta)]
