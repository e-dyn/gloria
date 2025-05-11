"""
Package-wide used type aliases
"""

# Standard Library
from typing import Iterable, Literal, Mapping, Union

# Third Party
import pandas as pd
from typing_extensions import TypeAlias

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
