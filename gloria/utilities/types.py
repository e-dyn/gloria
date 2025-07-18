# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Package-wide used type aliases
"""

# Standard Library
from typing import Annotated, Iterable, Literal, Mapping, Union

# Third Party
import pandas as pd
from pydantic import BeforeValidator
from typing_extensions import TypeAlias

# Gloria
from gloria.utilities.misc import convert_to_timedelta, convert_to_timestamp

# The strings representing implemented backend models
Distribution: TypeAlias = Literal[
    "binomial",
    "normal",
    "poisson",
    "negative binomial",
    "gamma",
    "beta",
    "beta-binomial",
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

# Annotaded Timedelta and Timestamp type for validation
Timedelta = Annotated[pd.Timedelta, BeforeValidator(convert_to_timedelta)]
Timestamp = Annotated[pd.Timestamp, BeforeValidator(convert_to_timestamp)]

# Timedelta like type including strings
TimedeltaLike = Union[pd.Timedelta, str]
