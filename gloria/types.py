"""
Package-wide used type aliases
"""

# Standard Library
from typing import Literal

# Third Party
from typing_extensions import TypeAlias

# The strings representing implemented backend models
Distribution: TypeAlias = Literal[
    "binomial constant n", "normal", "poisson", "negative binomial"
]

# Mode in which regressors are added to the model
RegressorMode: TypeAlias = Literal["additive", "multiplicative"]

# Allowed dtype kinds
DTypeKind: TypeAlias = Literal["b", "i", "u", "f"]
