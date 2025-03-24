"""
Definition of gloria specific Errors
"""

# Standard Library
from typing import Optional

# Third Party
from typing_extensions import Self


class NotFittedError(RuntimeError):
    """
    Raised when an operation expects a fitted Gloria instance but got an
    unfitted one.
    """

    def __init__(self: Self, message: Optional[str] = None) -> None:
        if message is None:
            message = "Gloria model has not been fit yet."
        super().__init__(message)


class FittedError(RuntimeError):
    """
    Raised when an operation expects an unfitted Gloria instance but got a
    fitted one.
    """

    def __init__(self: Self, message: Optional[str] = None) -> None:
        if message is None:
            message = "Gloria model has been fit before."
        super().__init__(message)
