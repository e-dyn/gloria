"""
Definition of Event base class and its implementations
"""

### --- Module Imports --- ###
# Standard Library
from abc import ABC, abstractmethod
from typing import Any, Type, Union

# Third Party
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Self

### --- Global Constants Definitions --- ###


### --- Class and Function Definitions --- ###
class Event(BaseModel, ABC):
    """
    Abstract base class for all events
    """

    model_config = ConfigDict(
        # All events will use some sort of pd.Timestamp or pd.Timedelta
        arbitrary_types_allowed=True,
    )

    @property
    def _event_type(self: Self):
        """
        Returns name of the event class.
        """
        return type(self).__name__

    @abstractmethod
    def generate(
        self: Self, timestamps: pd.Series, t_start: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single instance of the event.

        Parameters
        ----------
        timestamps : pd.Series
            The input timestamps as independent variable
        t_start : pd.Timestamp
            Location of the event

        Raises
        ------
        NotImplementedError
            In case the inheriting Event class did not implement the generate
            method

        Returns
        -------
        pd.Series
            The output time series including the event.
        """
        pass

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the event to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the event type. All other event fields will
            be added by event child classes.
        """
        event_dict = {"event_type": self._event_type}
        return event_dict

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[Self], event_dict: dict[str, Any]) -> Self:
        """
        Forward declaration of class method for static type checking.
        See details in event_from_dict().
        """
        pass

    @classmethod
    def check_for_missing_keys(
        cls: Type[Self], event_dict: dict[str, Any]
    ) -> None:
        """
        Confirms that all required fields for the requested event type are
        found in the event dictionary.

        Parameters
        ----------
        event_dict : dict[str, Any]
            Dictionary containing all event fields

        Raises
        ------
        KeyError
            Raised if any keys are missing

        Returns
        -------
        None
        """
        # Use sets to find the difference between event model fields and passed
        # dictionary keys
        required_fields = {
            name
            for name, info in cls.model_fields.items()
            if info.is_required()
        }
        missing_keys = required_fields - set(event_dict.keys())
        # If any is missing, raise an error.
        if missing_keys:
            missing_keys_str = ", ".join([f"'{key}'" for key in missing_keys])
            raise KeyError(
                f"Key(s) {missing_keys_str} required for event of type "
                f"'{cls.__name__}' but not found in event dictionary."
            )


class BoxCar(Event):
    """
    A BoxCar shaped event
    """

    # Duration of of boxcar window
    duration: pd.Timedelta

    @field_validator("duration", mode="before")
    @classmethod
    def validate_duration(
        cls: Type[Self], duration: Union[pd.Timedelta, str]
    ) -> pd.Timedelta:
        # Third Party
        from pandas._libs.tslibs.parsing import DateParseError

        try:
            return pd.Timedelta(duration)
        except DateParseError as e:
            raise ValueError("Could not parse input sampling period.") from e

    def generate(
        self: Self, timestamps: pd.Series, t_start: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single boxcar event.

        Parameters
        ----------
        timestamps : pd.Series
            The input timestamps as independent variable
        t_start : pd.Timestamp
            Location of the boxcar's rising edge

        Returns
        -------
        pd.Series
            The output time series including the boxcar event with amplitude 1.
        """
        mask = (timestamps >= t_start) & (timestamps < t_start + self.duration)
        return mask * 1

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the BoxCar event to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all event fields including event type

        """
        # Start with event type
        event_dict = super().to_dict()
        # Add additional fields
        event_dict["duration"] = str(self.duration)
        return event_dict

    @classmethod
    def from_dict(cls: Type[Self], event_dict: dict[str, Any]) -> Self:
        """
        Creates BoxCar event instance from a dictionary that holds the event
        fields.

        Parameters
        ----------
        event_dict : dict[str, Any]
            Dictionary containing all event fields

        Returns
        -------
        BoxCar
            BoxCar instance with fields from event_dict
        """
        # Convert duration string to pd.Timedelta
        event_dict["duration"] = pd.Timedelta(event_dict["duration"])
        return cls(**event_dict)


class Gaussian(Event):
    """
    A Gaussian shaped event
    """

    # Duration of of boxcar window
    sigma: pd.Timedelta

    @field_validator("sigma", mode="before")
    @classmethod
    def validate_sigma(
        cls: Type[Self], sigma: Union[pd.Timedelta, str]
    ) -> pd.Timedelta:
        # Third Party
        from pandas._libs.tslibs.parsing import DateParseError

        try:
            return pd.Timedelta(sigma)
        except DateParseError as e:
            raise ValueError("Could not parse input sampling period.") from e

    def generate(
        self: Self, timestamps: pd.Series, t_start: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single Gaussian event.

        Parameters
        ----------
        timestamps : pd.Series
            The input timestamps as independent variable.
        t_start : pd.Timestamp
            Location of the Gaussian's maximum.

        Returns
        -------
        pd.Series
            The output time series including the Gaussian event with amplitude
            1.
        """
        # normalize the input timestamps
        t = (timestamps - t_start) / self.sigma
        # Evaluate the Gaussian
        return np.exp(-0.5 * t**2)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Gaussian event to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all event fields including event type.
        """
        # Start with event type
        event_dict = super().to_dict()
        # Add additional fields
        event_dict["sigma"] = str(self.sigma)
        return event_dict

    @classmethod
    def from_dict(cls: Type[Self], event_dict: dict[str, Any]) -> Self:
        """
        Creates Gaussian event instance from a dictionary that holds the event
        fields.

        Parameters
        ----------
        event_dict : dict[str, Any]
            Dictionary containing all event fields

        Returns
        -------
        BoxCar
            BoxCar instance with fields from event_dict
        """
        # Convert sigma string to pd.Timedelta
        event_dict["sigma"] = pd.Timedelta(event_dict["sigma"])
        return cls(**event_dict)


class SuperGaussian(Event):
    """
    A super-Gaussian shaped event (or Higher Order Gaussian)
    """

    # Duration of of boxcar window
    sigma: pd.Timedelta
    order: float = Field(ge=1, default=1.0)

    @field_validator("sigma", mode="before")
    @classmethod
    def validate_sigma(
        cls: Type[Self], sigma: Union[pd.Timedelta, str]
    ) -> pd.Timedelta:
        # Third Party
        from pandas._libs.tslibs.parsing import DateParseError

        try:
            return pd.Timedelta(sigma)
        except DateParseError as e:
            raise ValueError("Could not parse input sampling period.") from e

    def generate(
        self: Self, timestamps: pd.Series, t_start: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single Gaussian event.

        Parameters
        ----------
        timestamps : pd.Series
            The input timestamps as independent variable.
        t_start : pd.Timestamp
            Location of the Gaussian's maximum.

        Returns
        -------
        pd.Series
            The output time series including the Gaussian event with amplitude
            1.
        """
        # normalize the input timestamps
        t = (timestamps - t_start) / self.sigma
        # Evaluate the Gaussian
        return np.exp(-((0.5 * t**2) ** self.order))

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Gaussian event to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all event fields including event type.
        """
        # Start with event type
        event_dict = super().to_dict()
        # Add additional fields
        event_dict["sigma"] = str(self.sigma)
        event_dict["order"] = self.order
        return event_dict

    @classmethod
    def from_dict(cls: Type[Self], event_dict: dict[str, Any]) -> Self:
        """
        Creates Gaussian event instance from a dictionary that holds the event
        fields.

        Parameters
        ----------
        event_dict : dict[str, Any]
            Dictionary containing all event fields

        Returns
        -------
        BoxCar
            BoxCar instance with fields from event_dict
        """
        # Convert sigma string to pd.Timedelta
        event_dict["sigma"] = pd.Timedelta(event_dict["sigma"])
        return cls(**event_dict)


# A map of Event class names to actual classes
EVENT_MAP: dict[str, Type[Event]] = {
    "BoxCar": BoxCar,
    "Gaussian": Gaussian,
    "SuperGaussian": SuperGaussian,
}


def event_from_dict(cls: Type[Event], event_dict: dict[str, Any]) -> Event:
    """
    Identifies the appropriate event type calls its from_dict() method.

    Parameters
    ----------
    event_dict : dict[str, Any]
        Dictionary containing all event fields including event type.

    Raises
    ------
    NotImplementedError
        Is raised in case the event type stored in event_dict does not
        correspond to any event class.

    Returns
    -------
    Event
        The appropriate event constructed from the event_dict fields.
    """
    event_dict = event_dict.copy()
    # Get the event type
    if "event_type" not in event_dict:
        raise KeyError("The input dictionary must have the key 'event_type'.")
    event_type = event_dict.pop("event_type")
    # Check that the event type exists
    try:
        event_class = EVENT_MAP[event_type]
    except KeyError as e:
        raise NotImplementedError(
            f"Event Type '{event_type}' does not exist."
        ) from e
    # Ensure that event dictionary contains all required fields.
    event_class.check_for_missing_keys(event_dict)
    # Call the from_dict() method of the correct event
    return event_class.from_dict(event_dict)


# Add event_from_dict() as class method to the Event base class, so it can
# always called as Event.from_dict(event_dict) with any dictionary as long as
# it contains the event_type field.
Event.from_dict = classmethod(event_from_dict)  # type: ignore
