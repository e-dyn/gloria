"""
Defintion of Regressor classes used by the Gloria Model
"""

### --- Module Imports --- ###
# Standard Library
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Optional, Type

# Third Party
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

# Gloria
from gloria.events import Event

# Inhouse Packages
from gloria.utilities.constants import _DELIM


### --- Class and Function Definitions --- ###
class Regressor(BaseModel, ABC):
    """
    Base class for adding regressors to the Gloria model and creating the
    respective feature matrix
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # class attributes that all regressors have in common
    name: str
    prior_scale: float = Field(gt=0)

    @property
    def _regressor_type(self: Self) -> str:
        """
        Returns name of the regressor class.
        """
        return type(self).__name__

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Regressor to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields including regressor type
        """

        regressor_dict = {
            k: self.__dict__[k] for k in Regressor.model_fields.keys()
        }

        # Add regressor_type holding the regressor class name.
        regressor_dict["regressor_type"] = self._regressor_type

        return regressor_dict

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Forward declaration of class method for static type checking.
        See details in regressor_from_dict().
        """
        pass

    @classmethod
    def check_for_missing_keys(
        cls: Type[Self], regressor_dict: dict[str, Any]
    ) -> None:
        """
        Confirms that all required fields for the requested regressor type are
        found in the regressor dictionary.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Raises
        ------
        KeyError
            Raised if any keys are missing

        Returns
        -------
        None
        """
        # Use sets to find the difference between regressor model fields and
        # passed dictionary keys
        required_fields = {
            name
            for name, info in cls.model_fields.items()
            if info.is_required()
        }
        missing_keys = required_fields - set(regressor_dict.keys())
        # If any is missing, raise an error.
        if missing_keys:
            missing_keys_str = ", ".join([f"'{key}'" for key in missing_keys])
            raise KeyError(
                f"Key(s) {missing_keys_str} required for regressors"
                f" of type {cls.__name__} but not found in "
                "regressor dictionary."
            )

    @abstractmethod
    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix along with prior scales for a given integer
        time vector

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated
        regressor: pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for ExternalRegressor

        Raises
        ------
        NotImplementedError
            In case the child regressor did not implement the make_feature()
            method yet

        Returns
        -------
        pd.DataFrame
            Contains the feature matrix
        dict
            A map for 'feature matrix column name' -> 'prior_scale'
        """
        pass


class ExternalRegressor(Regressor):
    """
    Used to add external regressors to the Gloria model and create its
    feature matrix
    """

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates ExternalRegressor instance from a dictionary that holds the
        regressor fields.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        ExternalRegressor
            ExternalRegressor instance with fields from regressor_dict
        """
        return cls(**regressor_dict)

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix along with prior scales for a given integer
        time vector

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for ExternalRegressor

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        """
        if not isinstance(regressor, pd.Series):
            raise TypeError("External Regressor must be pandas Series.")

        # the provided regressor must have a value for each timestamp
        if t.shape[0] != regressor.shape[0]:
            raise ValueError(
                f"Provided data for extra Regressor {self.name}"
                " do not have same length as timestamp column."
            )
        # Prepare the outputs
        column = f"{self._regressor_type}{_DELIM}{self.name}"
        X = pd.DataFrame({column: regressor.values})
        prior_scales = {column: self.prior_scale}
        return X, prior_scales


class Seasonality(Regressor):
    """
    Used to add a seasonality regressors to the Gloria model and create its
    feature matrix

    Important: Period is unitless. That is, when called from Gloria, it will
    make seasonality features with a period in units of 1/sampling_frequency.
    """

    # Fundamental period in units of 1/sampling_frequency
    period: float = Field(gt=0)
    # Order up to which fourier components will be added to the feature matrix
    fourier_order: int = Field(ge=1)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Seasonality regressor to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields
        """
        # Parent class converts basic fields
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["period"] = self.period
        regressor_dict["fourier_order"] = self.fourier_order
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates Seasonality regressor instance from a dictionary that holds the
        regressor fields.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        Seasonality
            Seasonality regressor instance with fields from regressor_dict
        """
        return cls(**regressor_dict)

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix along with prior scales for a given integer
        time vector

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated. The timestamps have to be represented as integers.
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for ExternalRegressor

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        """
        # First construct column names, Note that in particular 'odd' and
        # 'even' must follow the same order as they are returned by
        # self.fourier_series()
        orders_str = map(str, range(1, self.fourier_order + 1))
        columns = [
            _DELIM.join(x)
            for x in product(
                [self._regressor_type],
                [self.name],
                ["odd", "even"],
                orders_str,
            )
        ]
        # Create the feature matrix
        X = pd.DataFrame(
            data=self.fourier_series(
                np.asarray(t), self.period, self.fourier_order
            ),
            columns=columns,
        )
        # Prepare prior_scales
        prior_scales = {col: self.prior_scale for col in columns}
        return X, prior_scales

    @staticmethod
    def fourier_series(
        t: np.ndarray, period: float, max_fourier_order: int
    ) -> np.ndarray:
        """
        Create a (2 X max_fourier_order) column array that contains alternating
        odd and even terms of fourier components up to the maximum order

        Parameters
        ----------
        t : np.ndarray
            Integer array at which the fourier Components are to be evaluated
        period : float
            Period duration in units of the integer array
        max_fourier_order : int
            Maximum order up to which Fourier components will be created

        Returns
        -------
        np.ndarray
            The array containing the Fourier components

        """
        # Calculate angular frequency
        w0 = 2 * np.pi / period
        # Two matrices of even and odd terms from fundamental mode up to
        # specified max_fourier_order
        odd = np.sin(
            w0 * t.reshape(-1, 1) * np.arange(1, max_fourier_order + 1)
        )
        even = np.cos(
            w0 * t.reshape(-1, 1) * np.arange(1, max_fourier_order + 1)
        )
        return np.hstack([odd, even])


class EventRegressor(Regressor):
    """
    A base class used to create a regressor based on an event
    """

    # Each EventRegressor must be associated with exactly one event
    event: Event

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the EventRegressor to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields
        """
        # Parent class converts basic fields
        regressor_dict = super().to_dict()
        # Additionally convert the event
        regressor_dict["event"] = self.event.to_dict()
        return regressor_dict

    @abstractmethod
    def get_impact(self: Self, t: pd.Series) -> float:
        """
        Calculates the fraction of overall events within the timestamp range
        """
        pass


class SingleEvent(EventRegressor):
    """
    An EventRegressor that produces the event exactly once at a given time.
    """

    # Single timestamp at which the event occurs
    t_start: pd.Timestamp

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the SingleEvent regressor to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["t_start"] = str(self.t_start)
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates SingleEvent regressor instance from a dictionary that holds the
        regressor fields.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        SingleEvent
            SingleEvent regressor instance with fields from regressor_dict
        """
        # Convert non-built-in types
        regressor_dict["t_start"] = pd.Timestamp(regressor_dict["t_start"])
        regressor_dict["event"] = Event.from_dict(regressor_dict["event"])
        return cls(**regressor_dict)

    def get_impact(self: Self, t: pd.Series) -> float:
        """
        Calculates the fraction of overall events within the timestamp range.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

        Returns
        -------
        impact : float
            Fraction of overall events within the timestamp range

        """
        impact = float(t.min() <= self.t_start <= t.max())
        return impact

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix along with prior scales for a given timestamp
        series.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for ExternalRegressor

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        """

        # First construct column name
        column = (
            f"{self._regressor_type}{_DELIM}{self.event._event_type}"
            f"{_DELIM}{self.name}"
        )
        # Create the feature matrix
        X = pd.DataFrame({column: self.event.generate(t, self.t_start)})
        # Prepare prior_scales
        prior_scales = {column: self.prior_scale}
        return X, prior_scales


class IntermittentEvent(EventRegressor):
    """
    An EventRegressor that produces the event at times given through a list.
    """

    # A list of timestamps at which the base events occur.
    t_list: list[pd.Timestamp] = []

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the IntermittentEvent regressor to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["t_list"] = [str(t) for t in self.t_list]
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates IntermittentEvent regressor instance from a dictionary that
        holds the regressor fields.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        IntermittentEvent
            IntermittentEvent regressor instance with fields from
            regressor_dict
        """
        # Convert non-built-in
        regressor_dict["event"] = Event.from_dict(regressor_dict["event"])
        # As t_list is optional, check if it is present
        if "t_list" in regressor_dict:
            try:
                regressor_dict["t_list"] = [
                    pd.Timestamp(t) for t in regressor_dict["t_list"]
                ]
            except Exception as e:
                raise TypeError(
                    "Field 't_list' of IntermittentEvent regressor must be a "
                    "list of objects that can be cast to a pandas timestamp."
                ) from e
        return cls(**regressor_dict)

    def get_impact(self: Self, t: pd.Series) -> float:
        """
        Calculates the fraction of overall events within the timestamp range.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

        Returns
        -------
        impact : float
            Fraction of overall events within the timestamp range

        """
        # In case no event is in the list, return zero to signal that no event
        # will be fitted
        if len(self.t_list) == 0:
            return 0.0
        # Count instances in t_list that are within the timestamp range
        impact = sum(float(t.min() <= t0 <= t.max()) for t0 in self.t_list)
        impact /= len(self.t_list)
        return impact

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix along with prior scales for a given timestamp
        series.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for ExternalRegressor

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        """

        # Drop index to ensure t aligns with all_events
        t = t.reset_index(drop=True)

        # First construct column name
        column = (
            f"{self._regressor_type}{_DELIM}{self.event._event_type}"
            f"{_DELIM}{self.name}"
        )

        # Loop through all start times in t_list, and accumulate the events
        all_events = pd.Series(0, index=range(t.shape[0]))

        for t_start in self.t_list:
            all_events += self.event.generate(t, t_start)

        # Create the feature matrix
        X = pd.DataFrame({column: all_events})

        # Prepare prior_scales
        prior_scales = {column: self.prior_scale}
        return X, prior_scales


class PeriodicEvent(SingleEvent):
    """
    An EventRegressor that produces periodically recurring events
    """

    # The periodicity of the base event
    period: pd.Timedelta

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the PeriodicEvent regressor to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["period"] = str(self.period)
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates PeriodicEvent regressor instance from a dictionary that
        holds the regressor fields.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        PeriodicEvent
            PeriodicEvent regressor instance with fields from
            regressor_dict
        """
        # Convert non-built-in fields
        regressor_dict["t_start"] = pd.Timestamp(regressor_dict["t_start"])
        regressor_dict["period"] = pd.Timedelta(regressor_dict["period"])
        regressor_dict["event"] = Event.from_dict(regressor_dict["event"])
        return cls(**regressor_dict)

    def get_t_list(self: Self, t: pd.Series) -> list[pd.Timestamp]:
        """
        Calculate all timestamps of periods within the range of input
        timestamps.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

        Returns
        -------
        t_list : list[pd.Timestamp]

        """
        # Calculate number of periods with respect to t_start necessary to
        # cover the entire given timestamp range.
        n_min = (t.min() - self.t_start) // self.period
        n_max = (t.max() - self.t_start) // self.period

        # Generate list of event start times
        t_list = [
            self.t_start + n * self.period for n in range(n_min, n_max + 1)
        ]
        # Remove timestamps outside of range of t
        t_list = [t0 for t0 in t_list if t.min() <= t0 <= t.max()]
        return t_list

    def get_impact(self: Self, t: pd.Series) -> float:
        """
        Calculates the fraction of overall events within the timestamp range.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

        Returns
        -------
        impact : float
            Fraction of overall events within the timestamp range

        """

        # Generate list of event start times
        t_list = self.get_t_list(t)

        # In case no event is in the list, return zero to signal that no event
        # will be fitted
        if len(t_list) == 0:
            return 0.0

        # Count instances in t_list that are within the timestamp range
        impact = sum(float(t.min() <= t0 <= t.max()) for t0 in t_list)
        impact /= len(t_list)
        return impact

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix along with prior scales for a given timestamp
        series.

        Parameters
        ----------
        t: pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for ExternalRegressor

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        """

        # Drop index to ensure t aligns with all_events
        t = t.reset_index(drop=True)

        # First construct column name
        column = (
            f"{self._regressor_type}{_DELIM}{self.event._event_type}"
            f"{_DELIM}{self.name}"
        )

        # Generate list of event start times
        t_list = self.get_t_list(t)

        # Loop through all start times in t_list, and accumulate the events
        all_events = pd.Series(0, index=range(t.shape[0]))
        for t_start in t_list:
            all_events += self.event.generate(t, t_start)

        # Create the feature matrix
        X = pd.DataFrame({column: all_events})
        # Prepare prior_scales
        prior_scales = {column: self.prior_scale}
        return X, prior_scales


# A map of Regressor class names to actual classes
def get_regressor_map() -> dict[str, Type[Regressor]]:
    """
    Returns a dictionary mapping regressor names as strings to actual classes.
    Creating of this map is encapsulated as function to avoid circular imports
    of the protocol modules and a number of linting errors.

    Returns
    -------
    regressor_map : dict[str, Regressor]
        A map 'protocol name' -> 'protocol class'

    """
    # Before creating the regressor map, import regressors that have been
    # defined in other modules
    # Gloria
    from gloria.protocols.calendric import Holiday

    # Create the map
    regressor_map: dict[str, Type[Regressor]] = {
        "Holiday": Holiday,
        "ExternalRegressor": ExternalRegressor,
        "Seasonality": Seasonality,
        "SingleEvent": SingleEvent,
        "IntermittentEvent": IntermittentEvent,
        "PeriodicEvent": PeriodicEvent,
    }
    return regressor_map


REGRESSOR_MAP = get_regressor_map()

# Filter those Regressors that are EventRegressors
EVENT_REGRESSORS = [
    k
    for k, v in REGRESSOR_MAP.items()
    if (issubclass(v, EventRegressor)) and (v != EventRegressor)
]


def regressor_from_dict(
    cls: Type[Regressor], regressor_dict: dict[str, Any]
) -> Regressor:
    """
    Identifies the appropriate regressor type calls its from_dict() method

    Parameters
    ----------
    regressor_dict : dict[str, Any]
        Dictionary containing all regressor fields including regressor type

    Raises
    ------
    NotImplementedError
        Is raised in case the regressor type stored in regressor_dict does not
        correspond to any regressor class

    Returns
    -------
    Regressor
        The appropriate regressor constructed from the regressor_dict fields.
    """
    regressor_dict = regressor_dict.copy()
    # Get the regressor type
    if "regressor_type" not in regressor_dict:
        raise KeyError(
            "The input dictionary must have the key 'regressor_type'."
        )
    regressor_type = regressor_dict.pop("regressor_type")
    # Check that the regressor type exists
    try:
        regressor_class = REGRESSOR_MAP[regressor_type]
    except KeyError as e:
        raise NotImplementedError(
            f"Regressor Type '{regressor_type}' does not exist."
        ) from e
    # Ensure that regressor dictionary contains all required fields.
    regressor_class.check_for_missing_keys(regressor_dict)
    # Call the from_dict() method of the correct regressor
    return regressor_class.from_dict(regressor_dict)


# Add regressor_from_dict() as class method to the Regressor base class, so
# it can always called as Regressor.from_dict(regressor_dict) with any
# dictionary as long as it contains the regressor_type field.
Regressor.from_dict = classmethod(regressor_from_dict)  # type: ignore
