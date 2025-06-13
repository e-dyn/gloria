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
from gloria.utilities.types import Timestamp


### --- Class and Function Definitions --- ###
class Regressor(BaseModel, ABC):
    """
    Base class for adding regressors to the Gloria model and creating the
    respective feature matrix

    Parameters
    ----------
    name : str
        A descriptive, unique name to identify the regressor
    prior_scale : float
        Parameter modulating the strength of the regressors. Larger values
        allow the model to fit larger impact, smaller values dampen the impact.
        Must be larger than zero.
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
        Converts the regressor to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields including an extra
            ``regressor_type`` key with the class name as value.
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
    A regressor based on user-provided data.

    The regressor is added to the :class:`Gloria` model using
    :meth:`~Gloria.add_external_regressor` and does not need to be handled
    directly by the user. Instead of synthesizing the regressor data, they must
    be provided to :meth:`~Gloria.fit` as part of the input data frame.

    Parameters
    ----------
    name : str
        A descriptive, unique name to identify the regressor
    prior_scale : float
        Parameter modulating the strength of the regressors. Larger values
        allow the model to fit larger impact, smaller values dampen the impact.
        Must be larger than zero.
    """

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates an ExternalRegressor object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the regressor.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields.

        Returns
        -------
        ExternalRegressor
            ExternalRegressor instance with fields from ``regressor_dict``.
        """
        return cls(**regressor_dict)

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Creates the feature matrix for the external regressor.


        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated. For ``ExternalRegressor`` this is only used to validate
            that the input ``regressor`` data and timestamps ``t`` have
            identical shapes.
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged.

        Returns
        -------
        X : pd.DataFrame
            The feature matrix containing the data of the regressor.
        prior_scales : dict
            A map for ``feature matrix column name`` → ``prior_scale``
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
    A regressor to model seasonality features from Fourier components.

    The regressor is added to the :class:`Gloria` model using
    :meth:`~Gloria.add_seasonality` and does not need to be handled
    directly by the user. The feature matrix produced by
    :meth:`~Seasonality.make_feature` contains :math:`2 \\cdot N` columns
    corresponding to the even and odd Fourier terms

    .. math::
        \\sum_{n=1}^{N}{\\sin\\left(\\frac{2\\pi n}{T} t\\right)
                        + \\cos\\left(\\frac{2\\pi n}{T} t\\right)}

    where :math:`T` is the fundamental Fourier period and :math:`N` is the
    maximum Fourier order to be included, controlled by the parameters
    ``period`` and ``order``, respectively.

    Parameters
    ----------
    name : str
        A descriptive, unique name to identify the regressor.
    prior_scale : float
        Parameter modulating the strength of the regressors. Larger values
        allow the model to fit larger seasonal oscillations, smaller values
        dampen the impact. Must be larger than zero.
    period : float
        Fundamental period of the seasonality component. Note that the period
        is unitless. It can be understood in units of ``sampling_period`` of
        the :class:`~gloria.Gloria` owning this seasonality. Must be larger
        than zero.
    fourier_order : int
        Maximum Fourier order of the underlying series. Even and odd Fourier
        terms from fundamental up to ``fourier_order`` will be used as
        regressors. Must be larger or equal to 1.


    .. warning::
        In a future version of Gloria, ``period`` will become a
        :class:`pandas.Timestamp` or ``str`` representing such. Where possible
        use :meth:`~Gloria.add_seasonality` instead of :class:`Seasonality`
        to avoid conflict.

    """

    # Fundamental period in units of 1/sampling_frequency
    period: float = Field(gt=0)
    # Order up to which fourier components will be added to the feature matrix
    fourier_order: int = Field(ge=1)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Seasonality regressor to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields including an extra
            ``regressor_type = "Seasonality"`` item.
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
        Creates an Seasonality object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the regressor.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields.

        Returns
        -------
        Seasonality
            Seasonality regressor instance with fields from ``regressor_dict``.
        """
        return cls(**regressor_dict)

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix for the seasonality regressor.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated. The timestamps have to be represented as integers in
            units of their sampling frequency.
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for
            :class:`ExternalRegressor`. Any input will be ignored for
            :class:`Seasonality`.

        Returns
        -------
        X : pd.DataFrame
            The feature matrix containing the data of the regressor.
        prior_scales : dict
            A map for ``feature matrix column name`` → ``prior_scale``


        .. warning::
            In a future version of Gloria, ``period`` will become a
            :class:`pandas.Timestamp` or ``str`` representing such and ``t``
            will be a :class:`pandas.Series` of timestamps.

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
        t: np.ndarray, period: float, fourier_order: int
    ) -> np.ndarray:
        """
        Creates an array of even and odd Fourier terms.

        The :class:`numpy.ndarray` output array has the following structure:

        * **Columns**: alternatingely odd and even Fourier terms up to the
          given maximum ``fourier_order``, resulting in :math:`2\\times`
          ``fourier_order`` columns.
        * **Rows**: Fourier terms evaluated at each timestamp, resulting in
          ``len(t)`` rows.


        Parameters
        ----------
        t : np.ndarray
            Integer array at which the Fourier components are evaluated.
        period : float
            Period duration in units of the integer array.
        fourier_order : int
            Maximum Fourier order up to which Fourier components will be
            created. Must be larger or equal 1.

        Returns
        -------
        np.ndarray
            The array containing the Fourier components


        .. warning::
            In a future version of Gloria, ``period`` will become a
            :class:`pandas.Timestamp` or ``str`` representing such and ``t``
            will be a :class:`pandas.Series` of timestamps.

        """
        # Calculate angular frequency
        w0 = 2 * np.pi / period
        # Two matrices of even and odd terms from fundamental mode up to
        # specified max_fourier_order
        odd = np.sin(w0 * t.reshape(-1, 1) * np.arange(1, fourier_order + 1))
        even = np.cos(w0 * t.reshape(-1, 1) * np.arange(1, fourier_order + 1))
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
    A regressor to model a single occurrence of an event.

    The regressor is added to the :class:`Gloria` model using
    :meth:`~Gloria.add_event` and does not need to be handled
    directly by the user.

    Parameters
    ----------
    name : str
        A descriptive, unique name to identify the regressor.
    prior_scale : float
        Parameter modulating the strength of the regressors. Larger values
        allow the model to fit larger a larger impact of the event, smaller
        values dampen the impact. Must be larger than zero.
    event : Event
        The event that occurs at ``t_start``. Allowed event types are described
        in the :ref:`ref-events` section.
    t_start : :class:`pandas.Timestamp` | str
        The timestamp at which ``event`` occurs. The exact meaning of
        ``t_start`` depends on the implementation details of the underlying
        ``event``, but typically refers to its mode.

    """

    # Single timestamp at which the event occurs
    t_start: Timestamp

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the single event regressor to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields including an extra
            ``regressor_type = "SingleEvent"`` item.
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["t_start"] = str(self.t_start)
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates an SingleEvent object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the regressor.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        SingleEvent
            SingleEvent regressor instance with fields from ``regressor_dict``
        """

        # Convert non-built-in types
        regressor_dict["t_start"] = pd.Timestamp(regressor_dict["t_start"])
        regressor_dict["event"] = Event.from_dict(regressor_dict["event"])
        return cls(**regressor_dict)

    def get_impact(self: Self, t: pd.Series) -> float:
        """
        Calculate fraction of overall events occurring within a timerange.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp`.

        Returns
        -------
        impact : float
            Fraction of overall events occurring between minimum and maximum
            date of ``t``.

        """
        impact = float(t.min() <= self.t_start <= t.max())
        return impact

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix for the single event regressor.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp` at which the regressor has to
            be evaluated
        regressor : :class:`pandas.Series`
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for
            :class:`ExternalRegressor`. Any input will be ignored for
            :class:`SingleEvent`.

        Returns
        -------
        X : :class:`pandas.DataFrame`
            The feature matrix containing the data of the regressor.
        prior_scales : dict
            A map for ``feature matrix column name`` → ``prior_scale``.
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
    A regressor to model reoccuring events at given times.

    The regressor is added to the :class:`Gloria` model using
    :meth:`~Gloria.add_event` and does not need to be handled
    directly by the user.

    Parameters
    ----------
    name : str
        A descriptive, unique name to identify the regressor.
    prior_scale : float
        Parameter modulating the strength of the regressors. Larger values
        allow the model to fit larger a larger impact of the event, smaller
        values dampen the impact. Must be larger than zero.
    event : Event
        The event that occurs at ``t_start``. Allowed event types are described
        in the :ref:`ref-events` section.
    t_list : list[:class:`pandas.Timestamp`] | list[str]
        A list of timestamps at which ``event`` occurs. The exact meaning of
        each timestamp in the list depends on implementation details of the
        underlying ``event``, but typically refers to its mode.

    """

    # A list of timestamps at which the base events occur.
    t_list: list[Timestamp] = []

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the intermittent event regressor to a JSON-serializable
        dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields including an extra
            ``regressor_type = "IntermittentEvent"`` item.
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["t_list"] = [str(t) for t in self.t_list]
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates an IntermittentEvent object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the regressor.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        IntermittentEvent
            IntermittentEvent regressor instance with fields from
            ``regressor_dict``
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
        Calculate fraction of overall events occurring within a timerange.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp`.

        Returns
        -------
        impact : float
            Fraction of overall events occurring between minimum and maximum
            date of ``t``.

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
        Create the feature matrix for the intermittent event regressor.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp` at which the regressor has to
            be evaluated
        regressor : :class:`pandas.Series`
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for
            :class:`ExternalRegressor`. Any input will be ignored for
            :class:`IntermittentEvent`.

        Returns
        -------
        X : :class:`pandas.DataFrame`
            The feature matrix containing the data of the regressor.
        prior_scales : dict
            A map for ``feature matrix column name`` → ``prior_scale``.
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
    A regressor to model periodically recurring events.

    The regressor is added to the :class:`Gloria` model using
    :meth:`~Gloria.add_event` and does not need to be handled
    directly by the user.

    Parameters
    ----------
    name : str
        A descriptive, unique name to identify the regressor.
    prior_scale : float
        Parameter modulating the strength of the regressors. Larger values
        allow the model to fit larger a larger impact of the event, smaller
        values dampen the impact. Must be larger than zero.
    event : Event
        The event that periodically occurs. Allowed event types are described
        in the :ref:`ref-events` section.
    t_start : :class:`pandas.Timestamp`
        An arbitrary timestamp at which ``event`` occurs. The event will be
        repeated forwards and backwards in time every ``period``. The exact
        meaning of ``t_start`` depends on the implementation details of the
        underlying ``event``, but typically refers to its mode.
    period : :class:`pandas.Timedelta`
        Periodicity of the periodic event regressor.

    """

    # The periodicity of the base event
    period: pd.Timedelta

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the periodic event regressor to a JSON-serializable
        dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields including an extra
            ``regressor_type = "PeriodicEvent"`` item.
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["period"] = str(self.period)
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates an PeriodocEvent object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the regressor.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        PeriodicEvent
            PeriodicEvent regressor instance with fields from
            ``regressor_dict``.
        """
        # Convert non-built-in fields
        regressor_dict["t_start"] = pd.Timestamp(regressor_dict["t_start"])
        regressor_dict["period"] = pd.Timedelta(regressor_dict["period"])
        regressor_dict["event"] = Event.from_dict(regressor_dict["event"])
        return cls(**regressor_dict)

    def get_t_list(self: Self, t: pd.Series) -> list[pd.Timestamp]:
        """
        Yields a list of timestamps of period starts within the range of
        input timestamps.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A pandas series of :class:`pandas.Timestamp`.

        Returns
        -------
        t_list : list[:class:`pandas.Timestamp`]
            A list of timestamps of period starts.

        """
        # Calculate number of periods with respect to t_start necessary to
        # cover the entire given timestamp range.
        n_margin = 2
        n_min = (t.min() - self.t_start) // self.period - n_margin
        n_max = (t.max() - self.t_start) // self.period + n_margin

        # Generate list of event start times
        t_list = [
            self.t_start + n * self.period for n in range(n_min, n_max + 1)
        ]
        return t_list

    def get_impact(self: Self, t: pd.Series) -> float:
        """
        Calculate fraction of overall events occurring within a timerange.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp`.

        Returns
        -------
        impact : float
            Fraction of overall events occurring between minimum and maximum
            date of ``t``.

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
        Create the feature matrix for the periodic event regressor.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp` at which the regressor has to
            be evaluated
        regressor : :class:`pandas.Series`
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for
            :class:`ExternalRegressor`. Any input will be ignored for
            :class:`PeriodicEvent`.

        Returns
        -------
        X : :class:`pandas.DataFrame`
            The feature matrix containing the data of the regressor.
        prior_scales : dict
            A map for ``feature matrix column name`` → ``prior_scale``.
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
