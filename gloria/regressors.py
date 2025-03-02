"""
TODO:
    - Include checkups that verify that an event has been trained. That 
      requires a flag per Regressor whether it was trained or not, a flag
      that tells the make_feature() method whether it was called from the
      Gloria.fit() method, as well as a routine that checks whether the 
      training timestamps properly cover the regressor, e.g. that at least one
      full cycle is covered. If not, set the has_been_trained flag to False and
      exclude the regressor from training. This is an important safe-guard: in
      the most extreme case of a regressor being completely out of the time
      range, the respective regressor matrix column is entirely zero, causing
      the fit to fail.
    - For consistency, The currently unitless period attribute of the seasonality
      regressor should be changed to type pd.Timedelta. Note that make_feauture
      and fourier_series methods have to be adapted, too, as well as the 
      respective methods of the Gloria interface.
    - Add Rule-based quasi-periodic EventRegressor that accepts strings like 
      "every first of the month" and parsing these rules into lists of 
      timestamps during make_feature
    - This affects all EventRegressors: if an event is long (high sigma, duration,...)
      it might leak into the timestamp range even the center of the event is
      outside. Therefore, it is usefull to define a rule that decides how far
      an event can lie outside the timestamp range without being discarded.
"""

### --- Module Imports --- ###
# Standard Library
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Any, Type, Union

# Third Party
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Self

# Inhouse Packages
from gloria.constants import _DELIM, _HOLIDAY
from gloria.events import Event, Gaussian, BoxCar


### --- Class and Function Definitions --- ###
class Regressor(BaseModel, ABC):
    """
    Base class for adding regressors to the Gloria model and creating the
    respective feature matrix
    """
    # class attributes that all regressors have in common
    name: str
    prior_scale: float = Field(gt=0)
    mode: Literal['additive', 'multiplicative']
    
    class Config:
        arbitrary_types_allowed=True
    
    
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
        regressor_dict['regressor_type'] = self._regressor_type
        
        return regressor_dict
    
    
    @classmethod
    def check_for_missing_keys(
            cls: Type[Self],
            regressor_dict: dict[str, Any]
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
        required_fields = {name for name, info in cls.model_fields.items()
                           if info.is_required()}
        missing_keys = required_fields - set(regressor_dict.keys())
        # If any is missing, raise an error.
        if missing_keys:
            missing_keys = ', '.join([f"'{key}'" for key in missing_keys])
            raise KeyError(f"Key(s) {missing_keys} required for regressors"
                           f" of type {cls.__name__} but not found in "
                           "regressor dictionary.")


    @abstractmethod
    def make_feature(
            self: Self,
            t: pd.Series
        ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Create the feature matrix along with prior scales and modes for a given
        integer time vector

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

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
        dict : TYPE
            A map for 'feature matrix column name' -> 'mode'
        """
        raise NotImplementedError("make_feature() method not implemented.")
        return pd.DataFrame(), dict(), dict()


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
        # Ensure that regressor dictionary contains all required fields.
        cls.check_for_missing_keys(regressor_dict)
        return cls(**regressor_dict)
    
    
    def make_feature(
            self: Self,
            t: pd.Series,
            regressor: pd.Series
        ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Create the feature matrix along with prior scales and modes for a given
        integer time vector

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        modes : dict
            A map for 'feature matrix column name' -> 'mode'
        """
        
        # the provided regressor must have a value for each timestamp
        if t.shape[0] != regressor.shape[0]:
            raise ValueError(f"Provided data for extra Regressor {self.name}"
                             " do not have same length as timestamp column.")
        # Prepare the outputs
        column = f'{self._regressor_type}{_DELIM}{self.name}'
        X = pd.DataFrame({column: regressor.values})
        prior_scales = {column: self.prior_scale}
        modes = {column: self.mode}
        return X, prior_scales, modes
        

class Seasonality(Regressor):
    """
    Used to add a seasonality regressors to the Gloria model and create its
    feature matrix
    
    Important: Period is unitless. That is, when called from Gloria, it will
    make seasonality features with a period in units of 1/sampling_frequency.
    """
    # Fundamental period in units of 1/sampling_frequency
    period: float = Field(gt = 0)
    # Order up to which fourier components will be added to the feature matrix
    fourier_order: int = Field(ge = 1)
    
    
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
        regressor_dict['period'] = self.period
        regressor_dict['fourier_order'] = self.fourier_order
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
        # Ensure that regressor dictionary contains all required fields.
        cls.check_for_missing_keys(regressor_dict)
        return cls(**regressor_dict)
    
    
    def make_feature(
            self: Self,
            t: pd.Series
        ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Create the feature matrix along with prior scales and modes for a given
        integer time vector

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated. The timestamps have to be represented as integers.

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        modes : dict
            A map for 'feature matrix column name' -> 'mode'
        """
        # First construct column names, Note that in particular 'odd' and
        # 'even' must follow the same order as they are returned by
        # self.fourier_series()
        orders_str = map(str, range(1, self.fourier_order+1))
        columns = [ 
            _DELIM.join(x) for x in product(
                [self._regressor_type],
                [self.name],
                ['odd', 'even'],
                orders_str
            )
        ]
        # Create the feature matrix
        X = pd.DataFrame(
            data = self.fourier_series(
                t.values,
                self.period,
                self.fourier_order
            ),
            columns = columns
        )
        # Prepare prior_scales and modes
        prior_scales = {col: self.prior_scale for col in columns}
        modes = {col: self.mode for col in columns}
        return X, prior_scales, modes
    
    
    @staticmethod
    def fourier_series(
            t: np.ndarray,
            period: float,
            max_fourier_order: int
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
        w0 = 2*np.pi / period
        odd = np.sin(w0 * t.reshape(-1,1) * np.arange(1, max_fourier_order+1))
        even = np.cos(w0 * t.reshape(-1,1) * np.arange(1, max_fourier_order+1))
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
        regressor_dict['event'] = self.event.to_dict()
        return regressor_dict
    

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
        regressor_dict['t_start'] = str(self.t_start)
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
        # Ensure that regressor dictionary contains all required fields.
        cls.check_for_missing_keys(regressor_dict)
        # Convert non-built-in types
        regressor_dict['t_start'] = pd.Timestamp(regressor_dict['t_start'])
        regressor_dict['event'] = Event.from_dict(regressor_dict['event'])
        return cls(**regressor_dict)


    def make_feature(
            self: Self,
            timestamps: pd.Series
        ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Create the feature matrix along with prior scales and modes for a given
        timestamp series.

        Parameters
        ----------
        timestamps : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        modes : dict
            A map for 'feature matrix column name' -> 'mode'
        """
        
        # First construct column name
        column  = (
            f'{self._regressor_type}{_DELIM}{self.event._event_type}'
            f'{_DELIM}{self.name}'
        )
        # Create the feature matrix
        X = pd.DataFrame({
            column: self.event.generate(timestamps, self.t_start)
        })
        # Prepare prior_scales and modes
        prior_scales = {column: self.prior_scale}
        modes = {column: self.mode}
        return X, prior_scales, modes
    

class IntermittentEvent(EventRegressor):
    """
    An EventRegressor that produces the event at times given through a list.
    """
    # A list of timestamps at which the base events occur.
    t_list: Optional[list[pd.Timestamp]] = []


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
        regressor_dict['t_list'] = [str(t) for t in self.t_list]
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
        # Ensure that regressor dictionary contains all required fields.
        cls.check_for_missing_keys(regressor_dict)
            
        # Convert non-built-in fields
        regressor_dict['t_list'] = [
            pd.Timestamp(t) for t in regressor_dict['t_list']
        ]
        regressor_dict['event'] = Event.from_dict(regressor_dict['event'])
        return cls(**regressor_dict)
        
    
    def make_feature(
            self: Self,
            timestamps: pd.Series
        ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Create the feature matrix along with prior scales and modes for a given
        timestamp series.

        Parameters
        ----------
        timestamps : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        modes : dict
            A map for 'feature matrix column name' -> 'mode'
        """
        
        # First construct column name
        column  = (
            f'{self._regressor_type}{_DELIM}{self.event._event_type}'
            f'{_DELIM}{self.name}'
        )
        
        # Loop through all start times in t_list, and accumulate the events
        all_events = pd.Series(0, index=range(timestamps.shape[0]))
        for t_start in self.t_list:
            all_events += self.event.generate(timestamps, t_start)
        
        # Create the feature matrix
        X = pd.DataFrame({column: all_events})
        # Prepare prior_scales and modes
        prior_scales = {column: self.prior_scale}
        modes = {column: self.mode}
        return X, prior_scales, modes


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
        regressor_dict['period'] = str(self.period)
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
        # Ensure that regressor dictionary contains all required fields.
        cls.check_for_missing_keys(regressor_dict)
        # Convert non-built-in fields
        regressor_dict['t_start'] = pd.Timestamp(regressor_dict['t_start'])
        regressor_dict['period'] = pd.Timedelta(regressor_dict['period'])
        regressor_dict['event'] = Event.from_dict(regressor_dict['event'])
        return cls(**regressor_dict)
        
    
    def make_feature(
            self: Self,
            timestamps: pd.Series
        ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Create the feature matrix along with prior scales and modes for a given
        timestamp series.

        Parameters
        ----------
        timestamps : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated

        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        modes : dict
            A map for 'feature matrix column name' -> 'mode'
        """
        
        # First construct column name
        column  = (
            f'{self._regressor_type}{_DELIM}{self.event._event_type}'
            f'{_DELIM}{self.name}'
        )

        # Calculate number of periods with respect to t_start necessary to
        # cover the entire given timestamp range. -3 / +4 are safety margins
        # for long events.
        n_min = (timestamps.min() - self.t_start) // self.period - 3
        n_max = (timestamps.max() - self.t_start) // self.period + 4
        
        # Generate list of event start times
        t_list = [self.t_start + n*self.period for n in range(n_min, n_max+1)]
        
        # Loop through all start times in t_list, and accumulate the events
        all_events = pd.Series(0, index=range(timestamps.shape[0]))
        for t_start in t_list:
            all_events += self.event.generate(timestamps, t_start)
        
        # Create the feature matrix
        X = pd.DataFrame({column: all_events})
        # Prepare prior_scales and modes
        prior_scales = {column: self.prior_scale}
        modes = {column: self.mode}
        return X, prior_scales, modes


# Before creating regressor maps, import regressors that have been defined in
# other modules
from gloria.protocols.calendric import Holiday


# A map of Regressor class names to actual classes
REGRESSOR_MAP = dict()
for k, v in globals().copy().items():
    try:
        if issubclass(v, Regressor) and (v not in (Regressor, EventRegressor)):
            REGRESSOR_MAP[k] = v
    except:
        pass

# Filter those Regressors that are EventRegressors
EVENT_REGRESSORS = [
    k for k, v in REGRESSOR_MAP.items()
    if (issubclass(v, EventRegressor)) and (v != EventRegressor)
]


def regressor_from_dict(
        cls: Type[Self],
        regressor_dict: dict[str, Any]
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
    if 'regressor_type' not in regressor_dict:
        raise KeyError("The input dictionary must have the key"
                       " 'regressor_type'")
    regressor_type = regressor_dict.pop('regressor_type')
    # Check that the regressor type exists
    if regressor_type not in REGRESSOR_MAP:
        raise NotImplementedError(f"Regressor Type {regressor_type} does not"
                                  " exist.")
    # Call the from_dict() method of the correct regressor
    return REGRESSOR_MAP[regressor_type].from_dict(regressor_dict)

# Add regressor_from_dict() as class method to the Regressor base class, so
# it can always called as Regressor.from_dict(regressor_dict) with any
# dictionary as long as it contains the regressor_type field.
Regressor.from_dict = classmethod(regressor_from_dict)
