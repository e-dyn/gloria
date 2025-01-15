# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:19:36 2024

@author: BeKa
"""

### --- Module Imports --- ###
# Standard Library
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Literal


# Third Party
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Self

# Inhouse Packages
from gloria.constants import _DELIM
from gloria.utilities import time_to_integer


def scale_holiday_dates(
        holidays: dict,
        t0: pd.Timestamp,
        sampling_delta: pd.Timedelta, 
        ):
    """
    Updates the 'date', 'lead_window' and 'lag_window' for all holidays 
    according to the sampling_delta
    """

    for holiday, data in holidays.items():
        date = data['date']
        lead = data['lead_window']
        lag= data['lag_window']
        data['date'] = time_to_integer(
                    time = date, 
                    t0 = t0, 
                    sampling_delta = sampling_delta
                    )
        data['lead_window'] = data['date'] - time_to_integer(
                    time = date - pd.Timedelta(f'{lead}d'), 
                    t0 = t0, 
                    sampling_delta = sampling_delta
                    )
        data['lag_window'] = time_to_integer(
                    time = date + pd.Timedelta(f'{lag}d'), 
                    t0 = t0, 
                    sampling_delta = sampling_delta
                    ) - data['date']

### --- Global Constants Definitions --- ###

### --- Class and Function Definitions --- ###
class AbstractRegressor(BaseModel, ABC):
    """
    Base class for adding regressors to the Gloria model and creating the
    respective feature matrix
    """
    # class attributes that all regressors have in common
    name: str
    prior_scale: float = Field(gt=0)
    mode: Literal['additive', 'multiplicative']

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
            The input timestamps as integers

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

        
class ExternalRegressor(AbstractRegressor):
    """
    Used to add external regressors to the Gloria model and create its
    feature matrix
    """
    
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
            The input timestamps as integers
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
        column = f'external{_DELIM}{self.name}'
        X = pd.DataFrame({column: regressor.values})
        prior_scales = {column: self.prior_scale}
        modes = {column: self.mode}
        return X, prior_scales, modes
        

class Seasonality(AbstractRegressor):
    """
    Used to add a seasonality regressors to the Gloria model and create its
    feature matrix
    
    Important: Period is unitless. That is, when called from Gloria, it will
    make seasonality features with a period in units of 1/sampling_frequency.
    """

    # Extra regressor parameters for seasonalities
    # Fundamental period in units of 1/sampling_frequency
    period: float = Field(gt = 0)
    # Order up to which fourier components will be added to the feature matrix
    fourier_order: int = Field(ge = 1)
    
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
            The input timestamps as integers

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
                ['season'],
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
    


class Holidays(AbstractRegressor):
    """
    TODO:
        - t0 und sampling_delta als Parameter der make_feature Methode, da
          diese vom Gloria Modell und Trainingsdaten abhängen
        - Feiertage müssen sich jährlich wiederholen innerhalb einer Spalte. Am
          besten indem eine Liste mit allen Feiertagen übergeben wird, d.h.
          ["24.12.2022", "24.12.2023", "24.12.2024"]
    """

    date: pd.Timestamp
    t0: pd.Timestamp
    sampling_delta: pd.Timedelta 
    lead_window: int = Field(ge = 0)
    lag_window: int = Field(ge = 0)
    
    class Config:
        arbitrary_types_allowed=True

    def make_feature(
            self: Self,
            x: pd.Series
        ) -> tuple[pd.DataFrame, dict, dict]:

        # Define column names
        column  = f'holiday{_DELIM}{self.name}'
        orders_str = map(str, range(-self.lead_window, self.lag_window+1))
        columns = [ 
            _DELIM.join(x) for x in product(
                ['holiday'],
                [self.name],
                orders_str
            )
        ]

        # Create the DataFrame with all values set to zero
        X = pd.DataFrame(0, index=range(x.shape[0]), columns=columns)

        # Fill in the corresponding lines with ones
        for col in X.columns:
            window_value = int(col.split(_DELIM)[-1])
            idx_min = time_to_integer(
                time = self.date + pd.Timedelta(window_value, 'd'), 
                t0 = self.t0, 
                sampling_delta = sampling_delta
            )
            idx_max = time_to_integer(
                time = self.date + pd.Timedelta(window_value + 1, 'd'), 
                t0 = self.t0, 
                sampling_delta = sampling_delta
            )
            if 0 <= idx_min < X.shape[0]:  # Ensure that the index is valid
                X.loc[idx_min:max(idx_min,idx_max-1), col] = 1
           
           
        prior_scales = {column: self.prior_scale}
        modes = {column: self.mode}
        return X, prior_scales, modes 
    


### --- Main Script --- ###
if __name__ == "__main__":
    TEST_SEASONALITIES = {
        'weekly': {
            'period': 14,
            'fourier_order': 3,
            'prior_scale': 3.,
            'mode': 'additive'
        },
        'quarterly': {
            'period': 365.25/2,
            'fourier_order': 3,
            'prior_scale': 5.0,
            'mode': 'additive'
        }
    }

    TEST_HOLIDAYS = {
        'Christmas': {
            'date': pd.Timestamp("2020-12-24"),
            'lead_window': 0,
            'lag_window': 2,
            'mode': 'additive',
            'prior_scale': 5.0,
        },
        'Easter': {
            'date': pd.Timestamp("2020-04-12"),
            'lead_window': 2,
            'lag_window': 1,
            'mode': 'additive',
            'prior_scale': 3.0,
        }
    }


    basepath = Path(__file__).parent
    
    sampling_delta = pd.Timedelta(2, 'd')
    holiday_sampling_factor = pd.Timedelta(1, 'd') / sampling_delta


    t = pd.Series(pd.date_range(
        start='2/1/2020',
        periods = 600,
        freq = '2 d'
    ))
    
    t_int = time_to_integer(
        time = t, 
        t0 = t.min(), 
        sampling_delta = sampling_delta
    )

    X = []
    prior_scales = dict()
    modes = dict()
    for name, props in TEST_HOLIDAYS.items():
        holidays = Holidays(
            name = name,
            prior_scale = props['prior_scale'],
            mode = props['mode'],
            date = props['date'], 
            t0 = t.min(),
            sampling_delta = sampling_delta,
            lead_window = props['lead_window'],
            lag_window = props['lag_window']
        )
        X_loc, prior_scales_loc, modes_loc = holidays.make_feature(t_int)
        X.append(X_loc)
        prior_scales = {**prior_scales, **prior_scales_loc}
        modes = {**modes, **modes_loc}
        
    X = pd.concat(X, axis = 1)
    X2=X.merge(t.to_frame(), left_index=True, right_index=True)
    
    