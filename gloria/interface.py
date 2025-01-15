"""
TODO:
    - implement plot functions
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path
from typing import Literal, Optional, Any
from time import time

# Third Party
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from gloria.constants import _T_CONVERSION, _DELIM, _T_INT
from gloria.models import ModelInputData, MODEL_MAP, ModelBackend
from gloria.regressors import ExternalRegressor, Seasonality
from gloria.utilities import time_to_integer
from typing_extensions import Self

# Inhouse Packages


### --- Global Constants Definitions --- ###



### --- Class and Function Definitions --- ###
class Gloria(BaseModel):
    """Gloria forecaster.

    Parameters
    ----------
    model : Literal[tuple(MODEL_MAP.keys())]
        The distribution model to be used. Can be any of 'poisson' or
        'binomial constant n'
    sampling_value : int
        Sampling frequency is defined by value and unit of corresponding
        sampling period, e.g. a sampling frequency of 0.5 Hz corresponds to
        a sampling period of 2 seconds [0.5 Hz = 1/(2 s)], leading to 
        sampling_value = 2 and sampling_unit = s.
    sampling_unit : Literal[tuple(_T_CONVERSION.keys())]
        See definition of sampling value. Must be a valid unit for pd.Timedelta
    timestamp_name : str, optional
        The name of the timestamp column as expected in the input data frame
        for the fit-method. The default is 'ds'.
    metric_name : str, optional
        The name of the metric column as expected in the input data frame for
        for the fit-method. The default is 'y'.
    changepoints : pd.Series, optional
        ist of dates at which to include potential changepoints. If not 
        specified (default), potential changepoints are selected automatically.
    n_changepoints : int, optional
        Number of potential changepoints to include. Not used if input 
        'changepoints' is supplied. If 'changepoints' is not supplied, then 
        n_changepoints potential changepoints are selected uniformly from the 
        first 'changepoint_range' proportion of the history. Must be positive
        integer, default is 25.
    changepoint_range : float, optional
        Proportion of history in which trend changepoints will be estimated. 
        Must be in range [0,1]. Defaults to 0.8 for the first 80%. Not used if
        'changepoints' is specified. 
    seasonality_mode : Literal['additive', 'multiplicative'], optional
        Whether seasonal components are treated as additive or multiplicative
        features by default. Can be overwritten if a mode is explicitly
        provided to the regressor. Default is 'additive'.
    seasonality_prior_scale : float, optional
        Parameter modulating the strength of the seasonality model. Larger 
        values allow the model to fit larger seasonal fluctuations, smaller
        values dampen the seasonality. Can be specified for individual
        seasonalities using add_seasonality. 
    changepoint_prior_scale : float, optional
        Parameter modulating the flexibility of the automatic changepoint 
        selection. Large values will allow many changepoints, small values will 
        allow few changepoints. Must be larger than 0. Default is 0.05
    interval_width : float, optional
        Width of the uncertainty intervals provided for the prediction. It is
        used for both uncertainty intervals of the expected value (fit) as
        well as the observed values (observed). Must be in range [0,1].
        Default is 0.8.
    uncertainty_samples : int, optional
        Number of simulated draws used to estimate uncertainty intervals of the
        trend in prediction periods that were not included in the historical
        data. Settings this value to 0 will disable uncertainty estimation.
        Must be greater equal to 0, Default is 1000.
    """
    model: Literal[tuple(MODEL_MAP.keys())]
    sampling_period: str = '1d'
    timestamp_name: str = 'ds'
    metric_name: str = 'y'
    changepoints: pd.Series = Field(default = None)
    n_changepoints: int = Field(ge = 0, default = 25)
    changepoint_range: float = Field(gt = 0, lt = 1, default = 0.8)
    seasonality_mode: Literal['additive', 'multiplicative'] = 'additive'
    seasonality_prior_scale: float = Field(gt = 0, default = 10.0)
    changepoint_prior_scale: float = Field(gt = 0, default = 0.05)
    interval_width: float = Field(gt = 0, lt = 1, default = 0.8)
    uncertainty_samples: int = Field(ge = 0, default = 1000)
    
    class Config:
        extra = 'allow'  # Allows setting extra attributes
        arbitrary_types_allowed = True # So the model accepts pd.Series
    
    
    def __init__(
            self: Self,
            *args: tuple[Any, ...],
            **kwargs: dict[str, Any]
        ) -> None:
        """
        Initializes Gloria object.

        Parameters
        ----------
        *args : tuple[Any, ...]
            Positional arguments passed through to Pydantic Model __init__()
        **kwargs : dict[str, Any]
            Keyword arguments passed through to Pydantic Model __init__()
        """
        # Call the __init__() method of the Pydantic Model for proper
        # initialization and validation
        super().__init__(*args, **kwargs)
        
        # Sanitize provided Changepoints
        if self.changepoints is not None:
            self.changepoints = pd.Series(
                pd.to_datetime(self.changepoints),
                name = self.timestamp_name
            )
            self.n_changepoints = len(self.changepoints)

        # Used for converting the timestamp column to an int colum and back
        # self.sampling_delta = pd.Timedelta(
        #     value = self.sampling_value,
        #     unit = self.sampling_unit
        # )
        self.sampling_delta = pd.to_timedelta(self.sampling_period)
        
        # Load model backend with stan adapter and predict methods
        self.model_backend = ModelBackend(model = self.model)
        
        # Set during fitting or by other methods
        self.external_regressors = []
        self.first_timestamp = None
        self.history = None
        self.modes = dict()
        self.prior_scales = dict()
        self.seasonalities = []
        self.X = pd.DataFrame()
        
        
    def validate_column_name(
            self: Self,
            name: str,
            check_seasonalities: bool = True,
            check_regressors: bool = True
        ) -> None:
        """
        Validates the name of a seasonality or regressor.

        Parameters
        ----------
        name : str
            The name to validate.
        check_seasonalities : bool, optional
            Check if name already used for seasonality. The default is True.
        check_regressors : bool, optional
            Check if name already used for regressor The default is True.

        Raises
        ------
        ValueError
            Raised in case the name is forbidden.
        """

        # The _DELIM constant is used for constructing intermediate column
        # names, hence it's not allowed to be used within given names
        if _DELIM in name:
            raise ValueError(f"Name cannot contain '{_DELIM}'")
        
        # Reserved names are column names generated in the prediction output
        reserved_names = ['fit', 'observed', 'trend']
        rn_l = [n + '_lower' for n in reserved_names]
        rn_u = [n + '_upper' for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend([
            self.timestamp_name,
            self.metric_name,
            _T_INT
        ])
        if name in reserved_names:
            raise ValueError(f"Name {name} is reserved.")
            
        if (check_seasonalities 
            and name in [s.name for s in self.seasonalities]):
            raise ValueError(f"Name {name} already used for a seasonality.")
        if (check_regressors 
            and name in [r.name for r in self.external_regressors]):
            raise ValueError(f"Name {name} already used for a regressor.")
            

    def add_external_regressor(
            self: Self,
            name: str,
            prior_scale: float = Field(gt = 0),
            mode: Optional[Literal['additive', 'multiplicative']] = None
        ) -> Self:
        """
        Add an external regressor to be used for fitting and predicting.

        Parameters
        ----------
        name : str
            Name of the regressor. The dataframe passed to 'fit' and 'predict' 
            must have a column with the specified name to be used as a 
            regressor.
        prior_scale : float
            The regression coefficient is given a prior with the specified 
            scale parameter. Decreasing the prior scale will add additional 
            regularization. Must be greater than 0.
        mode : Optional[Literal['additive', 'multiplicative']], optional
            Whether regressor is treated as additive or multiplicative
            feature. If None is provided (default), self.seasonality_mode is 
            used.

        Raises
        ------
        Exception
            Raised when method is called before fitting.
        ValueError
            Raised when prior scale or mode are not allowed values.

        Returns
        -------
        Self
            Updated Gloria object
        """        
        if self.history is not None:
            raise Exception(
                "Regressors must be added prior to model fitting."
            )
        # Check that regressor name can be used. An error is raised if not
        self.validate_column_name(name, check_regressors = False)
        # Validate and set prior_scale and mode
        if prior_scale <= 0:
            raise ValueError('Prior scale must be > 0')
        if mode is None:
            mode = self.seasonality_mode
        if mode not in ['additive', 'multiplicative']:
            raise ValueError("Mode must be 'additive' or 'multiplicative'")
        
        # Create Regressor and add it to the external regressor list
        self.external_regressors.append(ExternalRegressor(
            name = name,
            prior_scale = prior_scale,
            mode = mode
        ))
        return self

    
    def add_seasonality(
            self: Self,
            name: str,
            period: str,
            fourier_order: int,
            prior_scale: float = None,
            mode: Optional[Literal['additive', 'multiplicative']] = None
        ) -> Self:
        """
        Add a seasonality to be used for fitting and predicting.

        Parameters
        ----------
        name : str
            Name of the seasonality.
        period : str
            Fundamental period of the seasonality component. Should be a string
            that can be parsed by pd.to_datetime (eg. '1d' or '12 h')
        fourier_order : int
            All Fourier terms from fundamental up to fourier_order will be used
        prior_scale : float, optional
            The regression coefficient is given a prior with the specified 
            scale parameter. Decreasing the prior scale will add additional 
            regularization. If None is given self.seasonality_prior_scale will
            be used (default). Must be greater than 0.
        mode : Optional[Literal['additive', 'multiplicative']], optional
            Whether regressor is treated as additive or multiplicative
            feature. If None is provided (default), self.seasonality_mode is 
            used.

        Raises
        ------
        Exception
            Raised when method is called before fitting.
        ValueError
            Raised when prior scale, mode, or period are not allowed values.

        Returns
        -------
        Self
            Updated Gloria object
        """

        if self.history is not None:
            raise Exception(
                "Seasonalities must be added prior to model fitting."
            )
        # Check that seasonality name can be used. An error is raised if not
        self.validate_column_name(name, check_seasonalities = False)
        
        # Validate and set prior_scale, mode, and fourier_order
        if prior_scale is None:
            ps = self.seasonality_prior_scale
        else:
            ps = float(prior_scale)
        if ps <= 0:
            raise ValueError("Prior scale must be > 0")
        if (fourier_order <= 0) or (not isinstance(fourier_order, int)):
            raise ValueError('Fourier Order must be an integer > 0')
        if mode is None:
            mode = self.seasonality_mode
        if mode not in ['additive', 'multiplicative']:
            raise ValueError('mode must be "additive" or "multiplicative"')
        
        # Create seasonality regressor and add it to the seasonality regressor
        # list
        self.seasonalities.append(Seasonality(
            name = name,
            period = pd.to_timedelta(period) / self.sampling_delta,
            fourier_order = fourier_order,
            prior_scale = prior_scale,
            mode = mode
        ))
        return self


    def validate_dataframe(
            self: Self,
            df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Validates that the input data frame of the fitting-method adheres to
        all requirements.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that contains at the very least a timestamp column with
            name self.timestamp_name and a numeric column with name 
            self.metric_name. If external regressors were added to the model,
            the respective columns must be present as well.
            
        Returns
        -------
        pd.DataFrame
            Validated DataFrame that is reduced to timestamp, metric and 
            external regressor columns.
        """
        # Overall Frame validation
        if df.shape[0] < 2:
            raise ValueError('Dataframe has less than 2 non-NaN rows.')
            
        # Timestamp validation
        if self.timestamp_name not in df:
            raise KeyError(
                f"Timestamp column '{self.timestamp_name}' is missing from"
                " DataFrame."
            )
        if df.index.name == 'ds':
            raise KeyError(
                f"Timestamp '{self.timestamp_name}' is set as index but"
                " expected to be a column"
            )
        time = df[self.timestamp_name]
        if time.dtype.kind != 'M':
            raise TypeError(
                f"Timestamp column '{self.timestamp_name}' is not of type"
                "datetime."
            )
        if time.isnull().any():
            raise ValueError(
                f"Found NaN in timestamp column '{self.timestamp_name}'."
            )
        if time.dt.tz is not None:
            raise NotImplementedError(
                f"Timestamp column '{self.timestamp_name}' has timezone "
                "specified, which is not supported. Remove timezone."
            )
        if not time.is_monotonic_increasing:
            raise ValueError(
                f"Timestamp column '{self.timestamp_name}' is not sorted."
            )
        
        # Check that timestamps lie on the expected grid. Note that no error
        # will be raised as long as the data lie on multiples of the expected
        # grid, eg. it is accepted, if daily sampling is expected, but the data
        # are sampled every other day.
        sampling_is_valid = (
            (time - time.min()) / self.sampling_delta
        ).apply(float.is_integer).all()
        if not sampling_is_valid:
            raise ValueError(
                f"Timestamp column '{self.timestamp_name}' is not sampled with"
                f" expected frequency of 1 / ({self.sampling_value}"
                f" {self.sampling_unit})"
            )
        
        # Metric validation
        if self.metric_name not in df:
            raise KeyError(
                f"Metric column '{self.metric_name}' is missing from "
                "DataFrame."
            )
        mtype = df[self.metric_name].dtype.kind
        allowed_types = list(MODEL_MAP[self.model].kind)
        if mtype not in allowed_types:
            type_list = ', '.join([f"'{s}'" for s in allowed_types])
            raise TypeError(
                f"Metric column type is '{mtype}', but must be any of"
                f" {type_list} for selected model '{self.model}'."
            )
        if df[self.metric_name].isnull().any():
            raise ValueError(
                f"Found NaN in metric column '{self.metric_name}'."
            )

        # Regressor validation
        regressor_names = [r.name for r in self.external_regressors]
        for name in regressor_names:
            if name not in df:
                raise KeyError(
                    f"Regressor column '{name}' is missing from DataFrame."
                )
            if df[name].dtype.kind not in 'biuf':
                raise TypeError(f"Regressor column '{name}' is non-numeric.")
            if df[name].isnull().any():
                raise ValueError(f"Regressor column '{name}' contains NaN.")
                
        return df.loc[:,[
            self.timestamp_name,
            self.metric_name,
            *regressor_names
        ]].copy()

    
    def set_changepoints(self: Self) -> Self:
        """
        !!! n_changepoints = 0 and 1 is not yet handled correctly !!!
        
        Sets changepoints

        Sets changepoints to the dates and corresponding integer values of
        changepoints. The following cases are handled:
        1) The changepoints were passed in explicitly.
            A) They are empty.
            B) They are not empty, and need validation.
        2) We are generating a grid of them.
        3) The user prefers no changepoints be used.
        """
        
        # Validates explicitly provided changepoints. These must fall within
        # training data range
        if self.changepoints is not None:
            if len(self.changepoints) == 0:
                pass
            else:
                too_low = (self.changepoints.min() < 
                           self.history[self.timestamp_name].min())
                too_high = (self.changepoints.max() > 
                            self.history[self.timestamp_name].max())
                if too_low or too_high:
                    raise ValueError(
                        "Changepoints must fall within training data."
                    )
        # In case not explicit changepoints were provided, create a grid
        else:
            # Place potential changepoints evenly through first
            # 'changepoint_range' proportion of the history
            hist_size = int(np.floor(self.history.shape[0]
                                     * self.changepoint_range))
            # when there are more changepoints than data, reduce number of
            # changepoints accordingly
            if self.n_changepoints + 1 > hist_size:
                self.n_changepoints = hist_size - 1
            if self.n_changepoints > 0:
                # Create indices for the grid
                cp_indexes = (
                    np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                        .round()
                        .astype(int)
                )
                # Find corresponding timestamps and use them as changepoints
                self.changepoints = (
                    self.history.iloc[cp_indexes][self.timestamp_name].tail(-1)
                )
            # If no changepoints were requested
            else:
                # Set empty changepoints
                self.changepoints = pd.Series(
                    pd.to_datetime([]),
                    name = self.timestamp_name,
                    dtype = '<M8[ns]'
                )
        # Convert changepoints to corresponding integer values for model 
        # backend
        if len(self.changepoints) > 0:
            self.changepoints_int = time_to_integer(
                self.changepoints,
                self.first_timestamp,
                self.sampling_delta
            )
            self.changepoints_int = pd.Series(
                self.changepoints_int,
                name = _T_INT,
                dtype = int
            ).sort_values().reset_index(drop = True)
        else:
            # Dummy changepoint
            self.changepoints_int = pd.Series([0], name = _T_INT, dtype = int)
            
        return self
            
    
    
    def time_to_integer(self: Self, history: pd.DataFrame) -> pd.DataFrame:
        """
        Create a new column from timestamp column of input data frame that 
        contains corresponding integer values with respect to sampling_delta

        Parameters
        ----------
        history : pd.DataFrame
            Validated input data frame of fit method

        Returns
        -------
        history : pd.DataFrame
            Updated data frame

        """
        # Find and save first and last timestamp
        time = history[self.timestamp_name]
        self.first_timestamp = time.min()
        self.last_timestamp = time.max()

        # Convert to integer and update data frame
        time_as_int = time_to_integer(
            time,
            self.first_timestamp,
            self.sampling_delta
        )
        history[_T_INT] = time_as_int
        
        return history
    
    
    def make_all_features_given_time(
            self: Self,
            t_int: pd.Series
        ) -> tuple[pd.DataFrame, dict[str,float], dict[str,str]]:
        """
        Creates the feature matrix X containing all regressors used in the fit
        and for prediction. Also returns prior scales and modes for all
        features.

        Parameters
        ----------
        t_int : pd.Series
            An integer array corresponding to the timestamps at which the
            features are to be evaluated

        Returns
        -------
        X : pd.DataFrame
            Feature matrix with columns for the different features and rows 
            corresponding to the timestamps
        prior_scales : dict[str,float]
            A dictionary mapping feature -> prior scale
        modes : dict[str,str]
            A dictionary mapping feature -> mode

        """
        # Features are either seasonalities or external regressors. Both have
        # a make_feature() method. Therefore stitch all features together and
        # subsequently call the methods in a single loop
        
        # 1. Seasonalities
        make_features = [
            lambda s=s: s.make_feature(t_int) 
            for s in self.seasonalities
        ]
        # 2. External Regressors
        make_features.extend([
            lambda er=er: er.make_feature(
                t_int,
                self.history[er.name]
            )
            for er in self.external_regressors
        ])
    
        # Make the features and save all feature matrices along with prior
        # scales and modes.
        X_lst = []
        prior_scales = dict()
        modes = dict()
        for make_feature in make_features:
            X_loc, prior_scales_loc, modes_loc = make_feature()
            X_lst.append(X_loc)
            prior_scales = {**prior_scales, **prior_scales_loc}
            modes = {**modes, **modes_loc}
        
        # Concat the single feature matrices to a single overall matrix
        if X_lst:
            X = pd.concat(X_lst, axis = 1)
        else:
            X = pd.DataFrame()
        
        return X, prior_scales, modes
    
    def make_all_features(self: Self) -> Self:
        """
        Creates the feature matrix X containing all regressors used in the fit
        and for prediction. Also returns prior scales and modes for all
        features.
        
        The difference to self.make_all_features_given_time() is that this
        method is used explicitly for fitting and hence takes the timestamps
        of the historical data.

        Raises
        ------
        Exception
            If anyone tries to call the method prior to the fit

        Returns
        -------
        Self
            Updated Gloria object
        """
        # Mostly because timestamp column must have been converted using
        # time_to_integer
        if self.history is None:
            raise Exception('make_all_features() cannot be called directly.')
        
        self.X, self.prior_scales, self.modes = (
            self.make_all_features_given_time(self.history[_T_INT])
        )
           
        return self
    
    
    def preprocess(
            self: Self,
            data: pd.DataFrame
        ) -> None:
        """
        Validates input data and prepares the model with respect to the data.
    
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame that contains at the very least a timestamp column with
            name self.timestamp_name and a numeric column with name 
            self.metric_name. If external regressors were added to the model,
            the respective columns must be present as well.
        
        Returns
        -------
        ModelInputData
            Data as used by the model backend. Contains the inputs that all
            stan models have in common.
        """
        # Make sure the data adhere to all requirements
        self.history = self.validate_dataframe(data)
        self.history.sort_values(by = self.timestamp_name, inplace = True)
        # Add a colum with mapping the timestamps to integer values
        self.history = self.time_to_integer(self.history)
        # Create The feature matrix of all seasonal and external regressor
        # components
        self.make_all_features()
        
        # Set changepoints according to changepoint parameters set by user
        self.set_changepoints()
        
        mode_values = np.array(list(self.modes.values()))
        
        # Prepares the input data as used by the model backend
        return ModelInputData(
            T = self.history.shape[0],
            S = len(self.changepoints_int),
            K = self.X.shape[1],
            tau = self.changepoint_prior_scale,
            y = self.history[self.metric_name].values,
            t = self.history[_T_INT].values,
            t_change = self.changepoints_int.values,
            s_a = np.where(mode_values == 'additive', 1, 0),
            s_m = np.where(mode_values == 'multiplicative', 1, 0),
            X = self.X.values,
            sigmas = np.array(list(self.prior_scales.values()))
        )
        
        
    def fit(
            self: Self,
            data: pd.DataFrame,
            optimize_mode: Literal['MAP', 'MLE'] = 'MAP',
            sample: bool = True,
            **kwargs: dict[str, Any]
        ) -> Self:
        """
        Fits the Gloria model

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame that contains at the very least a timestamp column with
            name self.timestamp_name and a numeric column with name 
            self.metric_name. If external regressors were added to the model,
            the respective columns must be present as well.
        optimize_mode : Literal['MAP', 'MLE'], optional
            If 'MAP' (default), the optimization step yiels the Maximum A 
            Posteriori, if 'MLE' the Maximum Likehood Estimate
        sample : bool, optional
            If True (default), the optimization is followed by a sampling over
            the Laplace approximation around the posterior mode.
        **kwargs : dict[str, Any]
            Keywoard arguments that will be augmented and then passed through
            to the model backend

        Raises
        ------
        Exception
            Raised when the model is attempted to be fit more than once

        Returns
        -------
        Self
            Updated Gloria object
        """
        if self.history is not None:
            raise Exception('Gloria object can only be fit once. '
                            'Instantiate a new object.')
        
        input_data = self.preprocess(data)
        
        
        t0 = time()
        self.model_backend.fit(
            input_data,
            optimize_mode = optimize_mode,
            sample = sample,
            **kwargs
        )
        t1 = time()
        
        return self, t1-t0
    
    
    def predict(self, timestamps: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the fitted Gloria model.

        Parameters
        ----------
        timestamps : pd.Series
            A Series containing the timestamps for the prediction

        Returns
        -------
        prediction : pd.DataFrame
            A dataframe containing timestamps, predicted metric, trend, and
            lower and upper bounds.
        """
        # First convert to integer timestamps with respect to first timestamp
        # and sampling_delta of training data
        timestamps_int = time_to_integer(
            timestamps[self.timestamp_name],
            self.first_timestamp,
            self.sampling_delta
        )
        
        # Create the regressor matrix at desired timestamps
        X, _, _ = self.make_all_features_given_time(timestamps_int)
        
        # Call the prediction method of the model backend
        prediction = self.model_backend.predict(
            timestamps_int,
            X,
            self.interval_width,
            self.uncertainty_samples,
        )
        
        # Insert timestamp into result
        prediction.insert(0, self.timestamp_name, timestamps.values)
        
        return prediction
    
    
    def make_future_dataframe(
            self,
            periods: int,
            include_history: bool = True
        ) -> pd.DataFrame:
        """
        Convenience function to create a Series of timestamps to be used by the
        predict method.

        Parameters
        ----------
        periods : int
            Number of periods to forecast forward.
        include_history : bool, optional
            Boolean to include the historical dates in the data frame for
            predictions. The default is True.

        Raises
        ------
        Exception
            Can only be used after fitting

        Returns
        -------
        new_timestamps : pd.Series
            The Series that extends forward from the end of self.history for
            the requested number of periods.

        """
        if self.history is None: 
            raise Exception('Model has not been fit.')
        
        # Create series of timestamps extending forward from the training data
        new_timestamps = pd.Series(pd.date_range(
            start = self.last_timestamp + self.sampling_delta,
            periods = periods,
            freq = self.sampling_delta
        ))
        
        # If desired attach training timestamps at the beginning
        if include_history:
            new_timestamps = pd.concat([
                self.history[self.timestamp_name],
                new_timestamps
            ])
        
        return pd.DataFrame({self.timestamp_name: new_timestamps})


### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent