"""
Definition of the Gloria forecaster class

FUTURE IMPROVEMENTS:
    - Give option to do a pre-fit with poissonian model. Subsequently using the
      dispersion_calc() will show whether data are under or overdispersed. With
      that an appropriate model can be suggested to the user.
    - Give possibility to pass a list of (country, subdiv) pairs to
      CalendricData.
    - Include all Prophet output columns of the prediction to Gloria output
    - Changepoint protocols: estimate number of changepoints by considering
      slowest varying seasonality -> density of changepoints should not be able
      to interfere
    - Add Disclaimer and copyright note to all modules
    - Seasonality regressor works with integer timescale. All others with real
      timestamps. Unify.
    - Reevaluate configuration.py. Probably a utility like that only adds value
      if we offer automated pipelines, which we may not want to do. If we
      decide against configuration.py, move it out of the library and keep it
      as internal tool.
    - Currently, many places Timedeltas are just accepted as string. It would
      be more natural if they are also accepted as pd.Timedelta
    - Check whether event_prior_scale is serialized

ROBUSTNESS
    - Appropriate regressor scaling. Idea: (1) the piece-wise-linear estimation
      in calculate_initial_parameters also returns the residuals
        res = y_scaled - trend
      (2) Find the scale of the residuals, eg. their standard deviation
      (3) Set regressor scales to the residual scale
      Also consider this in conjunction with estimating initial values for all
      beta. And probably reconsider reparametrizing the Stan-models.
    - Browser Data with normal model caused stan optimization error
    - The data set '2025-02-19_binomial_test_n02.csv' cannot be fit with the
      normally distributed model, if the CalendricData protocol adds all
      holidays. The data itself only show an effect at Christmas. It can be
      fixed by reducing event_prior_scale or or decreasing the duration of the
      event (Gaussian with 5d doesn't work, with 2d it works)
    - Fitting Browser Data with a small number of changepoints and Poisson
      model fails with 'Error evaluating model log probability: Non-finite
      gradient.'

For Documentation
    - Summarize differences in features and API between Gloria and Prophet. For
      missing feature, describe workarounds. For additional features, show
      applications.
    - Check Docstrings for Errors, as signatures or defaults may have changed.
    - Currently the fit routine has a boolean argument sample that triggers
      Laplace sampling in the backend. Also, the predict routine has an
      argument n_samples that controlls how many samples are drawn for the
      trend uncertainty. These two parameters can be confused.
      Clearly document their difference, maybe consider better names for them
    - Notes on how regressors are added: overwrite by default, manually added
      regressors take precedence over protocol ones. However, only the name is
      checked, not the properties!
    - Make Note that multiplicative model currently only works as expected for
      normally distributed model
"""

### --- Module Imports --- ###
# Standard Library
import math
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Type, Union

# Third Party
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)
from typing_extensions import Self

# Gloria
import gloria.utilities.serialize as gs
from gloria.events import Event
from gloria.models import (
    MODEL_MAP,
    BinomialPopulation,
    ModelInputData,
    get_model_backend,
)
from gloria.plot import (
    add_changepoints_to_plot,
    plot_event_component,
    plot_seasonality_component,
    plot_trend_component,
)
from gloria.protocols.protocol_base import Protocol
from gloria.regressors import (
    EventRegressor,
    ExternalRegressor,
    Regressor,
    Seasonality,
)

# Inhouse Packages
from gloria.utilities.constants import (
    _BACKEND_DEFAULTS,
    _DELIM,
    _DTYPE_KIND,
    _GLORIA_DEFAULTS,
    _T_INT,
)
from gloria.utilities.errors import FittedError, NotFittedError
from gloria.utilities.logging import get_logger
from gloria.utilities.misc import time_to_integer
from gloria.utilities.types import Distribution, SeriesData


### --- Class and Function Definitions --- ###
class Gloria(BaseModel):
    """
    Gloria forecaster.

    Parameters
    ----------
    model : Distribution
        The distribution model to be used. Can be any of 'poisson',
        'binomial constant n', 'binomial vectorized n', 'negative binomial',
        or 'normal'.
    sampling_period : Union[pd.Timedelta, str]
        Minimum spacing between two adjacent samples either as pandas Timedelta
        or an equivalent string.
    timestamp_name : str, optional
        The name of the timestamp column as expected in the input data frame
        for the fit-method. The default is 'ds'.
    metric_name : str, optional
        The name of the expected metric column of the input data frame for the
        fit-method. The default is 'y'.
    population_name : str, optional
        The name of the name containing population size data for the model
        'binomial vectorized n'. The default is ''
    changepoints : pd.Series, optional
        List of timestamps at which to include potential changepoints. If not
        specified (default), potential changepoints are selected automatically.
    n_changepoints : int, optional
        Number of potential changepoints to include. Not used if input
        'changepoints' is supplied. If 'changepoints' is not supplied, then
        n_changepoints potential changepoints are selected uniformly from the
        first 'changepoint_range' proportion of the history. Must be a positive
        integer, default is 25.
    changepoint_range : float, optional
        Proportion of history in which trend changepoints will be estimated.
        Must be in range [0,1]. Defaults to 0.8 for the first 80%. Not used if
        'changepoints' is specified.
    seasonality_prior_scale : float, optional
        Parameter modulating the strength of the seasonality model. Larger
        values allow the model to fit larger seasonal fluctuations, smaller
        values dampen the seasonality. Can be specified for individual
        seasonalities using add_seasonality.
    event_prior_scale : float, optional
        Parameter modulating the strength of additional event regressors.
        Larger values allow the model to fit larger event impact, smaller
        values dampen the event impact. Can be specified for individual
        events using add_event.
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

    model_config = ConfigDict(
        # Allows setting extra attributes during initialization
        extra="allow",
        # So the model accepts pandas object as values
        arbitrary_types_allowed=True,
        # Use validation also when fields of an existing model are assigned
        validate_assignment=True,
    )

    model: Distribution = _GLORIA_DEFAULTS["model"]
    sampling_period: pd.Timedelta = _GLORIA_DEFAULTS["sampling_period"]
    timestamp_name: str = _GLORIA_DEFAULTS["timestamp_name"]
    metric_name: str = _GLORIA_DEFAULTS["metric_name"]
    population_name: str = _GLORIA_DEFAULTS["population_name"]
    changepoints: Optional[pd.Series] = _GLORIA_DEFAULTS["changepoints"]
    n_changepoints: int = Field(
        ge=0, default=_GLORIA_DEFAULTS["n_changepoints"]
    )
    changepoint_range: float = Field(
        gt=0, lt=1, default=_GLORIA_DEFAULTS["changepoint_range"]
    )
    seasonality_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["seasonality_prior_scale"]
    )
    event_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["event_prior_scale"]
    )
    changepoint_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["changepoint_prior_scale"]
    )
    interval_width: float = Field(
        gt=0, lt=1, default=_GLORIA_DEFAULTS["interval_width"]
    )
    uncertainty_samples: int = Field(
        ge=0, default=_GLORIA_DEFAULTS["uncertainty_samples"]
    )

    @field_validator("sampling_period", mode="before")
    @classmethod
    def validate_sampling_period(
        cls: Type[Self], sampling_period: Union[pd.Timedelta, str]
    ) -> pd.Timedelta:
        """
        Converts sampling period to a pandas Timedelta if it was passed as a
        string instead.
        """
        # Third Party
        from pandas._libs.tslibs.parsing import DateParseError

        try:
            s_period = pd.Timedelta(sampling_period)
        except (ValueError, DateParseError) as e:
            msg = "Could not parse input sampling period."
            get_logger().error(msg)
            raise ValueError(msg) from e

        if s_period <= pd.Timedelta(0):
            msg = "Sampling period must be positive and nonzero"
            get_logger().error(msg)
            raise ValueError(msg)

        return s_period

    @field_validator("population_name", mode="before")
    @classmethod
    def validate_population_name(
        cls: Type[Self],
        population_name: Optional[str],
        other_fields: ValidationInfo,
    ) -> str:
        """
        Check that the population name is set if the 'binomial vectorized n' is
        being used
        """
        model = other_fields.data["model"]
        population_name = "" if population_name is None else population_name
        if (model == "binomial vectorized n") and (population_name == ""):
            raise ValueError(
                "The name of population must be set for model 'binomial "
                "vectorized n' but is an empty string."
            )
        return population_name

    @field_validator("changepoints", mode="before")
    @classmethod
    def validate_changepoints(
        cls: Type[Self], changepoints: Optional[SeriesData]
    ) -> pd.Series:
        """
        Converts changepoints input to pd.Series
        """
        if changepoints is None:
            return changepoints

        try:
            changepoints = pd.Series(changepoints)
        except Exception as e:
            raise ValueError(
                "Input changepoints cannot be converted to a pandas Series."
            ) from e
        return changepoints

    def __init__(
        self: Self, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> None:
        """
        Initializes Gloria model.

        Parameters
        ----------
        *args : tuple[Any, ...]
            Positional arguments passed through to Pydantic Model __init__()
        **kwargs : dict[str, Any]
            Keyword arguments passed through to Pydantic Model __init__()
        """
        # Call the __init__() method of the pydantic model for proper
        # initialization and validation of model fields.
        super().__init__(*args, **kwargs)

        # Sanitize provided Changepoints
        if self.changepoints is not None:
            print(self.changepoints)
            self.changepoints = pd.Series(
                pd.to_datetime(self.changepoints), name=self.timestamp_name
            )
            self.n_changepoints = len(self.changepoints)

        # Instantiate the model backend that manages communication with the
        # Stan model and performs predictions based on the fit
        self.model_backend = get_model_backend(model=self.model)

        # The following attributes will be set during fitting or by other
        # methods
        # 1. All Regressors
        self.external_regressors: dict[str, ExternalRegressor] = dict()
        self.seasonalities: dict[str, Seasonality] = dict()
        self.events: dict[str, dict[str, Any]] = dict()
        # 2. Prior scales assigned to the regressors
        self.prior_scales: dict[str, float] = dict()
        # 3. A list of all protocols applied to the model
        self.protocols: list[Protocol] = []
        # 4. Input data to be fitted
        self.history: pd.DataFrame = pd.DataFrame()
        self.first_timestamp: pd.Timestamp = pd.Timestamp(0)
        # 5. A matrix of all regressors (columns) X timestamps (rows)
        self.X: pd.DataFrame = pd.DataFrame()

    @property
    def is_fitted(self: Self) -> bool:
        """
        Determines whether the Gloria model is fitted.

        Returns
        -------
        bool
            True if fitted, False otherwise.

        """
        return self.model_backend.fit_params != dict()

    def validate_column_name(
        self: Self,
        name: str,
        check_seasonalities: bool = True,
        check_events: bool = True,
        check_external_regressors: bool = True,
    ) -> None:
        """
        Validates the name of a seasonality, an event or an external regressor.

        Parameters
        ----------
        name : str
            The name to validate.
        check_seasonalities : bool, optional
            Check if name already used for seasonality. The default is True.
        check_events : bool, optional
            Check if name already used as an event. The default is True.
        check_external_regressors : bool, optional
            Check if name already used for regressor The default is True.

        Raises
        ------
        TypeError
            If the passed name is not a string
        ValueError
            Raised in case the name is not valid for any reason.
        """

        # Name must be a string
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        # The _DELIM constant is used for constructing intermediate column
        # names, hence it's not allowed to be used within given names
        if _DELIM in name:
            raise ValueError(f"Name cannot contain '{_DELIM}'")

        # Reserved names are either
        # 1. Column names generated by the prediction method:
        reserved_names = ["fit", "observed", "trend"]
        rn_l = [n + "_lower" for n in reserved_names]
        rn_u = [n + "_upper" for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        # 2. Column names that are expected to be part of the input data or
        #    generated by other methods
        reserved_names.extend(
            [self.timestamp_name, self.metric_name, _T_INT, _DTYPE_KIND]
        )
        if self.model == "binomial vectorized n":
            reserved_names.append(self.population_name)

        if name in reserved_names:
            raise ValueError(f"Name {name} is reserved.")

        # Apart from reserved names, also check if the name is alread in use
        # for the model. The "check_XY" flag is used to allow for overwriting,
        # ie. when adding a seasonality, check_seasonalities is False hence
        # no error will be raised.
        if check_seasonalities and name in self.seasonalities:
            raise ValueError(f"Name {name} already used for a seasonality.")
        if check_events and name in self.events:
            raise ValueError(f"Name {name} already used for an event.")
        if check_external_regressors and name in self.external_regressors:
            raise ValueError(f"Name {name} already used for a regressor.")

    def add_seasonality(
        self: Self,
        name: str,
        period: str,
        fourier_order: int,
        prior_scale: Optional[float] = None,
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

        Raises
        ------
        Exception
            Raised when method is called before fitting.
        ValueError
            Raised when prior scale, or period are not allowed values.

        Returns
        -------
        Gloria
            Updated Gloria object
        """
        if self.is_fitted:
            raise FittedError(
                "Seasonalities must be added prior to model fitting."
            )
        # Check that seasonality name can be used. An error is raised if not
        self.validate_column_name(name, check_seasonalities=False)
        # If name was already in use for a seasonality but no error was raised
        # during validation, the seasonality will be overwritten.
        # Important: for the logging info to be consistent with possible
        # validation errors, keep the following if-clause subsequent to the
        # validation.
        if name in self.seasonalities:
            get_logger().info(
                f"'{name}' is an existing seasonality. Overwriting with new"
                " configuration."
            )

        # Validate and set prior_scale and fourier_order
        if prior_scale is None:
            prior_scale = self.seasonality_prior_scale
        prior_scale = float(prior_scale)
        if prior_scale <= 0:
            raise ValueError("Prior scale must be > 0")
        if (fourier_order <= 0) or (not isinstance(fourier_order, int)):
            raise ValueError("Fourier Order must be an integer > 0")

        # Create seasonality regressor and add it to the seasonality regressor
        # list
        self.seasonalities[name] = Seasonality(
            name=name,
            period=pd.to_timedelta(period) / self.sampling_period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
        )
        return self

    def add_event(
        self: Self,
        name: str,
        regressor_type: str,
        event: Union[Event, dict[str, Any]],
        prior_scale: Optional[float] = None,
        include: Union[bool, Literal["auto"]] = "auto",
        **regressor_kwargs: Any,
    ) -> Self:
        """
        Add an event to be used for fitting and predicting.

        Parameters
        ----------
        name : str
            name of the event.
        regressor_type : str
            Type of the underlying event regressor.
        event : Union[Event, dict[str, Any]]
            The base event used by the event regressor.
        prior_scale : float
            The regression coefficient is given a prior with the specified
            scale parameter. Decreasing the prior scale will add additional
            regularization.
        **regressor_kwargs : dict[str, Any]
            Additional keyword arguments necessary to create the event
            regressor.

        Raises
        ------
        Exception
            Raised in case the method is called on a fitted Gloria model.
        ValueError
            Raised in case of invalid prior scales.

        Returns
        -------
        Gloria
            The Gloria model updated with the new event

        """

        if self.is_fitted:
            raise FittedError("Event must be added prior to model fitting.")

        # Check that event name can be used. An error is raised if not
        self.validate_column_name(name, check_events=False)
        # If name was already in use for an event but no error was raised
        # during validation, the event will be overwritten.
        # Important: for the logging info to be consistent with possible
        # validation errors, keep the following if-clause subsequent to the
        # validation.
        if name in self.events:
            get_logger().info(
                f"'{name}' is an existing event. Overwriting with new"
                " configuration."
            )

        # Validate and set prior_scale and include flag
        if prior_scale is None:
            prior_scale = self.event_prior_scale
        prior_scale = float(prior_scale)
        if prior_scale <= 0:
            raise ValueError("Prior scale must be > 0")
        if not (isinstance(include, bool) or include == "auto"):
            raise ValueError("include must be True, False, or 'auto'.")

        # As the event regressor is built from a dictionary, convert the event
        # to a dictionary in case it was an Event instance.
        if isinstance(event, Event):
            event = event.to_dict()

        # Build the regressor dictionary
        regressor_dict = {
            "name": name,
            "prior_scale": prior_scale,
            "regressor_type": regressor_type,
            "event": event,
            **regressor_kwargs,
        }

        # Build the event regressor and add it to the model
        new_regressor = Regressor.from_dict(regressor_dict)
        # Safety check for type
        if isinstance(new_regressor, EventRegressor):
            regressor_info = {"regressor": new_regressor, "include": include}
            self.events[name] = regressor_info
        else:
            raise TypeError(
                "The created regressor must be an EventRegressor"
                f" but is {type(new_regressor)}."
            )
        return self

    def add_external_regressor(
        self: Self,
        name: str,
        prior_scale: float,
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

        Raises
        ------
        Exception
            Raised when method is called before fitting.
        ValueError
            Raised when prior scale value is not allowed.

        Returns
        -------
        Gloria
            Updated Gloria object
        """
        if self.is_fitted:
            raise FittedError(
                "Regressors must be added prior to model fitting."
            )
        # Check that regressor name can be used. An error is raised if not
        self.validate_column_name(name, check_external_regressors=False)
        # If name was already in use for a regressor but no error was raised
        # during validation, the regressor will be overwritten.
        # Important: for the logging info to be consistent with possible
        # validation errors, keep the following if-clause subsequent to the
        # validation.
        if name in self.external_regressors:
            get_logger().info(
                f"'{name}' is an existing external regressor. Overwriting with"
                " new configuration."
            )
        # Validate and set prior_scale
        prior_scale = float(prior_scale)
        if prior_scale <= 0:
            raise ValueError("Prior scale must be > 0")

        # Create Regressor and add it to the external regressor list
        self.external_regressors[name] = ExternalRegressor(
            name=name, prior_scale=prior_scale
        )
        return self

    def add_protocol(self: Self, protocol: Protocol) -> Self:
        """
        Add a protocol to the Gloria model that provides additional routines
        for setting the model up during the fit.

        Parameters
        ----------
        protocol : Protocol
            The Protocol object to be added.

        Raises
        ------
        Exception
            Raised when method is called before fitting.
        TypeError
            Raised when the provided protocol is not a valid Protocol object

        Returns
        -------
        Gloria
            the updated Gloria model.

        """

        if self.is_fitted:
            raise FittedError(
                "Protocols must be added prior to model fitting."
            )

        # Simply check whether the input protocol is a Protocol object. If so,
        # it can safely be added to the model.
        if not isinstance(protocol, Protocol):
            raise TypeError(
                "The protocol must be of type 'Protocol'"
                f", but is {type(protocol)}."
            )

        p_type = protocol._protocol_type
        existing_types = set(p._protocol_type for p in self.protocols)
        if p_type in existing_types:
            get_logger().warning(
                f"The model already has a protocol of type {p_type}. Adding "
                "another one may lead to unexpected interference between these"
                " protocols."
            )
        self.protocols.append(protocol)

        return self

    def validate_metric_column(
        self: Self,
        df: pd.DataFrame,
        name: str,
        col_type: Literal["Metric", "Population"] = "Metric",
    ) -> None:
        """
        Validate that the metric column exists and contains only valid values.

        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame of data to be fitted.
        name : str
            The metric column name
        col_type : Literal["Metric", "Population"], optional
            Specifies whether the metric column or population column is to be
            validated. The default is "Metric".

        Raises
        ------
        KeyError
            Raised if the metric column doesn't exist in the DataFrame
        TypeError
            Raised if the metric columns dtype does not fit to the model
        ValueError
            Raised if there are any NaNs in the metric column

        """
        if name not in df:
            raise KeyError(
                f"{col_type} column '{name}' is missing from " "DataFrame."
            )
        m_dtype_kind = df[name].dtype.kind
        allowed_types = list(MODEL_MAP[self.model].kind)
        if m_dtype_kind not in allowed_types:
            type_list = ", ".join([f"'{s}'" for s in allowed_types])
            raise TypeError(
                f"{col_type} column '{name}' type is '{m_dtype_kind}', but "
                f"must be any of {type_list} for model '{self.model}'."
            )
        if df[name].isnull().any():
            raise ValueError(
                f"Found NaN in {col_type.lower()} column '{name}'."
            )

    def validate_dataframe(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates that the input data frame of the fitting-method adheres to
        all requirements.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that contains at the very least a timestamp column with
            name self.timestamp_name and a numeric column with name
            self.metric_name. If the Gloria model is 'binomial vectorized n'
            a column with name self.population.name must exists. If external
            regressors were added to the model, the respective columns must be
            present as well.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame that is reduced to timestamp, metric and
            external regressor columns.
        """
        # Overall Frame validation
        if df.shape[0] < 2:
            raise ValueError("Dataframe has less than 2 non-NaN rows.")

        # Timestamp validation
        if self.timestamp_name not in df:
            raise KeyError(
                f"Timestamp column '{self.timestamp_name}' is missing from"
                " DataFrame."
            )
        if df.index.name == self.timestamp_name:
            raise KeyError(
                f"Timestamp '{self.timestamp_name}' is set as index but"
                " expected to be a column"
            )
        time = df[self.timestamp_name]
        if time.dtype.kind != "M":
            raise TypeError(
                f"Timestamp column '{self.timestamp_name}' is not of type "
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
        # Check that timestamps lie on the expected grid.
        sample_multiples = (time - time.min()) / self.sampling_period
        sampling_is_valid = sample_multiples.apply(float.is_integer).all()
        if not sampling_is_valid:
            raise ValueError(
                f"Timestamp column '{self.timestamp_name}' is not sampled with"
                f" expected sampling period '{self.sampling_period}'"
            )
        # Note that no error will be raised as long as the data lie on
        # multiples of the expected grid, eg. it is accepted, if daily sampling
        # is expected, but the data are sampled every other day. A logger info
        # is issued if that's the case
        if (sample_multiples.diff() > 1).any():
            get_logger().info(
                "All timestamps are multiples of the sampling period, but gaps"
                " were found."
            )
        # Metric validation
        self.validate_metric_column(
            df=df, name=self.metric_name, col_type="Metric"
        )
        # Population validation
        if self.model == "binomial vectorized n":
            self.validate_metric_column(
                df=df, name=self.population_name, col_type="Population"
            )
            # Check values in the population column
            if (df[self.population_name] < df[self.metric_name]).any():
                raise ValueError(
                    "There are values in the metric column that exceed the "
                    "corresponding values in the population column, which is "
                    "not allowed for model 'binomial vectorized n'."
                )

        # Regressor validation
        for name in self.external_regressors:
            if name not in df:
                raise KeyError(
                    f"Regressor column '{name}' is missing from DataFrame."
                )
            if df[name].dtype.kind not in "biuf":
                raise TypeError(f"Regressor column '{name}' is non-numeric.")
            if df[name].isnull().any():
                raise ValueError(f"Regressor column '{name}' contains NaN.")

        history = df.loc[
            :,
            [
                self.timestamp_name,
                self.metric_name,
                *self.external_regressors.keys(),
            ],
        ].copy()

        if self.model == "binomial vectorized n":
            history[self.population_name] = df[self.population_name].copy()
        return history

    def set_changepoints(self: Self) -> Self:
        """
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
                too_low = (
                    self.changepoints.min()
                    < self.history[self.timestamp_name].min()
                )
                too_high = (
                    self.changepoints.max()
                    > self.history[self.timestamp_name].max()
                )
                if too_low or too_high:
                    raise ValueError(
                        "Changepoints must fall within training data."
                    )
        # In case not explicit changepoints were provided, create a grid
        else:
            # Place potential changepoints evenly through first
            # 'changepoint_range' proportion of the history
            hist_size = int(
                np.floor(self.history.shape[0] * self.changepoint_range)
            )
            # when there are more changepoints than data, reduce number of
            # changepoints accordingly
            if self.n_changepoints + 1 > hist_size:
                get_logger().warning(
                    f"Provided number of changepoints {self.n_changepoints} "
                    f"greater than number of observations in changepoint "
                    f"range. Using {hist_size - 1} instead. Consider reducing"
                    " n_changepoints."
                )
                self.n_changepoints = hist_size - 1

            get_logger().info(
                f"Distributing {self.n_changepoints} equidistant"
                " changepoints."
            )

            if self.n_changepoints > 0:
                # Create indices for the grid
                cp_indexes = (
                    np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                    .round()
                    .astype(int)
                )
                # Find corresponding timestamps and use them as changepoints
                self.changepoints = self.history.iloc[cp_indexes][
                    self.timestamp_name
                ].tail(-1)
            # If no changepoints were requested
            else:
                # Set empty changepoints
                self.changepoints = pd.Series(
                    pd.to_datetime([]),
                    name=self.timestamp_name,
                    dtype="<M8[ns]",
                )
        # Convert changepoints to corresponding integer values for model
        # backend
        if len(self.changepoints) > 0:
            changepoints_int_loc = time_to_integer(
                self.changepoints, self.first_timestamp, self.sampling_period
            )
            self.changepoints_int = (
                pd.Series(changepoints_int_loc, name=_T_INT, dtype=int)
                .sort_values()
                .reset_index(drop=True)
            )
        else:
            # Dummy changepoint
            self.changepoints_int = pd.Series([0], name=_T_INT, dtype=int)

        return self

    def time_to_integer(self: Self, history: pd.DataFrame) -> pd.DataFrame:
        """
        Create a new column from timestamp column of input data frame that
        contains corresponding integer values with respect to sampling_delta.

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
            time, self.first_timestamp, self.sampling_period
        )
        history[_T_INT] = time_as_int

        return history

    def make_all_features(
        self: Self, data: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Creates the feature matrix X containing all regressors used in the fit
        and for prediction. Also returns prior scales for all features.

        Parameters
        ----------
        data : Optional[pd.DataFrame], optional
            Input dataframe. It must contain at least a column with integer
            timestamps (for column name cf. to _T_INT constant) as well as
            the external regressor columns associated with the model. Default
            of data is None, in which case the model history will be used.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix with columns for the different features and rows
            corresponding to the timestamps
        prior_scales : dict[str,float]
            A dictionary mapping feature -> prior scale
        """
        # If no data were passed in, assume history as data. This is the case
        # if called during preprocessing
        if data is None:
            data = self.history

        # Features are either seasonalities or external regressors. Both have
        # a make_feature() method. Therefore stitch all features together and
        # subsequently call the methods in a single loop

        # 1. Seasonalities
        make_features = [
            lambda s=s: s.make_feature(data[_T_INT])
            for s in self.seasonalities.values()
        ]
        # 2. External Regressors
        make_features.extend(
            [
                lambda er=er: er.make_feature(data[_T_INT], data[er.name])
                for er in self.external_regressors.values()
            ]
        )
        # 3. Event Regressors
        for name in list(self.events.keys()):
            regressor = self.events[name]["regressor"]
            # Events can be excluded controlled by the include flag
            if not self.is_fitted:
                # Whether or not to include an event only needs to be evaluated
                # before the fit. Therefore skip this entire block post-fit
                include = self.events[name]["include"]
                # Exclude the if include=False
                if include is False:
                    continue
                # Otherwise calculate impact which quantifies how many events
                # of the regressor lie within the date range of the data.
                impact = regressor.get_impact(data[self.timestamp_name])
                # If the impact is below a threshold, fitting it may be unsafe.
                # Two cases are covered:
                # 1. include=True, ie the user wishes to include the event
                # under any circumstances. In this case only issue a warning
                if impact < 0.1 and include is True:
                    get_logger().warning(
                        f"Event '{regressor.name}' hardly occurs during "
                        "timerange of interest, which may lead to unreliable "
                        "or failing fits. Consider setting include='auto'."
                    )
                # 2. include="auto". Send a warning that the event will be
                # excluded and delete it from self.events
                elif impact < 0.1:
                    get_logger().warning(
                        f"Event '{regressor.name}' hardly occurs during "
                        "timerange of interest. Removing it from model. Set "
                        "include=True to overwrite this."
                    )
                    del self.events[name]
                    continue

            make_features.append(
                lambda reg=regressor: reg.make_feature(
                    data[self.timestamp_name]
                )
            )

        # Make the features and save all feature matrices along with prior
        # scales.
        X_lst = []
        prior_scales: dict[str, float] = dict()
        for make_feature in make_features:
            X_loc, prior_scales_loc = make_feature()
            X_lst.append(X_loc)
            prior_scales = {**prior_scales, **prior_scales_loc}

        # Concat the single feature matrices to a single overall matrix
        if X_lst:
            X = pd.concat(X_lst, axis=1)
        else:
            X = pd.DataFrame()

        return X, prior_scales

    def preprocess(self: Self, data: pd.DataFrame) -> ModelInputData:
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
        self.history.sort_values(by=self.timestamp_name, inplace=True)
        # Add a colum with mapping the timestamps to integer values
        self.history = self.time_to_integer(self.history)

        # Execute protocols to further set up the model
        for protocol in self.protocols:
            protocol.set_events(
                model=self, timestamps=self.history[self.timestamp_name]
            )
            protocol.set_seasonalities(
                model=self, timestamps=self.history[self.timestamp_name]
            )

        # Create The feature matrix of all seasonal and external regressor
        # components
        self.X, self.prior_scales = self.make_all_features()

        # Set changepoints according to changepoint parameters set by user
        self.set_changepoints()

        # Prepares the input data as used by the model backend
        input_data = ModelInputData(
            T=self.history.shape[0],
            S=len(self.changepoints_int),
            K=self.X.shape[1],
            tau=self.changepoint_prior_scale,
            y=np.asarray(self.history[self.metric_name]),
            t=np.asarray(self.history[_T_INT]),
            t_change=np.asarray(self.changepoints_int),
            X=self.X.values,
            sigmas=np.array(list(self.prior_scales.values())),
        )
        # Add population size for vectorized binomial model
        if self.model == "binomial vectorized n":
            input_data.N_vec = np.asarray(self.history[self.population_name])
        return input_data

    def fit(
        self: Self,
        data: pd.DataFrame,
        optimize_mode: Literal["MAP", "MLE"] = _BACKEND_DEFAULTS[
            "optimize_mode"
        ],
        sample: bool = _BACKEND_DEFAULTS["sample"],
        augmentation_config: Optional[BinomialPopulation] = None,
        **kwargs: dict[str, Any],
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
        augmentation_config : Optional[BinomialPopulation], optional
            Configuration parameters for the augment_data method. Currently, it
            is only required for the BinomialConstantN model. For all other
            models it defaults to None.
        **kwargs : dict[str, Any]
            Keywoard arguments that will be augmented and then passed through
            to the model backend

        Raises
        ------
        Exception
            Raised when the model is attempted to be fit more than once

        Returns
        -------
        Gloria
            Updated Gloria object
        """
        if self.is_fitted:
            raise FittedError(
                "Gloria object can only be fit once. Instantiate a new object."
            )

        # Prepare the model and input data
        get_logger().debug("Starting to preprocess input data.")
        input_data = self.preprocess(data)

        # Fit the model
        get_logger().debug("Handing over preprocessed data to model backend.")
        self.model_backend.fit(
            input_data,
            optimize_mode=optimize_mode,
            sample=sample,
            augmentation_config=augmentation_config,
            **kwargs,
        )

        return self

    def predict(self: Self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the fitted Gloria model.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe. It must contain at least a timestamp column named
            according to the models timestamp_name as well as the external
            regressor columns associated with the model.

        Returns
        -------
        prediction : pd.DataFrame
            A dataframe containing timestamps, predicted metric, trend, and
            lower and upper bounds.
        """
        if not self.is_fitted:
            raise NotFittedError("Can only predict using a fitted model.")
        # Validate and extract population to pass it to predict
        N_vec = None
        if self.model == "binomial vectorized n":
            if self.population_name not in data:
                raise KeyError(
                    "Prediction input data require a population size column "
                    f"'{self.population_name}' for vectorized binomial model."
                )
            N_vec = data[self.population_name].copy()

        # Validate external regressors
        missing_regressors = [
            f"'{name}'"
            for name in self.external_regressors
            if name not in data
        ]
        if missing_regressors:
            missing_regressors_str = ", ".join(missing_regressors)
            raise KeyError(
                "Prediction input data miss the external regressor column(s) "
                f"{missing_regressors_str}."
            )

        # First convert to integer timestamps with respect to first timestamp
        # and sampling_delta of training data
        data = data.copy()
        data[_T_INT] = time_to_integer(
            data[self.timestamp_name],
            self.first_timestamp,
            self.sampling_period,
        )

        # Create the regressor matrix at desired timestamps
        X, _ = self.make_all_features(data)

        # Call the prediction method of the model backend
        prediction = self.model_backend.predict(
            t=np.asarray(data[_T_INT]),
            X=np.asarray(X),
            interval_width=self.interval_width,
            n_samples=self.uncertainty_samples,
            N_vec=N_vec,
        )

        # Insert timestamp into result
        prediction.insert(
            0, self.timestamp_name, np.asarray(data[self.timestamp_name])
        )

        return prediction

    def make_future_dataframe(
        self: Self, periods: int = 1, include_history: bool = True
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
        if not self.is_fitted:
            raise NotFittedError()

        # Create series of timestamps extending forward from the training data
        new_timestamps = pd.Series(
            pd.date_range(
                start=self.last_timestamp + self.sampling_period,
                periods=periods,
                freq=self.sampling_period,
            )
        )

        # If desired attach training timestamps at the beginning
        if include_history:
            new_timestamps = pd.concat(
                [self.history[self.timestamp_name], new_timestamps]
            )

        return pd.DataFrame({self.timestamp_name: new_timestamps}).reset_index(
            drop=True
        )

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Convert Gloria model to a dictionary of JSON serializable types. The
        Gloria model must be fitted.

        Returns
        -------
        dict[str, Any]
            JSON serializable dictionary containing data of Gloria model.

        """
        # Cf. to serialization module for details.
        return gs.model_to_dict(self)

    def to_json(self: Self, filepath: Optional[Path] = None, **kwargs) -> str:
        """
        Serialises a fitted Gloria object and returns it as string. If desired
        the model is also dumped to a .json-file

        Parameters
        ----------
        filepath : Optional[Path], optional
            Filepath of the target .json-file. The default is None, in which
            case no file will be written.
        **kwargs : TYPE
            Keyword arguments which are passed through to json.dump()

        Returns
        -------
        str
            JSON string containing the model data.
        """
        # Cf. to serialization module for details.
        return gs.model_to_json(self, filepath=filepath, **kwargs)

    @staticmethod
    def from_dict(model_dict: dict[str, Any]) -> "Gloria":
        """
        Takes a dictionary as returned by model_to_dict() and restores the
        original Gloria model.

        Parameters
        ----------
        model_dict : dict[str, Any]
            Dictionary containing the Gloria object data

        Returns
        -------
        Gloria
            Input data converted to Gloria object.
        """
        # Cf. to serialization module for details.
        return gs.model_from_dict(model_dict)

    @staticmethod
    def from_json(
        model_json: Union[Path, str],
        return_as: Literal["dict", "model"] = "model",
    ) -> Union[dict[str, Any], "Gloria"]:
        """
        Takes a serialized Gloria model in json-format and converts it to a
        Gloria instance or dictionary.

        Parameters
        ----------
        model_json : Union[Path, str]
            Filepath of .json-model file or string containing the data
        return_as : Literal['dict', 'model'], optional
            If 'dict', the model is returned in dictionary format, if 'model'
            it is returned as Gloria instance. The default is 'model'.

        Returns
        -------
        Union[dict[str, Any], Gloria]
            Gloria object or dictionary representing it based on input json
            data.

        """
        # Cf. to serialization module for details.
        return gs.model_from_json(model_json, return_as)

    @staticmethod
    def Foster():
        with open(
            Path(__file__).parent / "foster.txt", "r", encoding="utf8"
        ) as f:
            print(f.read(), end="\n\n")
        print("  --------------------  ".center(70))
        print("| Here, take a cookie. |".center(70))
        print("  ====================  ".center(70))

    def plot(
        self: Self,
        fcst: pd.DataFrame,
        ax: Optional[sns] = None,
        uncertainty: bool = True,
        xlabel: str = "ds",
        ylabel: str = "y",
        figsize: Tuple[int, int] = (10, 6),
        show_changepoints: bool = False,
        include_legend: bool = False,
    ) -> plt.figure:
        """
        Plot the forecast of a Gloria model, including trend line, predictions,
        and confidence intervals.

        Parameters
        ----------
        m : Gloria
            A trained Gloria model. Must contain historical data in `m.history`

        fcst : pd.DataFrame
            DataFrame with forecast results, including the columns: 'ds',
            'yhat', 'trend', 'observed_lower', and 'observed_upper'.

        ax : sns.axes.Axes, optional
            An existing matplotlib axis to draw on. If None, a new figure
            and axis will be created.

        uncertainty : bool, default=True
            Whether to plot the uncertainty/confidence intervals.

        xlabel : str, default='ds'
            Label for the x-axis.

        ylabel : str, default='y'
            Label for the y-axis.

        figsize : tuple of int, default=(10, 6)
            Figure size in inches. Used only when creating a new figure.

        show_changepoints : bool, default=False
            Whether to display significant changepoints on the plot.

        include_legend : bool, default=False
            Whether to display a legend on the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the forecast plot.
        """
        # Check if a custom axis was passed
        user_provided_ax = ax is not None

        # Create new figure and axis if none provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor="w")
        else:
            fig = ax.get_figure()

        # Set Seaborn style and update plot aesthetics
        sns.set(style="whitegrid")
        plt.rcParams.update(
            {
                "font.size": 14,
                "font.family": "DejaVu Sans",
                "axes.titlesize": 18,
                "axes.labelsize": 16,
                "legend.fontsize": 14,
                "axes.edgecolor": "#333333",
                "axes.linewidth": 1.2,
            }
        )

        # Plot historical data as scatter points
        sns.scatterplot(
            x=self.history["ds"],
            y=self.history["y"],
            ax=ax,
            label="Data",
            color="#016a86",
            edgecolor="w",
            s=20,
            alpha=0.7,
        )

        # Plot the model's trend line
        ax.plot(
            fcst["ds"],
            fcst["trend"],
            color="#264653",
            linewidth=1.0,
            alpha=0.8,
            label="Trend",
        )

        # Plot the forecast line
        ax.plot(
            fcst["ds"],
            fcst["yhat"],
            color="#e6794a",
            linewidth=1.5,
            label="Fit",
        )

        # Plot the confidence interval (if enabled)
        if uncertainty:
            ax.fill_between(
                fcst["ds"],
                fcst["observed_lower"],
                fcst["observed_upper"],
                color="#819997",
                alpha=0.3,
                label="Confidence Interval",
            )

        if show_changepoints:
            add_changepoints_to_plot(self, fcst, ax)

        # Set date format for x-axis
        locator = AutoDateLocator(interval_multiples=False)
        formatter = AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Set axis labels
        ax.set_xlabel(xlabel, labelpad=15)
        ax.set_ylabel(ylabel, labelpad=15)

        # Add gridlines (only horizontal)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.grid(visible=False, axis="x")

        # Remove top and right spines for cleaner look
        sns.despine(ax=ax)

        # Remove default legend unless specified
        try:
            ax.get_legend().remove()
        except AttributeError:
            pass

        if include_legend:
            ax.legend(frameon=True, shadow=True, loc="best", fontsize=10)

        # Adjust layout if we created the figure
        if not user_provided_ax:
            fig.tight_layout()

        return fig

    def plot_components(
        self: Self,
        fcst: pd.DataFrame,
        uncertainty: bool = True,
        weekly_start: int = 0,
        figsize: Tuple[int, int] | None = None,
    ) -> plt.figure:
        """
        Plot forecast components of a Gloria model using a modern Seaborn style

        Parameters
        ----------
        m : Gloria
            A fitted Gloria model containing seasonalities, events, regressors,
            and trend.

        fcst : pd.DataFrame
            Forecast dataframe from the model, used for plotting trend
            and uncertainty.

        uncertainty : bool, optional, default=True
            Whether to include uncertainty intervals in the trend component
            plot.

        weekly_start : int, optional, default=0
            Starting day of the week (0=Monday) for weekly seasonal plots.

        figsize : tuple of float, optional
            Figure size as (width, height). If not provided, it is calculated
            automatically to arrange subplots in a nearly square grid.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib Figure containing all component subplots.
        """

        # Set Seaborn style and Matplotlib parameters for consistent aesthetics
        sns.set(style="whitegrid")
        plt.rcParams.update(
            {
                "font.size": 14,
                "font.family": "DejaVu Sans",
                "axes.titlesize": 18,
                "axes.labelsize": 16,
                "legend.fontsize": 14,
                "axes.edgecolor": "#333333",
                "axes.linewidth": 1.2,
            }
        )

        # Define components to plot: always include 'trend'
        components = ["trend"]

        # Add seasonalities detected in the model
        components.extend(self.model_extra["seasonalities"].keys())

        # Add events if available
        if self.model_extra["events"].keys():
            components.append("events")

        # Add external regressors if available
        if self.model_extra["external_regressors"].keys():
            components.append("external_regressors")

        npanel = len(components)

        # Calculate number of rows and columns for subplot
        # grid (as square as possible)
        ncols = math.floor(math.sqrt(npanel))
        nrows = math.ceil(npanel / ncols)

        # Automatically determine figure size if not specified
        if not figsize:
            figsize = (int(4.5 * ncols), int(3.2 * nrows))

        # Create subplots with white background
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="w")

        # Flatten axes array for easy iteration, handle single subplot case
        axes = axes.flatten() if npanel > 1 else [axes]

        # Loop over components and call corresponding plot functions
        for ax, plot_name in zip(axes, components):
            if plot_name == "trend":
                plot_trend_component(
                    m=self,
                    fcst=fcst,
                    component="trend",
                    ax=ax,
                    uncertainty=uncertainty,
                )
            elif plot_name in self.model_extra["seasonalities"].keys():
                plot_seasonality_component(
                    m=self,
                    component=plot_name,
                    start_offset=weekly_start,
                    period=int(
                        np.floor(
                            self.model_extra["seasonalities"][plot_name].period
                        )
                    ),
                    ax=ax,
                )
            elif plot_name in ["events", "external_regressors"]:
                plot_event_component(m=self, component=plot_name, ax=ax)

            # Visual tuning: grid only on y-axis, remove x-axis grid,
            # remove top/right spines
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.grid(visible=False, axis="x")
            sns.despine(ax=ax)

        # Adjust layout to prevent overlap
        fig.tight_layout()

        return fig


if __name__ == "__main__":
    Gloria.Foster()
