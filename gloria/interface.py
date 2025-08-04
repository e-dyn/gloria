# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Definition of the Gloria forecaster class
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path
from typing import Any, Collection, Literal, Optional, Type, Union, cast

# Third Party
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
from matplotlib.ticker import Formatter, Locator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from scipy.optimize import minimize
from typing_extensions import Self

# Gloria
import gloria.utilities.serialize as gs
from gloria.events import Event
from gloria.models import (
    MODEL_MAP,
    ModelInputData,
    distance_to_scale,
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
from gloria.utilities.configuration import (
    assemble_config,
    model_from_toml,
)

# Inhouse Packages
from gloria.utilities.constants import (
    _DELIM,
    _DTYPE_KIND,
    _FIT_DEFAULTS,
    _GLORIA_DEFAULTS,
    _LOAD_DATA_DEFAULTS,
    _PREDICT_DEFAULTS,
    _T_INT,
)
from gloria.utilities.errors import FittedError, NotFittedError
from gloria.utilities.logging import get_logger
from gloria.utilities.misc import cast_series_to_kind, time_to_integer
from gloria.utilities.types import Distribution, SeriesData, Timedelta


### --- Class and Function Definitions --- ###
class Gloria(BaseModel):
    """
    The Gloria forecaster object is the central hub for the entire modeling
    workflow.

    Gloria objects are initialized with parameters controlling the fit and
    prediction behaviour. Features such as ``seasonalities``, ``external
    regressors``, and ``events`` (or collection of such using ``protocols``)
    are added to Gloria objects. Once set up, :meth:`~Gloria.fit`,
    :meth:`~Gloria.predict`, or :meth:`~Gloria.plot` methods are available to
    fit the model to input data and visualize the results.

    Parameters
    ----------
    model : str
        The distribution model to be used. Can be any of ``"poisson"``,
        ``"binomial"``, ``"negative binomial"``, ``"gamma"``, ``"beta"``,
        ``"beta-binomial"``, or ``"normal"``. See :ref:`Model Selection
        <ref-model-selection>` tutorial for further information.
    sampling_period : Union[pd.Timedelta, str]
        Minimum spacing between two adjacent samples either as ``pd.Timedelta``
        or a compatible string such as ``"1d"`` or ``"20 min"``.
    timestamp_name : str, optional
        The name of the timestamp column as expected in the input data frame
        for :meth:`~Gloria.fit`.
    metric_name : str, optional
        The name of the expected metric column of the input data frame for
        :meth:`~Gloria.fit`.
    capacity_name : str, optional
        The name of the column containing capacity data for the models
        ``"binomial"`` and ``"beta-binomial"``.
    changepoints : pd.Series, optional
        List of timestamps at which to include potential changepoints. If not
        specified (default), potential changepoints are selected automatically.
    n_changepoints : int, optional
        Number of potential changepoints to include. Not used if input
        'changepoints' is supplied. If ``changepoints`` is not supplied, then
        ``n_changepoints`` potential changepoints are selected uniformly from
        the first ``changepoint_range`` proportion of the history. Must be a
        positive integer.
    changepoint_range : float, optional
        Proportion of history in which trend changepoints will be estimated.
        Must be in range [0,1]. Not used if ``changepoints`` is specified.
    seasonality_prior_scale : float, optional
        Parameter modulating the strength of the seasonality model. Larger
        values allow the model to fit larger seasonal fluctuations, smaller
        values dampen the seasonality. Can be specified for individual
        seasonalities using :meth:`add_seasonality`. Must be larger than 0.
    event_prior_scale : float, optional
        Parameter modulating the strength of additional event regressors.
        Larger values allow the model to fit larger event impact, smaller
        values dampen the event impact. Can be specified for individual
        events using :meth:`add_event`. Must be larger than 0.
    changepoint_prior_scale : float, optional
        Parameter modulating the flexibility of the automatic changepoint
        selection. Large values will allow many changepoints, small values will
        allow few changepoints. Must be larger than 0.
    dispersion_prior_scale : float, optional
        Parameter controlling the flexibility of the dispersion (i.e. allowed
        variance) of the model. Larger values allow more dispersion. This
        parameter does not affect the binomial and poisson model. Must be
        larger than one.
    interval_width : float, optional
        Width of the uncertainty intervals provided for the prediction. It is
        used for both uncertainty intervals of the expected value (fit) as
        well as the observed values (observed). Must be in range [0,1].
    uncertainty_samples : int, optional
        Number of simulated draws used to estimate uncertainty intervals of the
        *trend* in prediction periods that were not included in the historical
        data. Settings this value to 0 will disable uncertainty estimation.
        Must be greater equal to 0.
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
    sampling_period: Timedelta = _GLORIA_DEFAULTS["sampling_period"]
    timestamp_name: str = _GLORIA_DEFAULTS["timestamp_name"]
    metric_name: str = _GLORIA_DEFAULTS["metric_name"]
    capacity_name: str = _GLORIA_DEFAULTS["capacity_name"]
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
    dispersion_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["dispersion_prior_scale"]
    )
    interval_width: float = Field(
        gt=0, lt=1, default=_GLORIA_DEFAULTS["interval_width"]
    )
    uncertainty_samples: int = Field(
        ge=0, default=_GLORIA_DEFAULTS["uncertainty_samples"]
    )

    @field_validator("sampling_period")
    @classmethod
    def validate_sampling_period(
        cls: Type[Self], sampling_period: pd.Timedelta
    ) -> pd.Timedelta:
        """
        Converts sampling period to a pandas Timedelta if it was passed as a
        string instead.
        """

        if sampling_period <= pd.Timedelta(0):
            msg = "Sampling period must be positive and nonzero."
            get_logger().error(msg)
            raise ValueError(msg)

        return sampling_period

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
            self.changepoints = pd.Series(
                pd.to_datetime(self.changepoints), name=self.timestamp_name
            )
            self.n_changepoints = len(self.changepoints)

        # Instantiate the model backend that manages communication with the
        # Stan model and performs predictions based on the fit
        self.model_backend = get_model_backend(model=self.model)

        # If a capacity name was given, the capacity will be expected to be
        # part of the input data during fit() and predict(). The vectorized
        # attribute will indicate that.
        self.vectorized = (self.capacity_name != "") and self.model in (
            "binomial",
            "beta-binomial",
        )

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
        # 6. Dictionary holding kwargs passed to fit method
        self.fit_kwargs: dict[str, Any] = {}
        # 7. Configurations for fit and predict
        self._config: dict[str, Any] = {
            "fit": _FIT_DEFAULTS.copy(),
            "predict": _PREDICT_DEFAULTS.copy(),
            "load_data": _LOAD_DATA_DEFAULTS.copy(),
        }

    @property
    def is_fitted(self: Self) -> bool:
        """
        Determines whether the present :class:`Gloria` model is fitted.

        This property is *read-only*.

        Returns
        -------
        bool
            ``True`` if fitted, ``False`` otherwise.

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
            If the passed ``name`` is not a string
        ValueError
            Raised in case the ``name`` is not valid for any reason.
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
        if self.vectorized:
            reserved_names.append(self.capacity_name)

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
        Adds a seasonality to the Gloria object.

        From the seasonality, even and odd Fourier series components up to a
        user-defined maximum order will be generated and used as regressors
        during fitting and predicting.

        Parameters
        ----------
        name : str
            A descriptive name of the seasonality.
        period : str
            Fundamental period of the seasonality component. Should be a string
            compatible with ``pd.Timedelta`` (eg. ``"1d"`` or ``"12 h"``).
        fourier_order : int
            All Fourier terms from fundamental up to ``fourier_order`` will be
            used as regressors.
        prior_scale : float, optional
            The regression coefficient is given a prior with the specified
            scale parameter. Decreasing the prior scale will add additional
            regularization. If None is given self.seasonality_prior_scale will
            be used (default). Must be greater than 0.

        Raises
        ------
        :class:`~gloria.utilities.errors.FittedError`
            Raised in case the method is called on a fitted ``Gloria`` model.
        ValueError
            Raised when ``prior scale`` or ``period`` are not allowed values.

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
        Adds an event to the Gloria object.

        The event will be treated as a regressor during fitting and predicting.

        Parameters
        ----------
        name : str
            A descriptive name of the event.
        regressor_type : str
            Type of the underlying event regressor. Must be any of
            ``"SingleEvent"``, ``"IntermittentEvent"``, ``"PeriodicEvent"``,
            ``"Holiday"``
        event : Union[Event, dict[str, Any]]
            The base event used by the event regressor. Must be either of type
            :class:`Event` or a dictionary an event can be constructed from
            using :meth:`Event.from_dict`
        prior_scale : float
            The regression coefficient is given a prior with the specified
            scale parameter. Decreasing the prior scale will add additional
            regularization. Must be large than 0.
        include : Union[bool, Literal["auto"]]
            If set to ``"auto"`` (default), the event regressor will be
            excluded from the model during :meth:`fit`, if its overlap with the
            data is negligible. this behaviour can be overwritten by setting
            ``include`` to ``True`` or ``False``
        **regressor_kwargs : Any
            Additional keyword arguments necessary to create the event
            regressor specified by ``regressor_type``.

        Raises
        ------
        :class:`~gloria.utilities.errors.FittedError`
            Raised in case the method is called on a fitted ``Gloria`` model.
        ValueError
            Raised in case of invalid ``prior_scale`` or ``include`` values.

        Returns
        -------
        Gloria
            The ``Gloria`` model updated with the new event

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
        Add an external regressor to the Gloria object.

        The external regressor will  be used for fitting and predicting.

        Parameters
        ----------
        name : str
            A descriptive name of the regressor. The dataframes passed to
            :meth:`fit` and :meth:`predict` must have a column with the
            specified name. The values in these columns are used for the
            regressor.
        prior_scale : float
            The regression coefficient is given a prior with the specified
            scale parameter. Decreasing the prior scale will add additional
            regularization. Must be greater than 0.

        Raises
        ------
        :class:`~gloria.utilities.errors.FittedError`
            Raised in case the method is called on a fitted Gloria model.
        ValueError
            Raised in case of an invalid ``prior_scale`` value.

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
        Add a protocol to the Gloria object.

        Protocols provide additional, automated routines for setting up the
        model during :meth:`fit`. As of now, only the
        :class:`~gloria.CalendricData` protocol is implemented.

        Parameters
        ----------
        protocol : Protocol
            The Protocol object to be added.

        Raises
        ------
        :class:`~gloria.utilities.errors.FittedError`
            Raised in case the method is called on a fitted Gloria model.
        TypeError
            Raised when the provided ``protocol`` is not a valid Protocol
            object.

        Returns
        -------
        Gloria
            Updated Gloria model.

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
                "another one may lead to unexpected behaviour due to "
                "interferences."
            )
        self.protocols.append(protocol)

        return self

    def validate_metric_column(
        self: Self,
        df: pd.DataFrame,
        name: str,
        col_type: Literal["Metric", "Capacity"] = "Metric",
    ) -> None:
        """
        Validate that the metric column exists and contains only valid values.

        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame of data to be fitted.
        name : str
            The metric column name
        col_type : Literal["Metric", "Capacity"], optional
            Specifies whether the metric column or capacity column is to be
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
                f"{col_type} column '{name}' is missing from DataFrame."
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
            self.metric_name. If the Gloria models is 'binomial' and
            'beta-binomial' in vectorized capacity form are to be used, a
            column with name self.capacity_name must exists. If external
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
            get_logger().debug(
                "All timestamps are multiples of the sampling period, but gaps"
                " were found."
            )
        # Metric validation
        self.validate_metric_column(
            df=df, name=self.metric_name, col_type="Metric"
        )
        # Capcacity validation
        if self.vectorized:
            self.validate_metric_column(
                df=df, name=self.capacity_name, col_type="Capacity"
            )
            # Check values in the capacity column
            if (df[self.capacity_name] < df[self.metric_name]).any():
                raise ValueError(
                    "There are values in the metric column that exceed the "
                    "corresponding values in the capacity column, which is "
                    "not allowed for models 'binomial' and 'beta-binomial'."
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

        if self.vectorized:
            history[self.capacity_name] = df[self.capacity_name].copy()

        return history

    def set_changepoints(self: Self) -> Self:
        """
        Sets changepoints

        Sets changepoints to the dates and corresponding integer values of
        changepoints. The following cases are handled:

        1. The changepoints were passed in explicitly.
           a. They are empty.
           b. They are not empty, and need validation.
        2. We are generating a grid of them.
        3. The user prefers no changepoints be used.

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

            get_logger().debug(
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
        self.history.reset_index(drop=True, inplace=True)
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
            gamma=self.dispersion_prior_scale,
            y=np.asarray(self.history[self.metric_name]),
            t=np.asarray(self.history[_T_INT]),
            t_change=np.asarray(self.changepoints_int),
            X=self.X.values,
            sigmas=np.array(list(self.prior_scales.values())),
        )
        # Add capacity is vectorized, add it now
        if self.vectorized:
            input_data.capacity = np.asarray(self.history[self.capacity_name])
        return input_data

    def fit(
        self: Self,
        data: pd.DataFrame,
        toml_path: Optional[Union[str, Path]] = None,
        **kwargs: dict[str, Any],
    ) -> Self:
        """
        Fits the Gloria object.

        The fitting routine validates input data, sets up the model based on
        all input parameters, added regressors or protocols and eventually
        calls the model backend for the actual fitting.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame containing timestamp and metric columns named
            according to ``self.timestamp_name`` and ``self.metric_name``,
            respectively. If *external regressors* were added to the  model,
            the respective columns must be present as well.
        toml_path : Optional[Union[str, Path]], optional
            Path to an optional configuration TOML file that contains a section
            keyed by ``[fit]``. If *None* (default), TOML-configuration is
            skipped. TOML configuration precedes model settings saved in
            ``self._config`` as well as default settings.
        optimize_mode : str, optional
            If ``"MAP"`` (default), the optimization step yiels the Maximum A
            Posteriori estimation, if ``"MLE"`` a Maximum Likehood estimation.
        sample : bool, optional
            If ``True`` (default), the optimization is followed by a sampling
            over the Laplace approximation around the posterior mode.
        capacity : int, optional
            An upper bound used for ``binomial`` and ``beta-binomial`` models.
            Note that ``capacity`` must be >= all values in the metric column
            of ``data``. Default is ``None``. Specifying ``capacity`` is
            mutually exclusive with providing a  ``capacity_mode`` and
            ``capacity_value`` pair.
        capacity_mode : str, optional
            A method used to estimate the capacity. Two modes are available:
            - ``"factor"``: The capacity is the maximum of the response
              variable times ``capacity_value``
            - ``"scale"``: The capacity is optimized such that the response
              variable is distributed around the expectation value
              :math:`N \\times p` with :math:`N=` capacity and :math:`p=`
              ``capacity_value``. This mode is the default using
              ``capacity_value=0.5``.
        capacity_value : float, optional
            A value associated with the selected ``capacity_mode``:
            - If ``capacity_mode = "factor"``, ``capacity_value`` must be
              :math:`\\ge` 1
            - If ``capacity_mode = "factor"``, ``capacity_value`` must be in
              :math:`[0,1]`.

        Raises
        ------
        :class:`~gloria.utilities.errors.FittedError`
            Raised in case the method is called on a fitted ``Gloria`` model.

        Returns
        -------
        Gloria
            Updated Gloria object.

        Notes
        -----
        The configuration of the fit method via its parameters is composed in
        four layers, each one overriding the previous:

        1. **Model defaults** - the baseline configuration with defaults given
           above.
        2. **Global TOML file** - key-value pairs in the ``[fit]`` table of the
           TOML file passed to :meth:`Gloria.from_toml` if the current Gloria
           instance was created this way.
        3. **Local TOML file** - key-value pairs in the ``[fit]`` table of the
           TOML file provided for ``toml_path``.
        4. **Keyword overrides** - additional arguments supplied directly to
           the method take highest precedence.

        """
        if self.is_fitted:
            raise FittedError(
                "Gloria object can only be fit once. Instantiate a new object."
            )

        # Assemble overall config from different layers
        config = assemble_config(
            method="fit", model=self, toml_path=toml_path, **kwargs
        )

        self.fit_kwargs = dict(
            optimize_mode=config["optimize_mode"],
            sample=config["sample"],
            capacity=config["capacity"],
            capacity_mode=config["capacity_mode"],
            capacity_value=config["capacity_value"],
        )

        # Prepare the model and input data
        get_logger().debug("Starting to preprocess input data.")
        input_data = self.preprocess(data)

        # Fit the model
        get_logger().debug("Handing over preprocessed data to model backend.")
        self.model_backend.fit(
            input_data,
            optimize_mode=config["optimize_mode"],
            sample=config["sample"],
            capacity=config["capacity"],
            capacity_mode=config["capacity_mode"],
            capacity_value=config["capacity_value"],
            vectorized=self.vectorized,
        )

        return self

    def predict(
        self: Self,
        data: Optional[pd.DataFrame] = None,
        toml_path: Optional[Union[str, Path]] = None,
        **kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Generate forecasts from a *fitted* :class:`Gloria` model.

        Two usage patterns are supported:

        1. **Explicit input dataframe** - ``data`` contains future
           (or historical) timestamps plus any required external-regressor
           columns.

        2. **Auto-generated future dataframe** - leave ``data`` as ``None`` and
           supply the helper kwargs ``periods`` and/or ``include_history``.
           This shortcut only works when the model has *no* external
           regressors.


        Parameters
        ----------
        data : Optional[pd.DataFrame], optional
            A pandas DataFrame containing timestamp and metric columns named
            according to ``self.timestamp_name`` and ``self.metric_name``,
            respectively. If *external regressors* were added to the  model,
            the respective columns must be present as well. If ``None``, a
            future dataframe is produced with :meth:`make_future_dataframe`.
        toml_path : Optional[Union[str, Path]], optional
            Path to a TOML file whose ``[predict]`` section should be merged
            into the configuration. Ignored when ``None``.
        periods : int
            Number of future steps to generate. Must be a positive integer.
            Measured in units of``self.sampling_period``. The default is ``1``.
        include_history : bool, optional
            If ``True`` (default), the returned frame includes the historical
            dates that wereseen during fitting; if ``False`` it contains only
            the future portion.

        Returns
        -------
        prediction : pd.DataFrame
            A dataframe containing timestamps, predicted metric, trend, and
            lower and upper bounds.

        Notes
        -----
        The configuration of the predict method via ``periods`` and ``include``
        is composed in four layers, each one overriding the previous:

        1. **Model defaults** - the baseline configuration with defaults given
           above.
        2. **Global TOML file** - key-value pairs in the ``[predict]`` table of
           the TOML file passed to :meth:`Gloria.from_toml` if the current
           Gloria instance was created this way.
        3. **Local TOML file** - key-value pairs in the ``[predict]`` table of
           the TOML file provided for ``toml_path``.
        4. **Keyword overrides** - additional arguments supplied directly to
           the method take highest precedence.

        """
        if not self.is_fitted:
            raise NotFittedError("Can only predict using a fitted model.")

        # If there is no data input for the prediction, try to make a new
        # future dataframe
        if data is None:
            if len(self.external_regressors) > 0:
                raise ValueError(
                    "If the model has external regressors, data must be "
                    "explicitly provided."
                )
            # Assemble overall config from different layers
            config = assemble_config(
                method="predict", model=self, toml_path=toml_path, **kwargs
            )
            # Make future dataframe
            data = self.make_future_dataframe(**config)

        # At this point 'data' is a pd.DataFrame. Let MyPy know
        data = cast(pd.DataFrame, data)

        # Validate and extract capacity to pass it to predict
        capacity_vec = None
        if self.vectorized:
            self.validate_metric_column(
                df=data, name=self.capacity_name, col_type="Capacity"
            )
            capacity_vec = data[self.capacity_name].copy()

        # Validate external regressors
        # 1. Collect missing columns
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

        # 2. Check that columns are numeric and don't contain NaN
        for name in self.external_regressors:
            if data[name].dtype.kind not in "biuf":
                raise TypeError(f"Regressor column '{name}' is non-numeric.")
            if data[name].isnull().any():
                raise ValueError(f"Regressor column '{name}' contains NaN.")

        # First convert to integer timestamps with respect to first timestamp
        # and sampling_delta of training data
        data = data.copy().reset_index(drop=True)
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
            capacity_vec=capacity_vec,
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
        Build a timestamp skeleton that extends the training horizon.

        This helper is typically used when you plan to call :meth:`predict`.
        It produces a frame with a single column, named according to
        ``self.timestamp_name``, whose values

        * start one sampling step after the last training timestamp, and
        * continue for ``periods`` intervals spaced by
          ``self.sampling_period``.

        If ``include_history`` is ``True`` the original training timestamps are
        prepended, yielding a contiguous timeline from the first observed point
        up to the requested forecast horizon.

        Parameters
        ----------
        periods : int
            Number of future steps to generate. Must be a positive integer.
            Measured in units of ``self.sampling_period``. The default is
            ``1``.
        include_history : bool, optional
            If ``True`` (default), the returned frame includes the historical
            dates that wereseen during fitting; if ``False`` it contains only
            the future portion.

        Raises
        ------
        NotFittedError
            The model has not been fitted yet.
        TypeError
            If ``preriods`` is not an integer.
        ValueError
            If ``periods`` is < 1.

        Returns
        -------
        future_df : pd.DataFrame
            A dataframe with a single column ``self.timestamp_name`` containing
            ``pd.Timestamps``. It can be passed directly to :meth:`predict`
            if the model has no external regressors. When the model relies on
            external regressors you must merge the appropriate regressor
            columns into ``future_df`` before forecasting.

        """
        if not self.is_fitted:
            raise NotFittedError()

        if not isinstance(periods, int):
            raise TypeError(
                "Argument 'periods' must be an integer but is "
                f"{type(periods)}."
            )
        if periods < 1:
            raise ValueError("Argument 'periods' must be >= 1.")

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

    def load_data(
        self: Self,
        toml_path: Optional[Union[str, Path]] = None,
        **kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Load and configure the time-series input data for fit method.

        Reads a .csv-file that must contain at least two columns: a timestamp
        and a metric column named according to ``self.timestamp_name`` and
        ``self.metric_name``, respectively. The timestamp column is converted
        to a series of ``pd.Timestamps`` and the metric column is cast to
        ``dtype_kind``.

        Parameters
        ----------
        toml_path : Optional[Union[str, Path]], optional
            Path to a TOML file whose ``[load_data]`` section overrides the
            model defaults. Ignored when ``None``.
        source : Union[str, Path]
            Location of the CSV file to load the input data from. This key must
            be provided.
        dtype_kind : bool, optional
            Desired *kind* of the metric column as accepted by NumPy
            (``"u"`` unsigned int, ``"i"`` signed int, ``"f"`` float, ``"b"``
            boolean). If omitted, the metric dtype is cast to float.

        Returns
        -------
        data : pandas.DataFrame
            The preprocessed dataframe ready for modelling

        Notes
        -----
        The configuration of the ``load_data`` method via ``source`` and
        ``dtype_kind`` is composed in four layers, each one overriding the
        previous:

        1. **Model defaults** - the baseline configuration with defaults given
           above.
        2. **Global TOML file** - key-value pairs in the ``[load_data]`` table
           of the TOML file passed to :meth:`Gloria.from_toml` if the current
           Gloria instance was created this way.
        3. **Local TOML file** - key-value pairs in the ``[load_data]`` table
           of the TOML file provided for ``toml_path``.
        4. **Keyword overrides** - additional arguments supplied directly to
           the method take highest precedence.

        """

        # Assemble overall config from different layers
        config = assemble_config(
            method="load_data", model=self, toml_path=toml_path, **kwargs
        )

        # Read data
        data = pd.read_csv(config["source"])

        # Convert timestamp column to actual timestamps
        data[self.timestamp_name] = pd.to_datetime(data[self.timestamp_name])

        # Convert data to type required by model
        data[self.metric_name] = cast_series_to_kind(
            data[self.metric_name], config["dtype_kind"]
        )

        return data

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts Gloria object to a dictionary of JSON serializable types.

        Only works on fitted Gloria objects.

        The method calls :func:`model_to_dict` on ``self``.

        Returns
        -------
        dict[str, Any]
            JSON serializable dictionary containing data of Gloria model.

        """
        # Cf. to serialization module for details.
        return gs.model_to_dict(self)

    def to_json(
        self: Self, filepath: Optional[Path] = None, **kwargs: Any
    ) -> str:
        """
        Converts Gloria object to a JSON string.

        Only works on fitted Gloria objects. If desired the model is
        additionally dumped to a .json-file.

        The method calls :func:`model_to_json` on ``self``.

        Parameters
        ----------
        filepath : Optional[Path], optional
            Filepath of the target .json-file. If ``None`` (default) no output-
            file will be written.
        **kwargs : Any
            Keyword arguments which are passed through to :func:`json.dump` and
            :func:`json.dumps`

        Raises
        ------
        ValueError
            In case the given filepath does not have .json extension.

        Returns
        -------
        str
            JSON string containing the model data of the fitted Gloria object.


        """
        # Cf. to serialization module for details.
        return gs.model_to_json(self, filepath=filepath, **kwargs)

    @staticmethod
    def from_dict(model_dict: dict[str, Any]) -> "Gloria":
        """
        Restores a fitted Gloria model from a dictionary.

        The input dictionary must be the output of :func:`model_to_dict` or
        :meth:`Gloria.to_dict`.

        The method calls :func:`model_from_dict` on ``self``.

        Parameters
        ----------
        model_dict : dict[str, Any]
            Dictionary containing the Gloria object data.

        Returns
        -------
        Gloria
            Input data converted to a fitted Gloria object.
        """
        # Cf. to serialization module for details.
        return gs.model_from_dict(model_dict)

    @staticmethod
    def from_json(
        model_json: Union[Path, str],
        return_as: Literal["dict", "model"] = "model",
    ) -> Union[dict[str, Any], "Gloria"]:
        """
        Restores a fitted Gloria model from a json string or file.

        The input json string must be the output of :func:`model_to_json` or
        :meth:`Gloria.to_json`. If the input is a json-file, its contents is
        first read to a json string.

        The method calls :func:`model_from_json` on ``self``.

        Parameters
        ----------
        model_json : Union[Path, str]
            Filepath of .json-model file or string containing the data.
        return_as : Literal['dict', 'model'], optional
            If ``dict`` (default), the model is returned in dictionary format,
            if ``model`` as fitted Gloria object.

        Raises
        ------
        ValueError
            Two ValueErrors are possible:
            1. In case the given filepath does not have .json extension
            2. If ``return_as`` is neither ``"dict"`` nor ``"model"``

        Returns
        -------
        Union[dict[str, Any], Gloria]
            Gloria object or dictionary representing the Gloria object based on
            input json data.

        """
        # Cf. to serialization module for details.
        return gs.model_from_json(model_json, return_as)

    @staticmethod
    def from_toml(
        toml_path: Union[str, Path],
        ignore: Union[Collection[str], str] = set(),
        **kwargs: dict[str, Any],
    ) -> "Gloria":
        """
        Instantiate and configure a Gloria object from a TOML configuration
        file.

        The TOML file is expected to have the following top-level tables /
        arrays-of-tables (all are optional except ``[model]``):

        * ``[model]`` - keyword arguments passed directly to the
          :class:`Gloria` constructor.
        * ``[[external_regressors]]`` - one table per regressor; each is
          forwarded to :meth:`~Gloria.add_external_regressor`.
        * ``[[seasonalities]]`` - one table per seasonality; each is
          forwarded to :meth:`~Gloria.add_seasonality`.
        * ``[[events]]`` - one table per event; each is forwarded to
          :meth:`~Gloria.add_event`.
        * ``[[protocols]]`` - one table per protocol. Each table **must**
          contain a ``type`` key that maps to a protocol class name; the
          remaining keys are passed to that class before calling
          :meth:`~Gloria.add_protocol`.

        Defaults as defined in :class:`Gloria` constructor or respective
        methods are used for all keys not provided in the TOML file. ``kwargs``
        can be used to overwrite keys found in the ``[model]`` table.


        Parameters
        ----------
        toml_path : Union[str, Path]
            Path to the TOML file containing the model specification.
        ignore : Union[Collection[str],str], optional
            Which top-level sections of the file to skip. Valid values are
            ``"external_regressors"``, ``"seasonalities"``, ``"events"``, and
            ``"protocols"``. The special value ``"all"`` suppresses every
            optional section. May be given as a single string or any iterable
            of strings.
        **kwargs : dict[str, Any]
            Keyword arguments that override or extend the ``[model]`` table.
            Only keys that are valid fields of Gloria (i.e. that appear in
            Gloria.model_fields) are retained; others are silently dropped.

        Returns
        -------
        Gloria
            A fully initialized Gloria instance.


        .. seealso::

            :func:`model_from_toml`
                An alias

        Notes
        -----
        Precedence order for :class:`Gloria` constructor arguments from highest
        to lowest is:

        1. Values supplied via ``kwargs``
        2. Values found in the TOML ``[model]`` table
        3. Gloria's own defaults

        """
        # Cf. model_from_toml for details.
        return model_from_toml(toml_path, ignore, **kwargs)

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
        ax: Optional[plt.Axes] = None,
        uncertainty: Optional[bool] = True,
        show_changepoints: Optional[bool] = False,
        include_legend: Optional[bool] = False,
        mark_anomalies: Optional[bool] = False,
        show_capacity: Optional[bool] = False,
        plot_kwargs: Optional[dict[str, Any]] = None,
        rcparams_kwargs: Optional[dict[str, Any]] = None,
        style_kwargs: Optional[dict[str, Any]] = None,
        scatter_kwargs: Optional[dict[str, Any]] = None,
        trend_kwargs: Optional[dict[str, Any]] = None,
        forecast_kwargs: Optional[dict[str, Any]] = None,
        interval_kwargs: Optional[dict[str, Any]] = None,
        xlabel_kwargs: Optional[dict[str, Any]] = None,
        ylabel_kwargs: Optional[dict[str, Any]] = None,
        grid_y_kwargs: Optional[dict[str, Any]] = None,
        despine_kwargs: Optional[dict[str, Any]] = None,
        ticklabel_kwargs: Optional[dict[str, Any]] = None,
        anomaly_kwargs: Optional[dict[str, Any]] = None,
        capacity_kwargs: Optional[dict[str, Any]] = None,
        date_locator: Optional[Locator] = None,
        date_formatter: Optional[Formatter] = None,
    ) -> plt.Figure:
        """
        Plot the forecast, trend, and observed data with extensive
        customization options.

        Parameters
        ----------
        fcst : pandas.DataFrame
            Forecast DataFrame containing:
              - Timestamp column (matching `self.timestamp_name`)
              - 'yhat' (predicted values)
              - 'trend' (trend component)
              - 'observed_lower' and 'observed_upper' (confidence intervals)

        ax : matplotlib.axes.Axes, optional
            Existing matplotlib axis to draw on.
            If None, a new figure and axis will be created.

        uncertainty : bool, default=True
            Whether to plot the confidence interval bands.

        show_changepoints : bool, default=False
            Whether to annotate changepoints in the forecast.

        include_legend : bool, default=False
            Whether to include a legend in the plot.

        plot_kwargs : dict, optional
            Arguments for `plt.subplots()` if creating a new figure.

        rcparams_kwargs : dict, optional
            Overrides for matplotlib rcParams to control global styling.

        style_kwargs : dict, optional
            Keyword arguments passed to `sns.set()` for Seaborn style
            configuration.

        scatter_kwargs : dict, optional
            Styling for the historical data scatter plot (`sns.scatterplot`).

        trend_kwargs : dict, optional
            Styling for the trend line (`ax.plot`).

        forecast_kwargs : dict, optional
            Styling for the forecast line (`ax.plot`).

        interval_kwargs : dict, optional
            Styling for the confidence interval area (`ax.fill_between`).

        xlabel_kwargs : dict, optional
            Settings for the x-axis label (`ax.set_xlabel`).

        ylabel_kwargs : dict, optional
            Settings for the y-axis label (`ax.set_ylabel`).

        grid_y_kwargs : dict, optional
            Settings for the y-axis gridlines (`ax.grid`).

        despine_kwargs : dict, optional
            Arguments to `sns.despine()` for removing spines.

        ticklabel_kwargs : dict, optional
            Settings for customizing tick labels (rotation, alignment,
            fontsize).

        anomaly_kwargs: dict, optional
            Styling for the anomaly data scatter plot (`sns.scatterplot`).

        capacity_kwargs: dict, optional
            Styling for the capcity line (`ax.plot`).

        date_locator : matplotlib.ticker.Locator, optional
            Locator for x-axis ticks. Defaults to `AutoDateLocator`.

        date_formatter : matplotlib.ticker.Formatter, optional
            Formatter for x-axis tick labels. Defaults to `AutoDateFormatter`.

        Returns
        -------
        None

        Raises
        ------
        NotFittedError
            If the model has not been fitted before calling this method.

        Notes
        -----
        This method is designed for flexible, reproducible, and highly
        customizable visualization of forecasts and their uncertainty
        intervals. You can control nearly every aspect of the figure appearance
        via the provided keyword argument dictionaries.

        Examples
        --------
        Basic usage:
            >>> model.plot(fcst)

        Custom scatter point styling:
            >>> model.plot(fcst, scatter_kwargs={"s": 40,
                                                       "color": "purple"})

        Specifying figure size and dpi:
            >>> model.plot(fcst, plot_kwargs={"figsize": (12, 8),
                                                    "dpi": 200})
        """

        if not self.is_fitted:
            raise NotFittedError()

        # Initialize all kwargs to empty dicts if None
        plot_kwargs = plot_kwargs or {}
        rcparams_kwargs = rcparams_kwargs or {}
        style_kwargs = style_kwargs or {}
        scatter_kwargs = scatter_kwargs or {}
        trend_kwargs = trend_kwargs or {}
        forecast_kwargs = forecast_kwargs or {}
        interval_kwargs = interval_kwargs or {}
        xlabel_kwargs = xlabel_kwargs or {}
        ylabel_kwargs = ylabel_kwargs or {}
        grid_y_kwargs = grid_y_kwargs or {}
        despine_kwargs = despine_kwargs or {}
        ticklabel_kwargs = ticklabel_kwargs or {}
        anomaly_kwargs = anomaly_kwargs or {}
        capacity_kwargs = capacity_kwargs or {}

        # Set default Seaborn style
        style_defaults = dict(style="whitegrid")
        style_defaults.update(style_kwargs)
        sns.set(**style_defaults)

        # Prepare rcParams overrides
        rcparams_defaults = {
            "font.size": 14,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
        }
        rcparams_defaults.update(rcparams_kwargs)

        # Use context managers to isolate styling
        with plt.rc_context(rc=rcparams_defaults):
            with sns.axes_style(style_defaults):
                # Default axis labels if not provided
                if "xlabel" not in xlabel_kwargs:
                    xlabel_kwargs["xlabel"] = self.timestamp_name
                if "ylabel" not in ylabel_kwargs:
                    ylabel_kwargs["ylabel"] = self.metric_name

                # Create figure and axis if none provided
                user_provided_ax = ax is not None
                if ax is None:
                    plot_defaults = dict(
                        figsize=(10, 6), dpi=150, facecolor="w"
                    )
                    plot_defaults.update(plot_kwargs)
                    fig, ax = plt.subplots(**plot_defaults)
                else:
                    fig = ax.get_figure()

                # Default styles for each plot element
                scatter_defaults = dict(
                    color="#016a86",
                    edgecolor="w",
                    s=20,
                    alpha=0.7,
                    label="Observed",
                    ax=ax,
                )
                scatter_defaults.update(scatter_kwargs)

                trend_defaults = dict(
                    color="#264653",
                    linewidth=1.0,
                    alpha=0.8,
                    label="Trend",
                )
                trend_defaults.update(trend_kwargs)

                forecast_defaults = dict(
                    color="#e6794a",
                    linewidth=1.5,
                    label="Forecast",
                )
                forecast_defaults.update(forecast_kwargs)

                interval_defaults = dict(
                    color="#819997",
                    alpha=0.3,
                    label="Confidence Interval",
                )
                interval_defaults.update(interval_kwargs)

                # Plot observed data as scatter
                sns.scatterplot(
                    x=self.history[self.timestamp_name],
                    y=self.history[self.metric_name],
                    **scatter_defaults,
                )

                # Plot trend line
                ax.plot(
                    fcst[self.timestamp_name],
                    fcst["trend"],
                    **trend_defaults,
                )

                # Plot forecast line
                ax.plot(
                    fcst[self.timestamp_name],
                    fcst["yhat"],
                    **forecast_defaults,
                )

                # Plot uncertainty intervals if enabled
                if uncertainty:
                    ax.fill_between(
                        fcst[self.timestamp_name],
                        fcst["observed_lower"],
                        fcst["observed_upper"],
                        **interval_defaults,
                    )

                # Add changepoints if enabled
                if show_changepoints:
                    add_changepoints_to_plot(self, fcst, ax)

                # Configure date ticks
                locator = date_locator or AutoDateLocator(
                    interval_multiples=False
                )
                formatter = date_formatter or AutoDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                # Axis labels
                ax.set_xlabel(**xlabel_kwargs)
                ax.set_ylabel(**ylabel_kwargs)

                # Y-axis grid
                grid_y_defaults = dict(
                    visible=True,
                    axis="y",
                    linestyle="--",
                    alpha=0.3,
                )
                grid_y_defaults.update(grid_y_kwargs)
                ax.grid(**grid_y_defaults)

                # Hide X-axis grid
                ax.grid(visible=False, axis="x")

                # Remove top and right spines
                sns.despine(ax=ax, **despine_kwargs)

                # Customize tick labels
                ticklabel_defaults = dict(
                    rotation=45,
                    horizontalalignment="right",
                )
                ticklabel_defaults.update(ticklabel_kwargs)
                for label in ax.get_xticklabels():
                    label.set_rotation(ticklabel_defaults.get("rotation", 0))
                    label.set_horizontalalignment(
                        ticklabel_defaults.get("horizontalalignment", "center")
                    )
                    if "fontsize" in ticklabel_defaults:
                        label.set_fontsize(ticklabel_defaults["fontsize"])
                    if "color" in ticklabel_defaults:
                        label.set_color(ticklabel_defaults["color"])

                # Apply tight layout if figure was created here
                if not user_provided_ax:
                    fig.tight_layout()

                # Add a red scatter plot showing the anomalies if requested
                if mark_anomalies:
                    # Default styles for each plot element
                    anomaly_defaults = dict(
                        color="#ff0000",
                        edgecolor="w",
                        s=20,
                        alpha=0.7,
                        label="Anomalies",
                        ax=ax,
                    )
                    anomaly_defaults.update(anomaly_kwargs)

                    # Extra boolean column that is true if a data point is
                    # outside the confidence interval
                    fcst["is_anomaly"] = (
                        fcst["observed_upper"] < self.history[self.metric_name]
                    ) | (  # upper anomalies
                        fcst["observed_lower"] > self.history[self.metric_name]
                    )  # lower anomalies

                    sns.scatterplot(
                        x=self.history.loc[
                            fcst["is_anomaly"], self.timestamp_name
                        ],
                        y=self.history.loc[
                            fcst["is_anomaly"], self.metric_name
                        ],
                        **anomaly_defaults,
                    )

                # Add a line showing the capacity if requested
                if show_capacity and self.model in {
                    "binomial",
                    "beta-binomial",
                }:

                    # Extract capacity-related settings
                    fit_kwargs = self.model_extra.get("fit_kwargs", {})
                    capacity = fit_kwargs.get("capacity")
                    capacity_mode = fit_kwargs.get("capacity_mode")
                    capacity_value = fit_kwargs.get("capacity_value")

                    if capacity_mode == "vectorized":
                        # Plot capacity directly if capacity is vectorized
                        capacity_defaults = {
                            "color": "grey",
                            "linestyle": "--",
                            "label": self.capacity_name,
                        }
                        capacity_defaults.update(capacity_kwargs)

                        ax.plot(
                            self.history[self.timestamp_name],
                            self.history[self.capacity_name],
                            **capacity_defaults,
                        )

                    elif (
                        capacity_mode in {"constant", "factor", "scale"}
                        or capacity is not None
                    ):
                        # Plot a constant or derived capacity line
                        capacity_defaults = {
                            "color": "grey",
                            "linestyle": "--",
                            "label": "capacity",
                        }
                        capacity_defaults.update(capacity_kwargs)

                        # Determine the capacity value based on mode
                        if capacity_mode == "scale":
                            # Scale capacity_value to match metric scale
                            res = minimize(
                                lambda f: distance_to_scale(
                                    f,
                                    self.history[self.metric_name],
                                    capacity_value,
                                ),
                                x0=1 / capacity_value,
                                bounds=[(1, None)],
                            )
                            plot_value = int(
                                np.ceil(
                                    res.x[0]
                                    * max(self.history[self.metric_name])
                                )
                            )

                        elif capacity_mode == "factor":
                            # Multiply capacity_value by max metric
                            plot_value = capacity_value * np.ceil(
                                max(self.history[self.metric_name])
                            )

                        else:
                            # Use constant capacity or fallback value
                            plot_value = capacity_value or capacity
                            print(plot_value)
                        # Plot the capacity line
                        ax.plot(
                            self.history[self.timestamp_name],
                            [plot_value] * len(self.history),
                            **capacity_defaults,
                        )

                elif show_capacity and self.model not in {
                    "binomial",
                    "beta-binomial",
                }:
                    # Warn user if capacity is requested for incompatible
                    # model
                    print(
                        "Capacities are only considered with the binomial "
                        "or beta-binomial model. It is therefore not an "
                        "option to display the capacities."
                    )

                # Remove Seaborn's default legend
                try:
                    ax.get_legend().remove()
                except AttributeError:
                    pass

                # Add legend if requested
                if include_legend:
                    ax.legend(
                        frameon=True, shadow=True, loc="best", fontsize=10
                    )

                plt.show()
        return

    def plot_components(
        self: Self,
        fcst: pd.DataFrame,
        uncertainty: bool = True,
        weekly_start: int = 0,
        plot_kwargs: Optional[dict[str, Any]] = None,
        line_kwargs: Optional[dict[str, Any]] = None,
        interval_kwargs: Optional[dict[str, Any]] = None,
        xlabel_kwargs: Optional[dict[str, Any]] = None,
        ylabel_kwargs: Optional[dict[str, Any]] = None,
        grid_y_kwargs: Optional[dict[str, Any]] = None,
        despine_kwargs: Optional[dict[str, Any]] = None,
        ticklabel_kwargs: Optional[dict[str, Any]] = None,
        rcparams_kwargs: Optional[dict[str, Any]] = None,
        style_kwargs: Optional[dict[str, Any]] = None,
    ) -> plt.Figure:
        """
        Plot forecast components of a Gloria model using a modern Seaborn
        style, with global kwargs applied to all subplots.

        Parameters
        ----------
        fcst : :class:`pandas.DataFrame`
            Forecast DataFrame from the model, used for plotting trend and
            uncertainty.
        uncertainty : bool, default True
            Whether to include uncertainty intervals in the trend component
            plot.
        weekly_start : int, default 0
            Starting day of the week (0=Monday) for weekly seasonal plots.
        plot_kwargs : dict, optional
            Keyword arguments passed to matplotlib.subplots() for figure and
            axes creation (e.g., figsize, dpi).
        line_kwargs : dict, optional
            Styling kwargs for lines in all components (e.g., color, linewidth)
        interval_kwargs : dict, optional
            Styling kwargs for uncertainty intervals in all components
            (e.g., alpha, color).
        xlabel_kwargs : dict, optional
            Keyword arguments for x-axis labels in all components.
        ylabel_kwargs : dict, optional
            Keyword arguments for y-axis labels in all components.
        grid_y_kwargs : dict, optional
            Keyword arguments for customizing the y-axis grid appearance.
        despine_kwargs : dict, optional
            Keyword arguments passed to seaborn.despine() for spine removal.
        ticklabel_kwargs : dict, optional
            Keyword arguments to customize tick labels in all components.
        rcparams_kwargs : dict, optional
            Matplotlib rcParams overrides for all components (e.g., font sizes)
        style_kwargs : dict, optional
            Seaborn style kwargs for all components (e.g., style presets).

        Returns
        -------
        None
        """
        # Third Party
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        # Initialize default kwargs dictionaries
        plot_kwargs = plot_kwargs or {}
        line_kwargs = line_kwargs or {}
        interval_kwargs = interval_kwargs or {}
        xlabel_kwargs = xlabel_kwargs or {}
        ylabel_kwargs = ylabel_kwargs or {}
        grid_y_kwargs = grid_y_kwargs or {}
        despine_kwargs = despine_kwargs or {}
        ticklabel_kwargs = ticklabel_kwargs or {}
        rcparams_kwargs = rcparams_kwargs or {}
        style_kwargs = style_kwargs or {}

        # Set Seaborn style and Matplotlib parameters for consistent aesthetics
        style_defaults = {"style": "whitegrid"}
        style_defaults.update(style_kwargs)
        sns.set(**style_defaults)

        rcparams_defaults = {
            "font.size": 14,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
        }
        rcparams_defaults.update(rcparams_kwargs)
        plt.rcParams.update(rcparams_defaults)

        # Use context managers to isolate styling
        with plt.rc_context(rc=rcparams_defaults):
            with sns.axes_style(style_defaults):

                # Define components to plot: always include 'trend'
                components = ["trend"]
                components.extend(self.model_extra["seasonalities"].keys())
                if self.model_extra["events"].keys():
                    components.append("events")
                if self.model_extra["external_regressors"].keys():
                    components.append("external_regressors")

                npanel = len(components)
                ncols = int(np.floor(np.sqrt(npanel)))
                nrows = int(np.ceil(npanel / ncols))

                # Default subplots configuration
                plot_defaults = {
                    "nrows": nrows,
                    "ncols": ncols,
                    "figsize": (int(4.5 * ncols), int(3.2 * nrows)),
                    "facecolor": "w",
                    "dpi": 150,
                }
                plot_defaults.update(plot_kwargs)

                # Create figure and axes
                fig, axes = plt.subplots(**plot_defaults)

                if isinstance(axes, np.ndarray):
                    axes = axes.flatten()
                else:
                    axes = [axes]

                for ax, comp in zip(axes, components):
                    if comp == "trend":
                        plot_trend_component(
                            m=self,
                            fcst=fcst,
                            component="trend",
                            ax=ax,
                            uncertainty=uncertainty,
                            plot_kwargs=plot_kwargs,
                            line_kwargs=line_kwargs,
                            interval_kwargs=interval_kwargs,
                            xlabel_kwargs=xlabel_kwargs,
                            ylabel_kwargs=ylabel_kwargs,
                            grid_y_kwargs=grid_y_kwargs,
                            ticklabel_kwargs=ticklabel_kwargs,
                            rcparams_kwargs=rcparams_kwargs,
                            style_kwargs=style_kwargs,
                        )
                    elif comp in self.model_extra["seasonalities"].keys():
                        plot_seasonality_component(
                            m=self,
                            component=comp,
                            start_offset=weekly_start,
                            period=int(
                                np.rint(
                                    self.model_extra["seasonalities"][
                                        comp
                                    ].period
                                )
                            ),
                            ax=ax,
                            plot_kwargs=plot_kwargs,
                            line_kwargs=line_kwargs,
                            interval_kwargs=interval_kwargs,
                            xlabel_kwargs=xlabel_kwargs,
                            ylabel_kwargs=ylabel_kwargs,
                            grid_y_kwargs=grid_y_kwargs,
                            ticklabel_kwargs=ticklabel_kwargs,
                            rcparams_kwargs=rcparams_kwargs,
                            style_kwargs=style_kwargs,
                        )
                    elif comp in ["events", "external_regressors"]:
                        plot_event_component(
                            m=self,
                            component=comp,
                            ax=ax,
                            plot_kwargs=plot_kwargs,
                            line_kwargs=line_kwargs,
                            interval_kwargs=interval_kwargs,
                            xlabel_kwargs=xlabel_kwargs,
                            ylabel_kwargs=ylabel_kwargs,
                            grid_y_kwargs=grid_y_kwargs,
                            ticklabel_kwargs=ticklabel_kwargs,
                            rcparams_kwargs=rcparams_kwargs,
                            style_kwargs=style_kwargs,
                        )

                    # Remove top and right spines for each subplot
                    sns.despine(ax=ax, **despine_kwargs)

                # Hide empty axes if any
                for ax in axes[npanel:]:
                    ax.set_visible(False)

                fig.tight_layout()

                plt.show()
        return

    ...
