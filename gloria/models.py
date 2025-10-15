# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This Module defines the Backend classes for all distribution models that can be
used in Gloria.
"""

### --- Module Imports --- ###
# Standard Library
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Type, Union, cast

# Third Party
import numpy as np
import pandas as pd
from cmdstanpy import (
    CmdStanLaplace,
    CmdStanMLE,
    CmdStanModel,
    set_cmdstan_path,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta, betabinom, binom, gamma, nbinom, norm, poisson
from typing_extensions import Self, TypeAlias

# Gloria
from gloria.utilities.constants import _CMDSTAN_VERSION

# Inhouse Packages
from gloria.utilities.errors import NotFittedError
from gloria.utilities.logging import get_logger
from gloria.utilities.types import Distribution

### --- Global Constants Definitions --- ###
BASEPATH = Path(__file__).parent


### --- Class and Function Definitions --- ###
def distance_to_scale(f, y: np.ndarray, value) -> float:
    """
    This function yields the distance between desired scale and data
    normalized to a capacity.
    """
    N = f * max(y)
    p = y / N
    return ((p - value) ** 2).sum()


def get_capacity(y: np.ndarray, mode: str, value: Union[int, float]) -> int:
    """
    Estimate a capacity suitable for binomial or beta-binomial models
    based on the response y and one of three selection modes.

    Parameters
    ----------
    y : np.ndarray
        Array of observed response variable data
    mode : str
        Mode for determining the capacity. Must be one of:
        - ``"constant"`` : Use a fixed capacity equal to ``value``.
        - ``"factor"`` : Set capacity to ``ceil(max(y) * value)``.
        - ``"scale"`` : Optimize capacity such that the implied success
          probability ``p = y / N`` is close to ``value``.
    value : int or float
        Mode-dependent parameter:
        - For ``"constant"``, the capacity (must be ≥ max(y)).
        - For ``"factor"``, a scaling factor ≥ 1 applied to max(y).
        - For ``"scale"``, the target mean success probability ``p``.

    Returns
    -------
    int
        Estimated capacity

    Raises
    ------
    ValueError
        If ``"constant"`` mode is selected and the capacity is less than
        max(y).

    """

    # Find maximum of data
    y_max = y.max()

    # Determine capacity depending on mode
    if mode == "constant":
        if value < y_max:
            raise ValueError(
                f"The capacity (={value}) must be an integer >= y_max "
                f"(={y_max})."
            )
        capacity = int(value)
    elif mode == "factor":
        capacity = int(np.ceil(y_max * value))
    elif mode == "scale":
        # Minimize distance_to_scale() with respect to the factor f. f
        # determines the capacity via N = y_max * f.
        res = minimize(
            lambda f: distance_to_scale(f, y, value),
            x0=1 / value,
            bounds=[(1, None)],
        )
        capacity = int(np.ceil(res.x[0] * y_max))

    return capacity


class LinkPair(BaseModel):
    """
    Link function pairs connection the expectation value to Stan's GLM
    predictors

    link = transforming expectation value to predictor
    inverse = transforming predictor to expectation value
    """

    link: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


LINK_FUNC_MAP = {
    "id": LinkPair(
        link=lambda x: x.astype(float), inverse=lambda x: x.astype(float)
    ),
    "log": LinkPair(link=lambda x: np.log(x), inverse=lambda x: np.exp(x)),
    "logit": LinkPair(link=lambda x: logit(x), inverse=lambda x: expit(x)),
}


class BinomialCapacity(BaseModel):
    """
    Configuration parameters used by the augment_data method of the model
    BinomialConstantN and BetaBinomialConstantN to determine the capacity
    size.
    """

    mode: Literal["constant", "factor", "scale"]
    value: Union[int, float]

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, value: Union[int, float], info
    ) -> Union[int, float]:
        """
        Validates the value pass along with the capacity size estimation
        method.
        """
        # Safeguard if validation of mode already failed
        if "mode" not in info.data:
            raise ValueError(
                "Can't validate 'value' field as 'mode' was invalid."
            )
        if info.data["mode"] == "constant":
            if not isinstance(value, int):
                raise ValueError(
                    "In capacity mode 'constant' the capacity"
                    f" value (={value}) must be an integer."
                )
        elif info.data["mode"] == "factor":
            if value < 1:
                raise ValueError(
                    "In capacity mode 'factor' the capacity "
                    f"value (={value}) must be >= 1."
                )
        elif info.data["mode"] == "scale":
            if (value >= 1) or (value <= 0):
                raise ValueError(
                    "In capacity mode 'scale' the capacity "
                    f"value (={value}) must be 0 < value < 1."
                )
        return value

    @classmethod
    def from_parameters(cls, capacity, capacity_mode, capacity_value):
        cap_is_given = capacity is not None
        mode_is_given = (
            capacity_mode is not None and capacity_value is not None
        )
        mode_is_incomplete = (capacity_mode is not None) ^ (
            capacity_value is not None
        )

        if mode_is_incomplete:
            raise ValueError(
                "Provide either both 'capacity_mode' and 'capacity_value', "
                "or neither."
            )
        if not (cap_is_given ^ mode_is_given):
            raise ValueError(
                "Provide either 'capacity' or a 'capacity_mode' / "
                "'capacity_value' pair."
            )
        if cap_is_given:
            capacity_mode = "constant"
            capacity_value = capacity

        return cls(mode=capacity_mode, value=capacity_value)


class ModelParams(BaseModel):
    """
    A container for the fitting parameter of each model. Additional model-
    dependent parameters have to be added within the model
    """

    model_config = ConfigDict(
        # Allows setting extra attributes during initialization
        extra="allow",
        # So the model accepts pandas object as values
        arbitrary_types_allowed=True,
    )

    k: float = 0  # Base trend growth rate
    m: float = 0  # Trend offset
    delta: np.ndarray = np.array([])  # Trend rate adjustments, length S
    beta: np.ndarray = np.array([])  # Slope for y, length K


class ModelInputData(BaseModel):
    """
    A container for the input data of each model. Additional model-dependent
    parameters have to be added within the model
    """

    model_config = ConfigDict(
        # Allows setting extra attributes during initialization
        extra="allow",
        # So the model accepts pandas object as values
        arbitrary_types_allowed=True,
    )

    T: int = Field(ge=0, default=0)  # Number of time periods
    S: int = Field(ge=0, default=0)  # Number of changepoints
    K: int = Field(ge=0, default=0)  # Number of regressors
    tau: float = Field(gt=0, default=3)  # Scale on changepoints prior
    gamma: float = Field(gt=0, default=3)  # Scale on dispersion proxy prior
    y: np.ndarray = np.array([])  # Time series
    t: np.ndarray = np.array([])  # Time as integer vector
    # Times of trend changepoints as integers
    t_change: np.ndarray = np.array([])
    X: np.ndarray = np.array([[]])  # Regressors
    sigmas: np.ndarray = np.array([])  # Scale on seasonality prior
    linked_offset: Optional[float] = None  # Data offset on linked scale
    linked_scale: Optional[float] = None  # Data scale on linked scale

    @field_validator("S")
    @classmethod
    def validate_S(cls, S: int, info: ValidationInfo) -> int:
        if S > info.data["T"]:
            raise ValueError(
                "Number of changepoints must be less or"
                " equal number of data points."
            )
        return S

    @field_validator("y")
    @classmethod
    def validate_y_shape(
        cls, y: np.ndarray, info: ValidationInfo
    ) -> np.ndarray:
        if len(y.shape) != 1:
            raise ValueError("Data array must be 1d-ndarray.")
        if info.data["T"] != len(y):
            raise ValueError("Length of y does not equal specified T")
        return y

    @field_validator("t")
    @classmethod
    def validate_t_shape(
        cls, t: np.ndarray, info: ValidationInfo
    ) -> np.ndarray:
        if len(t.shape) != 1:
            raise ValueError("Timestamp array must be 1d-ndarray.")
        if info.data["T"] != len(t):
            raise ValueError("Length of t does not equal specified T")
        return t

    @field_validator("t_change")
    @classmethod
    def validate_t_change_shape(
        cls, t_change: np.ndarray, info: ValidationInfo
    ) -> np.ndarray:
        if len(t_change.shape) != 1:
            raise ValueError("Changepoint array must be 1d-ndarray.")
        if info.data["S"] != len(t_change):
            raise ValueError("Length of t_change does not equal specified S")
        return t_change

    @field_validator("X")
    @classmethod
    def validate_X_shape(
        cls, X: np.ndarray, info: ValidationInfo
    ) -> np.ndarray:
        if len(X.shape) != 2:
            raise ValueError("Regressor matrix X must be 2d-ndarray.")
        # In case there are no regressors
        if X.shape[1] == 0:
            return X
        if info.data["T"] != X.shape[0]:
            raise ValueError(
                "Regressor matrix X must have same number of rows"
                " as timestamp."
            )
        if info.data["K"] != X.shape[1]:
            raise ValueError(
                "Regressor matrix X must have same number of"
                " columns as specified K."
            )
        return X

    @field_validator("sigmas")
    @classmethod
    def validate_sigmas(
        cls, sigmas: np.ndarray, info: ValidationInfo
    ) -> np.ndarray:
        if len(sigmas.shape) != 1:
            raise ValueError("Sigmas array must be 1d-ndarray.")
        if info.data["K"] != len(sigmas):
            raise ValueError("Length of sigmas does not equal specified K.")
        if not np.all(sigmas > 0):
            raise ValueError("All elements in sigmas must be greater than 0.")
        return sigmas


class Uncertainty(BaseModel):
    """
    Small container class for holding trend uncertainties
    """

    model_config = ConfigDict(
        # So the model accepts pandas object as values
        arbitrary_types_allowed=True,
    )

    lower: np.ndarray
    upper: np.ndarray


class ModelBackendBase(ABC):
    """
    Abstract base clase for the model backend.

    The model backend is in charge of passing data and model parameters to the
    stan code as well as distribution model dependent prediction
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = Path()
    # Kind of data (integer, float, ...). Is used for data validation
    kind = ""  # must be any combination of "biuf"
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["id"]

    def yhat_func(
        self: Self, linked_arg: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Produces the predicted values yhat.

        Parameters
        ----------
        linked_arg : np.ndarray
            Linked GLM output
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution.

        Returns
        -------
        np.ndarray
            Predicted values
        """
        # The base class yhat_func is simply an identity function, which can be
        # used by many models (normal, poisson, ...). Others like binomial need
        # their own implementation.
        return linked_arg

    def quant_func(
        self: Self, level: float, yhat: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution
        """
        return np.array([])

    def __init__(self: Self, model_name: str, install=True) -> None:
        """
        Initialize the model backend.

        Parameters
        ----------
        model_name : str
            Name of the model. Must match any of the keys in MODEL_MAP. This
            will be validated by the ModelBackend class
        """
        # Set explicit local CmdStan path to avoid conflicts with other CmdStan
        # installations
        models_path = Path(__file__).parent / "stan_models"
        cmdstan_path = models_path / f"cmdstan-{_CMDSTAN_VERSION}"
        set_cmdstan_path(str(cmdstan_path))
        # Initialize the Stan model
        self.model = CmdStanModel(
            stan_file=self.stan_file,  # Keep explicit stan_file for dev
            exe_file=self.stan_file.with_suffix(".exe"),
        )
        # Silence cmdstanpy logger
        stan_logger = logging.getLogger("cmdstanpy")
        stan_logger.setLevel(logging.CRITICAL)
        for handler in stan_logger.handlers:
            handler.setLevel(logging.CRITICAL)
        # Set the model name as attribute
        self.model_name = model_name
        # The following attributes are evaluated and set during fitting. For
        # the time being initialize them with defaults.
        self.stan_data = ModelInputData()
        self.stan_inits = ModelParams()
        self.linked_offset: float = 0.0
        self.linked_scale: float = 1.0
        # The type hint helps MyPy to recognize that the stan_fit objects have
        # the stan_variables() method. the '#type:ignore' let's us initialize
        # it with None
        self.stan_fit: Union[CmdStanMLE, CmdStanLaplace] = None  # type: ignore
        self.use_laplace = False
        self.fit_params: dict[str, Any] = dict()

    @abstractmethod
    def preprocess(
        self: Self, stan_data: ModelInputData, **kwargs: Any
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface

        Raises
        ------
        NotImplementedError
            Will be raised if the child-model class did not implement this
            method

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """
        pass

    def normalize_data(
        self: Self,
        y: np.ndarray,
        capacity: Optional[Union[int, np.ndarray]] = None,
        lower_bound: bool = False,
        upper_bound: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """
        Normalize response variable y, apply the model`s link function, and
        re-scale the result to the open unit interval ``(0, 1)``.

        The routine performs three steps:

        1. **Optional normalization** - If ``N`` is provided, divide ``y`` by
           ``N`` to convert counts to rates.
        2. **Boundary adjustment** - Replace exact 0 or 1 with a small offset
           when ``lower_bound`` or ``upper_bound`` is set to ``True``.
        3. **Link + scaling** - Apply the link function, then rescale the
           result to the ``(0, 1)`` interval.

        Parameters
        ----------
        y : np.ndarray
            Raw response variable.  Shape can be any, provided it broadcasts
            with ``N`` if ``N`` is an array.
        capacity : Optional[Union[int, np.ndarray]]
            Capacity size(s) for normalisation.  If ``None`` (default) no
            division is performed.  If an array is given it must have the same
            shape as ``y``.
        lower_bound : bool, optional
            Set to ``True`` when the response is bounded below at zero.
            Exact zeros are replaced with ``1e-10`` before the link
            transformation to avoid ``-inf``.
        upper_bound : bool, optional
            Set to ``True`` when the response is bounded above at one.
            Exact ones are replaced with ``1 - 1e-10`` before the link
            transformation to avoid ``+inf``.

        Returns
        -------
        y_scaled : TYPE
            The linked response min-max-scaled to lie strictly in ``(0, 1)``.
        linked_offset : TYPE
            The minimum of the linked, *un*-scaled response; add this to
            reverse the min-max scaling.
        linked_scale : TYPE
            The range (max - min) of the linked, un-scaled response; multiply
            by this to reverse the min-max scaling.
        """

        y_scaled = y.copy()

        # Normalize data with respect to capacity
        if capacity is not None:
            p = np.full(y_scaled.shape, 1e-10)
            # This safe-divide is necessary for models with vectorized N as
            # N = 0 can occur here
            y_scaled = np.divide(
                y_scaled, capacity, out=p, where=(capacity != 0)
            )
        # Replace true zeros with a small value as link function diverges
        # otherwise
        if lower_bound:
            y_scaled = np.where(y_scaled == 0, 1e-10, y_scaled)
        # Same for upper bounds. Note that all models with upper bound are
        # normalized between zero and one at this point
        if upper_bound:
            y_scaled = np.where(y_scaled == 1, 1 - 1e-10, y_scaled)

        # Apply link function
        y_scaled = self.link_pair.link(y_scaled)  # type: ignore[attr-defined]

        # Find offset and scale of data on linked scale
        linked_offset = np.min(y_scaled)
        linked_scale = np.max(y_scaled) - linked_offset
        y_scaled = (y_scaled - linked_offset) / linked_scale

        return y_scaled, linked_offset, linked_scale

    def initial_trend_parameters(
        self: Self, y_scaled: np.ndarray, stan_data: ModelInputData
    ) -> ModelParams:
        """
        Infers an estimation of the fit parameters k, m, delta from
        the data.

        Parameters
        ----------
        y_scaled : np.ndarray
            The input y-data scaled to the GLM depending on the model, eg.
            logit(y / N) for the binomial model
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface.

        Returns
        -------
        ModelParams
            Contains the estimations
        """

        t = stan_data.t

        # For models where y is unsigned, a cast to a signed type is necessary.
        # Otherwise the subtraction in calculating k can cause an overflow.
        y_scaled = y_scaled.copy().astype(float)

        # Step 1: Estimation of k and m, such that a straight line passes from
        # first and last data point
        T = t[-1] - t[0]
        k = (y_scaled[-1] - y_scaled[0]) / T
        m = y_scaled[0] - k * y_scaled[-1]

        # Step 2: Fit the clinear trend with changepoints to estimate delta.
        # self.piecewise_linear corresponds to the clinear function.

        def trend_optimizer(x: np.ndarray) -> float:
            """
            An optimizable function that is used to find a set of parameters
            minimizing the residual sum of squares for the trend model.
            """
            return float(
                (
                    (
                        self.piecewise_linear(
                            t,  # Timestamps as integer
                            stan_data.t_change,  # Changepoints
                            x[0],  # Trend offset
                            x[1],  # Base trend growth rate
                            x[2:],  # Trend rate adjustments
                        )
                        - y_scaled
                    )
                    ** 2
                ).sum()
            )

        # Optimize initial parameters
        res = minimize(trend_optimizer, x0=[m, k, *np.zeros(stan_data.S)])

        # Extract parameters
        m, k, delta = res.x[0], res.x[1], res.x[2:]

        # Restrict parameters to allowed range
        k = min(max(res.x[1], -0.5), 0.5)
        if m < 0:
            k_old = k
            k = k + m / stan_data.t_change[0]
            m = 0
            delta[0] = delta[0] + (k_old - k)
        elif m > 1:
            k_old = k
            k = k + (m - 1) / stan_data.t_change[0]
            m = 1
            delta[0] = delta[0] + (k_old - k)

        # Return initial parameters. Beta is left as zero. It can be
        # additionally pre-fitted using the pre-optimize flag in the interface
        # fit method. In that case the model backend fit method will use a
        # MAP estimate.
        return ModelParams(m=m, k=k, delta=delta, beta=np.zeros(stan_data.K))

    def estimate_variance(
        self: Self, stan_data: ModelInputData, trend_params: ModelParams
    ) -> float:
        """
        Estimate an upper bound for the variance based on the residuals between
        the observed data and the predicted trend.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface.
        trend_params : ModelParams
            Contains parameters from initial trend parameter guess.

        Returns
        -------
        float
            The upper bound of variance estimated from the residuals between
            data and trend.
        """

        # Based on the initial parameter guesses for the trend, make a trend
        # prediction

        # 1. Argument of the GLM
        trend_arg, _ = self.predict_regression(
            stan_data.t, stan_data.X, trend_params.model_dump()
        )

        # 2. Apply the inverse link
        trend_linked = self.link_pair.inverse(trend_arg)

        # 3. Get the expectation value
        trend = self.yhat_func(trend_linked)

        # Assume rediuals between data and trend are due to dispersion and
        # define this as upper bound for variance. Factor 1.5 for some leeway.
        variance_max = 1.5 * (stan_data.y - trend).std() ** 2

        return variance_max

    def fit(
        self: Self,
        stan_data: ModelInputData,
        optimize_mode: Literal["MAP", "MLE"],
        use_laplace: bool,
        capacity: Optional[int] = None,
        capacity_mode: Optional[str] = None,
        capacity_value: Optional[float] = None,
        vectorized: bool = False,
    ) -> Union[CmdStanMLE, CmdStanLaplace]:
        """
        Calculates initial parameters and fits the model to the input data.

        Parameters
        ----------
        stan_data : ModelInputData
            An object that holds the input data required by the data-block of
            the stan model.
        optimize_mode : Literal['MAP', 'MLE'], optional
            If 'MAP' (default), the optimization step yiels the Maximum A
            Posteriori, if 'MLE' the Maximum Likehood Estimate
        use_laplace : bool, optional
            If True (default), the optimization is followed by a sampling over
            the Laplace approximation around the posterior mode.
        capacity : int, optional
            An upper bound used for ``binomial`` and ``beta-binomial`` models.
            Specifying ``capacity`` is mutually exclusive with providing a
            ``capacity_mode`` and ``capacity_value`` pair.
        capacity_mode : str, optional
            A method used to estimate the capacity. Must be eitherr ``"scale"``
            or ``"factor"``.
        capacity_value : float, optional
            A value associated with the selected ``capacity_mode``.
        vectorized : bool
            If True, the capacity is already part of stan_data as a numpy
            array. If False the capacity must be constructed from capacity
            parameters

        Returns
        -------
        Union[CmdStanMLE, CmdStanLaplace]
            The fitted CmdStanModel object that holds the fitted parameters
        """

        jacobian = True if optimize_mode == "MAP" else False

        # Set raw version of stan_data
        # Do not remove: This step may seem unnecessary, but methods called
        # by self.preprocess() need self.stan_data
        self.stan_data = stan_data

        # The input stan_data only include data that all models have in common
        # The preprocess method adds additional data that are model dependent.
        # Additionally it estimates initial guesses for model parameters m, k,
        # and delta
        self.stan_data, self.stan_inits = self.preprocess(
            stan_data,
            # Pass capacity parameters as kwargs as only binomial and beta-
            # binomial models process them at all
            capacity=capacity,
            capacity_mode=capacity_mode,
            capacity_value=capacity_value,
            vectorized=vectorized,
        )

        # Scale regressors. The goal is to give each regressor a similar impact
        # on the model
        if stan_data.X.size:
            # Calculate the regressor strength as somewhat equivalent to a
            # physical field strength (sqrt of a signal's delivered power)
            reg_strength = np.sqrt((stan_data.X**2).sum(axis=0))
            # Normalization of the strength with respect to its median ensures
            # that most regressors won't be rescaled but only the too weak or
            # too strong.
            q = reg_strength / np.median(reg_strength)

            stan_data.X = stan_data.X / q

        # If the user wishes also initialize beta via an MAP estimation
        get_logger().debug(
            f"Optimizing model parameters using {optimize_mode}."
        )

        # Make local copy of stan data and vectorize capacity. Only the Stan-
        # files need the vectorized form even though the capacity is a single
        # value. Therefore, it can be discarded after the fit.
        stan_data_opt = self.stan_data.model_copy()
        if hasattr(stan_data_opt, "capacity") and not vectorized:
            stan_data_opt.capacity = (
                np.ones(stan_data_opt.T, dtype=int) * stan_data_opt.capacity
            )

        optimize_args = dict(
            data=stan_data_opt.model_dump(),
            inits=self.stan_inits.model_dump(),
            algorithm="BFGS",
            iter=int(1e4),
            jacobian=jacobian,
        )
        run_newton = True

        # Look for the largest possible initial step length that doesn't
        # lead to a fail of the line search
        get_logger().info("Starting optimization.")
        for init_alpha in [10 ** (-4 - i / 2) for i in range(0, 2 * 4)]:
            optimize_args["init_alpha"] = init_alpha
            try:
                # Do cast for mypy: cmdstanpy only accepts mappings, not dicts
                optimized_model = self.model.optimize(
                    **cast(Mapping[str, Any], optimize_args)
                )
            except RuntimeError:
                # If init_alpha fails, try the next one
                get_logger().debug(
                    f"Optimization with init_alpha={init_alpha} failed. "
                    "Moving to next."
                )
                continue
            else:
                # If it worked, finish loop and skip Newton fallback
                run_newton = False
                break

        if run_newton:
            # Remove init_alpha
            del optimize_args["init_alpha"]
            # Fall back on Newton
            get_logger().warning(
                "Optimization terminated abnormally. Falling back to Newton."
            )
            optimize_args["algorithm"] = "Newton"
            # Do cast for mypy: cmdstanpy only accepts mappings, not dicts
            optimized_model = self.model.optimize(
                **cast(Mapping[str, Any], optimize_args)
            )

        if use_laplace:
            get_logger().info("Starting Laplace sampling.")
            self.stan_fit = self.model.laplace_sample(
                data=stan_data.dict(), mode=optimized_model, jacobian=jacobian
            )
            self.use_laplace = True

        else:
            self.stan_fit = optimized_model
            self.use_laplace = False

        # Save relevant fit parameters in dictionary
        self.fit_params = {
            k: v
            for k, v in self.stan_fit.stan_variables().items()
            if k != "trend"
        }

        # Scale back both regressors and fit parameters
        if stan_data.X.size:
            self.stan_data.X *= q
            self.fit_params["beta"] /= q

        return self.stan_fit

    def predict(
        self: Self,
        t: np.ndarray,
        X: np.ndarray,
        interval_width: float,
        trend_samples: int,
        capacity_vec: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Based on the fitted model parameters predicts values and uncertainties
        for given timestamps.

        Parameters
        ----------
        t : np.ndarray
            Timestamps as integer values
        X : np.ndarray
            Overall feature matrix
        interval_width : float
            Confidence interval width: Must fall in [0, 1]
        trend_samples : int
            Number of samples to draw from
        capacity_vec : Optional[np.ndarray], optional
            Vectorized capacity - only relevant for models 'binomial' and
            'beta-binomial'. Default is None.

        Raises
        ------
        ValueError
            Is raised if the error was not fitted prior to prediction

        Returns
        -------
        result : pd.DataFrame
            Dataframe containing all predicted metrics, including
            uncertainties. The columns include:

            * yhat/trend: mean predicted value for overall model or trend
            * yhat/trend_upper/lower: uncertainty intervals for mean
              predicted values with respect to specified interval_width
            * observed_upper/lower: uncertainty intervals for observed
              values
            * '_linked' versions of all quantities except for 'observed'.
        """

        if self.fit_params == dict():
            raise NotFittedError("Can't predict prior to fit.")

        # Get optimized parameters (or their samples) from fit
        params_dict = self.fit_params

        # Calculate lower and upper percentile level from interval_width
        lower_level = (1 - interval_width) / 2
        upper_level = lower_level + interval_width

        # Evaluate trend uncertainties. As these only use the mean over the
        # sampled parameters, it is sufficient to call this function only once
        # outside the loop
        trend_uncertainty = self.trend_uncertainty(
            t, interval_width, trend_samples
        )

        # If we drew samples using the Laplace algorithm, self.use_laplace is
        # True. In this case we are able to get yhat uppers and lowers.
        if self.use_laplace:
            get_logger().info(
                "Evaluate model at all samples for yhat upper "
                "and lower bounds."
            )
            # Change dictionary of lists to list of dictionaries for looping
            params = [
                dict(zip(params_dict.keys(), t))
                for t in zip(*params_dict.values())
            ]

            # Single out scale parameter
            scale = (
                params_dict["scale"].mean() if "scale" in params_dict else None
            )

            # For each parameter sample produced by the fit method, calculate
            # the trend and overall yhat arguments and collect them in lists.
            yhat_arg_lst = []
            trend_arg_lst = []

            for pars in params:
                trend_arg, yhat_arg = self.predict_regression(t, X, pars)
                yhat_arg_lst.append(yhat_arg)
                trend_arg_lst.append(trend_arg)

            # Evaluate mean from the arguments as well as their upper and lower
            # percentiles for uncertainties
            yhat_args = np.array(yhat_arg_lst)
            trend_args = np.array(trend_arg_lst)

            yhat_arg = yhat_args.mean(axis=0)
            yhat_lower_arg = self.percentile(
                yhat_args, 100 * lower_level, axis=0  # type: ignore[arg-type]
            )
            yhat_upper_arg = self.percentile(
                yhat_args, 100 * upper_level, axis=0  # type: ignore[arg-type]
            )
            trend_arg = trend_args.mean(axis=0)

            # For the actual predictions, plug the arguments to the link
            # function and the yhat function
            yhat_linked = self.link_pair.inverse(yhat_arg)
            yhat_linked_lower = self.link_pair.inverse(
                yhat_lower_arg + trend_uncertainty.lower
            )
            yhat_linked_upper = self.link_pair.inverse(
                yhat_upper_arg + trend_uncertainty.upper
            )
            yhat = self.yhat_func(
                yhat_linked, scale=scale, capacity_vec=capacity_vec
            )
            yhat_lower = self.yhat_func(
                yhat_linked_lower, scale=scale, capacity_vec=capacity_vec
            )
            yhat_upper = self.yhat_func(
                yhat_linked_upper, scale=scale, capacity_vec=capacity_vec
            )
        else:
            trend_arg, yhat_arg = self.predict_regression(t, X, params_dict)
            scale = params_dict["scale"] if "scale" in params_dict else None

            yhat_linked = self.link_pair.inverse(yhat_arg)
            yhat_linked_lower = yhat_linked
            yhat_linked_upper = yhat_linked
            yhat = self.yhat_func(
                yhat_linked, scale=scale, capacity_vec=capacity_vec
            )
            yhat_lower = yhat
            yhat_upper = yhat

        trend_linked = self.link_pair.inverse(trend_arg)
        trend_linked_lower = self.link_pair.inverse(
            trend_arg + trend_uncertainty.lower
        )
        trend_linked_upper = self.link_pair.inverse(
            trend_arg + trend_uncertainty.upper
        )
        trend = self.yhat_func(
            trend_linked, scale=scale, capacity_vec=capacity_vec
        )
        trend_lower = self.yhat_func(
            trend_linked_lower, scale=scale, capacity_vec=capacity_vec
        )
        trend_upper = self.yhat_func(
            trend_linked_upper, scale=scale, capacity_vec=capacity_vec
        )
        # For the observed uncertainties, we need to plug the yhats into
        # the actual distribution function and evaluate their respective
        # quantiles
        observed_lower = self.quant_func(
            lower_level,
            yhat - trend + trend_lower,
            scale=scale,
            capacity_vec=capacity_vec,
        )
        observed_upper = self.quant_func(
            upper_level,
            yhat - trend + trend_upper,
            scale=scale,
            capacity_vec=capacity_vec,
        )

        # Reconstruct
        result = pd.DataFrame(
            {
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
                "yhat_linked": yhat_linked,
                "yhat_linked_lower": yhat_linked_lower,
                "yhat_linked_upper": yhat_linked_upper,
                "observed_lower": observed_lower,
                "observed_upper": observed_upper,
                "trend": trend,
                "trend_lower": trend_lower,
                "trend_upper": trend_upper,
                "trend_linked": trend_linked,
                "trend_linked_lower": trend_linked_lower,
                "trend_linked_upper": trend_linked_upper,
            }
        )

        return result

    def predict_regression(
        self: Self,
        t: np.ndarray,
        X: np.ndarray,
        pars: dict[str, Union[float, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate both trend and GLM argument from fitted model parameters

        Parameters
        ----------
        t : np.ndarray
            Timestamps as integer values
        X : np.ndarray
            Overall feature matrix
        pars : dict[str, Union[float, np.ndarray]]
            Dictionary containing initial rate k and offset m as well as rate
            changes delta

        Returns
        -------
        trend : np.ndarray
            The frend function
        np.ndarray
            Argument of the GLM

        """
        # First calculate the trend
        trend = self.predict_trend(t, pars)

        # If there are not regressors, we are already finished
        if self.stan_data.K == 0:
            return trend, trend
        # Otherwise calculate feature matrix for both additive and
        # multiplicative features
        beta = pars["beta"]
        Xb = np.matmul(X, beta)

        return (
            self.linked_offset + self.linked_scale * trend,
            self.linked_offset + self.linked_scale * (trend + Xb),
        )

    def predict_trend(
        self: Self, t: np.ndarray, pars: dict[str, Union[float, np.ndarray]]
    ) -> np.ndarray:
        """
        Predict the trend based on model parameters

        Parameters
        ----------
        t : np.ndarray
            Timestamps as integer values
        pars : dict[str, Union[float, np.ndarray]]
            Dictionary containing initial rate k and offset m as well as rate
            changes delta

        Returns
        -------
        trend : np.ndarray
            Predicted trend
        """
        # Get changepoints from input data, note that therefore this method
        # only works for historical data
        changepoints_int = self.stan_data.t_change

        # Extract fit parameters
        m = cast(float, pars["m"])
        k = cast(float, pars["k"])
        deltas = cast(np.ndarray, pars["delta"])

        # Get the trend
        trend = self.piecewise_linear(t, changepoints_int, m, k, deltas)

        return trend

    def piecewise_linear(
        self: Self,
        t: np.ndarray,
        changepoints_int: np.ndarray,
        m: float,
        k: float,
        deltas: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the piecewise linear trend function

        Parameters
        ----------
        t : np.ndarray
            Timestamps as integer values
        changepoints_int : np.ndarray
            Timestamps of changepoints as integer values
        m : float
            Trend offset
        k : float
            Base trend growth rate
        deltas : np.ndarray
            Trend rate adjustments, length S

        Returns
        -------
        np.ndarray
            The calculated trend
        """
        # Calculate the changepoint matrix times respective rate change
        deltas_t = (changepoints_int[None, :] <= t[..., None]) * deltas
        # Summing yields the rate for each timestamp
        k_t = deltas_t.sum(axis=1) + k
        # Offset per timestamp
        m_t = (deltas_t * -changepoints_int).sum(axis=1) + m
        return k_t * t + m_t

    def trend_uncertainty(
        self: Self, t: np.ndarray, interval_width: float, trend_samples: int
    ) -> Uncertainty:
        """
        Generates upper and lower bound estimations for the trend prediction.


        Parameters
        ----------
        t : np.ndarray
            Timestamps as integers
        interval_width : float
            Confidence interval width: Must fall in [0, 1]
        trend_samples : int
            Number of samples to draw from

        Returns
        -------
        upper : np.ndarray
            Upper bound of trend uncertainty
        lower : np.ndarray
            Lower bound of trend uncertainty

        """

        # If no samples were requested, return simply zero
        if trend_samples == 0:
            upper = np.zeros(t.shape)
            lower = np.zeros(t.shape)
            return Uncertainty(upper=upper, lower=lower)

        # Get the mean delta as Laplace distribution MLE
        mean_delta = np.abs(self.fit_params["delta"]).mean()

        # Separate into historic timestamps (zero uncertainty)
        t_history = t[t <= self.stan_data.t.max()]
        # ... and future timestamps (non-zero uncertainty)
        t_future = t[t > self.stan_data.t.max()]
        T_future = len(t_future)

        # Probability of finding a changepoint at a single timestamp
        likelihood = len(self.stan_data.t_change) / self.stan_data.T

        # Randomly choose timestamps with rate changes over all samples
        bool_slope_change = (
            np.random.uniform(size=(trend_samples, T_future)) < likelihood
        )
        # A matrix full of rate changes drawn from the Laplace distribution
        shift_values = np.random.laplace(
            scale=mean_delta, size=bool_slope_change.shape
        )
        # Multiplication of both yields the rate change at the changepoints,
        # otherwise zero
        shift_matrix = bool_slope_change * shift_values
        # First cumulative sum generates the rates at each timestamp
        # Second cumulative sum generates the y-values at each timestamp
        uncertainties = shift_matrix.cumsum(axis=1).cumsum(axis=1)

        # Get upper and lower bounds from percentiles
        lower_level = (1 - interval_width) / 2
        upper_level = lower_level + interval_width
        upper = np.percentile(uncertainties, 100 * upper_level, axis=0)
        lower = np.percentile(uncertainties, 100 * lower_level, axis=0)

        # Stitch together past and future uncertainties
        past_uncertainty = np.zeros(t_history.shape)
        upper = np.concatenate([past_uncertainty, upper])
        lower = np.concatenate([past_uncertainty, lower])

        return Uncertainty(upper=upper, lower=lower)

    def percentile(
        self: Self,
        a: np.ndarray,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """
        We rely on np.nanpercentile in the rare instances where there
        are a small number of bad samples with MCMC that contain NaNs.
        However, since np.nanpercentile is far slower than np.percentile,
        we only fall back to it if the array contains NaNs.
        """
        fn = np.nanpercentile if np.isnan(a).any() else np.percentile
        return fn(a, *args, **kwargs)  # type: ignore


class Binomial(ModelBackendBase):
    """
    Implementation of model backend for binomial distribution
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/binomial.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "bu"  # must be any combination of 'biuf'
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["logit"]

    def yhat_func(
        self: Self,
        linked_arg: np.ndarray,
        capacity_vec: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Produces the predicted values yhat

        Parameters
        ----------
        linked_arg : np.ndarray
            Linked GLM output
        capacity_vec : np.ndarray
            An array containing the capacity for each timestamp. Only used for
            prediction.

        Returns
        -------
        np.ndarray
            Predicted values

        """
        # In vectorized capacity mode the capacity is saved in stan_data for
        # fitting, but for predicting it's in capacity_vec
        capacity = (
            self.stan_data.capacity if capacity_vec is None else capacity_vec
        )
        return capacity * linked_arg

    def quant_func(
        self: Self,
        level: float,
        yhat: np.ndarray,
        capacity_vec: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        capacity_vec : np.ndarray
            An array containing the capacity for each timestamp. Only used for
            prediction.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        # In vectorized capacity mode the capacity is saved in stan_data for
        # fitting, but for predicting it's in capacity_vec
        capacity = (
            self.stan_data.capacity if capacity_vec is None else capacity_vec
        )
        return binom.ppf(level, capacity, yhat / capacity)

    def preprocess(
        self: Self,
        stan_data: ModelInputData,
        capacity: Optional[int] = None,
        capacity_mode: Optional[str] = None,
        capacity_value: Optional[float] = None,
        vectorized: bool = False,
        **kwargs: Any,
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface
        capacity : int, optional
            An upper bound used for ``binomial`` and ``beta-binomial`` models.
            Specifying ``capacity`` is mutually exclusive with providing a
            ``capacity_mode`` and ``capacity_value`` pair.
        capacity_mode : str, optional
            A method used to estimate the capacity. Must be eitherr ``"scale"``
            or ``"factor"``.
        capacity_value : float, optional
            A value associated with the selected ``capacity_mode``.
        vectorized : bool
            If True, the capacity is already part of stan_data as a numpy
            array. If False the capacity must be constructed from capacity
            parameters

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """

        ## -- 1. Augment stan_data -- ##

        if not vectorized:
            # Validate capacity parameters
            capacity_settings = BinomialCapacity.from_parameters(
                capacity=capacity,
                capacity_mode=capacity_mode,
                capacity_value=capacity_value,
            )

            # Get capacity size depending on selected mode
            stan_data.capacity = get_capacity(
                y=stan_data.y,
                mode=capacity_settings.mode,
                value=capacity_settings.value,
            )  # type: ignore[attr-defined]

        ## -- 2. Calculate initial parameter guesses -- ##
        # Get response variable on link scale and normalize
        y_scaled, self.linked_offset, self.linked_scale = self.normalize_data(
            y=stan_data.y,
            capacity=stan_data.capacity,
            lower_bound=True,
            upper_bound=True,
        )

        # Save normalization parameters
        stan_data.linked_offset = self.linked_offset
        stan_data.linked_scale = self.linked_scale

        # Calculate the parameters
        ini_params = self.initial_trend_parameters(y_scaled, stan_data)
        return stan_data, ini_params


class Normal(ModelBackendBase):
    """
    Implementation of model backend for normal distribution
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/normal.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "biuf"  # must be any combination of 'biuf'
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["id"]

    def quant_func(
        self: Self,
        level: float,
        yhat: np.ndarray,
        scale: float = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution. Equals observation noise for
            normal distribution.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        return norm.ppf(level, loc=yhat, scale=scale)

    def preprocess(
        self: Self, stan_data: ModelInputData, **kwargs: Any
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """

        ## -- 1. Augment stan_data -- ##
        # Nothing to augment

        ## -- 2. Calculate initial parameter guesses -- ##
        # Get response variable on link scale and normalize
        y_scaled, self.linked_offset, self.linked_scale = self.normalize_data(
            y=stan_data.y
        )

        # Save normalization parameters
        stan_data.linked_offset = self.linked_offset
        stan_data.linked_scale = self.linked_scale

        # Call the parent class parameter estimation method
        ini_params = self.initial_trend_parameters(y_scaled, stan_data)
        variance_max = self.estimate_variance(stan_data, ini_params)

        stan_data.variance_max = variance_max

        return stan_data, ini_params


class Poisson(ModelBackendBase):
    """
    Implementation of model backend for poisson distribution
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/poisson.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "bu"  # must be any combination of 'biuf'
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["log"]

    def quant_func(
        self: Self, level: float, yhat: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution. None for Poisson distribution.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        return poisson.ppf(level, yhat)

    def preprocess(
        self: Self, stan_data: ModelInputData, **kwargs: Any
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """
        ## -- 1. Augment stan_data -- ##
        # Nothing to augment

        ## -- 2. Calculate initial parameter guesses -- ##
        # Get response variable on link scale and normalize
        y_scaled, self.linked_offset, self.linked_scale = self.normalize_data(
            y=stan_data.y, lower_bound=True
        )

        # Save normalization parameters
        stan_data.linked_offset = self.linked_offset
        stan_data.linked_scale = self.linked_scale

        # Call the parent class parameter estimation method
        ini_params = self.initial_trend_parameters(y_scaled, stan_data)

        # Get an upper bound on the variance
        stan_data.variance_max = self.estimate_variance(stan_data, ini_params)

        return stan_data, ini_params


class NegativeBinomial(ModelBackendBase):
    """
    Implementation of model backend for negative binomial distribution
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/negative_binomial.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "bu"  # must be any combination of 'biuf'
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["log"]

    def quant_func(
        self: Self,
        level: float,
        yhat: np.ndarray,
        scale: float = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        # Calculate success probability. Note that phi has the meaning of
        # number of successes
        p = scale / (scale + yhat)
        return nbinom.ppf(level, n=scale, p=p)

    def preprocess(
        self: Self, stan_data: ModelInputData, **kwargs: Any
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """
        ## -- 1. Augment stan_data -- ##
        # Nothing to augment

        ## -- 2. Calculate initial parameter guesses -- ##
        # Apply inverse link function for y-value scaling. As the data are
        # scaled using the natural logarithm, zeros need to be replaced.
        y_scaled, self.linked_offset, self.linked_scale = self.normalize_data(
            y=stan_data.y, lower_bound=True
        )

        # Save normalization parameters
        stan_data.linked_offset = self.linked_offset
        stan_data.linked_scale = self.linked_scale

        # Call the parent class parameter estimation method
        ini_params = self.initial_trend_parameters(y_scaled, stan_data)

        # Get an upper bound on the variance
        stan_data.variance_max = self.estimate_variance(stan_data, ini_params)

        return stan_data, ini_params


class Gamma(ModelBackendBase):
    """
    Implementation of model backend for gamma distribution
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/gamma.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "biuf"  # must be any combination of 'biuf'
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["log"]

    def quant_func(
        self: Self,
        level: float,
        yhat: np.ndarray,
        scale: float = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        return gamma.ppf(level, yhat * scale, scale=1 / scale)

    def preprocess(
        self: Self, stan_data: ModelInputData, **kwargs: Any
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """
        ## -- 1. Augment stan_data -- ##
        # Nothing to augment

        ## -- 2. Calculate initial parameter guesses -- ##
        # Get response variable on link scale and normalize
        y_scaled, self.linked_offset, self.linked_scale = self.normalize_data(
            y=stan_data.y, lower_bound=True
        )

        # Save normalization parameters
        stan_data.linked_offset = self.linked_offset
        stan_data.linked_scale = self.linked_scale

        # Call the parent class parameter estimation method
        ini_params = self.initial_trend_parameters(y_scaled, stan_data)

        # Get an upper bound on the variance
        stan_data.variance_max = self.estimate_variance(stan_data, ini_params)

        return stan_data, ini_params


class Beta(ModelBackendBase):
    """
    Implementation of model backend for beta distribution
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/beta.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "f"  # must be any combination of 'biuf'
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["logit"]

    def quant_func(
        self: Self,
        level: float,
        yhat: np.ndarray,
        scale: float = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        # Calculate relation between standard beta distribution with parameters
        # a and b and parametrization according to Stan's beta_proportion with
        # parameters yhat and scale
        a = yhat * scale
        b = (1 - yhat) * scale
        return beta.ppf(level, a, b)

    def preprocess(
        self: Self, stan_data: ModelInputData, **kwargs: Any
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """
        ## -- 1. Augment stan_data -- ##
        # Nothing to augment

        ## -- 2. Calculate initial parameter guesses -- ##
        # Get response variable on link scale and normalize
        y_scaled, self.linked_offset, self.linked_scale = self.normalize_data(
            y=stan_data.y, lower_bound=True, upper_bound=True
        )

        # Save normalization parameters
        stan_data.linked_offset = self.linked_offset
        stan_data.linked_scale = self.linked_scale

        # Call the parent class parameter estimation method
        ini_params = self.initial_trend_parameters(y_scaled, stan_data)

        # Get an upper bound on the variance
        stan_data.variance_max = self.estimate_variance(stan_data, ini_params)

        return stan_data, ini_params


class BetaBinomial(ModelBackendBase):
    """
    Implementation of model backend for beta-binomial distribution
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/beta_binomial.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "bu"  # must be any combination of 'biuf'
    # Pair of 'link function'/'inverse link function'
    link_pair = LINK_FUNC_MAP["logit"]

    def yhat_func(
        self: Self,
        linked_arg: np.ndarray,
        scale: float = 1,
        capacity_vec: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Produces the predicted values yhat

        Parameters
        ----------
        linked_arg : np.ndarray
            Linked GLM output
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution.

        Returns
        -------
        np.ndarray
            Predicted values

        """
        # In vectorized capacity mode the capacity is saved in stan_data for
        # fitting, but for predicting it's in capacity_vec
        capacity = (
            self.stan_data.capacity if capacity_vec is None else capacity_vec
        )
        return capacity * linked_arg

    def quant_func(
        self: Self,
        level: float,
        yhat: np.ndarray,
        scale: float = 1,
        capacity_vec: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        scale : Union[float, np.ndarray, None]
            Scale parameter of the distribution.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        # In vectorized capacity mode the capacity is saved in stan_data for
        # fitting, but for predicting it's in capacity_vec
        capacity = (
            self.stan_data.capacity if capacity_vec is None else capacity_vec
        )

        # Average kappa if laplace was used
        if self.use_laplace:
            kappa = self.fit_params["kappa"].mean()
        else:
            kappa = self.fit_params["kappa"]

        # Do not use scale parameter from fit but calculate it based on
        # capacity and kappa. This way, scale will have the correct shape.
        scale = 4 * (capacity - 1) / (capacity * kappa**2) - 1

        # Calculate success probability
        p = yhat / capacity
        # Relate Stan model parameters to Scipy parameters for Beta-Binomial
        a = p * scale
        b = (1 - p) * scale
        return betabinom.ppf(level, capacity, a, b)

    def preprocess(
        self: Self,
        stan_data: ModelInputData,
        capacity: Optional[int] = None,
        capacity_mode: Optional[str] = None,
        capacity_value: Optional[float] = None,
        vectorized: bool = False,
        **kwargs: Any,
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface.
        capacity : int, optional
            An upper bound used for ``binomial`` and ``beta-binomial`` models.
            Specifying ``capacity`` is mutually exclusive with providing a
            ``capacity_mode`` and ``capacity_value`` pair.
        capacity_mode : str, optional
            A method used to estimate the capacity. Must be eitherr ``"scale"``
            or ``"factor"``.
        capacity_value : float, optional
            A value associated with the selected ``capacity_mode``.
        vectorized : bool
            If True, the capacity is already part of stan_data as a numpy
            array. If False the capacity must be constructed from capacity
            parameters

        Returns
        -------
        ModelInputData
            Updated stan_data
        ModelParams
            Guesses for the model parameters depending on the data

        """

        ## -- 1. Augment stan_data -- ##

        if not vectorized:
            # Validate capacity parameters
            capacity_settings = BinomialCapacity.from_parameters(
                capacity=capacity,
                capacity_mode=capacity_mode,
                capacity_value=capacity_value,
            )

            # Get capacity size depending on selected mode
            stan_data.capacity = get_capacity(
                y=stan_data.y,
                mode=capacity_settings.mode,
                value=capacity_settings.value,
            )  # type: ignore[attr-defined]

        ## -- 2. Calculate initial parameter guesses -- ##
        # Get response variable on link scale and normalize
        y_scaled, self.linked_offset, self.linked_scale = self.normalize_data(
            y=stan_data.y,
            capacity=stan_data.capacity,
            lower_bound=True,
            upper_bound=True,
        )

        # Save normalization parameters
        stan_data.linked_offset = self.linked_offset
        stan_data.linked_scale = self.linked_scale

        # Calculate the parameters
        ini_params = self.initial_trend_parameters(y_scaled, stan_data)

        # No need to define an upper bound on the variance for beta-binomial
        # model. The model itself has a clear upper bound.

        return stan_data, ini_params


# A TypeAlias for all existing Model Backends
ModelBackend: TypeAlias = Union[
    Binomial, Normal, Poisson, NegativeBinomial, Gamma, Beta, BetaBinomial
]

# Map model names to respective model backend classes
MODEL_MAP: dict[str, Type[ModelBackendBase]] = {
    "binomial": Binomial,
    "poisson": Poisson,
    "normal": Normal,
    "negative binomial": NegativeBinomial,
    "gamma": Gamma,
    "beta": Beta,
    "beta-binomial": BetaBinomial,
}


def get_model_backend(model: Distribution, **kwargs: Any) -> ModelBackend:
    """
    Creates a Model Backend Instance for the desired distribution type

    Parameters
    ----------
    model : Distribution
        The string representation of the desired distribution type
    **kwargs : dict[str, Any]
        Keyword arguments passed through to the model backend class constructor

    Raises
    ------
    NotImplementedError
        Raised if the requested model doesn't exist.

    Returns
    -------
    ModelBackend
        The instantiated model backend object

    """
    if model not in MODEL_MAP:
        raise NotImplementedError(f"Model {model} is not supported.")
    return cast(ModelBackend, MODEL_MAP[model](model, **kwargs))
