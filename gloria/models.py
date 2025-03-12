"""
This Module defines the Backend classes for all distribution models that can be
used in Gloria.

ModelBackendBase:
    - Defines the basic interface for all backend classes including fitting and
      prediction methods as well as initial parameter estimation and input data
      augmentation
BinomialConstantN, Poisson, Normal:
    - Implementations of ModelBackendBase for different distribution models.
      Their differences are mostly found in appropriatly scaling the data with
      respect to the model's link functions.
BinomialPopulation, ModelParams, ModelInputData:
    - Classes that help validating input data passed from Gloria


TODO:
    - Docstring for the module
    - Appropriate regressor scaling. Idea: (1) the piece-wise-linear estimation
      in calculate_initial_parameters also returns the residuals
        res = y_scaled - trend
      (2) Find the scale of the residuals, eg. their standard deviation
      (3) Set regressor scales to the residual scale
      Also consider this in conjunction with estimating initial values for all
      beta. And probably reconsider reparametrizing the Stan-models.
    - Repair regressor scaling when there is no regressor
"""

### --- Module Imports --- ###
# Standard Library
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union, cast

# Third Party
import numpy as np
import pandas as pd
from cmdstanpy import (
    CmdStanLaplace,
    CmdStanMLE,
    CmdStanModel,
    install_cmdstan,
    set_cmdstan_path,
)
from pydantic import BaseModel, Field, field_validator
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binom, nbinom, norm, poisson
from typing_extensions import Self, TypeAlias

# Gloria
# Inhouse Packages
from gloria.constants import _CMDSTAN_VERSION
from gloria.types import Distribution
from gloria.utilities import calculate_dispersion, get_logger

stan_logger = logging.getLogger("cmdstanpy")
stan_logger.setLevel(logging.ERROR)

### --- Global Constants Definitions --- ###
BASEPATH = Path(__file__).parent


### --- Class and Function Definitions --- ###
class BinomialPopulation(BaseModel):
    """
    Configuration parameters used by the augment_data method of the model
    BinomialConstantN to determine the population size. For more info cf the
    method docstring.
    """

    mode: Literal["constant", "factor", "scale"]
    value: Union[int, float]

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: int, info) -> int:
        """
        Validates the value pass along with the population size estimation
        method.
        """
        if info.data["mode"] == "constant":
            if not isinstance(value, int):
                raise ValueError(
                    "In population mode 'constant' the population"
                    f" value (={value}) must be an integer."
                )
        elif info.data["mode"] == "factor":
            if value < 1:
                raise ValueError(
                    "In population mode 'factor' the population "
                    f"value (={value}) must be >= 1."
                )
        elif info.data["mode"] == "scale":
            if (value >= 1) or (value <= 0):
                raise ValueError(
                    "In population mode 'scale' the population "
                    f"value (={value}) must be 0 < value < 1."
                )
        return value


class ModelParams(BaseModel):
    """
    A container for the fitting parameter of each model. Additional model-
    dependent parameters have to be added within the model
    """

    k: float = 0  # Base trend growth rate
    m: float = 0  # Trend offset
    delta: np.ndarray = np.array([])  # Trend rate adjustments, length S
    beta: np.ndarray = np.array([])  # Slope for y, length K

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class ModelInputData(BaseModel):
    """
    A container for the input data of each model. Additional model-dependent
    parameters have to be added within the model
    """

    T: int = Field(ge=0, default=0)  # Number of time periods
    S: int = Field(ge=0, default=0)  # Number of changepoints
    K: int = Field(ge=0, default=0)  # Number of regressors
    tau: float = Field(gt=0, default=1)  # Scale on changepoints prior
    y: np.ndarray = np.array([])  # Time series
    t: np.ndarray = np.array([])  # Time as integer vector
    t_change: np.ndarray = np.array(
        []
    )  # Times of trend changepoints as integers
    X: np.ndarray = np.array([[]])  # Regressors
    sigmas: np.ndarray = np.array([])  # Scale on seasonality prior
    s_a: np.ndarray = np.array([])  # Indicator of additive features
    s_m: np.ndarray = np.array([])  # Indicator of multiplicative features

    @field_validator("S")
    @classmethod
    def validate_S(cls, S: int, info) -> int:
        if S > info.data["T"]:
            raise ValueError(
                "Number of changepoints must be less or"
                " equal number of data points."
            )
        return S

    @field_validator("y")
    @classmethod
    def validate_y_shape(cls, y: np.ndarray, info) -> np.ndarray:
        if len(y.shape) != 1:
            raise ValueError("Data array must be 1d-ndarray.")
        if info.data["T"] != len(y):
            raise ValueError("Length of y does not equal specified T")
        return y

    @field_validator("t")
    @classmethod
    def validate_t_shape(cls, t: np.ndarray, info) -> np.ndarray:
        if len(t.shape) != 1:
            raise ValueError("Timestamp array must be 1d-ndarray.")
        if info.data["T"] != len(t):
            raise ValueError("Length of t does not equal specified T")
        return t

    @field_validator("t_change")
    @classmethod
    def validate_t_change_shape(cls, t_change: np.ndarray, info) -> np.ndarray:
        if len(t_change.shape) != 1:
            raise ValueError("Changepoint array must be 1d-ndarray.")
        if info.data["S"] != len(t_change):
            raise ValueError("Length of t_change does not equal specified S")
        return t_change

    @field_validator("X")
    @classmethod
    def validate_X_shape(cls, X: np.ndarray, info) -> np.ndarray:
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
    def validate_sigmas(cls, sigmas: np.ndarray, info) -> np.ndarray:
        if len(sigmas.shape) != 1:
            raise ValueError("Sigmas array must be 1d-ndarray.")
        if info.data["K"] != len(sigmas):
            raise ValueError("Length of sigmas does not equal specified K.")
        if not np.all(sigmas > 0):
            raise ValueError("All elements in sigmas must be greater than 0.")
        return sigmas

    @field_validator("s_a")
    @classmethod
    def validate_s_a(cls, s_a: np.ndarray, info) -> np.ndarray:
        if len(s_a.shape) != 1:
            raise ValueError("s_a array must be 1d-ndarray.")
        if info.data["K"] != len(s_a):
            raise ValueError("Length of s_a does not equal specified K.")
        if not all([s in [0, 1] for s in s_a]):
            raise ValueError("All elements of s_a must be either 0 or 1.")
        return s_a

    @field_validator("s_m")
    @classmethod
    def validate_s_m(cls, s_m: np.ndarray, info) -> np.ndarray:
        if len(s_m.shape) != 1:
            raise ValueError("s_m array must be 1d-ndarray.")
        if info.data["K"] != len(s_m):
            raise ValueError("Length of s_m does not equal specified K.")
        if not all((info.data["s_a"] + s_m) == 1):
            raise ValueError("s_m must be complimentary to s_a.")
        return s_m

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class Uncertainty(BaseModel):
    """
    Small container class for holding trend uncertainties
    """

    lower: np.ndarray
    upper: np.ndarray

    class Config:
        arbitrary_types_allowed = True


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

    def link_func(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Link function as used in Stan model.
        """
        return np.array([])

    def yhat_func(self: Self, linked_arg: np.ndarray) -> np.ndarray:
        """
        Produces the predicted values yhat
        """
        return np.array([])

    def quant_func(
        self: Self, level: float, yhat: np.ndarray, **kwargs: dict[str, Any]
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
        # If not yet installed, install CmdStan with desired version
        if not cmdstan_path.is_dir():
            get_logger().info(
                f"Cannot find cmdstan version {_CMDSTAN_VERSION}"
                ". Installing now."
            )
            install_cmdstan(version=_CMDSTAN_VERSION, dir=str(models_path))
        set_cmdstan_path(str(cmdstan_path))
        # Initialize the Stan model
        self.model = CmdStanModel(stan_file=self.stan_file)
        # Set the model name as attribute
        self.model_name = model_name
        # The following attributes are evaluated and set during fitting. For
        # the time being initialize them with None.
        self.stan_data = ModelInputData()
        self.stan_inits = ModelParams()
        # The type hint helps MyPy to recognize that the stan_fit objects have
        # the stan_variables() method. the '#type:ignore' let's us initialize
        # it with None
        self.stan_fit: Union[CmdStanMLE, CmdStanLaplace] = None  # type: ignore
        self.sample = False
        self.fit_params: dict[str, Any] = dict()

    @abstractmethod
    def preprocess(
        self: Self,
        stan_data: ModelInputData,
        augmentation_config: Optional[BinomialPopulation] = None,
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface
        augmentation_config : Optional[BinomialPopulation], optional
            Configuration parameters for the augmentation process. Currently,
            it is only required for the BinomialConstantN model. For all other
            models it defaults to None.

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
        raise NotImplementedError("preprocess() method not implemented.")
        return stan_data, ModelParams()

    def calculate_initial_parameters(
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

        # Step 1: Estimation of k and m, such that a straight line passes from
        # first and last data point
        T = t[-1] - t[0]
        k = (y_scaled[-1] - y_scaled[0]) / T
        m = y_scaled[0] - k * y_scaled[-1]

        # Step 2: Fit the clinear trend with changepoints to estimate delta
        # self.piecewise_linear corresponds to the clinear function.

        # trend_optimizer: an optimizable function that is used to find
        # a set of parameters minimizing the residual sum of squares for the
        # trend model.
        def trend_optimizer(x: np.ndarray) -> float:
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

        # Return initial parameters. Beta is left as zero. It can be
        # additionally pre-fitted using the pre-optimize flag in the interface
        # fit method. In that case the model backend fit method will use a
        # MAP estimate.
        return ModelParams(
            m=res.x[0],
            k=res.x[1],
            delta=np.array(res.x[2:]),
            beta=np.zeros(stan_data.K),
        )

    def fit(
        self: Self,
        stan_data: ModelInputData,
        optimize_mode: Literal["MAP", "MLE"] = "MAP",
        sample: bool = True,
        augmentation_config: Optional[BinomialPopulation] = None,
        **kwargs: dict[str, Any],
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
        sample : bool, optional
            If True (default), the optimization is followed by a sampling over
            the Laplace approximation around the posterior mode.
        augmentation_config : Optional[BinomialPopulation], optional
            Configuration parameters for the augment_data method. Currently, it
            is only required for the BinomialConstantN model. For all other
            models it defaults to None.
        **kwargs : dict[str, Any]
            Additional arguments that are passed to the fit method

        Returns
        -------
        Union[CmdStanMLE, CmdStanLaplace]
            The fitted CmdStanModel object that holds the fitted parameters
        """

        jacobian = True if optimize_mode == "MAP" else False

        # Scale regressors. The idea is to give each regressor a similar
        # impact on the model, hence the normalization to its own root sum of
        # squares. The additional normalization of the factor with respect to
        # its median ensures that most regressors won't be rescaled but only
        # the too weak or too strong.
        factor = np.sqrt((stan_data.X**2).sum(axis=0))

        factor /= np.median(factor)

        stan_data.X = stan_data.X / factor

        # The input stan_data only include data that all models have in common
        # The preprocess method adds additional data that are model dependent.
        # Additionally it estimates initial guesses for model parameters m, k,
        # and delta
        self.stan_data, self.stan_inits = self.preprocess(
            stan_data, augmentation_config
        )

        # If the user wishes also initialize beta via an MAP estimation
        get_logger().debug(
            "Optimizing model parameters using" f" {optimize_mode}."
        )
        optimized_model = self.model.optimize(
            data=stan_data.dict(),
            inits=self.stan_inits.dict(),
            iter=int(1e4),
            jacobian=jacobian,
        )

        if sample:
            get_logger().info(
                f"Starting Laplace sampling with {sample} " "samples."
            )
            self.stan_fit = self.model.laplace_sample(
                data=stan_data.dict(),
                mode=optimized_model,
                jacobian=jacobian,
                **kwargs,  # type: ignore
            )
            self.sample = True

        else:
            self.stan_fit = optimized_model
            self.sample = False

        # Save relevant fit parameters in dictionary
        self.fit_params = {
            k: v
            for k, v in self.stan_fit.stan_variables().items()
            if k != "trend"
        }
        # Scale back both regressors and fit parameters
        self.stan_data.X *= factor
        self.fit_params["beta"] /= factor

        # In case of the normal model the data were normalized by the Stan
        # model. Therefore the optimized model parameters need to be scaled
        # back
        if self.model_name == "normal":
            y_min = stan_data.y.min()
            y_max = stan_data.y.max()
            for k in self.fit_params.keys():
                self.fit_params[k] *= y_max - y_min
            self.fit_params["m"] += y_min

        return self.stan_fit

    def predict(
        self: Self,
        t: np.ndarray,
        X: np.ndarray,
        interval_width: float,
        n_samples: int,
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
        n_samples : int
            Number of samples to draw from

        Raises
        ------
        ValueError
            Is raised if the error was not fitted prior to prediction

        Returns
        -------
        result : pd.DataFrame
            Dataframe containing all predicted metrics, including
            uncertainties. The columns include:
                - yhat/trend: mean predicted value for overall model or trend
                - yhat/trend_upper/lower: uncertainty intervals for mean
                  predicted values with respect to specified interval_width
                - observed_upper/lower: uncertainty intervals for observed
                  values
        """

        if self.fit_params is None:
            raise ValueError("Can't predict prior to fit.")

        # Get optimized parameters (or their samples) from fit
        params_dict = self.fit_params

        # Calculate lower and upper percentile level from interval_width
        lower_level = (1 - interval_width) / 2
        upper_level = lower_level + interval_width

        # Evaluate trend uncertainties. As these only use the mean over the
        # sampled parameters, it is sufficient to call this function only once
        # outside the loop
        trend_uncertainty = self.trend_uncertainty(
            t, interval_width, n_samples
        )

        # If we drew samples using the Laplace algorithm, self.sample is True
        # In this case we are able to get yhat uppers and lowers.
        if self.sample:
            get_logger().info(
                "Evaluate model at all samples for yhat upper "
                "and lower bounds."
            )
            # Change dictionary of lists to list of dictionaries for looping
            params = [
                dict(zip(params_dict.keys(), t))
                for t in zip(*params_dict.values())
            ]

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
            yhat = self.yhat_func(self.link_func(yhat_arg))
            yhat_lower = self.yhat_func(
                self.link_func(yhat_lower_arg + trend_uncertainty.lower)
            )
            yhat_upper = self.yhat_func(
                self.link_func(yhat_upper_arg + trend_uncertainty.upper)
            )
        else:
            trend_arg, yhat_arg = self.predict_regression(t, X, params_dict)

            yhat = self.yhat_func(self.link_func(yhat_arg))
            yhat_lower = yhat
            yhat_upper = yhat

        trend = self.yhat_func(self.link_func(trend_arg))
        trend_lower = self.yhat_func(
            self.link_func(trend_arg + trend_uncertainty.lower)
        )
        trend_upper = self.yhat_func(
            self.link_func(trend_arg + trend_uncertainty.upper)
        )
        # For the observed uncertainties, we need to plug the yhats into
        # the actual distribution function and evaluate their respective
        # quantiles
        quant_kwargs = dict()
        if self.model_name == "normal":
            if isinstance(self.stan_fit, CmdStanLaplace):
                sigma = np.array([p["sigma_obs"] for p in params]).mean()
                quant_kwargs["sigma"] = sigma
            else:
                quant_kwargs["sigma"] = params_dict["sigma_obs"]
        elif self.model_name == "negative binomial":
            if isinstance(self.stan_fit, CmdStanLaplace):
                phi = np.array([p["phi"] for p in params]).mean()
                quant_kwargs["phi"] = phi
            else:
                quant_kwargs["phi"] = params_dict["phi"]
        observed_lower = self.quant_func(
            lower_level, yhat - trend + trend_lower, **quant_kwargs
        )
        observed_upper = self.quant_func(
            upper_level, yhat - trend + trend_upper, **quant_kwargs
        )

        # Reconstruct
        result = pd.DataFrame(
            {
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
                "observed_lower": observed_lower,
                "observed_upper": observed_upper,
                "trend": trend,
                "trend_lower": trend_lower,
                "trend_upper": trend_upper,
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
        Xb_a = np.matmul(X, beta * self.stan_data.s_a)
        Xb_m = np.matmul(X, beta * self.stan_data.s_m)

        return trend, trend * (1 + Xb_m) + Xb_a

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
        self: Self, t: np.ndarray, interval_width: float, n_samples: int
    ) -> Uncertainty:
        """
        Generates upper and lower bound estimations for the trend prediction.


        Parameters
        ----------
        t : np.ndarray
            Timestamps as integers
        interval_width : float
            Confidence interval width: Must fall in [0, 1]
        n_samples : int
            Number of samples to draw from

        Returns
        -------
        upper : np.ndarray
            Upper bound of trend uncertainty
        lower : np.ndarray
            Lower bound of trend uncertainty

        """

        # If no samples were requested, return simply zero
        if n_samples == 0:
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
            np.random.uniform(size=(n_samples, T_future)) < likelihood
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


class BinomialConstantN(ModelBackendBase):
    """
    Implementation of model backend for binomial distribution with constant N
    """

    # These class attributes must be defined by each model backend
    # Location of the stan file
    stan_file = BASEPATH / "stan_models/binomial_constant_n.stan"
    # Kind of data (integer, float, ...). Is used for data validation
    kind = "bu"  # must be any combination of 'biuf'

    def link_func(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Link function as used in Stan model

        Parameters
        ----------
        x : np.ndarray
            Output of the GLM

        Returns
        -------
        np.ndarray
            Linked GLM output

        """
        return expit(x)

    def yhat_func(self: Self, linked_arg: np.ndarray) -> np.ndarray:
        """
        Produces the predicted values yhat

        Parameters
        ----------
        linked_arg : np.ndarray
            Linked GLM output

        Returns
        -------
        np.ndarray
            Predicted values

        """
        return self.stan_data.N * linked_arg  # type: ignore[attr-defined]

    def quant_func(
        self: Self, level: float, yhat: np.ndarray, **kwargs: dict[str, Any]
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        return binom.ppf(level, self.stan_data.N, yhat / self.stan_data.N)  # type: ignore[attr-defined]

    def preprocess(
        self: Self,
        stan_data: ModelInputData,
        augmentation_config: Optional[BinomialPopulation] = None,
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface
        augmentation_config : BinomialPopulation
            Contains configuration parameters 'mode' and 'value' used to
            determine the population size. Three modes are supported:
            1. 'constant': value equals the population size
            2. 'factor': the population size is the maximum of y times value
            3. 'scale': the population size is a value optimized such that the
                data are distributed around the expectation value N*p with
                p = value and N = population size

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

        ## -- 1. Augment stan_data -- ##
        def distance_to_scale(f, y: np.ndarray) -> float:
            """
            This function yields the distance between desired scale and data
            normalized to a fixed population size N.
            """
            N = f * y_max
            p = y / N
            return ((p - value) ** 2).sum()

        if augmentation_config is None:
            raise ValueError(
                "Missing configuration argument for augmentation method."
            )

        # Prepare data
        y = stan_data.y
        y_max = y.max()
        mode = augmentation_config.mode
        value = augmentation_config.value
        # Determine population size depending on mode
        if mode == "constant":
            if value < y_max:
                raise ValueError(
                    "In population mode 'constant' the population"
                    f" value (={value}) cannot be smaller than "
                    f"y_max (={y_max})"
                )
            population = value
        elif mode == "factor":
            population = int(np.ceil(y_max * value))
        elif mode == "scale":
            # Minimize distance_to_scale() with respect to the factor f. f
            # determines the population size N via N = y_max * f.
            res = minimize(
                lambda f: distance_to_scale(f, y),
                x0=1 / value,
                bounds=[(1, None)],
            )
            population = int(np.ceil(res.x[0] * y_max))

        # Estimate a constant population size N for the binomial model
        stan_data.N = population  # type: ignore[attr-defined]

        ## -- 2. Calculate initial parameter guesses -- ##
        # Apply inverse link function for y-value scaling. Replacing the zeros
        # with small values prevents underflow during scaling
        y_scaled = np.where(stan_data.y == 0, 1e-10, stan_data.y)
        y_scaled = logit(y_scaled / stan_data.N)  # type: ignore[attr-defined]

        # Calculate the parameters
        ini_params = self.calculate_initial_parameters(y_scaled, stan_data)
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

    def link_func(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Link function as used in Stan model

        Parameters
        ----------
        x : np.ndarray
            Output of the GLM

        Returns
        -------
        np.ndarray
            Linked GLM output

        """
        return x

    def yhat_func(self: Self, linked_arg: np.ndarray) -> np.ndarray:
        """
        Produces the predicted values yhat

        Parameters
        ----------
        linked_arg : np.ndarray
            Linked GLM output

        Returns
        -------
        np.ndarray
            Predicted values

        """
        return linked_arg

    def quant_func(
        self: Self, level: float, yhat: np.ndarray, **kwargs: dict[str, Any]
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.
        kwargs : dict[str, Any]
            Must contain "sigma" as scale of the normal distribution

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        return norm.ppf(level, loc=yhat, scale=kwargs["sigma"])

    def preprocess(
        self: Self,
        stan_data: ModelInputData,
        augmentation_config: Optional[BinomialPopulation] = None,
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface
        augmentation_config : Optional[BinomialPopulation], optional
            Configuration parameters for the augmentation process. Currently,
            it is only required for the BinomialConstantN model. For all other
            models it defaults to None.

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
        ## -- 1. Augment stan_data -- ##
        # Add observed noise
        # stan_data.sigma_obs = 1  # type: ignore[attr-defined]

        ## -- 2. Calculate initial parameter guesses -- ##
        # No scaling needed for normal distribution model as its link function
        # is the identity-function.

        # Call the parent class parameter estimation method
        ini_params = self.calculate_initial_parameters(stan_data.y, stan_data)
        # The initial guess for the noise necessary for normal distribution
        ini_params.sigma_obs = 1  # type: ignore[attr-defined]
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

    def link_func(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Link function as used in Stan model

        Parameters
        ----------
        x : np.ndarray
            Output of the GLM

        Returns
        -------
        np.ndarray
            Linked GLM output

        """
        return np.exp(x)

    def yhat_func(self: Self, linked_arg: np.ndarray) -> np.ndarray:
        """
        Produces the predicted values yhat

        Parameters
        ----------
        linked_arg : np.ndarray
            Linked GLM output

        Returns
        -------
        np.ndarray
            Predicted values

        """
        return linked_arg

    def quant_func(
        self: Self, level: float, yhat: np.ndarray, **kwargs: dict[str, Any]
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        return poisson.ppf(level, yhat)

    def preprocess(
        self: Self,
        stan_data: ModelInputData,
        augmentation_config: Optional[BinomialPopulation] = None,
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface
        augmentation_config : Optional[BinomialPopulation], optional
            Configuration parameters for the augmentation process. Currently,
            it is only required for the BinomialConstantN model. For all other
            models it defaults to None.

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
        ## -- 1. Augment stan_data -- ##
        # Nothing to augment for Poisson model

        ## -- 2. Calculate initial parameter guesses -- ##
        # Apply inverse link function for y-value scaling. As the data are
        # scaled using the natural logarithm, zeros need to be replaced.
        y_scaled = np.where(stan_data.y == 0, 1e-10, stan_data.y)
        y_scaled = np.log(y_scaled)

        # Call the parent class parameter estimation method
        ini_params = self.calculate_initial_parameters(y_scaled, stan_data)
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

    def link_func(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Link function as used in Stan model

        Parameters
        ----------
        x : np.ndarray
            Output of the GLM

        Returns
        -------
        np.ndarray
            Linked GLM output

        """
        return np.exp(x)

    def yhat_func(self: Self, linked_arg: np.ndarray) -> np.ndarray:
        """
        Produces the predicted values yhat

        Parameters
        ----------
        linked_arg : np.ndarray
            Linked GLM output

        Returns
        -------
        np.ndarray
            Predicted values

        """
        return linked_arg

    def quant_func(
        self: Self, level: float, yhat: np.ndarray, **kwargs: dict[str, Any]
    ) -> np.ndarray:
        """
        Quantile function of the underlying distribution

        Parameters
        ----------
        level : float
            Level of confidence in (0,1)
        yhat : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            Quantile at given level

        """
        # Calculate success probability. Note that phi has the meaning of
        # number of successes
        p = kwargs["phi"] / (kwargs["phi"] + yhat)
        return nbinom.ppf(level, n=kwargs["phi"], p=p)

    def preprocess(
        self: Self,
        stan_data: ModelInputData,
        augmentation_config: Optional[BinomialPopulation] = None,
    ) -> tuple[ModelInputData, ModelParams]:
        """
        Augment the input data for the stan model with model dependent data
        and calculate initial guesses for model parameters.

        Parameters
        ----------
        stan_data : ModelInputData
            Model agnostic input data provided by the forecaster interface
        augmentation_config : Optional[BinomialPopulation], optional
            Configuration parameters for the augmentation process. Currently,
            it is only required for the BinomialConstantN model. For all other
            models it defaults to None.

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
        ## -- 1. Augment stan_data -- ##
        m_poisson = Poisson(model_name="poisson")
        m_poisson.fit(
            stan_data=stan_data,
            optimize_mode="MLE",
            sample=False,
            augmentation_config=None,
        )
        result = m_poisson.predict(
            t=stan_data.t,
            X=stan_data.X,
            interval_width=0.8,
            n_samples=0,
        )
        dof = 2 + stan_data.S + stan_data.K
        _, phi = calculate_dispersion(
            y_obs=stan_data.y, y_model=result.yhat, dof=dof
        )

        if phi < 0:
            get_logger().warning(
                "Found negative dispersion factor, indicating underdispersed "
                "data. Setting phi = abs(phi). Consider using Poisson or "
                "Binomial model."
            )
            phi = abs(phi)
        stan_data.phi_est = phi

        ## -- 2. Calculate initial parameter guesses -- ##
        # Apply inverse link function for y-value scaling. As the data are
        # scaled using the natural logarithm, zeros need to be replaced.
        y_scaled = np.where(stan_data.y == 0, 1e-10, stan_data.y)
        y_scaled = np.log(y_scaled)

        # Call the parent class parameter estimation method
        ini_params = self.calculate_initial_parameters(y_scaled, stan_data)
        # The initial guess for dispersion parameter phi
        ini_params.phi_raw = 0
        return stan_data, ini_params


# A TypeAlias for all existing Model Backends
ModelBackend: TypeAlias = Union[
    BinomialConstantN, Normal, Poisson, NegativeBinomial
]

# Map model names to respective model backend classes
MODEL_MAP: dict[str, Type[ModelBackendBase]] = {
    "binomial constant n": BinomialConstantN,
    "poisson": Poisson,
    "normal": Normal,
    "negative binomial": NegativeBinomial,
}


def get_model_backend(
    model: Distribution, **kwargs: dict[str, Any]
) -> ModelBackend:
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
