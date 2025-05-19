# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:56:28 2024

@author: BeKa
"""

### --- Module Imports --- ###
# Standard Library
import json
import re
from pathlib import Path
from typing import Any

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.special import expit, logit
from typing_extensions import Self

# Gloria
from gloria.regressors import EVENT_REGRESSORS, Regressor, Seasonality
from gloria.utilities.constants import _T_INT
from gloria.utilities.misc import time_to_integer

### --- Global Constants Definitions --- ###
START = pd.Timestamp("20.12.2020")
N_DAYS = 400
FREQ = "1 d"
T_NAME = "ds"
METRIC_NAME = "y"
SUFFIX = "test"
OUTPUT_DIR = "simulated_data"
CHANGEPOINT_DENSITY = 0.02
ANOMALY_DENSITY = 0.01
ANOMALY_STRENGTH = 0.1
SAVE_FIGURE = True
ADD_CHRISTMAS = True

MODEL = "normal"
BETA_SCALE = 1
Y_MIN = -3
Y_MAX = 3
N = 1000
SIGMA = 0.5
SEASON_TO_TREND = 10
HOLIDAY_TO_SEASONS = 10


### --- Class and Function Definitions --- ###

SEASONALITIES = {
    "weekly": {"period": "7d", "fourier_order": 1},
    # 'quarterly': {
    #     'period': f'{365.25/4}d',
    #     'fourier_order': 3
    # },
    "monthly": {"period": f"{365.25/12}d", "fourier_order": 1},
    # 'yearly': {
    #     'period': '365.25d',
    #     'fourier_order': 10
    # },
}

if Y_MAX <= Y_MIN:
    raise ValueError("Y_MAX must be larger than Y_MIN")
if (MODEL == "binomial") and (N < Y_MAX):
    raise ValueError("N must be larger or equal not Y_MAX")
if (MODEL in ("binomial", "poisson")) and (Y_MIN <= 0):
    raise ValueError("Y_MIN must be positive and non-zero.")

MODELFUN_MAP = {
    "poisson": {
        "link": lambda x: np.exp(x),
        "yhat": lambda x: x,
        "link_inv": lambda x: np.log(x),
        "yhat_inv": lambda x: x,
        "dist": lambda x: np.random.poisson(lam=x),
    },
    "binomial": {
        "link": lambda x: expit(x),
        "yhat": lambda x: N * x,
        "link_inv": lambda x: logit(x),
        "yhat_inv": lambda x: x / N,
        "dist": lambda x: np.random.binomial(N, p=x),
    },
    "normal": {
        "link": lambda x: x,
        "yhat": lambda x: x,
        "link_inv": lambda x: x,
        "yhat_inv": lambda x: x,
        "dist": lambda x: np.random.normal(loc=x, scale=SIGMA),
    },
}


class Simulator(BaseModel):
    model: str
    sampling_period: str = "1d"
    timestamp_name: str = "ds"
    metric_name: str = "y"
    n_changepoints: int = Field(ge=0, default=10)

    class Config:
        extra = "allow"  # Allows setting extra attributes
        arbitrary_types_allowed = True  # So the model accepts pd.Series

    def __init__(
        self: Self, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> None:

        super().__init__(*args, **kwargs)

        self.link = MODELFUN_MAP[self.model]["link"]
        self.yhat_fun = MODELFUN_MAP[self.model]["yhat"]
        self.link_inv = MODELFUN_MAP[self.model]["link_inv"]
        self.yhat_inv = MODELFUN_MAP[self.model]["yhat_inv"]
        self.dist = MODELFUN_MAP[self.model]["dist"]

        self.sampling_delta = pd.to_timedelta(self.sampling_period)

        # Set later on
        self.seasonalities = []
        self.events = []
        self.X = pd.DataFrame()
        self.pars = dict()

    def add_seasonality(
        self: Self, name: str, period: str, fourier_order: int
    ) -> Self:

        if (fourier_order <= 0) or (not isinstance(fourier_order, int)):
            raise ValueError("Fourier Order must be an integer > 0")

        self.seasonalities.append(
            Seasonality(
                name=name,
                period=pd.to_timedelta(period) / self.sampling_delta,
                fourier_order=fourier_order,
                prior_scale=0.1,
            )
        )
        return self

    def add_event(
        self: Self,
        name: str,
        prior_scale: float,
        regressor_type: str,
        event: dict[str, Any],
        **regressor_kwargs: dict[str, Any],
    ) -> Self:

        # Validate and set prior_scale, and fourier_order
        if prior_scale is None:
            prior_scale = self.event_prior_scale
        prior_scale = float(prior_scale)
        if prior_scale <= 0:
            raise ValueError("Prior scale must be > 0")

        if regressor_type not in EVENT_REGRESSORS:
            raise TypeError(
                f"The passed regressor type '{regressor_type}' is"
                " not a registered event regressor."
            )

        if not isinstance(event, dict):
            raise TypeError(
                "Input 'event' must be a dictionary containing"
                " base event parameters."
            )

        if "event_type" not in event:
            raise KeyError("Event Type is not specified in event dictionary.")

        regressor_dict = {
            "name": name,
            "prior_scale": prior_scale,
            "regressor_type": regressor_type,
            "event": event,
            **regressor_kwargs,
        }
        self.events.append(Regressor.from_dict(regressor_dict))
        return self

    def make_all_features_given_time(
        self: Self, t_int: pd.Series
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:

        # 1. Seasonalities
        make_features = [
            lambda s=s: s.make_feature(t_int) for s in self.seasonalities
        ]

        # 3. Event Regressors
        make_features.extend(
            [
                lambda event=event: event.make_feature(self.t)
                for event in self.events
            ]
        )

        # Make the features and save all feature matrices along with prior
        # scales
        X_lst = []
        prior_scales = dict()
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

    @staticmethod
    def clinear(x, breakpoints, parameters, offset=0, noise=None):
        all_edges = np.insert(breakpoints, [0, len(breakpoints)], [0, x.max()])

        # Create the matrix using broadcasting
        M = (x[:, None] >= all_edges[:-1]) & (x[:, None] < all_edges[1:])
        # Fill the last column
        M[-1, -1] = True
        output = np.cumsum(np.sum(M * parameters, axis=1)).astype(float)
        output += offset

        if noise:
            output += noise * np.random.randn(*output.shape)
        return output

    def set_changepoints(self: Self, t) -> Self:

        self.n_changepoints = np.random.poisson(
            self.t_int.size * CHANGEPOINT_DENSITY
        )
        self.changepoints_int = (
            pd.Series(
                np.random.choice(
                    self.t_int, self.n_changepoints, replace=False
                ),
                name=_T_INT,
                dtype=int,
            )
            .sort_values()
            .reset_index(drop=True)
        )
        self.changepoints = pd.Series(
            t.loc[self.changepoints_int.values],
            name=self.timestamp_name,
            dtype="<M8[ns]",
        )
        if len(self.changepoints) == 0:
            # Dummy changepoint
            self.changepoints_int = pd.Series([0], name=_T_INT, dtype=int)

        return self

    def set_anomalies(self: Self, t) -> Self:

        self.n_anomalies = np.random.poisson(self.t_int.size * ANOMALY_DENSITY)
        self.anomalies_int = (
            pd.Series(
                np.random.choice(self.t_int, self.n_anomalies, replace=False),
                name="t_ano_int",
                dtype=int,
            )
            .sort_values()
            .reset_index(drop=True)
        )
        self.anomalies = pd.Series(
            t.loc[self.anomalies_int.values], name="t_ano", dtype="<M8[ns]"
        )

        return self

    def simulate(self, t):
        def rescale(y, y_min, y_max, z_min, z_max):
            y_norm = (y - y_min) / (y_max - y_min)
            return z_min + y_norm * (z_max - z_min)

        self.first_timestamp = t.min()
        self.last_timestamp = t.max()
        self.t_int = time_to_integer(t, t.min(), self.sampling_delta)
        self.t = t
        (
            self.X,
            _,
        ) = self.make_all_features_given_time(self.t_int)
        self.K = self.X.shape[1]

        self.history = pd.DataFrame({T_NAME: t, _T_INT: self.t_int})

        self.set_changepoints(t)
        self.set_anomalies(t)
        self.pars["m"] = np.random.random()
        self.pars["k"] = np.random.random() - 0.5
        self.pars["delta"] = np.random.random(self.n_changepoints) - 0.5

        trend_arg = self.clinear(
            np.array(self.t_int),
            np.array(self.changepoints_int),
            [self.pars["k"], *self.pars["delta"]],
            offset=self.pars["m"],
        )

        self.pars["beta"] = np.random.random(self.K) * SEASON_TO_TREND
        is_holiday = pd.Series(self.X.columns).str.contains("Holiday")
        is_holiday *= HOLIDAY_TO_SEASONS - 1
        is_holiday += 1
        self.pars["beta"] *= is_holiday.values

        if self.K == 0:
            arg = trend_arg
        else:
            Xb = np.matmul(self.X, self.pars["beta"])
            arg = trend_arg + Xb
        arg_min = self.link_inv(self.yhat_inv(Y_MIN))
        arg_max = self.link_inv(self.yhat_inv(Y_MAX))

        self.arg = rescale(arg, arg.min(), arg.max(), arg_min, arg_max)
        self.trend_arg = rescale(
            trend_arg, arg.min(), arg.max(), arg_min, arg_max
        )

        self.trend = self.yhat_fun(self.link(self.trend_arg))
        self.yhat = self.yhat_fun(self.link(self.arg))
        ano_mask = self.t_int.isin(self.anomalies_int)
        ano_deviation = np.random.normal(
            scale=ANOMALY_STRENGTH, size=self.n_anomalies
        )
        ano_deviation = np.where(ano_deviation < -1, -1, ano_deviation)

        self.observed = pd.Series(self.dist(self.link(self.arg)))
        self.y_anos = pd.Series([0] * self.t_int.size)
        self.y_anos.loc[ano_mask] = (
            self.observed.loc[self.anomalies_int.values] * ano_deviation
        ).astype(int)
        self.observed = self.observed + self.y_anos

        result = pd.DataFrame(
            {
                self.timestamp_name: t,
                self.metric_name: self.observed,
                self.metric_name + "_yhat": self.yhat,
                "trend": self.trend,
                "is_changepoint": self.t_int.isin(self.changepoints_int),
                "is_anomaly": ano_mask,
                "ano_deviation": self.y_anos,
            }
        )

        return result


### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent

    while True:
        time = pd.Series(pd.date_range(start=START, periods=N_DAYS, freq=FREQ))

        model = Simulator(
            model=MODEL,
            sampling_period=FREQ,
            timestamp_name=T_NAME,
            metric_name=METRIC_NAME,
        )

        seasons = np.random.choice(
            np.array(list(SEASONALITIES.keys())),
            np.random.randint(low=1, high=len(SEASONALITIES) + 1),
            replace=False,
        )
        for name in seasons:
            model.add_seasonality(name, **SEASONALITIES[name])

        if ADD_CHRISTMAS:
            model.add_event(
                name="Christmas Day",
                prior_scale=10,
                regressor_type="Holiday",
                event={"event_type": "Gaussian", "sigma": "3 d"},
                country="DE",
            )

        df = model.simulate(time)
        df.y = expit(df.y)
        df.y_yhat = expit(df.y_yhat)
        df.trend = expit(df.trend)

        settings = {
            "model": MODEL,
            "number of days": N_DAYS,
            "sampling period": FREQ,
            "included seasons": [s.name for s in model.seasonalities],
            "changepoint density": CHANGEPOINT_DENSITY,
            "anomaly density": ANOMALY_DENSITY,
            "anomaly strength": ANOMALY_STRENGTH,
            "number of anomalies": model.n_anomalies,
            "number of changepoints": model.n_changepoints,
            "y_min": Y_MIN,
            "y_max": Y_MAX,
            "population N": N,
            "sigma observed": SIGMA,
        }
        formatted_text = "\n".join(
            f"{key}: {value}" for key, value in settings.items()
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(df[T_NAME], df[METRIC_NAME], "o", label="data")
        ax.plot(
            df.loc[df.is_anomaly, T_NAME],
            df.loc[df.is_anomaly, METRIC_NAME],
            "o",
            color="green",
            label="anomalies",
        )
        changepoints = df.loc[df.is_changepoint, T_NAME]
        plt.vlines(
            changepoints,
            df[METRIC_NAME].min(),
            df[METRIC_NAME].max(),
            linestyle="--",
            color="gray",
        )
        ax.plot(df[T_NAME], df["trend"], "black", label="trend")
        ax.plot(
            df[T_NAME],
            df[METRIC_NAME + "_yhat"],
            "red",
            label="expectation value",
        )
        plt.legend()

        plt.subplots_adjust(
            right=0.75
        )  # Adjust figure to make space for the text
        fig.text(
            0.78,
            0.5,
            formatted_text,
            va="center",
            fontsize=10,
            transform=fig.transFigure,
        )

        plt.show()

        while True:
            save = input("Save? [y/n] ")
            if save not in ["y", "n"]:
                print("hää? Bitte 'y' oder 'n'")
                continue
            save = True if save == "y" else False
            break

        if save:
            (basepath / OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            filebase = f"{pd.Timestamp.today().date()}_{MODEL}_{SUFFIX}_n"
            paths = (basepath / OUTPUT_DIR).glob(f"{filebase}*.csv")
            # Extract the two-digit numbers
            pattern = filebase + r"(\d{2})\.csv$"
            digits = []
            for path in paths:
                match = re.search(pattern, path.name)
                if match:
                    digits.append(int(match.group(1)))
            if not digits:
                n = 0
            else:
                n = max(digits) + 1
            df.to_csv(basepath / OUTPUT_DIR / f"{filebase}{n:02}.csv")

            settings["filename"] = f"{filebase}{n:02}.csv"

            with open(
                basepath / OUTPUT_DIR / f"{filebase}{n:02}.json", "w"
            ) as outfile:
                json.dump(settings, outfile, indent=4)

            if SAVE_FIGURE:
                fig.savefig(
                    basepath / OUTPUT_DIR / f"{filebase}{n:02}.png", dpi=150
                )

        while True:
            another_one = input("Want another time series? [y/n] ")
            if another_one not in ["y", "n"]:
                print("hää? Bitte 'y' oder 'n'")
                continue
            another_one = True if another_one == "y" else False
            break

        if not another_one:
            break
