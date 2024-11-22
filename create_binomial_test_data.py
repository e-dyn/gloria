# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:46:50 2024

@author: BeKa
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path
import math
from abc import ABC, abstractmethod
from typing import Union

# Third Party
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
from scipy.special  import expit
from matplotlib.ticker import MaxNLocator
from pydantic import BaseModel, Field, field_validator, Extra
# Inhouse Packages

### --- Global Constants Definitions --- ###
M = 5                           # Number of time series
N_REAL = 1000                   # Real population of first time series
DAYS = 120                      # Number of time stamps
PASS_OBSERVED_POPULATION = True # True/False: pass observed/modeled population to next state

# Seasonalities used to generate observed data and as regressors for fit
TEST_SEASONALITIES = {
    'weekly': {
        'period': 7,
        'fourier_order': 3
    },
    'quarterly': {
        'period': 365.25/4,
        'fourier_order': 3
    }
}

# Stan Settings
FIT_METHOD = "pathfinder"       # Either "pathfinder" or "sample"

# Figure Settings
CONFIDENCE_LEVEL = 80
AX_WIDTH = 6
AX_HEIGHT = 4
VERTICAL_GAP = 1
HORIZONTAL_GAP = 1

### --- Class and Function Definitions --- ###
def get_fourier_regressors(ds, T0, n_max):
    w0 = 2*np.pi / T0
    odd = np.sin(w0*ds.reshape(-1,1)*np.arange(1,n_max+1))
    even = np.cos(w0*ds.reshape(-1,1)*np.arange(1,n_max+1))
    return np.hstack([even, odd])

def validate_population(N, ds):
    if isinstance(N, int):
        assert N > 0, "Population size must be a non-zero positive integer"
    else:
        assert len(N) == len(ds), "If population is a 1D array, it must have the same length as ds"
        assert (N >= 0).all(), f"Population size vector must contain only positive integers: {N}"
        assert N.dtype == int, "Population size vector contains non-integer values"
    return N

def get_time_series(ds, N, X = None, alpha = None, beta = None, p = None, mode = "model"):
    if mode == "model":
        f = lambda n,p: n*p
    elif mode == "observed":
        f = lambda n,p: np.random.binomial(n, p)
    else:
        raise ValueError("'mode' must be either 'model' or 'observed'")
    validate_population(N, ds)

    if p is not None:
        assert len(N) == len(ds), "For model with constant p, population size must be vectorized"
        return f(N, p)
    
    for parameter, name in zip([alpha, beta, X],["alpha", "beta", "X"]):
        assert parameter is not None, f"For generalized linear models, parameter {name} must not be None."
    assert beta.shape[0] == X.shape[1], "number of regressors in beta and X do not agree"
    
    return f(N, expit(alpha+(beta*X).sum(axis = 1)))

class StanData(BaseModel):
    T: int = Field(gt=0)
    t: Union[pd.Series, np.ndarray]
    y: Union[pd.Series, np.ndarray]
    N: Union[int, pd.Series, np.ndarray]
    
    @field_validator('t')
    @classmethod
    def validate_len_of_t(cls, t: pd.Series, info) -> pd.Series:
        assert info.data['T'] == len(ds), "Length of t does not equal specified T"
        return t
        
    @field_validator('y')
    @classmethod
    def validate_len_of_y(cls, y: pd.Series, info) -> pd.Series:
        assert info.data['T'] == len(y), "Length of y does not equal specified T"
        return y
        
    @field_validator('N')
    @classmethod
    def validate_N(cls, N: Union[int, pd.Series], info) -> Union[int, pd.Series]:
        return validate_population(N, info.data['t'])
    
    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

class RawModel(ABC):
    fit_methods = {
        "sample": lambda m: m.sample,
        "pathfinder": lambda m: m.pathfinder
    }
    
    def __init__(
            self,
            seasonalities = TEST_SEASONALITIES
    ):
        self.seasonalities = seasonalities
        self.X = None
        
    def add_seasonality(self, name, period, fourier_order):
        self.seasonalities[name] = {
            'period': period,
            'fourier_order': fourier_order
        }
        return self
    
    def _add_regressors(self, t):
        X_list = []
        for name, props in self.seasonalities.items():
            X_list.append(get_fourier_regressors(t, props['period'], props['fourier_order']))
        self.X = np.hstack(X_list)
        return self
    
    @abstractmethod
    def simulate_data(self):
        pass
    
    @abstractmethod
    def fit(self, data):
        pass
    
    def _prepare_fit(self, data, population, fit_method):
        if fit_method not in self.fit_methods.keys():
            raise ValueError("Fit method not supported")

        return StanData(
            T=len(data['ds']),
            t=data['ds'],
            y=data['y'],
            N=population
        )
    
    def _get_yhat(self, data, population, params, has_regressors = True):
        # change dictionary of lists to list of dictionaries
        params = [dict(zip(params.keys(),t)) for t in zip(*params.values())]
        X = dict()
        if has_regressors:
            X['X'] = self.X
        yhat_lst = []
        for pars in params:
            yhat_lst.append(
                get_time_series(
                    data['ds'], N=population, **pars, **X
                )
            )
            
        yhats = np.array(yhat_lst)
        
        data['yhat'] = yhats.mean(axis=0)
        data['yhat_lower'] = np.percentile(yhats, lower_level, axis=0)
        data['yhat_upper'] = np.percentile(yhats, upper_level, axis=0)
        
        return data
            
    
class GLM_ConstantN(RawModel):
    name = "constant N"
    stan_file = "stan_models/edd_glm_N_constant.stan"
    
    def simulate_data(self, ds, N, mode = "model", **params):
        if self.X is None:
            raise ValueError("Create regressors first.")
        return get_time_series(ds, N, mode=mode, X=self.X, **params)
    
    def fit(self, data, population, show_console=False, fit_method = "sample"):
        stan_data = self._prepare_fit(data, population, fit_method)
        stan_data.X = self.X
        stan_data.K = self.X.shape[1]
        
        stan_model = CmdStanModel(stan_file=self.stan_file)
        
        fit = self.fit_methods[fit_method](stan_model)(
            data=stan_data.dict(),
            show_console=show_console
        )
                
        return self._get_yhat(data, population, fit.stan_variables(), has_regressors=True)


class GLM_VectorizedN(RawModel):
    name = "vectorized N"
    stan_file = "stan_models/edd_glm_N_vectorized.stan"
    
    def simulate_data(self, ds, N, mode = "model", **params):
        if self.X is None:
            raise ValueError("Create regressors first.")
        return get_time_series(ds, N, mode=mode, X=self.X, **params)
    
    def fit(self, data, population, show_console=False, fit_method = "sample"):
        stan_data = self._prepare_fit(data, population, fit_method)
        stan_data.X = self.X
        stan_data.K = self.X.shape[1]
        
        stan_model = CmdStanModel(stan_file=self.stan_file)
        
        fit = self.fit_methods[fit_method](stan_model)(
            data=stan_data.dict(),
            show_console=show_console
        )
                
        return self._get_yhat(data, population, fit.stan_variables(), has_regressors=True)


class ConstantP(RawModel):
    name = "constant p"
    stan_file = "stan_models/edd_p_constant.stan"
    
    def simulate_data(self, ds, N, p, mode="model"):
        return get_time_series(ds, N, p=p, mode=mode)
    
    def fit(self, data, population, show_console=False, fit_method = "sample"):
        stan_data = self._prepare_fit(data, population, fit_method)
        
        stan_model = CmdStanModel(stan_file=self.stan_file)
        
        fit = self.fit_methods[fit_method](stan_model)(
            data=stan_data.dict(),
            show_console=show_console
        )
        
        return self._get_yhat(data, population, fit.stan_variables(), has_regressors=True)


class Oracle:
    def __new__(cls, model, **kwargs):
        model_map = {
            GLM_ConstantN.name: GLM_ConstantN,
            GLM_VectorizedN.name: GLM_VectorizedN,
            ConstantP.name: ConstantP,
        }
        if model not in model_map:
            raise ValueError(f"Model {model} is not supported")
        return model_map[model](**kwargs)


round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def clinear(x, breakpoints, parameters, offset = 0, noise = None):
    all_edges = np.insert(breakpoints,[0,len(breakpoints)],[0,x.max()])

    # Create the matrix using broadcasting
    M = (x[:, None] >= all_edges[:-1]) & (x[:, None] < all_edges[1:])
    # Fill the last column
    M[-1,-1] = True
    output = np.cumsum(np.sum(M*parameters, axis = 1)).astype(float)
    output += offset
    
    if noise:
        output += noise * np.random.randn(*output.shape)
    return output


### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent
    # np.random.seed(3)

    ds = np.arange(0,DAYS)
    m = Oracle(model = 'constant N')
    

    # # True model parameters
    n_regressors = sum([props['fourier_order'] for props in m.seasonalities.values()])*2
    alpha0 = np.array([-2.3])
    beta0 = 0.1*np.random.randn(n_regressors)
    
    m = m._add_regressors(ds)
    trend = clinear(ds, [35,78,95], [-0.015,0.01,0.02,-0.008], )

    data = m.simulate_data(ds, N_REAL, mode = "observed", alpha = trend, beta = beta0)
    plt.plot(ds,data)
    plt.plot(ds,expit(trend)*N_REAL)
    df = pd.DataFrame({'time': ds, 'observed': data, 'trend': trend}).to_csv('data_with_trend.csv', index = False)
    # # Figure preparations
    # n_rows = int(np.ceil(np.sqrt(M)))
    # n_cols = int(np.ceil(M/n_rows))
    # fig_width = AX_WIDTH*n_cols + (n_cols-1)*HORIZONTAL_GAP
    # fig_height = AX_HEIGHT*n_rows + (n_rows-1)*VERTICAL_GAP
    # fig, ax = plt.subplots(n_rows,n_cols, figsize=(fig_width, fig_height))
    # lower_level = (100 - CONFIDENCE_LEVEL) / 2
    # upper_level = lower_level + CONFIDENCE_LEVEL
    
    # y_real = []
    # y_obs = []
    # for i in range(0,M):
    #     models[i] = models[i]._add_regressors(ds)
    #     if i == 0:
    #         population = N_REAL
    #     elif PASS_OBSERVED_POPULATION:
    #         population = y_obs[i-1]
    #     else:
    #         population = y_real[i-1].astype(int)
    #     if models[i].name == "constant p":
    #         params = {'p': 0.5}
    #     else:
    #         params = {'alpha': alpha0[i], 'beta': beta0[i]}
    #     y_real.append(models[i].simulate_data(ds, population, **params))
    #     y_obs.append(models[i].simulate_data(ds, population, mode = "observed", **params))
        
    #     data = pd.DataFrame({'ds': ds, 'y': y_obs[-1]})
    #     data = models[i].fit(data, population, fit_method = FIT_METHOD)
        
    #     row = i // n_cols
    #     col = i % n_cols
    #     ax[row, col].plot(ds,y_real[i], 'black', label = 'true')
    #     ax[row, col].plot(ds,y_obs[i], 'o', label = 'data')
    #     ax[row, col].plot(ds, data['yhat'], 'red', label = 'fit')
    #     ax[row, col].fill_between(ds, data['yhat_lower'], data['yhat_upper'], color = 'gray', alpha=0.3, label = 'ci')
    #     ax[row, col].set_ylabel('y', fontsize=12)
    #     ax[row, col].set_title(f'State {i+1}, Method: {models[i].name}')
    #     ax[row, col].yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # last_row = (M-1) // n_cols
    # for col in range(0,n_cols):
    #     ax[last_row, col].set_xlabel('ds', fontsize=12)
    # ax[0, 0].legend()
    # plt.tight_layout()
    # plt.show()
    
