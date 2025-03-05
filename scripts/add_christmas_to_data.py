# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:51:11 2025

@author: BeKa
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd

# Gloria
from gloria.events import Gaussian
from gloria.holidays import get_holidays

# Inhouse Packages
from gloria.regressors import Holiday

### --- Global Constants Definitions --- ###


### --- Class and Function Definitions --- ###


### --- Main Script --- ###
if __name__ == "__main__":
    filepath = (
        Path(__file__).parent
        / "simulated_data/2025-02-19_binomial_test_n00.csv"
    )

    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["ds"])

    christmas = Holiday(
        name="Christmas Day",
        prior_scale=1,
        mode="additive",
        event=Gaussian(sigma="3d"),
        country="DE",
    )

    X, _, _ = christmas.make_feature(df["ds"])
