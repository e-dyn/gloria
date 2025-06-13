# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:14:13 2025

@author: pwollgarten
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path

# Third Party
import pandas as pd

# Gloria
# Inhouse Packages
from gloria import CalendricData, Gloria, cast_series_to_kind

Path.cwd()
file = Path("data/real/Aquifer_Petrignano.csv")

df = pd.read_csv(file)
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

n_changepoints = 2
model = "normal"
metric_name = "Depth_to_Groundwater_P24"
regressor_name = "Rainfall_Bastia_Umbra"
timestamp_name = "Date"
sampling_period = "1 d"

df_gloria = df[
    (df["Date"] >= "2011-01-01")
    & (df["Date"] <= "2013-06-30")
    & (df[regressor_name].notna())
    & (df[metric_name].notna())
].reset_index(drop=True)

df_gloria = df_gloria.sort_values(by="Date")


# Without Regressor
my_model = Gloria(
    model=model,
    metric_name=metric_name,
    timestamp_name=timestamp_name,
    sampling_period=sampling_period,
    n_changepoints=n_changepoints,
)

protocol = CalendricData(yearly_seasonality=True, weekly_seasonality=False)

my_model.add_protocol(protocol)

my_model.fit(df_gloria)

future_dates = my_model.make_future_dataframe(periods=90)

forecast = my_model.predict(future_dates)

my_model.plot(forecast, include_legend=True)


# With Rainfall Regressor

my_model = Gloria(
    model=model,
    metric_name=metric_name,
    timestamp_name=timestamp_name,
    sampling_period=sampling_period,
    n_changepoints=n_changepoints,
)

protocol = CalendricData(yearly_seasonality=True, weekly_seasonality=False)

my_model.add_protocol(protocol)

my_model.add_external_regressor(name=regressor_name, prior_scale=1.0)

my_model.fit(df_gloria)

future_dates = my_model.make_future_dataframe(periods=90)

# Führe den Join durch
future_dates = future_dates.merge(
    df[["Date", regressor_name]], left_on="Date", right_on="Date", how="left"
)

forecast = my_model.predict(future_dates)

my_model.plot(forecast, include_legend=True)
