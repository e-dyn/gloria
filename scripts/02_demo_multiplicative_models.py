"""
Demonstration of multiplicative models and workaround for working with monthly
data.

Make sure to place the data file in the correct relative path
"""

# Standard Library
### --- Module Imports --- ###
from pathlib import Path

# Third Party
import pandas as pd

# Gloria
from gloria import Gloria, cast_series_to_kind

# Filepath the current script is in
basepath = Path(__file__).parent

## -- 1. Data Preparation of monthly data -- ##

# Read the data
data = pd.read_csv("data/real/AirPassengers.csv")

# Rename columns for convenience
data.rename(
    {"Month": "date", "#Passengers": "passengers"}, axis=1, inplace=True
)

# Convert to datetime
data["date"] = pd.to_datetime(data["date"])

# Make a new monthly column with equidistant spacing as Gloria can't digest
# data which are not on an equidistant grid.
# Note: the timestamps do not always coincide with the start of a month
# Instead the difference between two subsequent timestamps is rather the
# average length of a month
freq = pd.Timedelta(f"{365.25/12}d")
data["date"] = pd.Series(
    pd.date_range(
        start=data["date"].min(), periods=len(data["date"]), freq=freq
    )
)

# Cast to unsigned data type so it is digestible by count models
data["passengers"] = cast_series_to_kind(data["passengers"], "u")

# Set up the model
# Rerun with model = "normal" and note the differences in the plot
# 1. Trend
#   - poisson: grows exponentially. retraces real trend
#   - normal: grows linearly. underestimates trend for small values
# 2. Seasonality
#   - poisson: amplitude grows with trend. good fit
#   - normal: amplitude stays constant. only fits on average for mid range values
m = Gloria(
    model="poisson",
    metric_name="passengers",
    timestamp_name="date",
    sampling_period=freq,
    n_changepoints=0,
)

# Add observed seasonalities
m.add_seasonality("yearly", "365.25 d", 4)

# Fit the model to the data
m.fit(data)

# Predict
prediction = m.predict()

# Plot
m.plot(prediction)
