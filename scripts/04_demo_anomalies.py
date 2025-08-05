"""
Demonstration of singling out potential anomaluous data points.

Based on daily pypi downloads of the prophet package.

Make sure to place the data file in the correct relative path
"""

# Standard Library
from pathlib import Path

# Third Party
import pandas as pd
import seaborn as sns

# Gloria
from gloria import CalendricData, Gloria, cast_series_to_kind

# Filepath the current script is in
basepath = Path(__file__).parent

# Read the data
data = pd.read_csv("data/real/pypi_downloads_pandas_prophet.csv")

# Convert to datetime
data["date"] = pd.to_datetime(data["date"])

# Convert metric to unsigned int for count data, change to "kilo-downloads"
data["downloads"] = cast_series_to_kind(data["downloads"] / 1000, "u")

# Extract only prophet downloads
data_prophet = data.loc[data["project"] == "prophet"]

# Create the model. Note the interval width of "0.95", ie. 95% of all data
# should be inside the interval
m = Gloria(
    model="negative binomial",
    metric_name="downloads",
    timestamp_name="date",
    sampling_period="1d",
    n_changepoints=0,
    interval_width=0.95,
)

# Create the protocol
calendric_protocol = CalendricData(country="US")

# Add the protocol
m.add_protocol(calendric_protocol)

# Fit the model to the data
m.fit(data_prophet)

# Predict and remove actual future prediction data point
prediction = m.predict().iloc[:-1]

# Plot
fig = m.plot(prediction, mark_anomalies=True, include_legend=True)
