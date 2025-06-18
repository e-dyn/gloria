"""
Demonstration of using events to model discontinuities using BoxCar events.

Choice of model:
1. Best fit: 'negative binomial' for overdispersed count data
2. Bad fit: Try to change to 'normal' model and see how it produces impossible
   negative predictions before the release of pydantic version 2.10
   on 2024-11-20

Modeling the discontinuity:
1. The demo shows pypi downloads of pydantic version 2.10
2. Before release (2024-11-20) there are zero downloads
3. After the release of version 2.11 (2025-03-27) the downloads significantly
   drop
4. We model this using BoxCar events as step like functions by putting their
   start date on the two releases, but extend their right edge beyond the data
5. We also allow changepoints at the release dates as each segment shows a
   different trend.
6. Note that we effectively use events to modify the trend. The trend column
   returned by predict() does not contain the "event-effect" and therefore does
   not quite look like the trend one would expect.

Make sure to place the data file in the correct relative path
"""

# Standard Library
### --- Module Imports --- ###
from pathlib import Path

# Third Party
import pandas as pd

# Gloria
from gloria import BoxCar, Gloria, cast_series_to_kind

# Filepath the current script is in
basepath = Path(__file__).parent

# Read the data
data = pd.read_csv(
    "data/real/pypi_downloads_pydantic_versioning.csv",
    dtype={"version": "string"},  # Ensure this column stays a string
)

# Convert to datetime
data["date"] = pd.to_datetime(data["date"])

# Convert metric to unsigned int for count data
data["downloads"] = cast_series_to_kind(data["downloads"], "u")

# Extract pydantic version 2.10
data_210 = data.loc[data["version"] == "2.10"]

# Create the model. Change model to "normal" to see it fail before the release
# of version 2.10
m = Gloria(
    model="negative binomial",
    metric_name="downloads",
    timestamp_name="date",
    sampling_period="1d",
    changepoints=["2024-11-20", "2025-03-27"],
)

# Add observed seasonalities
m.add_seasonality("weekly", "7 d", 3)

# Create the event profile to model step function
event = BoxCar(width="300d")

# Add event at release of 2.10
m.add_event(
    name="210 release",
    regressor_type="SingleEvent",
    event=event,
    t_start="2024-11-20",
)

# Add event at release of 2.11
m.add_event(
    name="211 release",
    regressor_type="SingleEvent",
    event=event,
    t_start="2025-03-27",
)

# Fit the model to the data
m.fit(data_210)

# Predict
prediction = m.predict()

# Plot
m.plot(prediction)
