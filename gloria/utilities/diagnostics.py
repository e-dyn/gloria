"""
Utilities for evaluating prophet models and tuning hyperparameters
"""

# Standard Library
from typing import Optional, Union

# Third Party
import numpy as np

### --- Module Imports --- ###
import pandas as pd
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_percentage_error,
)

# Gloria
from gloria.interface import Gloria
from gloria.models import get_model_backend
from gloria.utilities.errors import NotFittedError
from gloria.utilities.logging import get_logger
from gloria.utilities.misc import convert_to_timedelta
from gloria.utilities.types import TimedeltaLike

### --- Global Constants Definitions --- ###
PERFORMANCE_METRICS_MAP = {
    "mse": lambda g, m: mean_squared_error(g[m], g["yhat"]),
    "rmse": lambda g, m: mean_squared_error(g[m], g["yhat"], square_root=True),
    "mae": lambda g, m: mean_absolute_error(g[m], g["yhat"]),
    "mape": lambda g, m: mean_absolute_percentage_error(g[m], g["yhat"]),
    "smape": lambda g, m: mean_absolute_percentage_error(
        g[m], g["yhat"], symmetric=True
    ),
    "mdape": lambda g, m: median_absolute_percentage_error(g[m], g["yhat"]),
    "smdape": lambda g, m: median_absolute_percentage_error(
        g[m], g["yhat"], symmetric=True
    ),
    "coverage": lambda g, m: (
        (g[m] >= g["observed_lower"]) & (g[m] <= g["observed_upper"])
    ).mean(),
}


### --- Class and Function Definitions --- ###
def generate_cutoffs(
    timestamps: pd.Series,
    horizon: pd.Timedelta,
    initial: pd.Timedelta,
    period: pd.Timedelta,
    sampling_period: pd.Timedelta,
) -> list[pd.Timestamp]:
    """
    Generate cutoff timestamps for cross-validation purposes. The cutoffs
    respect both a requested minimal initial training period as well as a
    horizon to test the training on. The cutoff spacing follows a given period
    and the global cutoff offset is chosen such that each training has the
    maximally possible amount of training data available.

    Parameters
    ----------
    timestamps : pd.Series
        A series containing the timestamps of the underlying data set
    horizon : pd.Timedelta
        The duration of the forecast horizon following each cutoff
    initial : pd.Timedelta
        The minimal duration of the training period
    period : pd.Timedelta
        The spacing between the cutoffs
    sampling_period : pd.Timedelta
        Sampling period of the underlying data set.

    Raises
    ------
    ValueError
        Is raised in case the underlying data set is too short to host both the
        initial period and a single subsequent horizon.

    Returns
    -------
    list[pd.Timestamp]
        A list of viable cutoff dates in ascending order
    """

    # NOTE: the cutoff date is included in the training and not horizon
    # Choose the maximum cutoff such that exactly one horizon fits after
    max_cutoff = timestamps.max() - horizon
    # The min_cutoff is not necessary included in the cutoff list, but is
    # rather a lower limit for allowed cutoffs
    min_cutoff = timestamps.min() + initial - sampling_period

    # if max_cutoff < min_cutoff, we dont have enough data for a single
    # training period and horizon
    if max_cutoff < min_cutoff:
        raise ValueError(
            f"An initial period of {initial} and a horizon of {horizon} was "
            "requested, but data only span "
            f"{timestamps.max() - timestamps.min() + sampling_period}."
        )

    # How many periods fit after the initial window
    n_periods = (max_cutoff - min_cutoff + sampling_period) // period

    # Generate cutoffs from end to start
    cutoffs = [max_cutoff - n * period for n in range(n_periods + 1)]

    # And return the reversed list
    return cutoffs[::-1]


def gloria_copy(m: Gloria, cutoff: pd.Timestamp) -> Gloria:
    """
    Creates a deep copy of a fitted Gloria model but resets the backend and
    restricts it to timestamps before a given cutoff.

    Parameters
    ----------
    m : Gloria
        The fitted reference Gloria model.
    cutoff : pd.Timestamp
        Simulated forecast will start from this timestamp.

    Returns
    -------
    m_copy : TYPE
        The copied and reset Gloria model

    """
    # Copy the fitted model completely
    m_copy = m.model_copy(deep=True)
    # But reset model backend so it can be fit once more
    m_copy.model_backend = get_model_backend(model=m_copy.model)
    # Also erase protocols as they have been executed already
    m_copy.protocols = []
    # Remove changepoints after cutoff date
    m_copy.changepoints = m_copy.changepoints.loc[
        m_copy.changepoints <= cutoff
    ]

    return m_copy


def single_cutoff_forecast(
    model: Gloria,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
    predict_columns: list[str],
) -> pd.DataFrame:
    """
    Forecast for single cutoff. Used in cross validation function when
    evaluating for multiple cutoffs.

    Parameters
    ----------
    model : Gloria
        The fitted reference Gloria model
    cutoff : pd.Timestamp
        Simulated forecast will start from this timestamp.
    horizon : pd.Timedelta
        Duration of the horizon for forecasts
    predict_columns : list[str]
        Columns to be returned in output.

    Raises
    ------
    ValueError
        If there are insufficient data points in the initial period

    Returns
    -------
    pd.DataFrame
        A pd.DataFrame with forecast, actual value and cutoff.

    """

    # Extract historic data
    data = model.history.copy()
    # Copy configured Gloria model including additional fitting arguments
    # used for fitting 'model', but make it fittable again.
    m = gloria_copy(model, cutoff)
    # Create new history by restricting data to  timestamps prior to cutoff
    data_fit = data.loc[data[m.timestamp_name] <= cutoff]
    if data_fit.shape[0] < 2:
        raise ValueError(
            "Less than two datapoints before cutoff. Increase initial "
            "window."
        )
    # Refit the model
    m.fit(data_fit, **m.fit_kwargs)
    # Make future dataframe
    mask_predict = (data[m.timestamp_name] <= cutoff + horizon) & (
        data[m.timestamp_name] > cutoff
    )
    data_predict = data.loc[mask_predict]
    # Make the prediction
    result = m.predict(data_predict)

    return pd.concat(
        [
            result[predict_columns],
            data.loc[mask_predict, m.metric_name].reset_index(drop=True),
            pd.DataFrame({"cutoff": [cutoff] * len(data_predict)}),
        ],
        axis=1,
    )


def cross_validation(
    model: Gloria,
    horizon: TimedeltaLike,
    period: Optional[TimedeltaLike] = None,
    initial: Optional[TimedeltaLike] = None,
    cutoffs: Optional[list[pd.Timestamp]] = None,
    extra_output_columns: Optional[Union[list[str], str]] = None,
) -> pd.DataFrame:
    """
    Cross-Validation for time series.

    Computes forecasts from historical cutoff points, which user can input.
    If not provided, begins from (end - horizon) and works backwards, making
    cutoffs with a spacing of period until initial is reached.

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model : Gloria
        The fitted reference Gloria model
    horizon : TimedeltaLike
        Duration of the horizon for forecasts.
    period : Optional[TimedeltaLike], optional
        Simulated forecast will be done at every this period. If not provided,
        0.5 * horizon is used.. The default is None.
    initial : Optional[TimedeltaLike], optional
        The first training period will include at least this much data. If not
        provided, 3 * horizon is used. The default is None.
    cutoffs : Optional[list[pd.Timestamp]], optional
        list of pd.Timestamp specifying cutoffs to be used during cross
        validation. If not provided, they are generated as described above.
        The default is None.
    extra_output_columns : Optional[Union[list[str], str]], optional
        Columns to be returned in output extra to timestamp, 'yhat' and lower/
        upper confidence bounds.

    Raises
    ------
    TypeError
        If input model is not a valid Gloria model
    NotFittedError
        If input model was not fitted yet
    ValueError
        If input cutoff list does not respect boundaries given by historic data
        and horizon

    Returns
    -------
    pd.DataFrame
        The result data frame containing predicted values alongside measured
        data, desired columns and one additional cutoff column.

    """
    if not isinstance(model, Gloria):
        raise TypeError("The input model must be a valid Gloria model.")
    if not model.is_fitted:
        raise NotFittedError(
            "Cross validation requires a fitted Gloria model for all"
            " contextual parameters set during fitting."
        )

    # Timedelta parameters may be given as strings. Cast them to pd.Timedelta
    initial = convert_to_timedelta(initial)
    horizon = convert_to_timedelta(horizon)
    period = convert_to_timedelta(period)

    # Minimal set of columns in the output DataFrame
    predict_columns = [
        model.timestamp_name,
        "yhat",
        "observed_lower",
        "observed_upper",
    ]

    # Add yhat confidence interval when Laplace sampling was performed
    if model.fit_kwargs["sample"]:
        predict_columns.extend(["yhat_lower", "yhat_upper"])

    # Add any additional columns the user requested
    if extra_output_columns is not None:
        if isinstance(extra_output_columns, str):
            extra_output_columns = [extra_output_columns]
        predict_columns.extend(
            [c for c in extra_output_columns if c not in predict_columns]
        )

    # Find period of longest seasonality component
    max_period = (
        max(season.period for season in model.seasonalities.values())
        * model.sampling_period
    )

    # Historical data timestamps
    timestamps = model.history[model.timestamp_name]

    # Generate cutoffs if none were provided
    if cutoffs is None:
        # Set period
        period = 0.5 * horizon if pd.isnull(period) else period

        # Set initial
        initial = (
            max(3 * horizon, max_period) if pd.isnull(initial) else initial
        )

        # Compute Cutoffs
        cutoffs = generate_cutoffs(
            timestamps, horizon, initial, period, model.sampling_period
        )
    # If cutoffs were provided, validate them and re-compute initial
    else:
        # Minimum cutoff must be strictly greater than the min date in the
        # history
        if min(cutoffs) <= timestamps.min():
            raise ValueError(
                "Minimum cutoff value is not strictly greater than min date in"
                " history"
            )
        # Max cutoff must be smaller than (end date minus horizon)
        if max(cutoffs) > timestamps.max() - horizon:
            raise ValueError(
                "Maximum cutoff value is greater than end date minus horizon,"
                " no value for cross-validation remaining"
            )
        initial = cutoffs[0] - timestamps.min()

    # Initial fitting period should accomodate at least two cycles of the
    # longest seasonality component. If it doesn't, issue a warning.
    if initial < 2 * max_period:
        get_logger().warning(
            "The longest seasonality of the model has a period of "
            f"{max_period} which is larger than initial window. Consider "
            "increasing initial."
        )

    predicts = [
        single_cutoff_forecast(model, cutoff, horizon, predict_columns)
        for cutoff in cutoffs
    ]

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return pd.concat(predicts, axis=0).reset_index(drop=True)


def performance_metrics(
    df: pd.DataFrame,
    metric_name: str,
    timestamp_name: str,
    horizon_period: TimedeltaLike,
    performance_metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute performance metrics from cross-validation results.

    Computes a suite of performance metrics on the output of cross-validation.
    By default the following metrics are included:
    'mse': mean squared error
    'rmse': root mean squared error
    'mae': mean absolute error
    'mape': mean absolute percent error
    'smape': symmetric mean absolute percentage error
    'mdape': median absolute percent error
    'smdape': symmetric median absolute percent error
    'coverage': coverage of the upper and lower intervals

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe returned by cross_validation.
    metric_name : str
        The name of the metric column.
    timestamp_name : str
        The name of the timestamp column.
    horizon_period : TimedeltaLike
        Predictions will be aggregated in steps of horizon_period starting from
        the cutoff. E.g., if horizon_period = "3d", predictions between cutoff
        and third day after cutoff will be summarized under 3 day horizon,
        predictions between third and sixth day under 6 day horizon etc.
    performance_metrics : Optional[list[str]], optional
        The list of performance metrics that will be evaluated. If none is
        provided, all available metrics will be used. The default is None.

    Returns
    -------
    result : pd.DataFrame
        The result data frame with horizon as index and all performance metrics
        as columns. The additional column 'count' measures the number of
        predictions the performance metrics were calculated with.

    """

    # Validate metrics list
    if performance_metrics is None:
        # If none were provided, use all available metrics
        performance_metrics = list(PERFORMANCE_METRICS_MAP.keys())
    else:
        # If some were provided, validate them

        # Which of them are valid (= metrics known to Gloria)
        valid_metrics = set(performance_metrics).intersection(
            PERFORMANCE_METRICS_MAP.keys()
        )
        # All other metrics are invalid. Issue a warning
        invalid_metrics = set(performance_metrics) - (
            PERFORMANCE_METRICS_MAP.keys()
        )
        if invalid_metrics:
            im_str = ", ".join([f"'{im}'" for im in invalid_metrics])
            get_logger().warn(
                f"The performance metric(s) {im_str} are not known to Gloria "
                "and will be ignored."
            )
        # Look whether any metrics are duplicated and inform the user
        duplicates = set(
            m for m in valid_metrics if performance_metrics.count(m) > 1
        )
        if duplicates:
            dm_str = ", ".join([f"'{im}'" for im in duplicates])
            get_logger().warn(
                f"Duplicates found for the performance metric(s) {dm_str}. "
                "These will be evaluated only once."
            )
        # Bring metrics back to requested order
        performance_metrics = sorted(
            valid_metrics, key=lambda x: performance_metrics.index(x)
        )
    # Validation finished

    # Ensure horizon_period is a valid Timedelta object
    horizon_period = convert_to_timedelta(horizon_period)

    df_loc = df.copy()

    # Compute the forecast horizon in steps of horizon_period
    df_loc["horizon"] = (
        np.ceil((df_loc[timestamp_name] - df_loc["cutoff"]) / horizon_period)
        * horizon_period
    )

    # Add count column to show how many predictions the metric is based on
    result = df_loc.groupby("horizon").agg(count=("yhat", "count"))

    # Evaluate all metrics
    for pm in performance_metrics:
        result[pm] = df_loc.groupby("horizon")[
            [metric_name, "yhat", "observed_lower", "observed_upper"]
        ].apply(PERFORMANCE_METRICS_MAP[pm], metric_name)

    return result
