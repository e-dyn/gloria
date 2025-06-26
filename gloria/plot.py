# Standard Library
from typing import TYPE_CHECKING, Optional, Tuple

# Third Party
import numpy as np
import pandas as pd

try:
    # Third Party
    from matplotlib import pyplot as plt
    from matplotlib.artist import Artist
    from matplotlib.dates import (
        AutoDateFormatter,
        AutoDateLocator,
        num2date,
    )
    from matplotlib.ticker import FuncFormatter
    from pandas.plotting import deregister_matplotlib_converters

    deregister_matplotlib_converters()
except ImportError as err:
    raise ImportError(
        "Importing matplotlib failed." " Plotting will not work."
    ) from err

try:
    # Third Party
    import seaborn as sns
except ImportError as err:
    raise ImportError(
        "Importing seaborn failed." " Plotting will not work."
    ) from err

# Conditional import of Gloria for static type checking. Otherwise Gloria is
# forward-declared as 'Gloria' to avoid circular imports
if TYPE_CHECKING:
    # Gloria
    from gloria.interface import Gloria


def plot_trend_component(
    m: "Gloria",
    fcst: pd.DataFrame,
    component: str,
    ax: Optional[Artist] = None,
    uncertainty: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> list[Artist]:
    """
    Plot a single forecast component (e.g., trend)

    Parameters
    ----------
    m : Gloria model
        Fitted Gloria model containing uncertainty samples.
    fcst : pd.DataFrame
        Forecast DataFrame with predicted values and uncertainty bounds.
    name : str
        Name of the component to plot (e.g., 'trend').
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. Created if not provided.
    uncertainty : bool, optional
        Whether to plot uncertainty intervals (if available).
    figsize : tuple, optional
        Figure size (width, height) if ax is not provided.

    Returns
    -------
    list
        List of matplotlib artist objects created by the plot.
    """
    # Set Seaborn style and matplotlib parameters
    sns.set(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
        }
    )

    artists = []
    if ax is None:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)

    fcst_t = fcst[m.timestamp_name]

    # Plot main component line (e.g., trend)
    artists += ax.plot(
        fcst_t,
        fcst[component],
        linestyle="-",
        color="#264653",  # dark teal
        linewidth=1.5,
        label=component.capitalize(),
    )

    # Plot uncertainty interval if requested and available
    if uncertainty and m.uncertainty_samples:
        artists += [
            ax.fill_between(
                fcst_t,
                fcst[component + "_lower"],
                fcst[component + "_upper"],
                color="#819997",  # soft light blue
                alpha=0.3,
                label="Confidence Interval",
            )
        ]

    # Format x-axis with automatic date ticks
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Grid only on y-axis, despine top and right
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    sns.despine(ax=ax)

    # Axis labels and tick formatting
    ax.set_ylabel(component.capitalize(), labelpad=10)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    return artists


def plot_seasonality_component(
    m: "Gloria",
    component: str,
    period: int,
    ax: Optional[Artist] = None,
    start_offset: int = 0,
    figsize: Tuple[int, int] = (10, 6),
) -> list[Artist]:
    """
    Plot a custom seasonal component of the forecast.

    Parameters
    ----------
    m : Gloria model
        Fitted Gloria model.
    component : str
        Seasonality name (e.g., 'daily', 'weekly').
    period : int
        Number of time points in the seasonality period.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. Created if not provided.
    start_offset : int, optional
        Offset in days for the starting index of the seasonality.
    figsize : tuple, optional
        Figure size if ax is not provided.

    Returns
    -------
    list
        List of matplotlib artist objects created by the plot.
    """
    sns.set(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
        }
    )

    artists = []
    if ax is None:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)

    # Adjust offset only for weekly seasonality
    if component != "weekly":
        start_offset = 0
    df = get_seasonal_component_df(m, component, period, start_offset % 7)
  
    # Define date range for one seasonality period
    start_date = min(pd.to_datetime(df[m.timestamp_name]))
    end_date = start_date + pd.Timedelta(days=period)

    # Plot seasonal component line
    artists += ax.plot(
        df[m.timestamp_name],
        df[m.metric_name],
        linestyle="-",
        color="#264653",
        linewidth=1.5,
        label=component.capitalize(),
    )

    ax.axhline(y=0, color="#5c5c5c", linewidth=1.5, linestyle="--", alpha=0.7)

    # Light grid for context
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        
    ax.set_ylabel(component.capitalize())

   # Format x-axis depending on component
    x_dates = pd.to_datetime(df[m.timestamp_name])
    if component == "yearly":
        ax.set_xticks(x_dates)
        tick_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticklabels(tick_labels) 
    elif component == "quarterly":
        ax.set_xticks(x_dates)
        ax.set_xticklabels([f"Month {d+1}" for d in range(3)])
    elif component == "monthly":
        ax.set_xticks(x_dates)
        ax.set_xticklabels([f"{d+1}" for d in range(31)])  # 01-Jan
    elif component == "weekly":
        ax.set_xticks(x_dates)
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                    "Friday", "Saturday", "Sunday"]
        rotated_weekdays = weekdays[start_offset:] + weekdays[:start_offset]
        ax.set_xticklabels(rotated_weekdays)
    elif component == "daily":
        ax.set_xticks(x_dates)
        ax.set_xticklabels([f"{H}:00" for H in range(23)])  # 00:00, 01:00
    else:
        # Fallback: auto format
        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)


    # Rotate tick labels for clarity
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")


    return artists


def plot_event_component(
    m: "Gloria",
    component: str,
    ax: Optional[Artist] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Artist:
    """
    Plot event or external regressor component of the forecast.

    Parameters
    ----------
    m : Gloria model
        Fitted Gloria model.
    component : str
        Component name, either 'events' or 'external_regressors'.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. Created if not provided.
    figsize : tuple, optional
        Figure size if ax is not provided.

    Returns
    -------
    list
        List of matplotlib artist objects created by the plot.
    """
    sns.set(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
        }
    )

    artists = []
    if ax is None:
        fig = plt.figure(facecolor="w", figsize=figsize)
        ax = fig.add_subplot(111)

    df = get_event_component_df(m, component)

    # Plot main event/regressor line
    artists += ax.plot(
        df[m.timestamp_name],
        df[m.metric_name],
        linestyle="-",
        color="#264653",  # dark teal
        linewidth=1.5,
        label="Holidays" if component == "events" else "External Regressors",
    )

    # Format x-axis with automatic date ticks
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Grid only on y-axis, despine top and right
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    sns.despine(ax=ax)

    # Axis label and tick formatting
    label_name = (
        "Events + Holidays"
        if component == "events"
        else "External Regressors"
    )
    
    ax.set_ylabel(label_name, labelpad=10)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    return artists


def get_seasonal_component_df(
    m: "Gloria", component: str, period: int, start_offset: int = 0
) -> pd.DataFrame:
    """
    Extracts a seasonal component (e.g. 'weekly', 'monthly', 'yearly',
    'custom') as a DataFrame.

    Parameters
    ----------
    m : Gloria model
        Trained Gloria model.
    component : str
        Name of the seasonal component.
    period : int
        Number of time points in the period.
    start_offset : int, optional
        Offset in days for the starting index.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'ds' (date) and 'y' (seasonality value).
    """

    key_str = f"Seasonality__delim__{component}__delim__"

    # Filter relevant columns in design matrix
    filtered_columns = [col for col in m.X.columns if key_str in col]
    if not filtered_columns:
        raise ValueError(f"No columns found for component '{component}'.")

    X_component = m.X[filtered_columns]

    # Find indices of the relevant columns
    column_indices = [m.X.columns.get_loc(col) for col in filtered_columns]

    # Extract beta coefficients, if available
    if (
        hasattr(m, "model_backend")
        and hasattr(m.model_backend, "fit_params")
        and "beta" in m.model_backend.fit_params
    ):
        beta_all = np.array(m.model_backend.fit_params["beta"])
        if beta_all.ndim == 1 and max(column_indices) < len(beta_all):
            beta_component = beta_all[column_indices]
        else:
            raise ValueError(
                "Beta vector too short or incorrectly structured."
            )
    else:
        raise ValueError("No beta coefficients available in model backend.")

    # Calculate component values
    Xb = np.matmul(X_component, beta_component)

    period_start = get_period_start(m.history[m.timestamp_name], component)

    timerange = m.history[m.timestamp_name].iloc[
        period_start + start_offset : period_start + start_offset + period
    ]

    Xb = Xb.iloc[
        period_start + start_offset : period_start + start_offset + period
    ]
        
    return pd.DataFrame({m.timestamp_name: timerange, m.metric_name: Xb})


def get_event_component_df(m: "Gloria", component: str) -> pd.DataFrame:
    """
    Extracts an event or external regressor component as a DataFrame.

    Parameters
    ----------
    m : Gloria model
        Trained Gloria model.
    component : str
        Name of the component ('events' or 'external_regressors').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'ds' (date) and 'y' (component value).
    """
    
    if component == "events":
        component_names = [
            "Holiday",
            "SingleEvent",
            "IntermittentEvent",
            "PeriodicEvent",
        ]
    else:
        component_names = ["ExternalRegressor"]
    
    # Create all possible keys for filtering
    key_strs = [f"{name}__delim__" for name in component_names]
    
    # Filter all columns that contain one of the keys
    filtered_columns = [
        col for col in m.X.columns
        if any(key in col for key in key_strs)
    ]
    
    if not filtered_columns:
        raise ValueError(
            f"No columns found for component(s): {', '.join(component_names)}."
        )

    X_component = m.X[filtered_columns]

    # Find indices of the relevant columns
    column_indices = [m.X.columns.get_loc(col) for col in filtered_columns]

    # Extract beta coefficients, if available
    if (
        hasattr(m, "model_backend")
        and hasattr(m.model_backend, "fit_params")
        and "beta" in m.model_backend.fit_params
    ):
        beta_all = np.array(m.model_backend.fit_params["beta"])
        if beta_all.ndim == 1 and max(column_indices) < len(beta_all):
            beta_component = beta_all[column_indices]
        else:
            raise ValueError(
                "Beta vector too short or incorrectly structured."
            )
    else:
        raise ValueError("No beta coefficients available in model backend.")

    Xb = np.matmul(X_component, beta_component)

    days = m.history[m.timestamp_name]
    
    return pd.DataFrame({m.timestamp_name: days, m.metric_name: Xb})


def add_changepoints_to_plot(
    m: "Gloria",
    fcst: pd.DataFrame,
    ax: Artist,
    threshold: float = 0.01,
    cp_color: str = "#a76a48",
    cp_linestyle: str = "--",
) -> list[Artist]:
    """Add markers for significant changepoints to prophet forecast plot.

    Example:
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)

    Parameters
    ----------
    ax: axis on which to overlay changepoint markers.
    m: Gloria model.
    fcst: Forecast output from m.predict.
    threshold: Threshold on trend change magnitude for significance.
    cp_color: Color of changepoint markers.
    cp_linestyle: Linestyle for changepoint markers.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    # artists.append(ax.plot(fcst['ds'], fcst['trend'], c=cp_color))
    if m.changepoints is not None and len(m.changepoints) > 0:
        signif_changepoints = m.changepoints
    else:
        signif_changepoints = []
    artists += [
        ax.axvline(x=cp, c=cp_color, ls=cp_linestyle)
        for cp in signif_changepoints
    ]
    return artists


def get_period_start(dates: pd.Series, component: str):
    """
    Returns the index in `dates` where a new period starts,
    depending on the selected `component`.
    
    Parameters:
    -----------
    dates : pd.Series
        Series of datetime objects.
    component : str
        One of ['yearly', 'quarterly', 'monthly', 'weekly', 'daily'].
    
    Returns:
    --------
    pd.Index
        Index of row where a new period begins.
    """
    dates = pd.to_datetime(dates)  # Sicherstellen, dass es datetime ist

    if component == "yearly":
        # Vergleiche Jahr mit Vorgänger
        mask = dates.dt.year != dates.dt.year.shift(1)
    elif component == "quarterly":
        # Quartal erkennen: Q1=1, Q2=2, ...
        mask = dates.dt.quarter != dates.dt.quarter.shift(1)
    elif component == "monthly":
        # Monat erkennen
        mask = dates.dt.month != dates.dt.month.shift(1)
    elif component == "weekly":
        # Kalenderwoche vergleichen
        mask = dates.dt.isocalendar().week != dates.dt.isocalendar().week.shift(1)
    elif component == "daily":
        # Tag vergleichen
        mask = dates.dt.date != dates.dt.date.shift(1)
    else:
        raise ValueError(f"Unknown component: {component}")

    # Erster Index (0) ist immer ein Periodenstart, also setzen wir mask[0] = True
    mask.iloc[0] = False

    # Rückgabe der Indizes mit True
    return dates.index[mask][0]