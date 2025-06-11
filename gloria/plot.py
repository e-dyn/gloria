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

    fcst_t = fcst["ds"]

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
    start_date = min(pd.to_datetime(df["ds"]))
    end_date = start_date + pd.Timedelta(days=period)

    # Plot seasonal component line
    artists += ax.plot(
        df["ds"],
        df["y"],
        linestyle="-",
        color="#264653",
        linewidth=1.5,
        label=component.capitalize(),
    )

    ax.axhline(y=0, color="#5c5c5c", linewidth=1.5, linestyle="--", alpha=0.7)

    # Light grid for context
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)

    # Set x-ticks and format according to seasonality type
    n_ticks = 8
    xticks = pd.to_datetime(
        np.linspace(start_date.value, end_date.value, n_ticks)
    ).to_pydatetime()
    ax.set_xticks(xticks)

    if component == "yearly":
        month_starts = pd.date_range(start=start_date, end=end_date, freq="MS")
        ax.set_xticks(month_starts)
        fmt = FuncFormatter(
            lambda x, pos=None: "{dt:%b} {dt.day}".format(dt=num2date(x))
        )
        ax.xaxis.set_major_formatter(fmt)
    elif component == "weekly":
        ax.set_xlim(
            start_date - pd.Timedelta(hours=12),
            start_date + pd.Timedelta(days=period - 1, hours=12),
        )
        fmt = FuncFormatter(
            lambda x, pos=None: "{dt:%A}".format(dt=num2date(x))
        )
        ax.xaxis.set_major_formatter(fmt)
    elif component == "daily":
        fmt = FuncFormatter(
            lambda x, pos=None: "{dt:%T}".format(dt=num2date(x))
        )
        ax.xaxis.set_major_formatter(fmt)
    elif period <= 2:
        fmt = FuncFormatter(
            lambda x, pos=None: "{dt:%T}".format(dt=num2date(x))
        )
        ax.xaxis.set_major_formatter(fmt)
    else:
        fmt = FuncFormatter(
            lambda x, pos=None: "{:.0f}".format(
                1 + pos * (period - 1) / (n_ticks - 1)
            )
        )
        ax.xaxis.set_major_formatter(fmt)

    ax.set_ylabel(component.capitalize())

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
        df["ds"],
        df["y"],
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
    label_name = "Events" if component == "events" else "External Regressors"
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
    start_date = "2017-01-01"
    key_str = f"Seasonality__delim__{component}__delim__"

    # Filter relevant columns in design matrix
    filtered_columns = [col for col in m.X.columns if key_str in col]
    if not filtered_columns:
        raise ValueError(f"No columns found for component '{component}'.")

    X_component = m.X[filtered_columns]

    # Extract corresponding coefficients
    beta_component = [
        value for key, value in m.prior_scales.items() if key_str in key
    ]
    if not beta_component:
        raise ValueError(f"No coefficients found for component '{component}'.")

    # Calculate component values
    Xb = np.matmul(X_component, beta_component)

    # Create timeline for the seasonality period
    days = pd.date_range(start=start_date, periods=period) + pd.Timedelta(
        days=start_offset
    )

    relevant_y = Xb.iloc[start_offset : start_offset + period].reset_index(
        drop=True
    )

    return pd.DataFrame({"ds": days, "y": relevant_y})


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
        component_name = "Holiday"
    else:
        component_name = "ExternalRegressor"

    key_str = f"{component_name}__delim__"

    filtered_columns = [col for col in m.X.columns if key_str in col]
    if not filtered_columns:
        raise ValueError(f"No columns found for component {component_name}.")

    X_component = m.X[filtered_columns]

    beta_component = [
        value for key, value in m.prior_scales.items() if key_str in key
    ]
    if not beta_component:
        raise ValueError(
            f"No coefficients found for component {component_name}."
        )

    Xb = np.matmul(X_component, beta_component)

    days = m.history["ds"]

    return pd.DataFrame({"ds": days, "y": Xb})


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
