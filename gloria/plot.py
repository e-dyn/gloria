# Standard Library
import math
from typing import Optional, Tuple

# Third Party
import numpy as np
import pandas as pd

# Gloria
from gloria import Gloria

try:
    # Third Party
    from matplotlib import pyplot as plt
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


def plot(
    m: Gloria,
    fcst: pd.DataFrame,
    ax: Optional[sns] = None,
    uncertainty: bool = True,
    xlabel: str = "ds",
    ylabel: str = "y",
    figsize: Tuple[int, int] = (10, 6),
    show_changepoints: bool = False,
    include_legend: bool = False,
) -> plt.figure:
    """
    Plot the forecast of a Gloria model, including trend line, predictions,
    and confidence intervals.

    Parameters
    ----------
    m : Gloria
        A trained Gloria model. Must contain historical data in `m.history`.

    fcst : pd.DataFrame
        DataFrame with forecast results, including the columns: 'ds', 'yhat',
        'trend', 'observed_lower', and 'observed_upper'.

    ax : sns.axes.Axes, optional
        An existing matplotlib axis to draw on. If None, a new figure
        and axis will be created.

    uncertainty : bool, default=True
        Whether to plot the uncertainty/confidence intervals.

    xlabel : str, default='ds'
        Label for the x-axis.

    ylabel : str, default='y'
        Label for the y-axis.

    figsize : tuple of int, default=(10, 6)
        Figure size in inches. Used only when creating a new figure.

    show_changepoints : bool, default=False
        Whether to display significant changepoints on the plot.

    include_legend : bool, default=False
        Whether to display a legend on the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the forecast plot.
    """
    # Check if a custom axis was passed
    user_provided_ax = ax is not None

    # Create new figure and axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="w")
    else:
        fig = ax.get_figure()

    # Set Seaborn style and update plot aesthetics
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

    # Plot historical data as scatter points
    sns.scatterplot(
        x=m.history["ds"],
        y=m.history["y"],
        ax=ax,
        label="Data",
        color="#016a86",
        edgecolor="w",
        s=20,
        alpha=0.7,
    )

    # Plot the model's trend line
    ax.plot(
        fcst["ds"],
        fcst["trend"],
        color="#264653",
        linewidth=1.0,
        alpha=0.8,
        label="Trend",
    )

    # Plot the forecast line
    ax.plot(
        fcst["ds"], fcst["yhat"], color="#e6794a", linewidth=1.5, label="Fit"
    )

    # Plot the confidence interval (if enabled)
    if uncertainty:
        ax.fill_between(
            fcst["ds"],
            fcst["observed_lower"],
            fcst["observed_upper"],
            color="#819997",
            alpha=0.3,
            label="Confidence Interval",
        )

    if show_changepoints:
        add_changepoints_to_plot(m, fcst, ax)

    # Set date format for x-axis
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Set axis labels
    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_ylabel(ylabel, labelpad=15)

    # Add gridlines (only horizontal)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")

    # Remove top and right spines for cleaner look
    sns.despine(ax=ax)

    # Remove default legend unless specified
    try:
        ax.get_legend().remove()
    except AttributeError:
        pass

    if include_legend:
        ax.legend(frameon=True, shadow=True, loc="best", fontsize=10)

    # Adjust layout if we created the figure
    if not user_provided_ax:
        fig.tight_layout()

    return fig


def plot_components(
    m: Gloria,
    fcst: pd.DataFrame,
    uncertainty: bool = True,
    weekly_start: int = 0,
    figsize: Tuple[int, int] | None = None,
) -> plt.figure:
    """
    Plot forecast components of a Gloria model using a modern Seaborn style.

    Parameters
    ----------
    m : Gloria
        A fitted Gloria model containing seasonalities, events, regressors,
        and trend.

    fcst : pd.DataFrame
        Forecast dataframe from the model, used for plotting trend
        and uncertainty.

    uncertainty : bool, optional, default=True
        Whether to include uncertainty intervals in the trend component plot.

    weekly_start : int, optional, default=0
        Starting day of the week (0=Monday) for weekly seasonal plots.

    figsize : tuple of float, optional
        Figure size as (width, height). If not provided, it is calculated
        automatically to arrange subplots in a nearly square grid.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure containing all component subplots.
    """

    # Set Seaborn style and Matplotlib parameters for consistent aesthetics
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

    # Define components to plot: always include 'trend'
    components = ["trend"]

    # Add seasonalities detected in the model
    components.extend(m.model_extra["seasonalities"].keys())

    # Add events if available
    if m.model_extra["events"].keys():
        components.append("events")

    # Add external regressors if available
    if m.model_extra["external_regressors"].keys():
        components.append("external_regressors")

    npanel = len(components)

    # Calculate number of rows and columns for subplot
    # grid (as square as possible)
    ncols = math.floor(math.sqrt(npanel))
    nrows = math.ceil(npanel / ncols)

    # Automatically determine figure size if not specified
    if not figsize:
        figsize = (int(4.5 * ncols), int(3.2 * nrows))

    # Create subplots with white background
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="w")

    # Flatten axes array for easy iteration, handle single subplot case
    axes = axes.flatten() if npanel > 1 else [axes]

    # Loop over components and call corresponding plot functions
    for ax, plot_name in zip(axes, components):
        if plot_name == "trend":
            plot_trend_component(
                m=m,
                fcst=fcst,
                component="trend",
                ax=ax,
                uncertainty=uncertainty,
            )
        elif plot_name in m.model_extra["seasonalities"].keys():
            plot_seasonality_component(
                m=m,
                component=plot_name,
                start_offset=weekly_start,
                period=int(
                    np.floor(m.model_extra["seasonalities"][plot_name].period)
                ),
                ax=ax,
            )
        elif plot_name in ["events", "external_regressors"]:
            plot_event_component(m=m, component=plot_name, ax=ax)

        # Visual tuning: grid only on y-axis, remove x-axis grid,
        # remove top/right spines
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.grid(visible=False, axis="x")
        sns.despine(ax=ax)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    return fig


def plot_trend_component(
    m: Gloria,
    fcst: pd.DataFrame,
    component: str,
    ax: Optional[sns] = None,
    uncertainty: bool = True,
    figsize: Tuple[int, int] = (10, 6),
):
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
    m: Gloria,
    component: str,
    period: int,
    ax: Optional[sns] = None,
    start_offset: int = 0,
    figsize: Tuple[int, int] = (10, 6),
):
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
    m: Gloria,
    component: str,
    ax: Optional[sns] = None,
    figsize: Tuple[int, int] = (10, 6),
):
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
    m: Gloria, component: str, period: int, start_offset: int = 0
):
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


def get_event_component_df(m: Gloria, component: str):
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
    m: Gloria,
    fcst: pd.DataFrame,
    ax: sns,
    threshold: float = 0.01,
    cp_color: str = "#a76a48",
    cp_linestyle: str = "--",
):
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
