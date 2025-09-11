# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Standard Library
from typing import TYPE_CHECKING, Any, Optional, Tuple

# Third Party
import numpy as np
import pandas as pd

try:
    # Third Party
    from matplotlib import pyplot as plt
    from matplotlib.artist import Artist
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator
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
    ax: Optional[plt.Axes] = None,
    uncertainty: bool = True,
    plot_kwargs: Optional[dict[str, Any]] = None,
    line_kwargs: Optional[dict[str, Any]] = None,
    interval_kwargs: Optional[dict[str, Any]] = None,
    xlabel_kwargs: Optional[dict[str, Any]] = None,
    ylabel_kwargs: Optional[dict[str, Any]] = None,
    grid_y_kwargs: Optional[dict[str, Any]] = None,
    ticklabel_kwargs: Optional[dict[str, Any]] = None,
    rcparams_kwargs: Optional[dict[str, Any]] = None,
    style_kwargs: Optional[dict[str, Any]] = None,
) -> list[Artist]:
    """
    Plot the trend component of a forecast with extensive customization.

    This function visualizes the trend pattern extracted from a fitted
    Gloria model. It supports extensive customization of figure, axes, grid,
    labels, and line styles.

    Parameters
    ----------
    m : Gloria
        Fitted Gloria model containing uncertainty samples and configuration.
    fcst : pd.DataFrame
        Forecast DataFrame with predicted values and uncertainty bounds.
    component : str
        Name of the component to plot (e.g., "trend").
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. If not provided, a new figure and axes are
        created.
    uncertainty : bool, default True
        Whether to plot uncertainty intervals (if available).
    plot_kwargs : dict, optional
        Keyword arguments for figure creation if ax is None (e.g., figsize,
        dpi).
    line_kwargs : dict, optional
        Styling options for the main line plot (e.g., color, linewidth).
    interval_kwargs : dict, optional
        Styling options for the uncertainty interval (fill_between).
    xlabel_kwargs : dict, optional
        Keyword arguments for the x-axis label (ax.set_xlabel).
    ylabel_kwargs : dict, optional
        Keyword arguments for the y-axis label (ax.set_ylabel).
    grid_y_kwargs : dict, optional
        Keyword arguments for customizing the y-axis grid.
    ticklabel_kwargs : dict, optional
        Keyword arguments for tick label formatting (rotation, fontsize, etc.).
    rcparams_kwargs : dict, optional
        Matplotlib rcParams overrides for styling.
    style_kwargs : dict, optional
        Seaborn style configuration.

    Returns
    -------
    list of matplotlib.artist.Artist
        List of Matplotlib artist objects created by the plot.
    """

    # Initialize kwargs if None
    plot_kwargs = plot_kwargs or {}
    line_kwargs = line_kwargs or {}
    interval_kwargs = interval_kwargs or {}
    xlabel_kwargs = xlabel_kwargs or {}
    ylabel_kwargs = ylabel_kwargs or {}
    grid_y_kwargs = grid_y_kwargs or {}
    ticklabel_kwargs = ticklabel_kwargs or {}
    rcparams_kwargs = rcparams_kwargs or {}
    style_kwargs = style_kwargs or {}

    # Default seaborn style
    style_defaults = dict(style="whitegrid")
    style_defaults.update(style_kwargs)
    sns.set(**style_defaults)

    # rcParams defaults
    rcparams_defaults = {
        "font.size": 14,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
    }
    rcparams_defaults.update(rcparams_kwargs)

    with plt.rc_context(rc=rcparams_defaults):
        with sns.axes_style(style_defaults):

            # Default figure properties if ax is None
            plot_defaults = {
                "figsize": (10, 6),
                "facecolor": "w",
                "dpi": 150,
            }
            plot_defaults.update(plot_kwargs)

            artists = []
            if ax is None:
                fig = plt.figure(**plot_defaults)
                ax = fig.add_subplot(111)

            fcst_t = fcst[m.timestamp_name]

            # Main line styling defaults
            line_defaults = dict(
                linestyle="-",
                color="#264653",
                linewidth=1.5,
                label=component.capitalize(),
            )
            line_defaults.update(line_kwargs)

            # Plot main line
            artists += ax.plot(
                fcst_t,
                fcst[component],
                **line_defaults,
            )

            # Interval styling defaults
            interval_defaults = dict(
                color="#819997",
                alpha=0.3,
                label="Confidence Interval",
            )
            interval_defaults.update(interval_kwargs)

            if uncertainty and m.trend_samples:
                artists += [
                    ax.fill_between(
                        fcst_t,
                        fcst[f"{component}_lower"],
                        fcst[f"{component}_upper"],
                        **interval_defaults,
                    )
                ]

            # Configure date ticks
            locator = AutoDateLocator(interval_multiples=False)
            formatter = AutoDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # Set axis labels
            xlabel = xlabel_kwargs.pop("label", m.timestamp_name)
            ylabel = ylabel_kwargs.pop("label", component.capitalize())
            ax.set_xlabel(xlabel, **xlabel_kwargs)
            ax.set_ylabel(ylabel, **ylabel_kwargs)

            # Y-axis grid
            grid_y_defaults = dict(
                visible=True,
                axis="y",
                linestyle="--",
                alpha=0.3,
            )
            grid_y_defaults.update(grid_y_kwargs)
            ax.grid(**grid_y_defaults)
            ax.grid(visible=False, axis="x")

            # Customize tick labels
            ticklabel_defaults = dict(
                rotation=45,
                horizontalalignment="center",
            )
            ticklabel_defaults.update(ticklabel_kwargs)
            for label in ax.get_xticklabels():
                label.set_rotation(ticklabel_defaults.get("rotation", 0))
                label.set_horizontalalignment(
                    ticklabel_defaults.get("horizontalalignment", "center")
                )
                if "fontsize" in ticklabel_defaults:
                    label.set_fontsize(ticklabel_defaults["fontsize"])
                if "color" in ticklabel_defaults:
                    label.set_color(ticklabel_defaults["color"])

    return artists


def plot_seasonality_component(
    m: "Gloria",
    component: str,
    period: int,
    ax: Optional[Artist] = None,
    start_offset: int = 0,
    figsize: Tuple[int, int] = (10, 6),
    plot_kwargs: Optional[dict[str, Any]] = None,
    line_kwargs: Optional[dict[str, Any]] = None,
    interval_kwargs: Optional[dict[str, Any]] = None,
    xlabel_kwargs: Optional[dict[str, Any]] = None,
    ylabel_kwargs: Optional[dict[str, Any]] = None,
    grid_y_kwargs: Optional[dict[str, Any]] = None,
    ticklabel_kwargs: Optional[dict[str, Any]] = None,
    rcparams_kwargs: Optional[dict[str, Any]] = None,
    style_kwargs: Optional[dict[str, Any]] = None,
) -> list[Artist]:
    """
    Plot a seasonality component (e.g., weekly, yearly) with customizable
    styling.

    This function visualizes the seasonal pattern extracted from a fitted
    Gloria model, such as weekly or yearly seasonality. It supports extensive
    customization of figure, axes, grid, labels, and line styles.

    Parameters
    ----------
    m : Gloria
        The fitted Gloria model providing the seasonal component data.
    component : str
        The name of the seasonality component to plot.
        Supported values: "yearly", "quarterly", "monthly",
        "weekly", "daily".
    period : int
        The period length in days for the seasonal component.
    ax : matplotlib.axes.Axes, optional
        A Matplotlib Axes object to plot into. If None, a new figure
        and axes are created.
    start_offset : int, default 0
        Offset in days for shifting the start of the seasonal period.
        Only relevant when `component="weekly"`.
    figsize : tuple of int, default (10, 6)
        Size of the figure if `ax` is None.
    plot_kwargs : dict, optional
        Additional keyword arguments passed to `plt.figure()`
        (e.g., facecolor, dpi).
    line_kwargs : dict, optional
        Styling options for the main line plot (e.g., color, linewidth).
    interval_kwargs : dict, optional
        Styling options for the horizontal reference line at y=0
        (e.g., linestyle, alpha).
    xlabel_kwargs : dict, optional
        Additional keyword arguments for the x-axis label.
    ylabel_kwargs : dict, optional
        Additional keyword arguments for the y-axis label.
    grid_y_kwargs : dict, optional
        Additional keyword arguments for customizing the y-axis grid.
    ticklabel_kwargs : dict, optional
        Keyword arguments for formatting tick labels
        (e.g., rotation, fontsize).
    rcparams_kwargs : dict, optional
        Dictionary to override Matplotlib `rcParams`.
    style_kwargs : dict, optional
        Seaborn style configuration.

    Returns
    -------
    list of matplotlib.artist.Artist
        A list of Matplotlib Artist objects created by the plot.
    """
    # Third Party
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    # Initialize kwargs if None
    plot_kwargs = plot_kwargs or {}
    line_kwargs = line_kwargs or {}
    interval_kwargs = interval_kwargs or {}
    xlabel_kwargs = xlabel_kwargs or {}
    ylabel_kwargs = ylabel_kwargs or {}
    grid_y_kwargs = grid_y_kwargs or {}
    ticklabel_kwargs = ticklabel_kwargs or {}
    rcparams_kwargs = rcparams_kwargs or {}
    style_kwargs = style_kwargs or {}

    # Set Seaborn style and rcParams
    style_defaults = {"style": "whitegrid"}
    style_defaults.update(style_kwargs)
    sns.set(**style_defaults)

    rcparams_defaults = {
        "font.size": 14,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
    }
    rcparams_defaults.update(rcparams_kwargs)
    plt.rcParams.update(rcparams_defaults)

    with plt.rc_context(rc=rcparams_defaults):
        with sns.axes_style(style_defaults):

            # Subplots kwargs defaults
            plot_defaults = {
                "figsize": figsize,
                "facecolor": "w",
                "dpi": 150,
            }
            # Update defaults with user-provided plot_kwargs
            plot_defaults.update(plot_kwargs)

            artists = []
            if ax is None:
                fig = plt.figure(**plot_defaults)
                ax = fig.add_subplot(111)

            # If not weekly, ignore start_offset
            if component != "weekly":
                start_offset = 0

            df = get_seasonal_component_df(
                m, component, period, start_offset % 7
            )

            # Line styling
            line_defaults = {
                "linestyle": "-",
                "color": "#264653",
                "linewidth": 1.5,
            }
            line_defaults.update(line_kwargs)

            artists += ax.plot(
                df[m.timestamp_name],
                df[m.metric_name],
                **line_defaults,
                label=component.capitalize(),
            )

            # Horizontal reference line at y=0
            interval_defaults = {
                "color": "#5c5c5c",
                "linewidth": 1.5,
                "linestyle": "--",
                "alpha": 0.7,
            }
            interval_defaults.update(interval_kwargs)
            ax.axhline(y=0, **interval_defaults)

            # Y-axis grid
            grid_y_defaults = {
                "visible": True,
                "axis": "y",
                "linestyle": "--",
                "alpha": 0.3,
            }
            grid_y_defaults.update(grid_y_kwargs)
            ax.grid(**grid_y_defaults)
            ax.grid(visible=False, axis="x")

            # Y-axis label
            ylabel_defaults = {}
            ylabel_defaults.update(ylabel_kwargs)
            ax.set_ylabel(component.capitalize(), **ylabel_defaults)

            # X-ticks and labels
            x_dates = pd.to_datetime(df[m.timestamp_name])
            if component == "yearly":
                x_dates = x_dates.loc[x_dates.dt.day == 1]
                ax.set_xticks(x_dates)
                tick_labels = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                ax.set_xticklabels(tick_labels)
            elif component == "quarterly":
                x_dates = x_dates.loc[x_dates.dt.day == 1]
                ax.set_xticks(x_dates)
                ax.set_xticklabels(
                    [f"Month {m+1}" for m in range(len(x_dates))]
                )
            elif component == "monthly":
                # Get main month
                month = x_dates.dt.month.median()
                # Filter for main month and weekly ticks
                x_dates = x_dates.loc[
                    (x_dates.dt.day % 7 == 1) & (x_dates.dt.month == month)
                ]
                ax.set_xticks(x_dates)
                ax.set_xticklabels([f"Week {w//7+1}" for w in x_dates.dt.day])
            elif component == "weekly":
                ax.set_xticks(x_dates)
                weekdays = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                rotated_weekdays = (
                    weekdays[start_offset:] + weekdays[:start_offset]
                )
                ax.set_xticklabels(rotated_weekdays)
            elif component == "daily":
                ax.set_xticks(x_dates)
                ax.set_xticklabels([f"{H}:00" for H in range(24)])
            else:
                locator = AutoDateLocator()
                formatter = AutoDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

            # Tick label customization
            ticklabel_defaults = {
                "rotation": 45,
                "horizontalalignment": "center",
            }
            ticklabel_defaults.update(ticklabel_kwargs)

            for label in ax.get_xticklabels():
                label.set_rotation(ticklabel_defaults.get("rotation", 0))
                label.set_horizontalalignment(
                    ticklabel_defaults.get("horizontalalignment", "center")
                )
                # Additional custom attributes if supported
                for k, v in ticklabel_kwargs.items():
                    if hasattr(label, f"set_{k}"):
                        getattr(label, f"set_{k}")(v)

    return artists


def plot_event_component(
    m: "Gloria",
    component: str,
    ax: Optional[Artist] = None,
    figsize: Tuple[int, int] = (10, 6),
    line_kwargs: Optional[dict[str, Any]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    interval_kwargs: Optional[dict[str, Any]] = None,
    xlabel_kwargs: Optional[dict[str, Any]] = None,
    ylabel_kwargs: Optional[dict[str, Any]] = None,
    grid_y_kwargs: Optional[dict[str, Any]] = None,
    ticklabel_kwargs: Optional[dict[str, Any]] = None,
    rcparams_kwargs: Optional[dict[str, Any]] = None,
    style_kwargs: Optional[dict[str, Any]] = None,
) -> list[Artist]:
    """
    Plot an event or external regressor component with customizable styling.

    This function visualizes the time series contribution of events or
    external regressors extracted from a fitted Gloria model.

    Parameters
    ----------
    m : Gloria
        The fitted Gloria model providing the event component data.
    component : str
        The name of the event component ("events" or other external regressors)
    ax : matplotlib.axes.Axes, optional
        A Matplotlib Axes object to plot into. If None, a new figure
        and axes are created.
    figsize : tuple of int, default (10, 6)
        Size of the figure if `ax` is None.
    line_kwargs : dict, optional
        Styling options for the main line plot (e.g., color, linewidth).
    plot_kwargs : dict, optional
        Additional keyword arguments passed to `plt.figure()`.
    interval_kwargs : dict, optional
        Styling options for the horizontal reference line at y=0.
    xlabel_kwargs : dict, optional
        Additional keyword arguments for the x-axis label.
    ylabel_kwargs : dict, optional
        Additional keyword arguments for the y-axis label.
    grid_y_kwargs : dict, optional
        Additional keyword arguments for customizing the y-axis grid.
    ticklabel_kwargs : dict, optional
        Keyword arguments for formatting tick labels (e.g., rotation).
    rcparams_kwargs : dict, optional
        Dictionary to override Matplotlib `rcParams`.
    style_kwargs : dict, optional
        Seaborn style configuration.

    Returns
    -------
    list of matplotlib.artist.Artist
        A list of Matplotlib Artist objects created by the plot.
    """
    # Third Party
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    # Initialize kwargs if None
    plot_kwargs = plot_kwargs or {}
    line_kwargs = line_kwargs or {}
    interval_kwargs = interval_kwargs or {}
    xlabel_kwargs = xlabel_kwargs or {}
    ylabel_kwargs = ylabel_kwargs or {}
    grid_y_kwargs = grid_y_kwargs or {}
    ticklabel_kwargs = ticklabel_kwargs or {}
    rcparams_kwargs = rcparams_kwargs or {}
    style_kwargs = style_kwargs or {}

    # Set Seaborn style and rcParams
    style_defaults = {"style": "whitegrid"}
    style_defaults.update(style_kwargs)
    sns.set(**style_defaults)

    rcparams_defaults = {
        "font.size": 14,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
    }
    rcparams_defaults.update(rcparams_kwargs)
    plt.rcParams.update(rcparams_defaults)

    with plt.rc_context(rc=rcparams_defaults):
        with sns.axes_style(style_defaults):

            # Subplot kwargs defaults
            plot_defaults = {
                "figsize": figsize,
                "facecolor": "w",
                "dpi": 150,
            }
            # Update defaults with user-provided plot_kwargs
            plot_defaults.update(plot_kwargs)

            artists = []
            if ax is None:
                fig = plt.figure(**plot_defaults)
                ax = fig.add_subplot(111)

            # Load event component data
            df = get_event_component_df(m, component)

            # Line styling defaults
            line_defaults = {
                "linestyle": "-",
                "color": "#264653",
                "linewidth": 1.5,
                "label": (
                    "Holidays"
                    if component == "events"
                    else "External Regressors"
                ),
            }
            line_defaults.update(line_kwargs)

            artists += ax.plot(
                df[m.timestamp_name], df[m.metric_name], **line_defaults
            )

            # Set date locator and formatter
            locator = AutoDateLocator(interval_multiples=False)
            formatter = AutoDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # Horizontal reference line at y=0
            interval_defaults = {
                "color": "#5c5c5c",
                "linewidth": 1.5,
                "linestyle": "--",
                "alpha": 0.7,
            }
            interval_defaults.update(interval_kwargs)
            ax.axhline(y=0, **interval_defaults)

            # Y-axis grid
            grid_y_defaults = {
                "visible": True,
                "axis": "y",
                "linestyle": "--",
                "alpha": 0.3,
            }
            grid_y_defaults.update(grid_y_kwargs)
            ax.grid(**grid_y_defaults)
            ax.grid(visible=False, axis="x")

            # Y-axis label
            label_name = (
                "Events + Holidays"
                if component == "events"
                else "External Regressors"
            )
            ylabel_kw = {}
            ylabel_kw.update(ylabel_kwargs)
            ax.set_ylabel(label_name, **ylabel_kw)

            # Tick label styling
            ticklabel_defaults = {
                "rotation": 45,
                "horizontalalignment": "center",
            }
            ticklabel_defaults.update(ticklabel_kwargs)

            # Apply tick label styling
            for label in ax.get_xticklabels():
                label.set_rotation(ticklabel_defaults.get("rotation", 0))
                label.set_horizontalalignment(
                    ticklabel_defaults.get("horizontalalignment", "center")
                )
                # Apply additional tick label attributes if supported
                for k, v in ticklabel_kwargs.items():
                    if hasattr(label, f"set_{k}"):
                        getattr(label, f"set_{k}")(v)

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
        col for col in m.X.columns if any(key in col for key in key_strs)
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
        Index of the row where a new period begins.

    Raises:
    -------
    ValueError
        If an unknown component string is passed.
    """
    dates = pd.to_datetime(dates)  # Ensure the series is datetime type

    if component == "yearly":
        # Compare year with previous entry
        mask = dates.dt.year != dates.dt.year.shift(1)
    elif component == "quarterly":
        # Detect quarter: Q1=1, Q2=2, ...
        mask = dates.dt.quarter != dates.dt.quarter.shift(1)
    elif component == "monthly":
        # Detect month change
        mask = dates.dt.month != dates.dt.month.shift(1)
    elif component == "weekly":
        # Compare calendar week
        mask = (
            dates.dt.isocalendar().week != dates.dt.isocalendar().week.shift(1)
        )
    elif component == "daily":
        # Compare day change
        mask = dates.dt.date != dates.dt.date.shift(1)
    else:
        raise ValueError(f"Unknown component: {component}")

    # The first index (0) is always a period start,
    # so set mask[0] = False to exclude it from the result
    mask.iloc[0] = False

    # Return the index of the first True value in mask (start of new period)
    return dates.index[mask][0]
