# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Definition of the protocol for handling calendric data.
"""

### --- Module Imports --- ###
# Standard Library
from typing import TYPE_CHECKING, Any, Optional, Type, Union, cast

# Third Party
import holidays
import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from typing_extensions import Self

# Inhouse Packages
if TYPE_CHECKING:
    from gloria import Gloria

# Gloria
from gloria.profiles import BoxCar, Profile
from gloria.protocols.protocol_base import Protocol
from gloria.regressors import IntermittentEvent
from gloria.utilities.constants import _HOLIDAY
from gloria.utilities.logging import get_logger
from gloria.utilities.misc import infer_sampling_period

### --- Global Constants Definitions --- ###

DEFAULT_SEASONALITIES = {
    "yearly": {"period": pd.Timedelta("365.25d"), "default_order": 10},
    "quarterly": {"period": pd.Timedelta("365.25d") / 4, "default_order": 2},
    "monthly": {"period": pd.Timedelta("365.25d") / 12, "default_order": 3},
    "weekly": {"period": pd.Timedelta("7d"), "default_order": 3},
    "daily": {"period": pd.Timedelta("1d"), "default_order": 4},
}


### --- Class and Function Definitions --- ###


def get_holidays(
    country: str,
    subdiv: Optional[str] = None,
    timestamps: Optional[pd.Series] = None,
) -> tuple[holidays.HolidayBase, set[str]]:
    """
    Return a mapping of holiday dates to names for a specified country (and,
    optionally, one of its subdivisions).

    If ``timestamps`` is supplied, the result is limited to holidays that fall
    within that series. Otherwise, the full calendar provided by the
    ``holidays`` package is returned, covering years 1990 through 2100.

    Parameters
    ----------
    country : str
        Two-letter ISO
        `ISO 3166-1 alpha-2 code <https://tinyurl.com/msw8fajk>`_ of the
        country (e.g. ``"US"``, ``"DE"``).
    subdiv : str | None
        An optional `ISO 3166-2 subdivion code <https://tinyurl.com/2b432nrx>`_
        (e.g. state, province etc.). If ``None``, only nationwide holidays are
        considered.
    timestamps : :class:`pandas.Series` | None
        Series whose timestamp values define the time span of interest.
        Holidays outside this span are omitted. Supplying ``None`` returns the
        complete holiday calendar.

    Raises
    ------
    AttributeError
        If the requested ``country`` is not supported by the ``holidays``
        package.

    Returns
    -------
    all_holidays : holidays.HolidayBase
        Dictionary-like object mapping ``date`` → ``holiday_name`` for every
        holiday in the chosen region and time window.
    all_holiday_names : set[str]
        Unique set of holiday names contained in ``all_holidays``.
    """

    # Get the class according to requested country
    if not hasattr(holidays, country):
        raise AttributeError(
            f"Holidays in {country} are not currently " "supported!"
        )
    holiday_generator = getattr(holidays, country)

    # If no timestamps were given, take the entire available date range and
    # convert to years
    if timestamps is None:
        # Third Party
        from holidays.constants import DEFAULT_END_YEAR, DEFAULT_START_YEAR

        years = np.array(range(DEFAULT_START_YEAR, DEFAULT_END_YEAR + 1))
    else:
        years = timestamps.dt.year.unique()

    # Get all holidays for desired country and year-range
    all_holidays = holiday_generator(
        subdiv=subdiv, years=years, language="en_US"
    )

    # Get a set of all holiday names in the range. The split-and-join is a
    # safety measure for rare cases that two holidays share a the same date in
    # which case they are separated by a semi-colon.
    all_holiday_names = set("; ".join(all_holidays.values()).split("; "))

    return all_holidays, all_holiday_names


def make_holiday_dataframe(
    timestamps: pd.Series,
    country: str,
    subdiv=None,
    timestamp_name: str = "ds",
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of holidays that fall within the span covered by
    input timestamps.

    The result has one row per holiday occurrence and two columns:

    1. ``<timestamp_name>`` - the holiday date.
    2. ``holiday`` - the holiday`s common name.

    Parameters
    ----------
    timestamps : :class:`pandas.Series`
        Series whose timestamp values define the time span of interest.
        Holidays outside this span are omitted.
    country : str
        Two-letter ISO
        `ISO 3166-1 alpha-2 code <https://tinyurl.com/msw8fajk>`_ of the
        country (e.g. ``"US"``, ``"DE"``).
    subdiv : str | None
        An optional `ISO 3166-2 subdivion code <https://tinyurl.com/2b432nrx>`_
        (e.g. state, province etc.). If ``None``, only nationwide holidays are
        considered.
    timestamp_name : str | None
        Desired name for the timestamp column. The default is ``"ds"``.

    Returns
    -------
    :class:`pandas.DataFrame`
        Two-column DataFrame with the columns described above, sorted by date.
    """

    # First get the HolidayBase object and a set of all holiday names
    all_holidays, all_holiday_names = get_holidays(
        country=country, subdiv=subdiv, timestamps=timestamps
    )

    # Iterate through all holiday names and get respective dates, each stored
    # in a small DataFrame
    holiday_df_list = []
    for name in all_holiday_names:
        holiday_df_loc = pd.DataFrame(
            {timestamp_name: all_holidays.get_named(name), _HOLIDAY: name}
        )
        holiday_df_list.append(holiday_df_loc)

    # Make one overall DataFrame
    holiday_df = pd.concat(holiday_df_list)

    # Postprocess a little
    holiday_df[timestamp_name] = pd.to_datetime(holiday_df[timestamp_name])
    holiday_df.sort_values(by=timestamp_name, inplace=True)
    # Some holidays need to be removed as HolidayBase only returns full-year-
    # wise
    holiday_df = holiday_df.loc[
        (holiday_df[timestamp_name] >= timestamps.min())
        & (holiday_df[timestamp_name] <= timestamps.max())
    ].reset_index(drop=True)

    return holiday_df


class Holiday(IntermittentEvent):
    """
    A regressor to model events coinciding with public holidays.

    The regressor is added to the :class:`Gloria` model either using
    :meth:`~Gloria.add_event` or by adding the :class:`CalendricData` protocoll
    via :meth:`Gloria.add_protocol` and does not need to be handled directly by
    the user.

    Parameters
    ----------
    name : str
        A descriptive, unique name to identify the regressor. Note that the
        ``name`` must equal the desired public holiday name as registered in
        the `holiday <https://holidays.readthedocs.io/en/latest/>`_ package.
        The function :func:`get_holidays` may be used to inspect valid
        holiday names.
    prior_scale : float
        Parameter modulating the strength of the regressors. Larger values
        allow the model to fit a larger impact of the event, smaller
        values dampen the impact. Must be larger than zero.
    profile : Profile
        The profile that periodically occurs. Allowed profile types are
        described in the :ref:`ref-profiles` section.
    t_list : list[:class:`pandas.Timestamp`]
        A list of timestamps at which ``profile`` occurs. The exact meaning of
        each timestamp in the list depends on implementation details of the
        underlying ``profile``, but typically refers to its mode.

        .. note::
            A user provided ``t_list`` will be ignored and overwritten with an
            automatically generated list of holiday occurrences.
    country : str
        The `ISO 3166-1 alpha-2 code <https://tinyurl.com/msw8fajk>`_ of the
        holiday`s country.
    subdiv : str | None
        The `ISO 3166-2 code <https://tinyurl.com/2b432nrx>`_ code of the
        country`s subdivision, if applicable.

    """

    # Country the holiday stems from
    country: str
    # Desired Subdivison if any
    subdiv: Optional[str] = None

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the periodic event regressor to a JSON-serializable
        dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields including an extra
            ``regressor_type = "Holiday"`` item.
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Convert additional fields
        regressor_dict["country"] = self.country
        regressor_dict["subdiv"] = self.subdiv
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates an Holiday object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the regressor.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        PeriodicEvent
            PeriodicEvent regressor instance with fields from
            ``regressor_dict``
        """
        # Ensure that regressor dictionary contains all required fields.
        cls.check_for_missing_keys(regressor_dict)
        # Convert non-built-in fields
        regressor_dict["profile"] = Profile.from_dict(
            regressor_dict["profile"]
        )
        return cls(**regressor_dict)

    def get_t_list(self: Self, t: pd.Series) -> list[pd.Timestamp]:
        """
        Yields a list of timestamps of holiday occurrences within the range of
        input timestamps.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A pandas series of :class:`pandas.Timestamp`.

        Returns
        -------
        t_list : list[:class:`pandas.Timestamp`]
            A list of timestamps of holiday occurrences.

        """
        # A temporary timestamp name
        t_name = "dummy"

        # Create a DataFrame with all holidays in the desired timerange
        holiday_df = make_holiday_dataframe(
            timestamps=t,
            country=self.country,
            subdiv=self.subdiv,
            timestamp_name=t_name,
        )

        # Filter for the desired holiday saved in self.name
        t_list = (
            holiday_df[t_name].loc[holiday_df[_HOLIDAY] == self.name].to_list()
        )
        return t_list

    def get_impact(self: Self, t: pd.Series) -> float:
        """
        Calculate fraction of overall profiles occurring within a timerange.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp`.

        Returns
        -------
        impact : float
            Fraction of overall profiles occurring between minimum and maximum
            date of ``t``.

        """
        # Set list of all occurences of desired holiday
        t_list = self.get_t_list(t)
        # In case no profile is in the list, return zero to signal that no
        # profile will be fitted
        if len(t_list) == 0:
            return 0.0
        # Count instances in t_list that are within the timestamp range
        impact = sum(float(t.min() <= t0 <= t.max()) for t0 in t_list)
        impact /= len(t_list)
        return impact

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Create the feature matrix for the holiday regressor.

        Parameters
        ----------
        t : :class:`pandas.Series`
            A series of :class:`pandas.Timestamp` at which the regressor has to
            be evaluated
        regressor : :class:`pandas.Series`
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for
            :class:`ExternalRegressor`. Any input will be ignored for
            :class:`Holiday`.

        Returns
        -------
        X : :class:`pandas.DataFrame`
            The feature matrix containing the data of the regressor.
        prior_scales : dict
            A map for ``feature matrix column name`` → ``prior_scale``.
        """

        # Set list of all occurences of desired holiday
        self.t_list = self.get_t_list(t)

        # Once we have the list, the IntermittentEvent.make_feature() method
        # will take care of the rest.
        return super().make_feature(t)


class CalendricData(Protocol):
    """
    Manage calendar-driven seasonal cycles and public-holiday effects for a
    :class:`Gloria` forecaster.

    The protocol contributes:

    * **Seasonalities** - yearly, quarterly, monthly, weekly, and daily terms.
    * **Holidays** - :class:`Holiday` event regressors for every public
      holiday in ``country`` and (optionally) ``subdiv``.


    Parameters
    ----------
    country : str | None
        Two-letter ISO
        `ISO 3166-1 alpha-2 code <https://tinyurl.com/msw8fajk>`_ of the
        country (e.g. ``"US"``, ``"DE"``). If ``None`` (default), no holiday
        regressors are created.
    subdiv : str | None
        An optional `ISO 3166-2 subdivion code <https://tinyurl.com/2b432nrx>`_
        (e.g. state, province etc.). If ``None``, only nationwide holidays are
        considered.
    holiday_prior_scale : float | None
        Parameter modulating the strength of all holiday regressors. Larger
        values allow the model to fit larger holiday impact, smaller values
        dampen the impact. Must be larger than 0. If ``None`` (default),
        the forecaster's ``event_prior_scale`` is used.
    holiday_profile : :ref:`Profile <ref-profiles>`
        Profile object that defines the temporal shape of each holiday
        regressor. The default is a one-day :class:`BoxCar` profile replicating
        Prophet-style holiday regressors.
    seasonality_prior_scale : float | None
        Global strength parameter for every seasonality added by the protocol.
        Larger values permit stronger seasonal variation, smaller values dampen
        it. Must be larger than 0.If ``None`` (default), the forecaster's
        ``seasonality_prior_scale`` is used.
    yearly_seasonality, quarterly_seasonality, monthly_seasonality,\
    weekly_seasonality, daily_seasonality : bool | int | "auto"
        Configures how to add the respective seasonality to the model. Details
        see below.


    .. rubric:: Seasonality Options

    The behaviour of the seasonal components is controlled by the
    ``yearly_seasonality``, ``quarterly_seasonality``, ``monthly_seasonality``,
    ``weekly_seasonality``, and ``daily_seasonality`` parameters. Valid values
    are:

    * ``True``: add the seasonality with the default maximum Fourier order
      (see table below).
    * ``False``: do **not** add the seasonality.
    * ``"auto"``: add the seasonality if the data span at least two full
      cycles. Choose the smaller of the default order and the highest order
      allowed by the `Nyquist theorem <https://tinyurl.com/425tj4wb>`_ as
      maximum order.
    * ``integer >= 1``: add the seasonality with that integer as the maximum
      order.

    .. rubric:: Default Maximum Orders

    +-----------+------------+-------------------+
    | **Name**  | **Period** | **Default Order** |
    +===========+============+===================+
    | yearly    | 365.25 d   | 10                |
    +-----------+------------+-------------------+
    | quarterly | 91.31 d    | 2                 |
    +-----------+------------+-------------------+
    | monthly   | 30.44 d    | 3                 |
    +-----------+------------+-------------------+
    | weekly    | 7 d        | 3                 |
    +-----------+------------+-------------------+
    | daily     | 1 d        | 4                 |
    +-----------+------------+-------------------+


    .. admonition:: Note on Quarterly Seasonality
       :class: caution

       The quarterly component is a strict subset of the yearly component.
       It is therefore automatically disabled if the yearly seasonality is
       enabled, overriding the setting of ``quarterly_seasonality``.

    """

    country: Optional[str] = None
    subdiv: Optional[str] = None
    holiday_prior_scale: Optional[float] = Field(gt=0, default=None)
    holiday_profile: Profile = BoxCar(width=pd.Timedelta("1d"))
    seasonality_prior_scale: Optional[float] = Field(gt=0, default=None)
    yearly_seasonality: Union[bool, str, int] = "auto"
    quarterly_seasonality: Union[bool, str, int] = False
    monthly_seasonality: Union[bool, str, int] = False
    weekly_seasonality: Union[bool, str, int] = "auto"
    daily_seasonality: Union[bool, str, int] = "auto"

    @field_validator("holiday_profile", mode="before")
    @classmethod
    def validate_holiday_profile(
        cls: Type[Self], holiday_profile: Union[Profile, dict[str, Any]]
    ) -> Profile:
        """
        In case the input profile was given as a dictionary this
        before-validator attempts to convert it to an Profile.
        """
        try:
            if isinstance(holiday_profile, dict):
                return Profile.from_dict(holiday_profile)
        except Exception as e:
            raise ValueError(
                f"Creating profile from dictionary failed: {e}"
            ) from e
        return holiday_profile

    @field_validator(
        *(s + "_seasonality" for s in DEFAULT_SEASONALITIES.keys())
    )
    @classmethod
    def validate_seasonality_arg(
        cls: Type[Self], arg: Union[bool, str, int]
    ) -> Union[bool, str, int]:
        """
        Validates the xy_seasonality arguments, which must be 'auto', boolean,
        or an integer >=1.
        """
        if isinstance(arg, str) and arg == "auto":
            return arg
        if isinstance(arg, bool):
            return arg
        if isinstance(arg, int) and arg >= 1:
            return arg
        raise ValueError("Must be 'auto', a boolean, or an integer >= 0.")

    def set_events(
        self: Self, model: "Gloria", timestamps: pd.Series
    ) -> "Gloria":
        """
        Adds all holidays for specified country and subdivision to the Gloria
        object.

        Only holidays whose dates fall within the span covered by
        ``timestamps`` are added; all others are ignored.

        .. note::
          You may call :meth:`set_events` directly to add the holidays.
          When the protocol is registered via :meth:`Gloria.add_protocol`,
          however, it is invoked automatically during
          :meth:`Gloria.fit`, so an explicit call is rarely required.

        Parameters
        ----------
        model : :class:`Gloria`
            The Gloria model to be updated
        timestamps : :class:`pandas.Series`
            A Series of :class:`pandas.Timestamp`. Only holidays within the
            range set by ``timestamps`` will be added to the model.

        Returns
        -------
        :class:`Gloria`
            The updated Gloria model.

        """
        # If holiday parameters were not set for the protocol, take them from
        # the Gloria model
        ps = self.holiday_prior_scale
        ps = model.event_prior_scale if ps is None else ps

        if self.country is not None:
            # Get all holidays that occur in the range of timestamps
            holiday_df = make_holiday_dataframe(
                timestamps=timestamps, country=self.country, subdiv=self.subdiv
            )
            # Extract unique holiday names
            holiday_names = set(holiday_df[_HOLIDAY].unique())
            # Add all holidays
            for holiday in holiday_names:
                if holiday in model.events:
                    get_logger().info(
                        f"Skipping calendric protocol holiday '{holiday}' as "
                        "as it was added to the model before."
                    )
                    continue
                model.add_event(
                    name=holiday,
                    prior_scale=ps,
                    regressor_type="Holiday",
                    profile=self.holiday_profile,
                    country=self.country,
                    subdiv=self.subdiv,
                )

        return model

    def set_seasonalities(
        self: Self, model: "Gloria", timestamps: pd.Series
    ) -> "Gloria":
        """
        Adds yearly, quarterly, monthly, weekly, daily seasonalities to the
        Gloria object.

        The ruleset whether and how to add each seasonality is described in the
        :class:`CalendricData` constructor in detail.

        .. note::
          You may call :meth:`set_seasonalities` directly to add the features.
          When the protocol is registered via :meth:`Gloria.add_protocol`,
          however, it is invoked automatically during
          :meth:`Gloria.fit`, so an explicit call is rarely required.

        Parameters
        ----------
        model : :class:`Gloria`
            The Gloria model to be updated
        timestamps : :class:`pandas.Series`
            A Series of :class:`pandas.Timestamp`.

        Returns
        -------
        :class:`Gloria`
            The updated Gloria model.

        """
        # If seasonality parameters were not set for the protocol, take them
        # from the Gloria model
        ps = self.seasonality_prior_scale
        ps = model.seasonality_prior_scale if ps is None else ps

        # The q'th fraction of the data has a sampling period below or equal
        # to the inferred period. Distinguishing between the inferred period
        # and the Gloria model's sampling period helps to ensure that the data
        # are sufficiently fine-grained to fulfill Nyquist theorem
        inferred_sampling_period = infer_sampling_period(timestamps, q=0.3)
        # The timespan covered by the data
        timespan = (
            timestamps.max() - timestamps.min() + inferred_sampling_period
        )

        # Add quarterly only if yearly won't be added, as quarterly is a subset
        # of yearly. Therefore we have to check a number of conditions
        skip_quarterly = (
            # If yearly is simply turned on
            (self.yearly_seasonality is True)
            # In 'auto' mode yearly will be turned on, if the data span 2 years
            or (
                self.yearly_seasonality == "auto"
                and timespan / DEFAULT_SEASONALITIES["yearly"]["period"] >= 2
            )
            # If a maximum yearly order was provided, it only interferes with
            # quarterly if it was larger than 3
            or (
                isinstance(self.yearly_seasonality, int)
                and self.yearly_seasonality > 3
            )
        )

        # Add the seasonalities to the model
        for season, prop in DEFAULT_SEASONALITIES.items():
            if season in model.seasonalities:
                get_logger().info(
                    f"Skipping calendric protocol seasonality '{season}' as it"
                    " was added to the model before."
                )
                continue
            period_loc = cast(pd.Timedelta, prop["period"])
            default_order_loc = cast(int, prop["default_order"])
            # If yearly interferes with quarterly turn quarterly off by default
            if (season == "quarterly") and skip_quarterly:
                get_logger().info(
                    "Quarterly seasonality will not be added to "
                    "Gloria model due to interference with "
                    "yearly seasonality."
                )
                continue

            # Now differentiate the cases of the current's season add_mode
            add_mode = self.__dict__[season + "_seasonality"]

            if add_mode is True:
                fourier_order = default_order_loc
            elif add_mode is False:
                continue
            elif add_mode == "auto":
                # If the data don't accomodate two full cycles of the season's
                # fundamental period, don't add it at all and move on
                if timespan / period_loc < 2:
                    get_logger().info(
                        f"Disabling {season} season. Configure "
                        f"protocol with {season}_seasonality = "
                        "True to overwrite this."
                    )
                    continue
                # Maximum order fulfilling Nyquist sampling condition
                max_order = int(
                    np.floor(period_loc / (2 * inferred_sampling_period))
                )
                # add orders up to default_order but no higher than max_order
                fourier_order = min(default_order_loc, max_order)
                # fourier_order == 0 occurs if not even the fundamental period
                # fulfills Nyquist. In this case we skip the season alltogether
                if fourier_order == 0:
                    get_logger().info(
                        f"Disabling {season} season. Configure "
                        f"protocol with {season}_seasonality = "
                        "True to overwrite this."
                    )
                    continue
            else:
                # If none of the cases applied, add_mode can only be an integer
                # equaling the maximum fourier order directly requested by the
                # user
                fourier_order = add_mode

            model.add_seasonality(
                name=season,
                period=str(period_loc),
                fourier_order=fourier_order,
                prior_scale=ps,
            )

        return model

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the calendric data protocol to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all protocol fields.

        """
        protocol_dict = {
            # Base class adds the protocol_type
            **super().to_dict(),
            # Pydantic converts fields with built-in data types
            **self.model_dump(),
            # The holiday profile is a non-standard type
            "holiday_profile": self.holiday_profile.to_dict(),
        }
        return protocol_dict

    @classmethod
    def from_dict(cls: Type[Self], protocol_dict: dict[str, Any]) -> Self:
        """
        Creates CalendricData protocol from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the protocol.

        Parameters
        ----------
        protocol_dict : dict[str, Any]
            Dictionary containing all protocol fields

        Returns
        -------
        :class:`CalendricData`
            CalendricData protocol object with fields from ``protocol_dict``
        """
        # Ensure that protocol dictionary contains all required fields.
        cls.check_for_missing_keys(protocol_dict)
        # Create and return the CalendricData instance
        return cls(**protocol_dict)
