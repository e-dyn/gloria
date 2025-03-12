"""
Definition of the protocol for handling calendric data.

TODO:
    - Cache the results of make_holiday_dataframe as it is called once
      for each Holiday regressor with equal input parameters
    - Add case handling if a holiday is not part of timestamps, but the
      associated event is so broad that it laps into the data
    - Give possibility to pass a list of (country, subdiv) pairs to
      CalendricData.
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
from gloria.constants import _HOLIDAY
from gloria.events import BoxCar, Event
from gloria.protocols.protocol_base import Protocol
from gloria.regressors import IntermittentEvent
from gloria.types import RegressorMode
from gloria.utilities import get_logger, infer_sampling_period

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
    Returns all holidays for the specified country within the timerange of
    interest as HolidayBase object.

    Parameters
    ----------
    country : str
        Must be a valid ISO 3166-1 alpha-2 country code
    subdiv : str, optional
        The subdivision (e.g. state or province) as a ISO 3166-2 code or its
        alias
    timestamps : pd.Series, optional
        A pandas series of timestamps representing the range for which holidays
        should be returned. If None (default), all holidays registered with the
        holidays package are returned.

    Raises
    ------
    AttributeError
        In case there is no class for the given input country

    Returns
    -------
    all_holidays : holidays.HolidayBase
        A dictionary-like class containing all holidays of the input country
    all_holiday_names : set[str]
        A set of all holiday names in the desired range
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
    Returns all holidays for the specified country within the timerange of
    interest as pandas DataFrame.

    Parameters
    ----------
    timestamps : pd.Series
        A pandas series of timestamps representing the range for which holidays
        should be returned.
    country : str
        Must be a valid ISO 3166-1 alpha-2 country code
    subdiv : str, optional
        The subdivision (e.g. state or province) as a ISO 3166-2 code or its
        alias
    timestamp_name : str, optional
        Desired name for the timestamp column. The default is 'ds'.

    Returns
    -------
    holiday_df : TYPE
        Pandas DataFrame with timestamp column and holiday name column.
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
    An EventRegressor that produces recurring events coinciding with a holiday.

    Note that the name-field of the regressor must equal the desired holiday.
    """

    # Country the holiday stems from
    country: str
    # Desired Subdivison if any
    subdiv: Optional[str] = None

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Holiday regressor to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all regressor fields
        """
        # Parent class converts basic fields and base event
        regressor_dict = super().to_dict()
        # Remove t_list as it is reevaluated for ever make_feature execution
        regressor_dict.pop("t_list")
        # Convert additional fields
        regressor_dict["country"] = self.country
        regressor_dict["subdiv"] = self.subdiv
        return regressor_dict

    @classmethod
    def from_dict(cls: Type[Self], regressor_dict: dict[str, Any]) -> Self:
        """
        Creates Holiday regressor instance from a dictionary that holds the
        regressor fields.

        Parameters
        ----------
        regressor_dict : dict[str, Any]
            Dictionary containing all regressor fields

        Returns
        -------
        PeriodicEvent
            PeriodicEvent regressor instance with fields from
            regressor_dict
        """
        # Ensure that regressor dictionary contains all required fields.
        cls.check_for_missing_keys(regressor_dict)
        # Convert non-built-in fields
        regressor_dict["event"] = Event.from_dict(regressor_dict["event"])
        return cls(**regressor_dict)

    def make_feature(
        self: Self, t: pd.Series, regressor: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Create the feature matrix along with prior scales and modes for a given
        timestamp series.

        Parameters
        ----------
        t : pd.Series
            A pandas series of timestamps at which the regressor has to be
            evaluated
        regressor : pd.Series
            Contains the values for the regressor that will be added to the
            feature matrix unchanged. Only has effect for ExternalRegressor


        Returns
        -------
        X : pd.DataFrame
            Contains the feature matrix
        prior_scales : dict
            A map for 'feature matrix column name' -> 'prior_scale'
        modes : dict
            A map for 'feature matrix column name' -> 'mode'
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
        self.t_list = (
            holiday_df[t_name].loc[holiday_df[_HOLIDAY] == self.name].to_list()
        )

        # Once we have the list, the IntermittentEvent.make_feature() method
        # will take care of the rest.
        return super().make_feature(t)


class CalendricData(Protocol):
    """
    Protocol to add features to a Gloria model that follow calendric patterns.

    Seasonalities:
        Yearly, Quarterly, Monthly, Weekly, and Daily seasonalities will be
        added
    Holidays:
        All holidays for a given country and (optional) subdivision will be
        added

    Details are described in set_seasonalities() and set_events()
    """

    country: Optional[str] = None
    subdiv: Optional[str] = None
    holiday_mode: Optional[RegressorMode] = cast(RegressorMode, None)
    holiday_prior_scale: Optional[float] = Field(gt=0, default=None)
    holiday_event: Event = BoxCar(duration=pd.Timedelta("1d"))
    seasonality_mode: Optional[RegressorMode] = cast(RegressorMode, None)
    seasonality_prior_scale: Optional[float] = Field(gt=0, default=None)
    yearly_seasonality: Union[bool, str, int] = "auto"
    quarterly_seasonality: Union[bool, str, int] = False
    monthly_seasonality: Union[bool, str, int] = False
    weekly_seasonality: Union[bool, str, int] = "auto"
    daily_seasonality: Union[bool, str, int] = "auto"

    @field_validator("holiday_event", mode="before")
    @classmethod
    def validate_holiday_event(
        cls: Type[Self], holiday_event: Union[Event, dict[str, Any]]
    ) -> Event:
        """
        In case the input event was given as a dictionary this before-validator
        attempts to convert it to an Event.
        """
        try:
            if isinstance(holiday_event, dict):
                return Event.from_dict(holiday_event)
        except Exception as e:
            raise ValueError("Creating event from dictionary failed.") from e
        return holiday_event

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
        model.

        The method first checks which of the country-holidays can be found in
        the time period specified by timestamps and adds only those.

        Parameters
        ----------
        model : Gloria
            The Gloria model to be updated
        timestamps : pd.Series
            A pandas series of timestamps.

        Returns
        -------
        Gloria
            The updated Gloria model.

        """
        # If holiday parameters were not set for the protocol, take them from
        # the Gloria model
        ps = self.holiday_prior_scale
        ps = model.event_prior_scale if ps is None else ps
        mode = self.holiday_mode
        mode = model.event_mode if mode is None else mode

        if self.country is not None:
            # Get all holidays that occur in the range of timestamps
            holiday_df = make_holiday_dataframe(
                timestamps=timestamps, country=self.country, subdiv=self.subdiv
            )
            # Extract unique holiday names
            holiday_names = set(holiday_df[_HOLIDAY].unique())
            # Add all holidays
            for holiday in holiday_names:
                model.add_event(
                    name=holiday,
                    prior_scale=ps,
                    mode=mode,
                    regressor_type="Holiday",
                    event=self.holiday_event,
                    country=self.country,
                    subdiv=self.subdiv,
                )

        return model

    def set_seasonalities(
        self: Self, model: "Gloria", timestamps: pd.Series
    ) -> "Gloria":
        """
        Adds yearly, quarterly, monthly, weekly, daily seasonalities to the
        Gloria model.

        The seasonalities have the following fundamental periods and default
        maximum orders:

        Name       Period   Default Order
        ---------------------------------
        yearly     365.25d       10
        quarterly  91.31d         2
        monthly    30.44d         3
        weekly     7d             3
        daily      1d             4


        Whether and how the seasonalities are added depends on the mode
        specified in the <name>_seasonality field:
            - True: The seasonality is added using the default maximum order
            - False: the seasonality won't be added
            - 'auto': The seasonality will be added according to the rule set
              described below
            - integer >= 1: the seasonality will be added with the integer used
              as maximum order

        In 'auto' mode the logic is as follows
            2. Each seasonality is only added, if the data span at least two
               full cycles of the fundamental period.
            3. The maximum order is determined by the default maximum order or
               the highest order that satisfies the Nyquist sampling theorem,
               whichever is smaller.

        Note on quarterly seasonality:
        Quarterly is a full subset of yearly. Therefore it will only be added
        if yearly won't be added to the model. This rule is independent of the
        mode in which quarterly is added.

        Parameters
        ----------
        model : Gloria
            The Gloria model to be updated
        timestamps : pd.Series
            A pandas series of timestamps.

        Returns
        -------
        Gloria
            The updated Gloria model.

        """
        # If seasonality parameters were not set for the protocol, take them
        # from the Gloria model
        ps = self.seasonality_prior_scale
        ps = model.seasonality_prior_scale if ps is None else ps
        mode = self.seasonality_mode
        mode = model.seasonality_mode if mode is None else mode

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
                mode=mode,
            )

        return model

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the CalendricData protocol to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all protocol fields

        """
        protocol_dict = {
            # Base class adds the protocol_type
            **super().to_dict(),
            # Pydantic converts fields with built-in data types
            **self.model_dump(),
            # The holiday event is a non-standard type
            "holiday_event": self.holiday_event.to_dict(),
        }
        return protocol_dict

    @classmethod
    def from_dict(cls: Type[Self], protocol_dict: dict[str, Any]) -> Self:
        """
        Creates CalendricData protocol from a dictionary that holds the
        protocol fields.

        Parameters
        ----------
        protocol_dict : dict[str, Any]
            Dictionary containing all protocol fields

        Returns
        -------
        CalendricData
            CalendricData protocol instance with fields from protocol_dict
        """
        # Ensure that protocol dictionary contains all required fields.
        cls.check_for_missing_keys(protocol_dict)
        # Create and return the CalendricData instance
        return cls(**protocol_dict)
