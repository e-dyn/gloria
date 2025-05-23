"""
Utilities for evaluating prophet models and tuning hyperparameters
"""

# Third Party
### --- Module Imports --- ###
import pandas as pd

# Gloria
from gloria.interface import Gloria
from gloria.models import get_model_backend


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


def gloria_copy(m: Gloria) -> tuple[pd.DataFrame, Gloria]:
    """
    Creates a deep copy of a fitted Gloria model but resets the backend.

    Parameters
    ----------
    m : Gloria
        The fitted Gloria model.

    Returns
    -------
    m_copy : TYPE
        The copied and reset Gloria model
    data : TYPE
        Historic data extracted from the fitted input model.

    """
    # Copy the fitted model completely
    m_copy = m.model_copy(deep=True)
    # But reset model backend so it can be fit once more
    m_copy.model_backend = get_model_backend(model=m_copy.model)
    # Also erase protocols as they have been executed already
    m_copy.protocols = []
    # Extract historic data
    data = m_copy.history

    return m_copy, data
