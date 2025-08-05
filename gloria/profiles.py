# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Definition of Profile base class and its implementations
"""

### --- Module Imports --- ###
# Standard Library
from abc import ABC, abstractmethod
from typing import Any, Type

# Third Party
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)
from typing_extensions import Self

# Gloria
### --- Global Constants Definitions --- ###
from gloria.utilities.types import Timedelta


### --- Class and Function Definitions --- ###
class Profile(BaseModel, ABC):
    """
    Abstract base class for all profiles
    """

    model_config = ConfigDict(
        # All profiles will use some sort of pd.Timestamp or pd.Timedelta
        arbitrary_types_allowed=True,
    )

    @property
    def _profile_type(self: Self):
        """
        Returns name of the profile class.
        """
        return type(self).__name__

    @abstractmethod
    def generate(
        self: Self, timestamps: pd.Series, t_anchor: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single instance of the profile.

        Parameters
        ----------
        timestamps : :class:`pandas.Series`
            The input timestamps as independent variable
        t_anchor : :class:`pandas.Timestamp`
            Location of the profile

        Raises
        ------
        NotImplementedError
            In case the inheriting Profile class did not implement the generate
            method

        Returns
        -------
        :class:`pandas.Series`
            The output time series including the profile.
        """
        pass

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the profile to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the profile type. All other profile fields
            will be added by profile child classes.
        """
        profile_dict = {"profile_type": self._profile_type}
        return profile_dict

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[Self], profile_dict: dict[str, Any]) -> Self:
        """
        Forward declaration of class method for static type checking.
        See details in profile_from_dict().
        """
        pass

    @classmethod
    def check_for_missing_keys(
        cls: Type[Self], profile_dict: dict[str, Any]
    ) -> None:
        """
        Confirms that all required fields for the requested profile type are
        found in the profile dictionary.

        Parameters
        ----------
        profile_dict : dict[str, Any]
            Dictionary containing all profile fields

        Raises
        ------
        KeyError
            Raised if any keys are missing

        Returns
        -------
        None
        """
        # Use sets to find the difference between profile model fields and
        # passed dictionary keys
        required_fields = {
            name
            for name, info in cls.model_fields.items()
            if info.is_required()
        }
        missing_keys = required_fields - set(profile_dict.keys())
        # If any is missing, raise an error.
        if missing_keys:
            missing_keys_str = ", ".join([f"'{key}'" for key in missing_keys])
            raise KeyError(
                f"Key(s) {missing_keys_str} required for profile of type "
                f"'{cls.__name__}' but not found in profile dictionary."
            )


class BoxCar(Profile):
    """
    A BoxCar shaped profile.

    For a given time :math:`t` the profile can be described by

    .. math::
        f(t) = \\left\\{
            \\begin{array}{ll}
                1 & t_0 \\le t < t_0 + w \\\\
                0 & \\text{otherwise}
            \\end{array}
        \\right.

    with ``width=w`` being a constructor parameter and ``t_anchor=t_0`` the
    input of :meth:`~gloria.BoxCar.generate`. The following plot illustrates
    the boxcar function.

    .. image:: ../pics/example_boxcar.png
      :align: center
      :width: 500
      :alt: Example plot of a boxcar function.

    .. note::
      Setting the boxcar profile's ``width`` equal to the :class:`Gloria`
      model's``sampling_period`` yields a :math:`\\delta`-shaped regressor -
      identical to the holiday regressors used by
      `Prophet <https://facebook.github.io/prophet/>`_.

    Parameters
    ----------
    width : :class:`pandas.Timedelta` | str
        Temporal width of the boxcar function given as
        :class:`pandas.Timedelta` or string representing such.
    """

    width: Timedelta

    def generate(
        self: Self, timestamps: pd.Series, t_anchor: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single boxcar profile.

        Parameters
        ----------
        timestamps : :class:`pandas.Series`
            The input timestamps at which the boxcar profile is to be
            evaluated.
        t_anchor : :class:`pandas.Timestamp`
            Location of the boxcar's rising edge

        Returns
        -------
        :class:`pandas.Series`
            The output time series including the boxcar profile with amplitude
            1.
        """
        mask = (timestamps >= t_anchor) & (timestamps < t_anchor + self.width)
        return mask * 1

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the BoxCar profile to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all profile fields including an extra
            ``profile_type = "BoxCar"`` item.
        """
        # Start with profile type
        profile_dict = super().to_dict()
        # Add additional fields
        profile_dict["width"] = str(self.width)
        return profile_dict

    @classmethod
    def from_dict(cls: Type[Self], profile_dict: dict[str, Any]) -> Self:
        """
        Creates a BoxCar object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the profile.

        Parameters
        ----------
        profile_dict : dict[str, Any]
            Dictionary containing all profile fields

        Returns
        -------
        BoxCar
            BoxCar object with fields from ``profile_dict``
        """
        # Convert width string to pd.Timedelta
        profile_dict["width"] = pd.Timedelta(profile_dict["width"])
        return cls(**profile_dict)


class Gaussian(Profile):
    """
    A Gaussian shaped profile with ``order`` parameter for generating flat-top
    Gaussians.

    For a given time :math:`t` the profile can be described by

    .. math::
        f(t) = \\exp\\left(-\\left(
            \\frac{\\left(t-t_0\\right)^2}{2\\sigma^2}
        \\right)^n\\right)


    with ``width=sigma`` and ``order=n`` being constructor parameters as well
    as ``t_anchor=t_0`` the input of :meth:`~gloria.Gaussian.generate`. For
    :math:`n=1` the function is a simple Gaussian and for increasing :math:`n`
    its maximum region increasingly flattens. The following plot illustrates
    the Gaussian function for different :math:`n`.

    .. image:: ../pics/example_gaussian.png
      :align: center
      :width: 500
      :alt: Example plot of a Gaussian function.

    Parameters
    ----------
    width : :class:`pandas.Timedelta` | str
        Temporal width of the Gaussian function given as
        :class:`pandas.Timedelta` or string representing such.
    order : float
        Controls the flatness of the Gaussian function with ``order=1`` being a
        usual Gaussian and a flat-top function for increasing ``order``. Must
        be greater than 0.

    """

    width: Timedelta
    order: float = Field(gt=0, default=1.0)

    def generate(
        self: Self, timestamps: pd.Series, t_anchor: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single Gaussian profile.

        Parameters
        ----------
        timestamps : :class:`pandas.Series`
            The input timestamps at which the Gaussian profile is to be
            evaluated.
        t_anchor : :class:`pandas.Timestamp`
            Location of the Gaussian profile's mode.

        Returns
        -------
        :class:`pandas.Series`
            The output time series including the Gaussian profile with
            amplitude 1.
        """

        # normalize the input timestamps
        t = (timestamps - t_anchor) / self.width
        # Evaluate the Gaussian
        return np.exp(-((0.5 * t**2) ** self.order))

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Gaussian profile to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all profile fields including an extra
            ``profile_type = "Gaussian"`` item.
        """
        # Start with profile type
        profile_dict = super().to_dict()
        # Add additional fields
        profile_dict["width"] = str(self.width)
        profile_dict["order"] = self.order
        return profile_dict

    @classmethod
    def from_dict(cls: Type[Self], profile_dict: dict[str, Any]) -> Self:
        """
        Creates a Gaussian object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the profile.

        Parameters
        ----------
        profile_dict : dict[str, Any]
            Dictionary containing all profile fields

        Returns
        -------
        Gaussian
            Gaussian object with fields from ``profile_dict``
        """

        # Convert width string to pd.Timedelta
        profile_dict["width"] = pd.Timedelta(profile_dict["width"])
        return cls(**profile_dict)


class Cauchy(Profile):
    """
    A Cauchy shaped profile.

    For a given time :math:`t` the profile can be described by

    .. math::
        f(t) = \\frac{1}{4\\cdot \\left(t-t_0 \\right)^2 / w^2 + 1}


    with ``width=w`` being a constructor parameter as well as ``t_anchor=t_0``
    the input of :meth:`~gloria.Cauchy.generate`. The following plot
    illustrates the Cauchy function.

    .. image:: ../pics/example_cauchy.png
      :align: center
      :width: 500
      :alt: Example plot of a Cauchy function.

    Parameters
    ----------
    width : :class:`pandas.Timedelta` | str
        Temporal width of the Cauchy function given as
        :class:`pandas.Timedelta` or string representing such.

    """

    width: Timedelta

    def generate(
        self: Self, timestamps: pd.Series, t_anchor: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single Cauchy profile.

        Parameters
        ----------
        timestamps : :class:`pandas.Series`
            The input timestamps at which the Cauchy profile is to be
            evaluated.
        t_anchor : :class:`pandas.Timestamp`
            Location of the Cauchy profile's mode.

        Returns
        -------
        :class:`pandas.Series`
            The output time series including the Cauchy profile with amplitude
            1.
        """

        # normalize the input timestamps
        t = (timestamps - t_anchor) / self.width
        # Evaluate the Cauchy
        return 1 / (4 * t**2 + 1)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Cauchy profile to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all profile fields including an extra
            ``profile_type = "Cauchy"`` item.
        """
        # Start with profile type
        profile_dict = super().to_dict()
        # Add additional fields
        profile_dict["width"] = str(self.width)
        return profile_dict

    @classmethod
    def from_dict(cls: Type[Self], profile_dict: dict[str, Any]) -> Self:
        """
        Creates a Cauchy object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the profile.

        Parameters
        ----------
        profile_dict : dict[str, Any]
            Dictionary containing all profile fields

        Returns
        -------
        Cauchy
            Cauchy object with fields from ``profile_dict``
        """
        # Convert width string to pd.Timedelta
        profile_dict["width"] = pd.Timedelta(profile_dict["width"])
        return cls(**profile_dict)


class Exponential(Profile):
    """
    A two-sided exponential decay shaped profile.

    For a given time :math:`t` the profile can be described by

    .. math::
        f(t) = \\exp\\left(
            -\\log 2 \\left|\\frac{t-t_0}{w\\left(t\\right)}\\right|
        \\right).

    Here, :math:`w\\left(t\\right) = w_\\text{lead}` is the left-sided
    lead-width for :math:`t<t_0` and :math:`w\\left(t\\right) = w_\\text{lag}`
    is the right-sided lag-width for :math:`t\\ge t_0`, set by ``lead_width``
    and ``lag_width`` in the constructor, respectively. The parameter
    ``t_anchor=t_0`` is an input of :meth:`~gloria.Exponential.generate`. The
    following plot illustrates the two-sided exponential decay function.

    .. image:: ../pics/example_exponential.png
      :align: center
      :width: 500
      :alt: Example plot of a two-sided exponential decay function.

    Parameters
    ----------
    lead_width : :class:`pandas.Timedelta` | str
        Temporal left-sided lead-width of the exponential function given as
        :class:`pandas.Timedelta` or string representing such.
    lag_width : :class:`pandas.Timedelta` | str
        Temporal right-sided lag-width of the exponential function given as
        :class:`pandas.Timedelta` or string representing such.
    """

    # Widths of both exponential decay wings
    lead_width: Timedelta
    lag_width: Timedelta

    @field_validator("lead_width")
    @classmethod
    def validate_lead_width(
        cls: Type[Self], lead_width: Timedelta
    ) -> Timedelta:
        """
        If lead width is below zero, sets to zero and warn user
        """
        if lead_width < Timedelta(0):
            # Gloria
            from gloria.utilities.logging import get_logger

            get_logger().warning(
                "Lead width of exponential decay < 0 interpreted as lag decay."
                " Setting lead_width = 0."
            )
            lead_width = Timedelta(0)
        return lead_width

    @field_validator("lag_width")
    @classmethod
    def validate_lag_width(
        cls: Type[Self],
        lag_width: Timedelta,
        other_fields: ValidationInfo,
    ) -> Timedelta:
        """
        If lag width is below zero, sets to zero and warn user. Also check
        whether lag_width = lag_width = 0 and issue warning.

        :meta private:
        """
        if lag_width < Timedelta(0):
            # Gloria
            from gloria.utilities.logging import get_logger

            get_logger().warning(
                "Lag width of exponential decay profile < 0 interpreted as "
                "lead decay. Setting lag_width = 0."
            )
            lag_width = Timedelta(0)

        if (lag_width == Timedelta(0)) & (
            other_fields.data["lead_width"] == Timedelta(0)
        ):
            # Gloria
            from gloria.utilities.logging import get_logger

            get_logger().warning(
                "Lead and lag width of exponential decay profile = 0 - likely"
                " numerical issues during fitting."
            )

        return lag_width

    def generate(
        self: Self, timestamps: pd.Series, t_anchor: pd.Timestamp
    ) -> pd.Series:
        """
        Generate a time series with a single Exponential profile.

        Parameters
        ----------
        timestamps : :class:`pandas.Series`
            The input timestamps at which the Exponential profile is to be
            evaluated.
        t_anchor : :class:`pandas.Timestamp`
            Location of the Exponential profile's mode.

        Returns
        -------
        :class:`pandas.Series`
            The output time series including the Exponential profile with
            amplitude 1.
        """
        # Shift the input timestamps
        t = timestamps - t_anchor

        mask_lead = timestamps < t_anchor
        mask_lag = timestamps >= t_anchor

        # Create profile and fill with zeros
        y = np.zeros_like(timestamps, dtype=float)

        # Add the one-sided lead exponential
        if self.lead_width > pd.Timedelta(0):
            arg = np.log(2) * np.asarray(t[mask_lead] / self.lead_width)
            y[mask_lead] += np.exp(arg)
        # Add the one-sided lag exponential
        if self.lag_width > pd.Timedelta(0):
            arg = np.log(2) * np.asarray(t[mask_lag] / self.lag_width)
            y[mask_lag] += np.exp(-arg)

        return y

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Exponential profile to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all profile fields including an extra
            ``profile_type = "Exponential"`` item.
        """
        # Start with profile type
        profile_dict = super().to_dict()
        # Add additional fields
        profile_dict["lead_width"] = str(self.lead_width)
        profile_dict["lag_width"] = str(self.lag_width)
        return profile_dict

    @classmethod
    def from_dict(cls: Type[Self], profile_dict: dict[str, Any]) -> Self:
        """
        Creates a Exponential object from a dictionary.

        The key-value pairs of the dictionary must correspond to the
        constructor arguments of the profile.

        Parameters
        ----------
        profile_dict : dict[str, Any]
            Dictionary containing all profile fields

        Returns
        -------
        Exponential
            Exponential object with fields from ``profile_dict``
        """
        # Convert lead_width string to pd.Timedelta
        profile_dict["lead_width"] = pd.Timedelta(profile_dict["lead_width"])
        # Convert lag_width string to pd.Timedelta
        profile_dict["lag_width"] = pd.Timedelta(profile_dict["lag_width"])
        return cls(**profile_dict)


# A map of Profile class names to actual classes
PROFILE_MAP: dict[str, Type[Profile]] = {
    "BoxCar": BoxCar,
    "Gaussian": Gaussian,
    "Exponential": Exponential,
    "Cauchy": Cauchy,
}


def profile_from_dict(
    cls: Type[Profile], profile_dict: dict[str, Any]
) -> Profile:
    """
    Identifies the appropriate profile type calls its from_dict() method.

    Parameters
    ----------
    profile_dict : dict[str, Any]
        Dictionary containing all profile fields including profile type.

    Raises
    ------
    NotImplementedError
        Is raised in case the profile type stored in profile_dict does not
        correspond to any profile class.

    Returns
    -------
    Profile
        The appropriate profile constructed from the profile_dict fields.
    """
    profile_dict = profile_dict.copy()
    # Get the profile type
    if "profile_type" not in profile_dict:
        raise KeyError(
            "The input dictionary must have the key 'profile_type'."
        )
    profile_type = profile_dict.pop("profile_type")
    # Check that the profile type exists
    try:
        profile_class = PROFILE_MAP[profile_type]
    except KeyError as e:
        raise NotImplementedError(
            f"Profile Type '{profile_type}' does not exist."
        ) from e
    # Ensure that profile dictionary contains all required fields.
    profile_class.check_for_missing_keys(profile_dict)
    # Call the from_dict() method of the correct profile
    return profile_class.from_dict(profile_dict)


# Add profile_from_dict() as class method to the Profile base class, so it can
# always called as Profile.from_dict(profile_dict) with any dictionary as long
# as it contains the profile_type field.
Profile.from_dict = classmethod(profile_from_dict)  # type: ignore
