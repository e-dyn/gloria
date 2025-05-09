"""
Definition of the protocol interface
"""

### --- Module Imports --- ###
# Standard Library
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Type

# Third Party
import pandas as pd
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

# Inhouse Packages
if TYPE_CHECKING:
    # Gloria
    from gloria import Gloria

### --- Global Constants Definitions --- ###


### --- Class and Function Definitions --- ###


class Protocol(ABC, BaseModel):
    """
    Protocols can be added to Gloria models in order to configure them based
    on the type of data that the model is supposed to fit.

    This abstract base class defines the Protocol interface and some basic
    functionalities
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def _protocol_type(self: Self) -> str:
        """
        Returns name of the protocol class.
        """
        return type(self).__name__

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Converts the Protocol to a serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing only the protocol type. Keys corresponding to
            other model fields will be added by the subclasses.
        """

        # Add protocol_type holding the regressor class name.
        protocol_dict = {"protocol_type": self._protocol_type}

        return protocol_dict

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[Self], protocol_dict: dict[str, Any]) -> Self:
        """
        Forward declaration of class method for static type checking.
        See details in protocol_from_dict().
        """
        pass

    @classmethod
    def check_for_missing_keys(
        cls: Type[Self], protocol_dict: dict[str, Any]
    ) -> None:
        """
        Confirms that all required fields for the requested protocol type are
        found in the protocol dictionary.

        Parameters
        ----------
        protocol_dict : dict[str, Any]
            Dictionary containing all protocol fields

        Raises
        ------
        KeyError
            Raised if any keys are missing

        Returns
        -------
        None
        """
        # Use sets to find the difference between protocol model fields and
        # passed dictionary keys
        required_fields = {
            name
            for name, info in cls.model_fields.items()
            if info.is_required()
        }
        missing_keys = required_fields - set(protocol_dict.keys())
        # If any is missing, raise an error.
        if missing_keys:
            missing_keys_str = ", ".join([f"'{key}'" for key in missing_keys])
            raise KeyError(
                f"Key(s) {missing_keys_str} required for protocols"
                f" of type {cls.__name__} but not found in "
                "protocol dictionary."
            )

    @abstractmethod
    def set_seasonalities(
        self, model: "Gloria", timestamps: pd.Series
    ) -> "Gloria":
        """
        Determines valid seasonalities according to protocol and input
        timestamps and adds them to the model.

        Parameters
        ----------
        model : Gloria
            The model the protocol should be applied to.
        timestamps : pd.Series
            A pandas series containing timestamps.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def set_events(self, model: "Gloria", timestamps: pd.Series) -> "Gloria":
        """
        Determines valid events according to protocol and input timestamps and
        adds them to the model.

        Parameters
        ----------
        model : Gloria
            The model the protocol should be applied to.
        timestamps : pd.Series
            A pandas series containing timestamps.

        Returns
        -------
        None
        """
        pass


def get_protocol_map() -> dict[str, Type[Protocol]]:
    """
    Returns a dictionary mapping protocol names as strings to actual classes.
    Creating of this map is encapsulated as function to avoid circular imports
    of the protocol modules and a number of linting errors.

    Returns
    -------
    protocol_map : dict[str, Protocol]
        A map 'protocol name' -> 'protocol class'

    """
    # Before creating the protocol map, import protocols that have been defined
    # in other modules
    # Gloria
    from gloria.protocols.calendric import CalendricData

    # Create the map
    protocol_map: dict[str, Type[Protocol]] = {"CalendricData": CalendricData}
    return protocol_map


def protocol_from_dict(
    cls: Type[Protocol], protocol_dict: dict[str, Any]
) -> Protocol:
    """
    Identifies the appropriate protocol type calls its from_dict() method

    Parameters
    ----------
    protocol_dict : dict[str, Any]
        Dictionary containing all protocol fields including protocol type

    Raises
    ------
    NotImplementedError
        Is raised in case the protocol type stored in protocol_dict does not
        correspond to any protocol class

    Returns
    -------
    Protocol
        The appropriate protocol constructed from the protocol_dict fields.
    """
    PROTOCOL_MAP = get_protocol_map()
    protocol_dict = protocol_dict.copy()
    # Get the protocol type
    if "protocol_type" not in protocol_dict:
        raise KeyError(
            "The input dictionary must have the key" " 'protocol_type'"
        )
    protocol_type = protocol_dict.pop("protocol_type")
    # Check that the protocol type exists
    if protocol_type not in PROTOCOL_MAP:
        raise NotImplementedError(
            f"Protocol Type {protocol_type} does not" " exist."
        )
    # Call the from_dict() method of the correct regressor
    return PROTOCOL_MAP[protocol_type].from_dict(protocol_dict)


# Add protocol_from_dict() as class method to the Protocol base class, so
# it can always called as Protocol.from_dict(protocol_dict) with any
# dictionary as long as it contains the protocol_type field.
Protocol.from_dict = classmethod(protocol_from_dict)  # type: ignore
