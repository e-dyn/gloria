"""
This module contains classes to manage Gloria configurations as well as their
serialization and deserialization
"""

### --- Module Imports --- ###
# Standard Library
import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Literal,
    Optional,
    Type,
    Union,
)

# Third Party
import pandas as pd
import tomli
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import Self

# Conditional import of Gloria for static type checking. Otherwise Gloria is
# forward-declared as 'Gloria' to avoid circular imports
if TYPE_CHECKING:
    from gloria.interface import Gloria

# Gloria
from gloria.models import MODEL_MAP, BinomialPopulation
from gloria.protocols.protocol_base import get_protocol_map
from gloria.utilities.constants import _BACKEND_DEFAULTS, _GLORIA_DEFAULTS
from gloria.utilities.logging import get_logger
from gloria.utilities.types import Distribution, DTypeKind


### --- Class and Function Definitions --- ###
def model_from_toml(
    toml_path: Union[str, Path],
    ignore: Union[Collection[str], str] = set(),
    **kwargs: dict[str, Any],
) -> "Gloria":
    """
    Instantiate a Gloria model from a TOML configuration file and augment it
    with optional external regressors, seasonalities, events, and protocols.

    The TOML file is expected to have the following top-level tables /
    arrays-of-tables (all are optional except [model]):

    * [model] - keyword arguments passed directly to Gloria`s constructor.
    * [[external_regressors]] - one table per regressor; each is forwarded to
    Gloria.add_external_regressor().
    * [[seasonalities]] - one table per seasonality; each is
    forwarded to Gloria.add_seasonality().
    * [[events]] - one table per event; each is forwarded to
    Gloria.add_event().
    * [[protocols]] - one table per protocol. Each table **must** contain a
    ``type`` key that maps to a protocol class name; the remaining keys are
    passed to that class before calling `Gloria.add_protocol`.

    Parameters
    ----------
    toml_path : Union[str, Path]
        Path to the TOML file containing the model specification.
    ignore : Union[Sequence[str],str], optional
        Which top-level sections of the file to skip. Valid values are
        "external_regressors", "seasonalities", "events", and`"protocols". The
        special value "all" suppresses every optional section. May be given as
        a single string or any iterable of strings.
    **kwargs : dict[str, Any]
        Keyword arguments that override or extend the [model] table. Only keys
        that are valid fields of Gloria (i.e. that appear in
        Gloria.model_fields) are retained; others are silently dropped.

    Returns
    -------
    Gloria
        A fully initialised Gloria instance.

    Notes
    -----
    Precedence order for constructor arguments is:

    1. Values supplied via **kwargs
    2. Values found in the TOML [model] table
    3. Gloria`s own defaults
    """
    # Gloria
    from gloria.interface import Gloria

    # Remove keys from kwargs that are no valid Gloria fields
    kwargs = {k: v for k, v in kwargs.items() if k in Gloria.model_fields}

    # Make sure ignore is a set
    if isinstance(ignore, str):
        ignore = {ignore}
    else:
        ignore = set(ignore)

    # Extend set by all possible attributes if 'all' in ignore
    if "all" in ignore:
        ignore = set(ignore) | {
            "external_regressors",
            "seasonalities",
            "events",
            "protocols",
        }

    # Load configuration file
    with open(toml_path, mode="rb") as file:
        config = tomli.load(file)

    # Give precedence to individial settings in kwargs
    model_config = config["model"] | kwargs

    # Create Gloria model
    m = Gloria(**model_config)

    # Add external regressors
    if "external_regressors" not in ignore:
        for er in config.get("external_regressors", []):
            m.add_external_regressor(**er)

    # Add seasonalities
    if "seasonalities" not in ignore:
        for season in config.get("seasonalities", []):
            m.add_seasonality(**season)

    # Add events
    if "events" not in ignore:
        for event in config.get("events", []):
            # Create and add the protocol with the remaining configurations
            m.add_event(**event)

    # Add protocols
    if "protocols" not in ignore:
        for protocol in config.get("protocols", []):
            # Get protocol class using the 'type' key in of the protocol config
            ProtocolClass = get_protocol_map()[protocol.pop("type")]
            # Create and add the protocol with the remaining configurations
            m.add_protocol(ProtocolClass(**protocol))

    # Save fit and predict tables for later use
    m._config = {k: v for k, v in config.items() if k in ("fit", "predict")}

    return m


class DataConfig(BaseModel):
    """
    Configuration of the input data
    """

    data_source: str
    sampling_period: str
    timestamp_name: str


class MetricConfig(BaseModel):
    """
    Configuration of the Metric column of the input data
    """

    metric_name: str
    population_name: Optional[str] = None
    model: Distribution
    dtype_kind: DTypeKind
    augmentation_config: Optional[BinomialPopulation] = None

    @field_validator("dtype_kind")
    @classmethod
    def validate_model_kind(cls, dtype_kind: str, info: ValidationInfo) -> str:
        """
        Validates that the specified dtype_kind matches any of the allowed type
        specific to the model.
        """
        allowed_types = list(MODEL_MAP[info.data["model"]].kind)
        if dtype_kind not in allowed_types:
            type_list = ", ".join([f"'{s}'" for s in allowed_types])
            raise TypeError(
                f"dtype_kind was set to '{dtype_kind}', but must be any of"
                f" {type_list} for selected model '{info.data['model']}'."
            )
        return dtype_kind


class GloriaConfig(BaseModel):
    """
    Configuration of the Gloria model
    """

    n_changepoints: int = Field(
        ge=0, default=_GLORIA_DEFAULTS["n_changepoints"]
    )
    changepoints: Optional[list[str]] = _GLORIA_DEFAULTS["changepoints"]
    changepoint_range: float = Field(
        gt=0, lt=1, default=_GLORIA_DEFAULTS["changepoint_range"]
    )
    seasonality_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["seasonality_prior_scale"]
    )
    event_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["event_prior_scale"]
    )
    changepoint_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["changepoint_prior_scale"]
    )
    interval_width: float = Field(
        gt=0, lt=1, default=_GLORIA_DEFAULTS["interval_width"]
    )
    uncertainty_samples: int = Field(
        ge=0, default=_GLORIA_DEFAULTS["uncertainty_samples"]
    )
    optimize_mode: Literal["MAP", "MLE"] = _BACKEND_DEFAULTS["optimize_mode"]
    sample: bool = _BACKEND_DEFAULTS["sample"]

    @field_validator("changepoints")
    @classmethod
    def validate_changepoints(
        cls: Type[Self], changepoints: Optional[list[str]]
    ) -> pd.Series:
        """
        Converts sampling period to a pandas Timedelta if it was passed as a
        string instead.
        """
        # Third Party
        from pandas._libs.tslibs.parsing import DateParseError

        try:
            # pd.to_datetime returns None if changepoints were None
            changepoints = pd.to_datetime(pd.Series(changepoints))
        except (ValueError, DateParseError) as e:
            msg = "Could not parse input changepoints."
            get_logger().error(msg)
            raise ValueError(msg) from e

        return changepoints


class RunConfig(BaseModel):
    """
    Overall Configuration class
    """

    data_config: DataConfig
    metric_config: MetricConfig
    gloria_config: GloriaConfig

    def to_json(self, path: Path):
        config_dict = self.model_dump()
        with open(path, "w") as file:
            json.dump(config_dict, file, indent=4)

    @classmethod
    def load_json(cls, path: Path):
        with open(path, "r") as file:
            data = json.load(file)
        return cls(**data)
