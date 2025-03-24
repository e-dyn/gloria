"""
This module contains classes to manage Gloria configurations as well as their
serialization and deserialization
"""

### --- Module Imports --- ###
# Standard Library
import json
from pathlib import Path
from typing import Literal, Optional, cast

# Third Party
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo

# Gloria
from gloria.models import MODEL_MAP, BinomialPopulation
from gloria.utilities.constants import _BACKEND_DEFAULTS, _GLORIA_DEFAULTS
from gloria.utilities.types import Distribution, DTypeKind, RegressorMode

### --- Class and Function Definitions --- ###


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
    def validate_model_kind(
        cls, dtype_kind: str, info: FieldValidationInfo
    ) -> str:
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
    changepoint_range: float = Field(
        gt=0, lt=1, default=_GLORIA_DEFAULTS["changepoint_range"]
    )
    seasonality_mode: RegressorMode = cast(
        RegressorMode, _GLORIA_DEFAULTS["seasonality_mode"]
    )
    seasonality_prior_scale: float = Field(
        gt=0, default=_GLORIA_DEFAULTS["seasonality_prior_scale"]
    )
    event_mode: RegressorMode = cast(
        RegressorMode, _GLORIA_DEFAULTS["event_mode"]
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
