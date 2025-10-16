# Standard Library
from pathlib import Path
from typing import Any, Protocol

# Third Party
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pydantic import BaseModel

# Gloria
from gloria import Gloria
from gloria.models import ModelBackendBase


## Helper functions
class HasDict(Protocol):
    __dict__: dict[str, Any]


def compare_class_instances(
    actual: HasDict, desired: HasDict, ignore_attributes: set = set()
) -> None:
    """
    Helper function to recursively check the equality of all attributes of
    a Gloria model. Note that this function assumes that
        1. the attributes "stan_fit", "model" of the backend should not
           be compared
        2. the backend has an attribute "fit_params", whose nested numpy array
           elements need special handling


    Parameters
    ----------
    actual : HasDict
        Object that results from the deserialization
    desired : HasDict
        Object from the unserialized fitted Gloria model
    ignore_attributes : set, optional
        A set of attributes that should be ignored. The default is set().

    """

    # Test type equality of models
    assert isinstance(actual, type(desired))

    # Extract all attribute names that should be tested
    if isinstance(actual, BaseModel):
        attributes = (
            set(actual.model_fields.keys()) | set(actual.model_extra.keys())
        ) - ignore_attributes

        desired_attributes = (
            set(desired.model_fields.keys()) | set(desired.model_extra.keys())
        ) - ignore_attributes
    else:
        attributes = set(actual.__dict__.keys()) - ignore_attributes
        desired_attributes = set(desired.__dict__.keys()) - ignore_attributes

    # Test that models have same fields (one might have more extras)
    assert attributes == desired_attributes

    # Test attribute values for equality
    for attr in attributes:
        actual_val = getattr(actual, attr)
        desired_val = getattr(desired, attr)

        # Use numpy and pandas methods for equality checks
        if isinstance(actual_val, np.ndarray):
            np.allclose(actual_val, desired_val)
        elif isinstance(actual_val, pd.Series):
            assert_series_equal(actual_val, desired_val)
        elif isinstance(actual_val, pd.DataFrame):
            assert_frame_equal(actual_val, desired_val, check_dtype=False)
            for col in actual_val.columns:
                assert (
                    actual_val[col].dtype.kind == desired_val[col].dtype.kind
                )
        # Apply recursive checks for pydantic models and the model backend
        elif isinstance(actual_val, BaseModel):
            compare_class_instances(actual=actual_val, desired=desired_val)
        elif isinstance(actual_val, ModelBackendBase):
            compare_class_instances(
                actual=actual_val,
                desired=desired_val,
                ignore_attributes={"stan_fit", "model"},
            )
        # Special handling of fit_params
        elif attr == "fit_params":
            for k, v in actual_val.items():
                if isinstance(v, np.ndarray):
                    assert np.allclose(v, desired_val[k])
                else:
                    assert v == desired_val[k]
        # Everything else should be comparable by the default equality operator
        else:
            assert actual_val == desired_val


## FIXTURES ##


@pytest.fixture(scope="module")
def fitted_model():
    # Get path of config file
    toml_path = Path(__file__).parent / "run_configs/serialize.toml"
    # Get path of data
    source = Path(__file__).resolve().parent / "data/serialize_data.csv"
    # Construct Gloria model from TOML file
    model = Gloria.from_toml(toml_path=toml_path)
    # Load data using TOML options saved in model._config
    df = model.load_data(source=source)
    # Fit model using TOML options saved in model._config
    model.fit(df)
    return model


## TESTS ##


def test_gloria_attribute_equivalence(fitted_model):
    """
    Tests whether all attributes of a Gloria model are included in the
    serialize.GLORIA_ATTRIBUTES dictionary
    """
    # Gloria
    from gloria.utilities.serialize import GLORIA_ATTRIBUTES

    serialize_attributes = set(GLORIA_ATTRIBUTES.keys())
    model_attributes = set(fitted_model.model_fields.keys()) | set(
        fitted_model.model_extra.keys()
    )
    assert serialize_attributes == model_attributes


def test_backend_attribute_equivalence(fitted_model):
    """
    Tests whether all attributes of a Gloria model are included in the
    serialize.GLORIA_ATTRIBUTES dictionary
    """
    # Gloria
    from gloria.utilities.serialize import BACKEND_ATTRIBUTES

    serialize_attributes = set(BACKEND_ATTRIBUTES.keys())
    # The ignore_attributes are cmdstanpy object that are difficult to
    # serialize and not necessary for prediction. Therefore they are not
    # serialized
    ignore_attributes = {"stan_fit", "model"}
    backend_attributes = (
        set(fitted_model.model_backend.__dict__.keys()) - ignore_attributes
    )
    assert serialize_attributes == backend_attributes


def test_serialize_deserialize_model_equivalence(fitted_model):
    """
    Serialize and subsequently deserialize a fitted Gloria model and test
    whether all attributes are conserved under these operations.
    """
    deserialized_model = Gloria.from_json(fitted_model.to_json())

    compare_class_instances(actual=deserialized_model, desired=fitted_model)


if __name__ == "__main__":
    pytest.main()
