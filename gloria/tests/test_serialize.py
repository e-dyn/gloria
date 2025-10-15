# Standard Library
from pathlib import Path

# Third Party
import pytest

# Gloria
# default_model
from gloria import Gloria

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


if __name__ == "__main__":
    pytest.main()
