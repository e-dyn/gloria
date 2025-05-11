# Third Party
import pandas as pd
import pytest

# Gloria
# default_model
from gloria import Gloria
from gloria.utilities.constants import _GLORIA_DEFAULTS

## FIXTURES ##


@pytest.fixture
def default_model():
    return Gloria()


## TESTS ##


# default_model object creation
def test_default_model_pydantic_defaults(default_model):
    """
    Tests that a default Gloria model's fields have all the default values
    stored in _default_model_DEFAULTS.
    """
    # Iterate through all default values
    for attr, default_value in _GLORIA_DEFAULTS.items():
        model_value = default_model.__dict__[attr]
        assert default_value == model_value


def test_default_model_init_defaults(default_model):
    """
    Tests that a defaults Gloria model's attributes set during __ini__ have the
    correct default values.
    """

    # Fields initialized
    assert default_model.changepoints is None
    assert default_model.external_regressors == {}
    assert default_model.seasonalities == {}
    assert default_model.events == {}
    assert default_model.prior_scales == {}
    assert default_model.protocols == []
    assert default_model.first_timestamp == pd.Timestamp(0)
    pd.testing.assert_frame_equal(default_model.history, pd.DataFrame())
    pd.testing.assert_frame_equal(default_model.X, pd.DataFrame())


@pytest.mark.parametrize(
    "changepoints, timestamp_name",
    [
        (None, "a"),
        ([], "b"),
        (["2024-12-31"], "c"),
        (["2024-12-31", "1990-01-01"], "d"),
    ],
)
def test_changepoint_initialization(changepoints, timestamp_name):

    model = Gloria(changepoints=changepoints, timestamp_name=timestamp_name)

    if changepoints is None:
        n_changepoints = _GLORIA_DEFAULTS["n_changepoints"]
        assert model.changepoints is None
    else:
        changepoints = pd.Series(
            pd.to_datetime(changepoints), name=timestamp_name
        )
        n_changepoints = len(changepoints)
        pd.testing.assert_series_equal(model.changepoints, changepoints)

    assert model.n_changepoints == n_changepoints


if __name__ == "__main__":
    ...
