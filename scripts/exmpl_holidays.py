### --- Module Imports --- ###
# Standard Library
from pathlib import Path

# Third Party
import pandas as pd

# Gloria
# Inhouse Packages
from gloria import Gloria

### --- Global Constants Definitions --- ###
CONFIG_FILE = "run_config"
# Note: predicting after deserialization currently only works when there is no
# external regressor
INCLUDE_SERIALIZATION_STEP = True

### --- Class and Function Definitions --- ###


### --- Main Script --- ###
if __name__ == "__main__":
    basepath = Path(__file__).parent

    model = Gloria.from_toml(
        toml_path=basepath / f"run_configs/{CONFIG_FILE}.toml"
    )
    df = model.load_data()
    model.fit(df)

    # Demonstration of model serialization and deserialization work flow
    if INCLUDE_SERIALIZATION_STEP:
        # Save the model
        model_path = basepath / "models/test_model.json"
        model_json = model.to_json(filepath=model_path, indent=2)
        # And load it
        model_new = Gloria.from_json(model_json=model_path, return_as="model")

    data = model.make_future_dataframe(periods=100)
    data = (
        pd.concat([data, df["ano_deviation"]], axis=1)
        .reset_index(drop=True)
        .fillna(0)
    )
    result = model.predict(data)

    timestamp_name = model.timestamp_name
    metric_name = model.metric_name
    mask = (
        (result[timestamp_name] - result[timestamp_name].min())
        / model.sampling_period
    ).apply(float.is_integer)
    result = result[mask]

    model.plot(result)
    model.plot_components(result)

    # a = (
    #     (df.y <= result.observed_lower) | (df.y > result.observed_upper)
    # ).sum()
    # a /= len(df.y)
    # print(a)
