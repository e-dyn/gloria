"""
Demonstration of config TOML file use

Note that no parameter are set manually in this manually in this example

Make sure to place both data and config file in the correct relative paths

See also Tutorial on configuration files:
    https://e-dyn.github.io/gloria/user_guide/toml_config.html
"""

# Standard Library
from pathlib import Path

# Gloria
from gloria import Gloria

# Get path of config file
toml_path = Path(__file__).parent / "run_configs/run_config.toml"

# Construct Gloria model from TOML file
model = Gloria.from_toml(toml_path=toml_path)

# Load data using TOML options saved in model._config
df = model.load_data()

# Fit model using TOML options saved in model._config
model.fit(df, use_laplace=False)

serialized = model.to_dict(indent=2)

model_copy = Gloria.from_json(serialized)

# Make prediction using the deserialized model
result = model_copy.predict()

# Plot results
model_copy.plot(result)
