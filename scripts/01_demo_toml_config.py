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
model.fit(df)

# Make prediction using TOML options saved in model._config
result = model.predict()

# Plot results
model.plot(result)
