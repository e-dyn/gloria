.. _ref-toml-config:
.. currentmodule:: gloria


Configuration Files
===================

Working with Gloria often means adjusting many parameters to control how models behave. When experiments span multiple scripts or environments, keeping these settings consistent can quickly become an error-prone chore. To make this easier, Gloria supports configuration files in the `TOML <https://toml.io/en/>`_ format. In the following we show how to use these files and explain how Gloria resolves conflicts between multiply defined parameters.

Configuring Models with TOML
----------------------------

Consider the following example, which uses the common Gloria building blocks and workflow.

.. code-block:: python

    from gloria import Gloria, CalendricData, Gaussian

    # Construct Gloria model
    model = Gloria(
        model="poisson",
        sampling_period="1d",
        timestamp_name="ds",
        metric_name="y",
        n_changepoints=2
    )

    # Create protocol for calendric data and add to model
    protocol = CalendricData(
        country="US",
        subdiv="",
        monthly_seasonality="auto",
        yearly_seasonality=False,
        holiday_profile=Gaussian(width="5d")
    )
    model.add_protocol(protocol)

    # Load and prepare data
    df = model.load_data(
        source="https://raw.githubusercontent.com/e-dyn/gloria/refs/heads/main/scripts/data/simulated/2025-02-19_binomial_test_n02.csv",
        dtype_kind="u"
    )

    # Fit
    model.fit(df, optimize_mode="MAP", use_laplace=True)

    # Predict
    result = model.predict(periods=30)

    # Plot results
    model.plot(result)
    
In this example, we customize the :class:`Gloria` model instantiation along with the :meth:`~Gloria.load_data`, :meth:`~Gloria.fit`, and :meth:`~Gloria.predict` methods by passing several keyword arguments. Also, we add a :class:`CalendricData` protocol to include holiday events as well as some default seasonalities. All of this boilerplate can be factored out into a configuration file. Using such a file, the code simplifies considerably: [#f1]_

.. code-block:: python

    from pathlib import Path
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
    
To create a Gloria model from a configuration file, simply call :meth:`Gloria.from_toml` with the ``toml_path``. All keyword arguments and the calendric data protocol are now handled through the configuration file, while model and its methods behave exactly as before. The contents of the file are shown below

.. code-block:: toml

    [model]
    model = "poisson"
    sampling_period = "1d"
    timestamp_name = "ds"
    metric_name = "y"
    n_changepoints = 2
    
    [load_data]
    source = "https://raw.githubusercontent.com/e-dyn/gloria/refs/heads/main/scripts/data/simulated/2025-02-19_binomial_test_n02.csv"
    dtype_kind = "u"
    
    [fit]
    optimize_mode = "MAP"
    use_laplace = true
    
    [predict]
    periods = 30
    
    [[protocols]]
    type = "CalendricData"
    country = "US"
    subdiv = ""
    monthly_seasonality = "auto"
    yearly_seasonality = false
    
      [protocols.holiday_profile]
      profile_type = "Gaussian"
      width = "5d"
      
Details on valid TOML syntax can be found in the `official documentation <https://toml.io/en/>`_, but the most important feature is that TOML files are organized in *tables*, which are collections of key/value pairs and defined by headers with square brackets. Each table in the configuration file maps to a specific step in the Gloria workflow. For example, the ``[model]`` table provides the arguments to the :class:`Gloria` constructor, while ``[load_data]`` supplies the options passed to :meth:`~Gloria.load_data`. 

Some methods, such as :meth:`~Gloria.add_seasonality`, :meth:`~Gloria.add_external_regressor`, :meth:`~Gloria.add_event`, and :meth:`~Gloria.add_protocol`, can be called multiple times on a model. Their configuration is therefore expressed as arrays of tables, where each item corresponds to a single call of the respective method. In our example, this is used to add a :class:`CalendricData` protocol.

Parameter Precedence
--------------------

**Model Instantiation**

When a Gloria model is created using ``m = Gloria()``, it is initialized using sensible defaults. These defaults can be overwritten by passing keyword arguments, for example ``m = Gloria(model="poisson")``. Creating a model from a TOML file takes effect in between: parameters defined in the ``[model]`` table take precedence over built-in defaults, but you can still provide additional keyword arguments to overwrite the options from the configuration file:

.. code-block:: python

    m = Gloria.from_toml(
        toml_path=toml_path,
        n_changepoints=5       # <-- overwrites n_changepoints in TOML file
    )
    
**Gloria Methods**

For methods, the situation is slightly more involved. Method settings are stored within the model rather than in the method itself. You can inspect the values via the ``Gloria()._config`` field and observe how they are being overwritten by the settings in the corresponding method table of the configuration file:

.. code-block:: python

    # Default Gloria model
    m = Gloria()
    print(m._config["fit"]["use_laplace"]) # out: False; Laplace sampling off by default

    # TOML-configured Gloria model
    m = Gloria.from_toml(toml_path=toml_path)
    print(m._config["fit"]["use_laplace"]) # out: True; default setting overwritten

When calling a method, you can combine both a TOML file and keyword arguments for local overwrites:

.. code-block:: python

    # Overwrite fit defaults with TOML settings
    m = Gloria.from_toml(toml_path=model_toml)
    
    # Overwrite fit configurations with another TOML file and keyword arguments
    m.fit(data, toml_path=method_toml, use_laplace=False)
    
The overall precedence order for methods is **keyword arguments > local TOML > global TOML > defaults**.

.. rubric:: Footnotes

.. [#f1] Note that the configuration file needs to be located at ``./run_configs/run_config.toml`` relative to the path of this script. The contents of the TOML file can be found in the `GitHub <https://raw.githubusercontent.com/e-dyn/gloria/refs/heads/main/scripts/run_configs/run_config.toml>`_ repository, but is also given in this tutorial.
