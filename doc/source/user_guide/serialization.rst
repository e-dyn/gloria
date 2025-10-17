.. _ref-serialization:
.. currentmodule:: gloria


Model Serialization
===================

While unfitted models can be created from :ref:`TOML configuration files <ref-toml-config>`, Gloria also provides built-in helper methods to serialize and deserialize fitted models. A deserialized model retains all the information needed to generate forecasts based on its previous fit. This is useful in various scenarios, such as resuming work in a later session, transferring a model between machines, or passing it between stages of a processing pipeline.

Serialize to JSON
-----------------

The main method for serializing a model is :meth:`Gloria.to_json`, which converts a fitted model into a JSON-formatted string:

.. code-block:: python

    # Gloria model must be fitted
    model.fit(df)
    
    # Serialize the model to a JSON string
    serialized = model.to_json(indent=2)
    
    # Print the string
    print(serialized[:290] + "\n...")

This will print output similar to the following:

.. code-block:: console
    
    {
      "model": "poisson",
      "timestamp_name": "ds",
      "metric_name": "y",
      "capacity_name": "",
      "n_changepoints": 2,
      "changepoint_range": 0.8,
      "seasonality_prior_scale": 3,
      "event_prior_scale": 3,
      "changepoint_prior_scale": 3,
      "dispersion_prior_scale": 3,
      "interval_width": 0.8
    ...

The serialized model can then be loaded back using :meth:`Gloria.from_json`, which reconstructs the original model. The restored model can be used for prediction:

.. code-block:: python

    # Deserialize the model
    model_copy = Gloria.from_json(serialized)

    # Make prediction using the deserialized model
    result = model_copy.predict()

    # Plot results
    model_copy.plot(result)
    

Both :meth:`Gloria.to_json` and :meth:`Gloria.from_json` accept an optional ``filepath`` argument. If provided, the methods will write to or read from the specified file path instead of working with strings directly.
    
.. tip::

    * :meth:`Gloria.to_json` internally uses :func:`json.dumps` to generate the JSON string from a dictionary. Therefore, any keyword arguments accepted by :func:`json.dumps` can also be passed to :meth:`Gloria.to_json`.
    
Serialize to a Dictionary
-------------------------

Gloria models can also be serialized to and restored from Python dictionaries using :meth:`Gloria.to_dict` and :meth:`Gloria.from_dict`. The workflow is identical to working with JSON-serialized models.

.. note::
    
    Although :meth:`Gloria.to_dict` and :meth:`Gloria.from_dict` are exposed to users, they primarily serve as helper methods for their JSON counterparts. Consequently, non–JSON-serializable objects (such as NumPy arrays or pandas DataFrames) are not preserved in their original types within the dictionary. Instead, they are converted to serializable representations.
