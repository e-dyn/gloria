.. _ref-installation:
.. currentmodule:: gloria

Installation
============

.. note::

    Gloria requires Python version :math:`\ge` 3.9 and :math:`<` 3.13. If you do not have it installed, please refer to `python.org <https://www.python.org/>`_. 

Installing Gloria
-----------------
    

We recommend to install Gloria into a dedicated `virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments>`_. Once the environment is created and activated use [#f1]_

.. code-block:: console

    pip install gloria

to install the package. To verify it is correctly installed, start a Python REPL session by simply entering ``python`` into your shell and run the following commands:

.. code-block:: console

    >>> from gloria import Gloria
    >>> Gloria()
    
This will output a newly instantiated Gloria object similar to the following

.. code-block:: console

    >>> Gloria(model='normal', sampling_period=Timedelta('1 days 00:00:00'), timestamp_name='ds', metric_name='y', capacity_name='', changepoints=None, n_changepoints=25, changepoint_range=0.8, seasonality_prior_scale=3, event_prior_scale=3, changepoint_prior_scale=3, dispersion_prior_scale=3, interval_width=0.8, trend_samples=1000, model_backend=<gloria.models.Normal object at 0x0000015509551CD0>, vectorized=False, external_regressors={}, seasonalities={}, events={}, prior_scales={}, protocols=[], history=Empty DataFrame
    Columns: []
    Index: [], first_timestamp=Timestamp('1970-01-01 00:00:00'), X=Empty DataFrame
    Columns: []
    Index: [], fit_kwargs={})

.. rubric:: Footnotes

.. [#f1] The package installer *pip* is typically installed along with Python, in particular if it was installed via `python.org <https://www.python.org/>`_. If pip is not installed, you can follow the instructions found `here <https://pip.pypa.io/en/stable/installation/>`_.