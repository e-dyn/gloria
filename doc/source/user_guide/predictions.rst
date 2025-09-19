.. _ref-predictions:
.. currentmodule:: gloria


Predictions
===========

In the previous tutorials, we frequently used the *model → fit → predict* workflow. The output of :meth:`~Gloria.predict()` was usually passed directly into the :meth:`~Gloria.plot()` method without inspecting it further. In this tutorial, we will take a closer look at the predictions themselves, focusing on a key feature that sets Gloria apart from Prophet: the distinction between confidence intervals around the predicted expectation and intervals describing the variability of observed data.

Output Data Frame
-----------------

Let's start with a simple example from the :ref:`Basic Usage <ref-basic-usage>` tutorial:

.. code-block:: python

    import pandas as pd             # For loading and processing
    from gloria import Gloria

    # Load the data
    url = "https://raw.githubusercontent.com/e-dyn/gloria/main/scripts/data/real/AEP_hourly.csv"
    data = pd.read_csv(url)

    # Save the column names for later usage
    timestamp_name = "Datetime"
    metric_name = "AEP_MW"

    # Convert to datetime
    data[timestamp_name] = pd.to_datetime(data[timestamp_name])

    # Restrict data to last 14 days
    data = data.sort_values(by = "Datetime").tail(336)

    # Set up the model
    m = Gloria(
        model = "gamma",
        metric_name = metric_name,
        timestamp_name = timestamp_name,
        sampling_period = "1 h",
        n_changepoints = 0
    )

    # Add observed seasonalities
    m.add_seasonality(name="daily", period="24 h", fourier_order=2)
    m.add_seasonality(name="weekly", period="7 d", fourier_order=2)

    # Fit the model to the data
    m.fit(data)

    # Predict
    prediction = m.predict(periods=96)

    # Plot
    m.plot(prediction, include_legend=True)
    
With the plot shown here:

.. image:: pics/01_basic_usage_fig02.png
  :align: center
  :width: 700
  :alt: Plot of the Gloria fit and forecast of the power consumption data set.
  
The plot elements forecast, trend, and confidence intervals are all stored in the ``prediction`` data frame. This object is returned by :meth:`~Gloria.predict` and consumed by :meth:`~Gloria.plot`. We can explore its contents by printing the column names:

.. code-block:: python

    for col in prediction.columns:
        print(col)
        
* ``Datetime``: Timestamps at which the prediction was evaluated.
* ``yhat``: The overall prediction, shown as orange solid line.
* ``yhat_upper``/``yhat_lower``: Upper and lower bounds on the prediction at a confidence level defined by ``m.interval_width`` (default 80%). These bounds are **not** shown in the plot.
* ``observed_lower``/``observed_upper``: Bounds on the variability of the observed data, derived from the underlying distribution model at a quantile level specified by ``m.interval_width``. These bounds are displayed by the grey area.
* ``trend``: Trend component of the model, shown as black solid line.
* ``trend_upper``/``trend_lower``: Bounds corresponding to the trend contribution within ``observed_lower``/``observed_upper``.
* ``*_linked_*``: The respective quantities on the scale of the underlying generalized linear model, transformed by the *link-function*. See :ref:`Decomposition Types <ref-decomposition_types>` tutorial for details.

Some of these columns need further explanation, which is given in the following.

Confidence Bands vs Data Variability
------------------------------------

By default, Gloria's :meth:`~Gloria.fit` method yields a point estimate of all model parameters by performing a maximum a posteriori estimation. As a consequence, the prediction ``yhat`` is also just a point estimate and does not by itself include uncertainty. To add a confidence band around ``yhat`` you can use `Laplace sampling <https://mc-stan.org/docs/cmdstan-guide/laplace_sample_config.html>`_. This method draws samples from a normal approximation centered at the optimized mode. To trigger Laplace sampling, set the ``use_laplace=True`` flag while calling :meth:`~Gloria.fit`. When running the fit, Gloria's logger will inform you about the steps being taken:

.. code-block:: console
    
    09:49:57 - gloria - INFO - Starting optimization.
    09:49:57 - gloria - INFO - Starting Laplace sampling.
    09:49:58 - gloria - INFO - Evaluate model at all samples for yhat upper and lower bounds.

Here, Gloria first runs the optimization, then applies Laplace sampling, and finally calculates the upper and lower bounds for ``yhat``, saved in the ``yhat_upper`` and ``yhat_lower`` columns of the prediction output. These columns also exist when ``use_laplace=False`` is set. In that case, ``yhat``, ``yhat_upper``, and ``yhat_lower`` are identical.

.. note::
    
    The quality of Laplace sampling depends on how well the normal approximation agrees with the true a posteriori distribution.

In contrast, data variability is always computed, as it only requires the point estimate from the optimization step. Specifically, the optimized fit parameters are passed to the model's percent point function (see [#f1]_) along with the upper and lower confidence levels corresponding to the requested interval width. The results are stored in the ``observed_upper`` and ``observed_lower`` columns. Typically, the variability interval is much wider than the confidence band of ``yhat``, which is illustrated in the following plot of the first 100 data points: [#f2]_

.. image:: pics/predictions_ci_vs_variability.png
  :align: center
  :width: 700
  :alt: Comparison of confidence band with data variability

.. important::
    
    Keep in mind: in Prophet ``yhat_upper`` and ``yhat_lower`` represent data variability and therefore correspond to Gloria's ``observed_upper`` and ``observed_lower``.

Trend Predictions
-----------------

While Gloria assumes stationary seasonality and event patterns, the trend component is less predictable. Neither the location nor the size of changepoint rate changes can be known in advance. What we can specify, however, are *changepoint density* and *mean rate change*, as they have been learned during training. Gloria uses these parameters to predict the trend and its uncertainty in the same fashion as `Prophet <https://facebook.github.io/prophet/docs/uncertainty_intervals.html#uncertainty-in-the-trend>`_. The following figure shows the outcome of our example, when ``n_changepoints=8`` is set:

.. image:: pics/prediction_trend_uncertainty.png
  :align: center
  :width: 700
  :alt: Plot of the Gloria fit and forecast of the power consumption data set.

From the result we observe the following:

1. From the 8 changepoints in total, only 3 result in meaningful rate changes, due to the sparse prior on rate changes.
2. It is not possible to assume specific changepoints or rate changes in the forecast period. Therefore, the trend forecast is simply an extrapolation of the latest trend in the training period.
3. The variability intervals widen as the forecast extends further beyond the training data. This is the result of an additional *trend uncertainty* estimation step, which simulates possible future trends based on the known changepoint density and mean rate change. Their distribution contributes to the overall variability. The number of these simulations can be controlled by the Gloria input parameter ``trend_samples`` (1000 by default).

.. rubric:: Footnotes

.. [#f1] Gloria uses scipy's percent point function :meth:`scipy.stats.rv_continuous.ppf` for the respective distribution, e.g. :obj:`scipy.stats.gamma`.
.. [#f2] Note that Gloria's plot method has been modified for the sake of this figure. By default, the confidence band defined by ``yhat_upper`` and ``yhat_lower`` cannot be plotted.
