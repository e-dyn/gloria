.. _ref-seasonalities:
.. currentmodule:: gloria

Seasonalities
=============

The seasonal component of a time-series :math:`f_\text{seas}(t)` isolates the portion of the series that periodically repeats. A convenient way to approximate this component is a partial Fourier sum, which is an expansion into a weighted sum of sine and cosine functions according to 

.. math::
    f_\text{seas}(t) \approx \sum_{n=1}^{N}{a_n\sin\left(\frac{2\pi n}{T} t\right)
                    + b_n\cos\left(\frac{2\pi n}{T} t\right)}.
                    
Here :math:`T` is the fundamental period of the Fourier series, e.g. :math:`T` = 7 days for patterns that repeat on a weekly basis. The upper limit :math:`N` of the Fourier sum is known as its *order*, where higher orders are able to describe quicker changes in the overall pattern. Eventually, the parameters :math:`a_n` and :math:`b_n` are weighting factors that are specific to the seasonal component at hand and will be estimated by Gloria's fitting procedure.

To illustrate how the Fourier order shapes the seasonal fit, we use the power consumption data already seen in the :ref:`Basic Usage <ref-basic-usage>` tutorial. First, we condense the hourly power consumption data into weekly totals. This aggregation smooths out daily and intra-week fluctuations, leaving only the longer-term signal.

.. code-block:: python
    
    import pandas as pd
    
    # Load the data
    url = "https://raw.githubusercontent.com/e-dyn/gloria/main/scripts/data/real/AEP_hourly.csv"
    data = pd.read_csv(url)
    
    # Convert to datetime
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    
    # Aggregate hourly data to weekly data
    data = data.resample('W', on="Datetime").sum().reset_index().iloc[2:-1]

The resulting series spans more than 14 years, and we model it with just a yearly seasonal component.

.. code-block:: python

    from gloria import Gloria
    
    # Set up the Gloria model
    m = Gloria(
        model="gamma",
        metric_name="AEP_MW",
        timestamp_name="Datetime",
        sampling_period="7 d",
        n_changepoints=0,
    )
    
    # Add observed seasonalities
    m.add_seasonality(
        name="yearly",
        period="365.25d",
        fourier_order=2         # <-- Change Fourier order here
    )

    # Fit the model to the data
    m.fit(data)

    # Predict
    prediction = m.predict(periods=1)
    
    # Plot the results
    m.plot(prediction, include_legend=True)
    
We deliberatly choose ``fourier_order=2`` because the weekly data show two clear demand peaks each year: one in mid-winter and another in mid-summer. As the plot below confirms, an order 2 Fourier series captures this dominant annual pattern faithfully.

.. image:: pics/05_seasonalities_fig01.png
  :align: center
  :width: 700
  :alt: Fitting weekly power consumption data with a yearly seasonality of order 2
  
The next plot shows the result when raising the order from 2 to 10, which lets the yearly Fourier series capture fine-grained oscillations such as small mid-season bumps and dips that the lower order smoothed out.

.. image:: pics/05_seasonalities_fig02.png
  :align: center
  :width: 700
  :alt: Fitting weekly power consumption data with a yearly seasonality of order 10
  
.. warning::

  Increasing the order even further can be tempting to achieve a better result, but bears the risk of fitting noise rather than signal. Good practice is to validate an increase in the order by excluding overfitting on a test data set.
  
Configuring Seasonalities
-------------------------

When considering possible seasonalities and Fourier orders for your model, you can follow these guidelines

* **Minimum data span**: include a seasonal component only if your data cover at least two complete cycles. One period for learning the pattern, one more for confirming its stability. For instance, a weekly seasonality requires at least 2 weeks of data.
* **Maximum Fourier order**: the maximum order that can be estimated from data with sampling period :math:`\Delta t` is :math:`\lfloor T / (2 \Delta t) \rfloor` following the `Nyquist theorem <https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem>`_. For a weekly seasonality (:math:`T` = 7 days) on daily data (:math:`\Delta t` = 1 day), the order is capped at 3.
   
.. tip::

  When using the :class:`~gloria.CalendricData` protocol, these rules are automatically applied setting ``yearly_seasonality``, ``weekly_seasonality`` etc. to ``"auto"``. For more information, see the :ref:`Calendric Data <ref-calendric-data>` tutorial.
