.. _overview:
.. currentmodule:: gloria
.. autosummary::
   :template: autosummary/small_class.rst
   :toctree: index/

Overview
========

Time series are highly diverse and so is the ecosystem of tools to analyze them. If you know the features of your time series, you can develop a great model that nicely fits the data and yields precise predictions. However, chances are your model cannot be recycled for the next data set and you have to start over. A true generalist covering virtually all use-cases does not exist, but `Prophet <https://facebook.github.io/prophet>`_ is popular for having a well adjusted amount of abstraction and automation making it usable for a wide range of different time series.

Gloria is inspired by Prophet and aims at extending its capabilities. The core improvement is an extensive set of distributions. Where Prophet only uses the normal distribution, Gloria additionally offers the Poisson or gamma distribution and many more, which enables the user to actually respect the data type of their time series. This and other features are explained as part of the present section.

Data Types
----------

Depending on what a time series represents, the associated data can have different characteristics. The two essential features to consider are the underlying set of numbers (natural vs. real numbers) and possible constraints, i.e., minimum and maximum values the data can take. Here are a few examples:

#. If you send 100 marketing emails, the number of replies you receive is an integer between 0 and the total number of emails.
#. The percentage of website visitors who complete a purchase is represented by a real number between zero and one.
#. The number of accidents at an intersection is a positive integer that is not bounded above.
#. The measurement error of a temperature reading is a real number and (almost) unbounded.

It is important that these characteristics are captured by the model we use to describe the time series, which is usually done by choosing an appropriate probability distribution—that is, a function that assigns probabilities to observed values. The examples above could be described using binomial, beta, Poisson, and normal distributions, respectively.

While Gloria supports these and other distribution types, Prophet only supports the normal distribution. For many use cases, this is sufficient. Although time series are technically often bounded, the data often falls within a range that is far from the boundaries, making it reasonable to treat the data as unbounded. Similarly, it's often practical to approximate an integer value with a real number.

However, there are cases where the normal distribution approximation fails: hopefully, the intersection mentioned above isn’t particularly dangerous and only has around 5 accidents per year on average. It’s likely that sometimes there will be 10 accidents and in other years just one or two. If we model accidents with a normal distribution, we would find that a negative number of accidents is not only possible but even likely. Negative values in this context are meaningless, indicating that the normal distribution is the wrong choice.

Additive and Multiplicative Models
----------------------------------

Depending on the context, there are two main approaches to combining trend, seasonality, and noise in a model: purely additive or purely multiplicative. These describe the data as either

.. math:: y(t) = Trend + Seasonality + Noise

or 

.. math:: y(t) = Trend \cdot Seasonality \cdot Noise

Both Prophet and Gloria are based on generalized linear models (GLMs), which link probability distributions with additive combinations of relevant terms. However, except for normally distributed models, the allowable value ranges of all other models are constrained, as discussed earlier. Purely additive models can violate these constraints. To handle this, GLMs use a concept called a link function, which ensures that additive models still respect these constraints. As a side effect, GLMs for some distributions—such as Poisson or gamma—automatically become multiplicative models. In practice, it's often entirely sufficient to describe multiplicative time series using these distributions.

Since these distributions are part of Gloria’s toolkit, modeling additive or multiplicative relationships becomes natural by simply choosing the appropriate distribution.

Prophet, on the other hand, only supports the normal distribution, which does not require a link function. This means its models are always additive. Prophet attempts to address this by allowing users to specify multiplicative terms within the linear model. However, this only works for the interaction between the trend and regressors (such as seasonal effects), not for the noise term. The resulting model is effectively a hybrid:

.. math:: y(t) = Trend \cdot Seasonality + Noise

Sampling Rate
-------------

The sampling rate is the frequency at which new data points are collected in a time series — e.g., one data point per day or one per hour.

Prophet can handle non-daily sampling rates, but it was designed with daily data in mind. As a result, any sampling rate—and any seasonality added to the model—must be converted into daily units.

Gloria, by contrast, makes no assumption about a fundamental time base. The sampling rate is specified directly, such as "once per day" or "once every 53 seconds," and all model attributes are automatically interpreted in the units of that sampling rate.

Events
------

One of Prophet’s strengths is its ability to learn the effects of special events, like holidays, from the data and use them to improve forecast accuracy. Prophet provides the user with three main parameters: the date of the holiday, a lead window, and a lag window. For example, to model holiday shopping activity in the week before Christmas, one would set a 7-day lead window and a 0-day lag window. As a result, Prophet adjusts a separate parameter for each of these days. If the training data spans only two or three years—or if you want to model not just one week but five—this leads to a large number of parameters relative to a small amount of data, increasing the risk of overfitting.

Gloria addresses this problem using generalized events, which take on specialized but meaningful shapes—such as an exponential ramp-up to model Christmas shopping. Such curves can usually be described with very few parameters, regardless of whether they span a short or long period. A minor drawback is that the analyst needs to understand the data well enough to choose the appropriate event type.