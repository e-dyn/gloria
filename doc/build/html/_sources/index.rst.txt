.. Gloria documentation master file, created by
   sphinx-quickstart on Sun May 11 22:55:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gloria documentation
====================

**Version**: |version|

Gloria is a timeseries forecasting tool inspired by `Prophet <https://facebook.github.io/prophet/>`_. It extends Prophet's use of generalized linear models (GLMs) to handle a wide variety of data types and constraints in a statistically coherent way through appropriate distributions such as binomial, Poisson, beta, and others.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 🚀 Get Started
      :link: get_started/index
      :text-align: center

      Erste Schritte, Installation und Hello World.

   .. grid-item-card:: 📚 API reference
      :link: api/index
      :text-align: center

      Die Dokumentation der Funktionen und Klassen.

.. toctree::
   :maxdepth: 2
   
   get_started/index
   api/index