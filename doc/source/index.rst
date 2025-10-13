.. Gloria documentation master file, created by
   sphinx-quickstart on Sun May 11 22:55:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gloria documentation
====================

**Version**: |version|

**Gloria** is a flexible time series forecasting tool inspired by `Prophet <https://facebook.github.io/prophet/>`_.  
It enhances Prophet’s GLM-based structure with advanced probabilistic modeling using tailored distributions  
(e.g., binomial, Poisson, beta) – ideal for real-world forecasting under uncertainty.


.. _cards-clickable:

Get Started
............

.. grid:: 1 1 2 2
   :gutter: 4
   :margin: 0 0 2 0

   .. grid-item-card:: 🚀 Get Started
      :link: user_guide/index
      :link-type: doc

      **Learn by doing.**  
      Tutorials and workflows for building, training, and evaluating models.

   .. grid-item-card:: 🧠 API Reference
      :link: api/index
      :link-type: doc

      **Explore the internals.**  
      Detailed reference for functions, models, and utility classes.


Built for Professionals
........................
.. grid:: 1 1 2 6
   :gutter: 3
   :margin: 2 2 2 2
   
   .. grid-item-card:: Distributional Flexibility
      :text-align: center
      :columns: 4

      Go beyond the normal distribution and model count data (Poisson, 
      Binomial, Negative Binomial, Beta-Binomial), bounded rates (Beta), or 
      non-negative floats (Gamma) natively

   .. grid-item-card:: Any Time Grid
      :text-align: center
      :columns: 4

      Gloria handles arbitrary sampling intervals beyond daily

   .. grid-item-card:: Rich Event Modeling
      :text-align: center
      :columns: 4

      Parametric and extensible event library to handle holidays, campaigns, or
      maintenance windows - any event, any shape, for realistic impacts and 
      reduced overfitting.

   .. grid-item-card:: Fully Explainable
      :text-align: center
      :columns: 4

      Gloria's models are explicit, fully documented, and always inspectable.
      
   .. grid-item-card:: Composable Pipelines
      :text-align: center
      :columns: 4

      Gloria's modular design lets you build custom forecasting workflows by 
      combining and extending components like events, distributions, and 
      regressors.
      
   .. grid-item-card:: Modern Python Stack
      :text-align: center
      :columns: 4

      Type hints, pydantic for validation, and a clean API design reminiscent 
      of `Prophet <https://facebook.github.io/prophet/>`_, but with a much more
      maintainable and extensible codebase.    

Get Involved
.............

Gloria is Open Source and thrives through your ideas, usage, and feedback.  
Try it, contribute, or just explore:

- 🛠️ `GitHub Repository <https://github.com/e-dyn/gloria>`__
- 📬 `Issue Tracker <https://github.com/e-dyn/gloria/issues>`__
- 📄 `License MIT <https://github.com/e-dyn/gloria/blob/main/LICENSE>`__


Table of Contents
..................

.. toctree::
   :maxdepth: 2

   user_guide/index
   api/index
   release_notes/index