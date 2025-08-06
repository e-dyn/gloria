.. _ref-installation:

.. currentmodule:: gloria
.. autosummary::
   :template: autosummary/small_class.rst
   :toctree: get_started/

Installation
============

.. note::

    Gloria requires Python version :math:`\ge` 3.9 and :math:`<` 3.13. If you do not have it installed, please refer to `python.org <https://www.python.org/>`_. 

Installing Gloria
-----------------
    
Gloria resides in the public GitHub repository ``https://github.com/e-dyn/gloria``. 

We recommend to install Gloria into a dedicated `virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments>`_. Once the environment is created and activated use [#f1]_

.. code-block:: console

    pip install gloria

to clone and install the package in one step. To verify it is correctly installed, run the following command:

.. code-block:: console

    > pip show gloria
    Name: gloria
    Version: 0.1.0
    Summary: ...
    
Installing CmdStan
------------------

Gloria's backend employs the statistical programming language `Stan <https://mc-stan.org/>`_ and uses `CmdStan <https://mc-stan.org/docs/cmdstan-guide/>`_ as interface. As of now, Gloria needs to have a full installation of CmdStan, which will be automatically triggered once you create a Gloria model for the first time. However, prior to that a C++ toolchain consisting of a modern C++ compiler and GNU-Make utility need to be installed. Depending on your system, do one of the following [#f2]_

.. tab:: Windows
    
    The Python interface to CmdStan provides the necessary functionality to install the toolchain automatically. Nothing needs to be done on a Windows machine (that's a first).
    
.. tab:: MacOS

    The Xcode and Xcode command line tools must be installed. Xcode is available for free from the Mac App Store. To install the Xcode command line tools, run the shell command: ``xcode-select --install.``
    
.. tab:: Linux

    The required C++ compiler is ``g++ 4.9 3``. On most systems the GNU-Make utility is pre-installed and is the default ``make`` utility. There is usually a pre-installed C++ compiler as well, but not necessarily new enough.
    


With the toolchain ready, start a Python REPL session by simply entering ``python`` into your shell and run the following commands:

.. code-block:: console

    >>> from gloria import Gloria
    >>> Gloria()

Now you need a little patience while the CmdStan Toolchain and CmdStan itself are being installed. Once this is finished you are all set. You should now be able to use gloria inside your python applications.

.. rubric:: Footnotes

.. [#f1] The package installer *pip* is typically installed along with Python, in particular if it was installed via `python.org <https://www.python.org/>`_. If pip is not installed, you can follow the instructions found `here <https://pip.pypa.io/en/stable/installation/>`_.
.. [#f2] Instructions borrowed from `CmdStanPy C++ Toolchain Requirements <https://mc-stan.org/cmdstanpy/installation.html>`_.