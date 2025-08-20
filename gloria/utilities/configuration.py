# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains classes to manage Gloria configurations as well as their
serialization and deserialization
"""

### --- Module Imports --- ###
# Standard Library
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Optional,
    Union,
)

# Third Party
import tomli

# Conditional import of Gloria for static type checking. Otherwise Gloria is
# forward-declared as 'Gloria' to avoid circular imports
if TYPE_CHECKING:
    from gloria.interface import Gloria

# Gloria
from gloria.protocols.protocol_base import get_protocol_map
from gloria.utilities.logging import get_logger


### --- Class and Function Definitions --- ###
def model_from_toml(
    toml_path: Union[str, Path],
    ignore: Union[Collection[str], str] = set(),
    **kwargs: dict[str, Any],
) -> "Gloria":
    """
    Instantiate and configure a Gloria object from a TOML configuration file.

    The TOML file is expected to have the following top-level tables /
    arrays-of-tables (all are optional except ``[model]``):

    * ``[model]`` - keyword arguments passed directly to the :class:`Gloria`
      constructor.
    * ``[[external_regressors]]`` - one table per regressor; each is forwarded
      to :meth:`~Gloria.add_external_regressor`.
    * ``[[seasonalities]]`` - one table per seasonality; each is
      forwarded to :meth:`~Gloria.add_seasonality`.
    * ``[[events]]`` - one table per event; each is forwarded to
      :meth:`~Gloria.add_event`.
    * ``[[protocols]]`` - one table per protocol. Each table **must** contain a
      ``type`` key that maps to a protocol class name; the remaining keys are
      passed to that class before calling :meth:`~Gloria.add_protocol`.

    Defaults as defined in :class:`Gloria` constructor or respective methods
    are used for all keys not provided in the TOML file. ``kwargs`` can be used
    to overwrite keys found in the ``[model]`` table.


    Parameters
    ----------
    toml_path : Union[str, Path]
        Path to the TOML file containing the model specification.
    ignore : Union[Collection[str],str], optional
        Which top-level sections of the file to skip. Valid values are
        ``"external_regressors"``, ``"seasonalities"``, ``"events"``, and
        ``"protocols"``. The special value ``"all"`` suppresses every optional
        section. May be given as a single string or any iterable of strings.
    **kwargs : dict[str, Any]
        Keyword arguments that override or extend the ``[model]`` table. Only
        keys that are valid fields of Gloria (i.e. that appear in
        Gloria.model_fields) are retained; others are silently dropped.

    Returns
    -------
    Gloria
        A fully initialized Gloria instance.


    .. seealso::

        :meth:`~Gloria.from_toml`
            An alias

    Notes
    -----
    Precedence order for :class:`Gloria` constructor arguments from highest to
    lowest is:

    1. Values supplied via ``kwargs``
    2. Values found in the TOML ``[model]`` table
    3. Gloria's own defaults
    """
    # Make sure ignore is a set
    if isinstance(ignore, str):
        ignore = {ignore}
    else:
        ignore = set(ignore)

    # Extend set by all possible attributes if 'all' in ignore
    if "all" in ignore:
        ignore = set(ignore) | {
            "external_regressors",
            "seasonalities",
            "events",
            "protocols",
        }

    # Load configuration file
    with open(toml_path, mode="rb") as file:
        config = tomli.load(file)

    # Gloria
    from gloria.interface import Gloria

    # Remove keys from kwargs and config that are no valid Gloria fields
    kwargs = {k: v for k, v in kwargs.items() if k in Gloria.model_fields}

    if "model" not in config:
        get_logger().info("Model table missing from TOML configuration file.")

    model_config = {
        k: v
        for k, v in config.get("model", dict()).items()
        if k in Gloria.model_fields
    }

    # Give precedence to individial settings in kwargs
    model_config = model_config | kwargs

    # Create Gloria model
    m = Gloria(**model_config)

    # Add external regressors
    if "external_regressors" not in ignore:
        for er in config.get("external_regressors", []):
            m.add_external_regressor(**er)

    # Add seasonalities
    if "seasonalities" not in ignore:
        for season in config.get("seasonalities", []):
            m.add_seasonality(**season)

    # Add events
    if "events" not in ignore:
        for event in config.get("events", []):
            # Create and add the protocol with the remaining configurations
            m.add_event(**event)

    # Add protocols
    if "protocols" not in ignore:
        for protocol in config.get("protocols", []):
            # Get protocol class using the 'type' key in of the protocol config
            ProtocolClass = get_protocol_map()[protocol.pop("type")]
            # Create and add the protocol with the remaining configurations
            m.add_protocol(ProtocolClass(**protocol))

    # Update model's config with fit, predict, and load_data tables
    fit_config = filter_config_parameter("fit", config.get("fit", dict()))
    predict_config = filter_config_parameter(
        "predict", config.get("predict", dict())
    )
    load_config = filter_config_parameter(
        "load_data", config.get("load_data", dict())
    )

    m._config["fit"] = m._config["fit"] | fit_config
    m._config["predict"] = m._config["predict"] | predict_config
    m._config["load_data"] = m._config["load_data"] | load_config

    return m


ACCEPTED_PARS = {
    "fit": (
        "optimize_mode",
        "use_laplace",
        "capacity",
        "capacity_mode",
        "capacity_value",
    ),
    "predict": ("periods", "include_history"),
    "load_data": ("dtype_kind", "source"),
}


def filter_config_parameter(
    method: str, config: dict[str, Any]
) -> dict[str, Any]:
    """
    Extract the subset of configuration options that are valid for
    ``Gloria.fit`` or ``Gloria.predict`` methods.

    Parameters
    ----------
    method : str
        Name of the ``Gloria`` method whose accepted parameters should be
        included.
    config : dict[str, Any]
        Arbitrary keyword arguments intended for ``Gloria``. Keys that do not
        appear in the target method`s signature are silently discarded.

    Raises
    ------
    ValueError
        If *method* is not exactly ``'fit'`` or ``'predict'``.

    Returns
    -------
    dict[str, Any]
        A filtered copy of *config* containing only the parameters that the
        specified ``Gloria`` method accepts.

    Examples
    --------
    >>> cfg = {
    ...     'sample': True,
    ...     'optimize_method': 'MLE',
    ...     'the_answer': 42,  # not an accepted parameter
    ... }
    >>> filter_config_parameter('predict', cfg)
    {'sample': True, 'optimize_method': 'MLE'}

    """
    # Validate method input
    if method not in ("fit", "predict", "load_data"):
        raise ValueError(
            "Parameter 'method' must be either 'fit', 'predict', 'load_data'."
        )

    # Get available input parameters for the method
    return {k: v for k, v in config.items() if k in ACCEPTED_PARS[method]}


def assemble_config(
    method: str,
    model: "Gloria",
    toml_path: Optional[Union[str, Path]] = None,
    **kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the effective configuration dictionary for ``Gloria.fit`` or
    ``Gloria.predict`` or ``load_data``.

    The final configuration is composed in three layers, each one overriding
    the previous:

    1. **Model defaults** - the baseline stored in ``model._config[method]``.
    2. **TOML file** - if *toml_path* is given, key-value pairs from the file
       are merged into the baseline.
    3. **Keyword overrides** - additional arguments supplied directly to
       ``assemble_config`` via *kwargs* take highest precedence.


    Parameters
    ----------
    method : str
        Name of the ``Gloria`` method the configuration is intended for.
    model : "Gloria"
        Instance whose internal default configuration should be taken as
        baseline.
    toml_path : Optional[Union[str, Path]], optional
        Path to a TOML file that contains configuration sections keyed by the
        *method* name. If *None*, this layer is skipped. The default is None.
    **kwargs : dict[str, Any]
        Arbitrary keyword arguments that override earlier layers. Unsupported
        keys are ignored silently.

    Returns
    -------
    dict[str, Any]
        The fully assembled configuration dictionary that can be passed
        directly to the desired ``Gloria`` method.

    """

    def augment_capacity(
        config: dict[str, Any], method: str
    ) -> dict[str, Any]:
        """
        Once any capacity parameter is set for the fit method, the remaining
        ones need to be set to None. Otherwise merging the configs will only
        partially overwrite old capacity parameters.
        """
        capacity_pars = ("capacity", "capacity_mode", "capacity_value")
        # If no capacity parameter was set, return config as is
        is_capacity_set = any(par in config for par in capacity_pars)
        if not is_capacity_set:
            return config

        # Set all capacity parameters that are not present to None
        if method == "fit":
            for par in capacity_pars:
                if par not in config:
                    config[par] = None
        return config

    # Baseline with internal model configurations, if available.
    config = model._config.get(method, dict())
    config = augment_capacity(config, method)

    # 1. Update with TOML config
    if toml_path is not None:
        # Load configuration file
        with open(toml_path, mode="rb") as file:
            toml_config = tomli.load(file)

        # Get method table of config file
        toml_config = toml_config.get(method, dict())
        toml_config = augment_capacity(toml_config, method)

        # Overwrite config with keys in TOML config
        config = config | toml_config

    # 2. Update with kwarg config

    # Overwrite config with keys in TOML config
    kwargs = augment_capacity(kwargs, method)
    config = config | kwargs

    # Remove keys that have no corresponding arguments in the method
    config = filter_config_parameter(method, config)

    return config
