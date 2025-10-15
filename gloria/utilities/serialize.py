# Copyright (c) 2025 e-dynamics GmbH and affiliates
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Definition of functions and constants used for serializing and deserializing
fitted Gloria objects.
"""

### --- Module Imports --- ###
# Standard Library
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

# Third Party
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Inhouse Packages

# Conditional import of Gloria for static type checking. Otherwise Gloria is
# forward-declared as 'Gloria' to avoid circular imports
if TYPE_CHECKING:
    from gloria import Gloria

# Gloria
from gloria.models import (
    ModelBackend,
    ModelInputData,
    ModelParams,
    get_model_backend,
)
from gloria.protocols.protocol_base import Protocol
from gloria.regressors import Regressor
from gloria.utilities.constants import _DTYPE_KIND
from gloria.utilities.errors import NotFittedError
from gloria.utilities.misc import cast_series_to_kind
from gloria.utilities.types import Distribution

### --- Class and Function Definitions --- ###


# ident is used as get and set for all built-in formats
def ident(x: Any) -> Any:
    return x


def get_pdseries(data_in: pd.Series) -> dict[str, Any]:
    """
    Converts a pandas series to a dictionary of json-serializable data types.
    The dtype_kind is saved as well as additional keyword so the series can be
    appropriately cast during deserialization.

    Parameters
    ----------
    data_in : pd.Series
        Pandas series to be converted to dictionary

    Returns
    -------
    dict[str, Any]
        JSON serializable dictionary containing data of pandas series.
    """
    # Convert directly to json string, wich is again loaded into a
    # dictionary.
    dict_out = json.loads(data_in.to_json(orient="split", date_format="iso"))
    return {**dict_out, _DTYPE_KIND: data_in.dtype.kind}


def set_pdseries(dict_in: dict[str, Any]) -> pd.Series:
    """
    Takes a dictionary as returned by get_pdseries() and restores the original
    pandas series.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Dictionary containing the series data

    Returns
    -------
    data_out : pd.Series
        Input data converted to pandas series.
    """
    # Save the dtype kind
    dtype_kind = dict_in.pop(_DTYPE_KIND)
    # Create the series
    data_out = pd.Series(**dict_in)
    # Cast the series values to the correct dtype kind
    if dtype_kind == "M":
        data_out = pd.to_datetime(data_out)
    else:
        data_out = cast_series_to_kind(data_out, dtype_kind)
    return data_out


def get_pddataframe(data_in: pd.DataFrame) -> dict[str, Any]:
    """
    Converts a pandas dataframe to a dictionary of json-serializable data
    types.

    Parameters
    ----------
    data_in : pd.DataFrame
        Pandas dataframe to be converted to dictionary

    Returns
    -------
    dict_out : dict[str, Any]
        JSON serializable dictionary containing data of pandas dataframe.
    """
    # Convert to a dictionary series-wise
    dict_out = data_in.to_dict("series")
    # Convert each series individually
    dict_out = {str(col): get_pdseries(rows) for col, rows in dict_out.items()}
    return dict_out  # type: ignore


def set_pddataframe(dict_in: dict[str, Any]) -> pd.DataFrame:
    """
    Takes a dictionary as returned by get_pddataframe() and restores the
    original pandas dataframe.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Dictionary containing the dataframe data

    Returns
    -------
    data_out : pd.DataFrame
        Input data converted to pandas dataframe.
    """
    # Convert the values of the input dictionary to pandas series and construct
    # the dataframe from it.
    if not dict_in:
        return pd.DataFrame()
    data_out = pd.DataFrame(
        {col: set_pdseries(data) for col, data in dict_in.items()}
    )
    return data_out


def get_regressors(data_in: dict[str, Regressor]) -> list[dict[str, Any]]:
    """
    Converts a dictionary of Regressor objects to a list of dictionaries of
    json-serializable data types.

    Parameters
    ----------
    data_in : dict[str, Regressor]
        Dictionary of Regressor objects to be converted

    Returns
    -------
    list_out : list[dict[str, Any]]
        List of json-serializable dictionaries representing the input data.
    """
    list_out = [regressor.to_dict() for regressor in data_in.values()]
    return list_out


def set_regressors(list_in: list[dict[str, Any]]) -> dict[str, Regressor]:
    """
    Takes a list of dictionaries as returned by get_regressors() and
    restores the original dictionary of Regressor objects

    Parameters
    ----------
    list_in : list[dict[str, Any]]
        list of dictionaries containing the Regressor object data

    Returns
    -------
    data_out : dict[str, Regressor]
        Input data converted to list of Regressor objects
    """
    data_out = {
        regressor["name"]: Regressor.from_dict(regressor)
        for regressor in list_in
    }

    return data_out


def get_events(data_in: dict[str, Regressor]) -> list[dict[str, Any]]:
    """
    Converts a dictionary of EventRegressor objects to a list of json-
    serializable data types.

    Parameters
    ----------
    data_in : dict[str, Regressor]
        Dictionary of Regressor objects to be converted to dictionary

    Returns
    -------
    list_out : list[dict[str, Any]]
        List of json-serializable dictionaries representing the input data.
    """
    list_out = [
        {
            "regressor": event_dict["regressor"].to_dict(),
            "include": event_dict["include"],
        }
        for event_dict in data_in.values()
    ]
    return list_out


def set_events(list_in: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Takes a list of dictionaries as returned by get_regressors() and
    restores the original dictionary of Regressor objects

    Parameters
    ----------
    list_in : list[dict[str, Any]]
        list of dictionaries containing the Regressor object data

    Returns
    -------
    data_out : dict[str, Regressor]
        Input data converted to list of Regressor objects
    """
    data_out = {
        event_dict["regressor"]["name"]: {
            "regressor": Regressor.from_dict(event_dict["regressor"]),
            "include": event_dict["include"],
        }
        for event_dict in list_in
    }

    return data_out


def get_protocols(data_in: list[Protocol]) -> list[dict[str, Any]]:
    """
    Converts a list of Protocol objects to a dictionary of json-serializable
    data types.

    Parameters
    ----------
    data_in : list[Regressor]
        List of Protocol objects to be converted to dictionary

    Returns
    -------
    list_out : list[dict[str, Any]]
        JSON serializable dictionary containing data of the input list.
    """
    list_out = [protocol.to_dict() for protocol in data_in]
    return list_out


def set_protocols(list_in: list[dict[str, Any]]) -> list[Protocol]:
    """
    Takes a list of dictionaries as returned by get_protocols() and
    restores the original list of Protocol objects

    Parameters
    ----------
    list_in : list[dict[str, Any]]
        list of dictionaries containing the Protocol object data

    Returns
    -------
    data_out : list[Protocol]
        Input data converted to list of Protocol objects
    """
    data_out = [Protocol.from_dict(protocol) for protocol in list_in]

    return data_out


def get_backend(data_in: ModelBackend) -> dict[str, Any]:
    """
    Converts a ModelBackend object to a dictionary of json-serializable
    data types. Note that the stan_fit attribute is omitted.

    Parameters
    ----------
    data_in : ModelBackend
        Model backend object to be converted to dictionary

    Returns
    -------
    dict_out : dict[str, Any]
        JSON serializable dictionary containing data of the ModelBackend object
    """
    # Convert all attributes listed in BACKEND_ATTRIBUTES using the pre-defined
    # serialization functions.
    dict_out = {
        attribute: functions[0](getattr(data_in, attribute))
        for attribute, functions in BACKEND_ATTRIBUTES.items()
    }
    return dict_out


def set_backend(dict_in: dict[str, Any], model: Distribution) -> ModelBackend:
    """
    Takes a dictionary as returned by get_backend() and restores the original
    ModelBackend object.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Dictionary containing the ModelBackend object data
    model : Literal[tuple(MODEL_MAP.keys())]
        The distribution model to be used. Can be any of 'poisson',
        'binomial constant n' or 'normal'

    Returns
    -------
    data_out : ModelBackend
        Input data converted to ModelBackend object.
    """
    # Initialize the ModelBackend object with desired model
    data_out = get_model_backend(model=model)

    # Convert all attributes listed in BACKEND_ATTRIBUTES using the pre-defined
    # deserialization functions.
    for attribute, functions in BACKEND_ATTRIBUTES.items():
        setattr(data_out, attribute, functions[1](dict_in[attribute]))

    return data_out


def get_dict(data_in: dict[str, Any]) -> dict[str, Any]:
    """
    Converts a dictionary containing numpy arrays to a dictionary of json-
    serializable data types. On that account numpy arrays are converted to
    simple lists.

    IMPORTANT: the get_dict/set_dict assumes the following things
    - The only non-default data type are numpy arrays
    - Numpy arrays only occur in the dictionaries top-level
    - The top-level of the dictionary has no lists, as these would be confused
      with numpy arrays during the set_dict reconstruction step.

    Parameters
    ----------
    data_in : dict[str, Any]
        Input dictionary to be converted.

    Returns
    -------
    dict[str, Any]
        JSON serializable dictionary containing data of original dictionary.
    """

    def empty_nested_list(d):
        if d <= 0:
            return None  # Edge case for non-positive dimensions
        return [] if d == 1 else [empty_nested_list(d - 1)]

    def dump_nparray(x):
        """
        Creates list from np array. If the array is not empty, use native numpy
        function, otherwise create a nested empty list of input array
        dimensionality
        """
        if x.size:
            return x.tolist()
        else:
            dim = len(x.shape)
            return empty_nested_list(dim)

    # Convert numpy arrays using dump_nparray. All other objects leave as is.
    dict_out = {
        k: dump_nparray(v) if isinstance(v, np.ndarray) else v
        for k, v in data_in.items()
    }
    return dict_out


def set_dict(dict_in: dict[str, Any]) -> dict[str, Any]:
    """
    Takes a dictionary as returned by get_dict() and restores the original
    dictionary by converting lists to numpy arrays.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Input dictionary to be restored.

    Returns
    -------
    dict[str, Any]
        Dictionary with lists replaced by numpy arrays
    """
    # If it has an __iter__-method, it can be cast to a numpy array. All other
    # items leave as is
    data_out = {
        k: np.array(v) if hasattr(v, "__iter__") else v
        for k, v in dict_in.items()
    }
    return data_out


def get_pydantic(data_in: BaseModel) -> dict[str, Any]:
    """
    Converts a pydantic model to a dictionary of json-serializable data types.

    Parameters
    ----------
    data_in : BaseModel
        Input pydantic model to be converted.

    Returns
    -------
    dict[str, Any]
        JSON serializable dictionary containing data of original pydantic
        model.
    """
    return get_dict(data_in.model_dump())


def set_pydantic(
    dict_in: dict[str, Any], model_class: type[BaseModel]
) -> BaseModel:
    """
    Takes a dictionary as returned by get_pydantic() and restores the original
    pydantic model.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Input dictionary to be restored.

    Returns
    -------
    dict[str, Any]
        Restored pydantic model.
    """
    return model_class(**set_dict(dict_in))


def model_to_dict(model: "Gloria") -> dict[str, Any]:
    """
    Converts Gloria object to a dictionary of JSON serializable types.

    Only works on fitted Gloria objects.

    Parameters
    ----------
    model : Gloria
        A fitted :class:`Gloria` object.

    Returns
    -------
    dict[str, Any]
        JSON serializable dictionary containing data of Gloria object.


    .. seealso::

        :meth:`Gloria.to_dict`
            An alias that calls ``model_to_dict`` on ``self``
        :func:`~gloria.model_from_dict`
            Reconstructs the fitted Gloria object from the output of
            ``model_to_dict``.
        :func:`~gloria.model_to_json`
            Additionaly dumps the dictionary to a JSON string or file.
        :func:`~gloria.model_from_json`
            Reconstructs the fitted Gloria object from the output of
            ``model_to_json``.

    """
    if not model.is_fitted:
        raise NotFittedError("Only fitted Gloria objects can be serialized.")

    # Convert all attributes listed in GLORIA_ATTRIBUTES using the pre-defined
    # serialization functions.
    model_dict = {
        attribute: functions[0](getattr(model, attribute))
        for attribute, functions in GLORIA_ATTRIBUTES.items()
    }

    return model_dict


def model_from_dict(model_dict: dict[str, Any]) -> "Gloria":
    """
    Restores a fitted Gloria model from a dictionary.

    The input dictionary must be the output of :func:`model_to_dict` or
    :meth:`Gloria.to_dict`.

    Parameters
    ----------
    model_dict : dict[str, Any]
        Dictionary containing the Gloria object data

    Returns
    -------
    model : Gloria
        Input data converted to Gloria object.


    .. seealso::

        :meth:`Gloria.from_dict`
            An alias as static method.
        :func:`~gloria.model_to_dict`
            Converts Gloria object to a dictionary of JSON serializable types.
        :func:`~gloria.model_to_json`
            Converts the fitted Gloria object into a JSON string or file.
        :func:`~gloria.model_from_json`
            Reconstructs the fitted Gloria object from the output of
            ``model_to_json``.
    """
    # Do not import Gloria in global namespace as this would cause circular
    # imports
    # Gloria
    from gloria.interface import Gloria

    # Create an empty Gloria instance
    model = Gloria()

    # Convert all attributes listed in GLORIA_ATTRIBUTES using the pre-defined
    # deserialization functions.
    for attribute, functions in GLORIA_ATTRIBUTES.items():
        # for the model_backend attribute, we also need to pass the model name
        # as it is needed to create the ModelBackend instance
        if attribute == "model_backend":
            setattr(
                model,
                attribute,
                functions[1](model_dict[attribute], model_dict["model"]),
            )
        else:
            setattr(model, attribute, functions[1](model_dict[attribute]))

    return model


def model_to_json(
    model: "Gloria",
    filepath: Optional[Union[Path, str]] = None,
    **kwargs: Any,
) -> str:
    """
    Converts a Gloria object to a JSON string.

    Only works on fitted Gloria objects. If desired the model is
    additionally dumped to a .json-file.

    Parameters
    ----------
    model : Gloria
        The fitted Gloria object.
    filepath : Optional[Union[Path, str]], optional
        Filepath of the target .json-file. If ``None`` (default) no output-
        file will be written.
    **kwargs : Any
        Keyword arguments which are passed through to :func:`json.dump` and
        :func:`json.dumps`

    Raises
    ------
    ValueError
        In case the given filepath does not have .json extension.

    Returns
    -------
    str
        JSON string containing the model data of the fitted Gloria object.




    .. seealso::

        :meth:`Gloria.to_json`
            An alias that calls ``model_to_json`` on ``self``
        :func:`~gloria.model_from_json`
            Reconstructs the fitted Gloria object from the output of
            ``model_to_json``.
        :func:`~gloria.model_to_dict`
            Converts the fitted Gloria object to a dictionary of JSON
            serializable types.
        :func:`~gloria.model_from_dict`
            Reconstructs the fitted Gloria object from the output of
            ``model_to_dict``.

    """
    # Convert model to json-serializavle dictionary
    model_dict = model_to_dict(model)
    # If a filepath was given
    if filepath:
        # Cast to Path object in case filepath is a string
        filepath = Path(filepath)
        # If filepath is not .json-file, raise error
        if filepath.suffix != ".json":
            raise ValueError("File extension must be .json.")
        # In case target folder doesn't exist, create it
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Dump the dictionary to json-file
        with open(filepath, "w") as file:
            json.dump(model_dict, file, **kwargs)  # type: ignore
    # And dump it to string anyway
    return json.dumps(model_dict, **kwargs)  # type: ignore


def model_from_json(
    model_json: Union[Path, str], return_as: Literal["dict", "model"] = "model"
) -> Union[dict[str, Any], "Gloria"]:
    """
    Restores a fitted Gloria model from a json string or file.

    The input json string must be the output of :func:`model_to_json` or
    :meth:`Gloria.to_json`. If the input is a json-file, its contents is
    first read to a json string.

    Parameters
    ----------
    model_json : Union[Path, str]
        Filepath of .json-model file or string containing the data.
    return_as : Literal['dict', 'model'], optional
        If ``dict`` (default), the model is returned in dictionary format,
        if ``model`` as fitted Gloria object.

    Raises
    ------
    ValueError
        Two ValueErrors are possible:
        1. In case the given filepath does not have .json extension
        2. If ``return_as`` is neither ``"dict"`` nor ``"model"``

    Returns
    -------
    Union[dict[str, Any], Gloria]
        Gloria object or dictionary representing it based on input json data.


    .. seealso::

        :meth:`Gloria.from_json`
            An alias static method.
        :func:`~gloria.model_to_json`
            Converts the fitted Gloria object into a JSON string or file.
        :func:`~gloria.model_to_dict`
            Converts the fitted Gloria object to a dictionary of JSON
            serializable types.
        :func:`~gloria.model_from_dict`
            Reconstructs the fitted Gloria object from the output of
            ``model_to_dict``.
    """
    # If json-data are a Path object, read the file
    if isinstance(model_json, Path):
        # If filepath is not .json-file, raise error
        if model_json.suffix != ".json":
            raise ValueError("File extension must be .json.")
        try:
            with open(model_json, "r") as file:
                model_dict = json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Specified model file '{model_json}' does not exist"
            ) from e
    # If json-data are a Path object, load the json-string
    else:
        model_dict = json.loads(model_json)

    # Return model as desired type
    if return_as == "model":
        return model_from_dict(model_dict)
    elif return_as == "dict":
        return model_dict
    else:
        raise ValueError(f"Return type '{return_as}' is not supported.")


### --- Global Constants Definitions --- ###

# Simple attributes are those that are built-in data types of the Glora object
# ie. they are json-serializable without further processing
SIMPLE_GLORIA_ATTRIBUTES = [
    "model",
    "timestamp_name",
    "metric_name",
    "capacity_name",
    "n_changepoints",
    "changepoint_range",
    "seasonality_prior_scale",
    "event_prior_scale",
    "changepoint_prior_scale",
    "dispersion_prior_scale",
    "interval_width",
    "trend_samples",
    "prior_scales",
    "fit_kwargs",
    "vectorized",
]

# All Gloria attributes we wish to serialize. The dictionary maps attribute
# name to a tuple of two functions. The first function is the serializer, the
# second one the deserializer
GLORIA_ATTRIBUTES: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    **{attribute: (ident, ident) for attribute in SIMPLE_GLORIA_ATTRIBUTES},
    "changepoints": (get_pdseries, set_pdseries),
    "changepoints_int": (get_pdseries, set_pdseries),
    "first_timestamp": (str, pd.Timestamp),
    "last_timestamp": (str, pd.Timestamp),
    "sampling_period": (str, pd.Timedelta),
    "history": (get_pddataframe, set_pddataframe),
    "X": (get_pddataframe, set_pddataframe),
    "seasonalities": (get_regressors, set_regressors),
    "protocols": (get_protocols, set_protocols),
    "model_backend": (get_backend, set_backend),
    "external_regressors": (get_regressors, set_regressors),
    "events": (get_events, set_events),
    "_config": (get_dict, set_dict),
}

# Same as Gloria attributes, only for the nested model_backend object
SIMPLE_BACKEND_ATTRIBUTES = [
    "model_name",
    "linked_scale",
    "linked_offset",
    "use_laplace",
]


BACKEND_ATTRIBUTES = {
    **{attribute: (ident, ident) for attribute in SIMPLE_BACKEND_ATTRIBUTES},
    "stan_data": (get_pydantic, lambda x: set_pydantic(x, ModelInputData)),
    "stan_inits": (get_pydantic, lambda x: set_pydantic(x, ModelParams)),
    "fit_params": (get_dict, set_dict),
}
