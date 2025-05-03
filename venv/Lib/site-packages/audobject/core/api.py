import os
import typing
import warnings

import oyaml as yaml

from audobject.core import define
from audobject.core.object import Object
import audobject.core.utils as utils


kwargs_deprecation_warning = (
    "The use of **kwargs is deprecated "
    "and will be removed with version 1.0.0. "
    "Use 'override_args' instead."
)


def from_dict(  # noqa: D417
        d: typing.Dict[str, typing.Any],
        root: str = None,
        *,
        auto_install: bool = False,
        override_args: typing.Dict[str, typing.Any] = None,
        **kwargs,
) -> 'Object':
    r"""Create object from dictionary.

    Args:
        d: dictionary representing the object
        root: if dictionary was read from a file, set to source directory
        auto_install: install missing packages needed to create the object
        override_args: override arguments in ``d`` or
            default values of hidden arguments

    Returns:
        object

    Raises:
        RuntimeError: if a mandatory argument of the object
            is missing in the dictionary ``d``

    """
    override_args = override_args or {}

    if kwargs:
        warnings.warn(
            kwargs_deprecation_warning,
            category=UserWarning,
            stacklevel=2,
        )
        for key, value in kwargs.items():
            override_args[key] = value

    name = next(iter(d))
    cls, version, installed_version = utils.get_class(
        name,
        auto_install,
    )

    params = {}
    for key, value in d[name].items():
        params[key] = _decode_value(
            value,
            auto_install,
            override_args,
        )

    object, params = utils.get_object(
        cls,
        version,
        installed_version,
        params,
        root,
        override_args,
    )

    if isinstance(object, Object):
        # create attribute to signal that object was loaded
        object.__dict__[define.OBJECT_LOADED] = None

    # If function has init decorator, add stream to parameters.
    # The additional parameter will be popped by the decorator
    # before the object is created.
    # If the class does not have a decorator
    # a TypeError will be raised since we call
    # the class with an unexpected argument.
    # Unfortunately, there is no straight-forward way
    # to check if a method of a class has a decorator
    # so we have to use a try-except block
    try:
        params[define.ROOT_ATTRIBUTE] = root
        object.__init__(**params)
    except TypeError:
        params.pop(define.ROOT_ATTRIBUTE)
        object.__init__(**params)

    return object


def from_yaml(  # noqa: D417
        path_or_stream: typing.Union[str, typing.IO],
        *,
        auto_install: bool = False,
        override_args: typing.Dict[str, typing.Any] = None,
        **kwargs,
) -> 'Object':
    r"""Create object from YAML file.

    Args:
        path_or_stream: file path or stream
        auto_install: install missing packages needed to create the object
        override_args: override arguments in the YAML file or
            default values of hidden arguments

    Returns:
        object

    """
    override_args = override_args or {}

    if kwargs:
        warnings.warn(
            kwargs_deprecation_warning,
            category=UserWarning,
            stacklevel=2,
        )
        for key, value in kwargs.items():
            override_args[key] = value

    if isinstance(path_or_stream, str):
        with open(path_or_stream, 'r') as fp:
            return from_yaml(
                fp,
                override_args=override_args,
                auto_install=auto_install,
            )
    return from_dict(
        yaml.load(path_or_stream, yaml.Loader),
        auto_install=auto_install,
        root=os.path.dirname(path_or_stream.name),
        override_args=override_args,
    )


def from_yaml_s(  # noqa: D417
        yaml_string: str,
        *,
        auto_install: bool = False,
        override_args: typing.Dict[str, typing.Any] = None,
        **kwargs,
) -> 'Object':
    r"""Create object from YAML string.

    Args:
        yaml_string: YAML string
        auto_install: install missing packages needed to create the object
        override_args: override arguments in the YAML string or
            default values of hidden arguments

    Returns:
        object

    """
    override_args = override_args or {}

    if kwargs:
        warnings.warn(
            kwargs_deprecation_warning,
            category=UserWarning,
            stacklevel=2,
        )
        for key, value in kwargs.items():
            override_args[key] = value

    return from_dict(
        yaml.load(yaml_string, yaml.Loader),
        auto_install=auto_install,
        override_args=override_args,
    )


def _decode_value(
        value_to_decode: typing.Any,
        auto_install: bool,
        override_args: typing.Dict[str, typing.Any],
) -> typing.Any:
    r"""Decode value."""
    if value_to_decode:  # not empty
        if isinstance(value_to_decode, list):
            return [
                _decode_value(
                    v,
                    auto_install,
                    override_args,
                ) for v in value_to_decode
            ]
        elif isinstance(value_to_decode, dict):
            name = next(iter(value_to_decode))
            if isinstance(name, Object) or utils.is_class(name):
                return from_dict(
                    value_to_decode,
                    auto_install=auto_install,
                    override_args=override_args,
                )
            else:
                return {
                    k: _decode_value(
                        v,
                        auto_install,
                        override_args,
                    ) for k, v in value_to_decode.items()
                }
    return value_to_decode
