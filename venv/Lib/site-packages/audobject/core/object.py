import collections.abc
import inspect
import os
import typing
import warnings

import oyaml as yaml

import audeer
from audobject.core import define
from audobject.core import resolver
from audobject.core import utils


class Object:
    r"""Base class for objects that can be serialized to YAML.

    Examples:
        >>> class Foo(Object):
        ...    def __init__(self, bar: str):
        ...        self.bar = bar
        >>> foo = Foo('hello object!')
        >>> print(foo)
        $audobject.core.object.Foo:
          bar: hello object!

    """
    def __init__(
            self,
            **kwargs,
    ):
        self.__dict__[define.KEYWORD_ARGUMENTS] = list(kwargs)

    @property
    def arguments(self) -> typing.Dict[str, typing.Any]:
        r"""Returns arguments that are serialized.

        Returns:
            Dictionary of arguments and their values.

        Raises:
            RuntimeError: if arguments are found that are not assigned to
                attributes of the same name

        Examples:
            >>> import audobject.testing
            >>> o = audobject.testing.TestObject('test', point=(1, 1))
            >>> o.arguments
            {'name': 'test', 'point': (1, 1)}

        """  # noqa: E501
        signature = inspect.signature(self.__init__)

        # non-keyword arguments from __init__
        names = [p.name for p in signature.parameters.values()
                 if not p.name == 'kwargs']

        # additional keyword arguments
        if define.KEYWORD_ARGUMENTS in self.__dict__:
            names.extend(self.__dict__[define.KEYWORD_ARGUMENTS])

        # remove hidden and borrowed attributes
        borrowed = self.borrowed_arguments
        hidden = self.hidden_arguments
        names = [
            name for name in names if (
                (name != 'self')
                and (not name.startswith('_'))
                and (name not in hidden)
                and (name not in borrowed.values())
            )
        ]

        # check for missing attributes
        missing = []
        for name in names:
            if (name not in self.__dict__) and (name not in borrowed):
                missing.append(name)
        if len(missing) > 0:
            raise RuntimeError(
                'Arguments '
                f'{missing} '
                'of '
                f'{self.__class__} '
                'not assigned to attributes of same name.'
            )

        # check borrowed attributes
        for key, value in borrowed.items():
            can_borrow = False
            if hasattr(self, value):
                if hasattr(self.__dict__[value], key):
                    can_borrow = True
                elif isinstance(self.__dict__[value], collections.abc.Mapping):
                    can_borrow = True
            if not can_borrow:
                raise RuntimeError(
                    'Cannot borrow attribute '
                    f"'{key}' "
                    'from '
                    f"'self.{value}'."
                )

        # pick arguments from self and borrowed attributes
        args = {
            name: self.__dict__[name] for name in names if (
                (name in self.__dict__)
            )
        }
        for key, value in borrowed.items():
            if hasattr(self.__dict__[value], key):
                args[key] = self.__dict__[value].__dict__[key]
            else:
                args[key] = self.__dict__[value][key]

        return args

    @property
    def borrowed_arguments(self) -> typing.Dict[str, str]:
        r"""Returns borrowed arguments.

        Returns:
            Dictionary with borrowed arguments.

        """
        if define.BORROWED_ATTRIBUTES in self.__dict__:
            borrowed = self.__dict__[define.BORROWED_ATTRIBUTES]
        else:
            borrowed = {}
        return borrowed

    @property
    def hidden_arguments(self) -> typing.List[str]:
        r"""Returns hidden arguments.

        Returns:
            List with names of hidden arguments.

        """
        if define.HIDDEN_ATTRIBUTES in self.__dict__:
            names = self.__dict__[define.HIDDEN_ATTRIBUTES]
        else:
            names = []
        return names

    @property
    def id(self) -> str:
        r"""Object identifier.

        The ID of an object ID is created from its non-hidden arguments.

        Returns:
            object identifier

        Examples:
            >>> class Foo(Object):
            ...    def __init__(self, bar: str):
            ...        self.bar = bar
            >>> foo1 = Foo('I am unique!')
            >>> foo1.id
            '893df240-babe-d796-cdf1-c436171b7a96'
            >>> foo2 = Foo('I am different!')
            >>> foo2.id
            '9303f2a5-bfc9-e5ff-0ffa-a9846e2d2190'
            >>> foo3 = Foo('I am unique!')
            >>> foo1.id == foo3.id
            True

        """
        string = self.to_yaml_s(include_version=False)
        return audeer.uid(from_string=string)

    @property
    def is_loaded_from_dict(self) -> bool:
        r"""Check if object was loaded from a dictionary.

        Returns ``True``
        if object was initialized
        from a dictionary,
        e.g. after loading it from a YAML file.

        Returns:
            ``True`` if object was loaded from a dictionary,
                otherwise ``False``

        """
        return define.OBJECT_LOADED in self.__dict__

    @staticmethod
    @audeer.deprecated(
        removal_version='1.0.0',
        alternative='audobject.from_dict',
    )
    def from_dict(  # noqa: D102
            d: typing.Dict[str, typing.Any],
            root: str = None,
            **kwargs,
    ) -> 'Object':  # pragma: no cover
        from audobject.core.api import from_dict
        return from_dict(d, root, **kwargs)

    @staticmethod
    @audeer.deprecated(
        removal_version='1.0.0',
        alternative='audobject.from_yaml',
    )
    def from_yaml(  # noqa: D102
            path_or_stream: typing.Union[str, typing.IO],
            **kwargs,
    ) -> 'Object':  # pragma: no cover
        from audobject.core.api import from_yaml
        return from_yaml(path_or_stream, **kwargs)

    @staticmethod
    @audeer.deprecated(
        removal_version='1.0.0',
        alternative='audobject.from_yaml_s',
    )
    def from_yaml_s(  # noqa: D102
            yaml_string: str,
            **kwargs,
    ) -> 'Object':  # pragma: no cover
        from audobject.core.api import from_yaml_s
        return from_yaml_s(yaml_string, **kwargs)

    @property
    def resolvers(self) -> typing.Dict[str, resolver.Base]:
        r"""Return resolvers.

        Returns:
            Dictionary with resolvers.

        """
        if hasattr(self, define.CUSTOM_VALUE_RESOLVERS):
            return self.__dict__[define.CUSTOM_VALUE_RESOLVERS]
        return {}

    @property
    def short_id(self) -> str:
        r"""Short object identifier.

        The short ID consists of eight characters
        and is created from its non-hidden arguments.

        Returns:
            short object identifier

        Examples:
            >>> class Foo(Object):
            ...    def __init__(self, bar: str):
            ...        self.bar = bar
            >>> foo1 = Foo('I am unique!')
            >>> foo1.id
            '893df240-babe-d796-cdf1-c436171b7a96'
            >>> foo1.short_id
            '171b7a96'
            >>> foo2 = Foo('I am different!')
            >>> foo2.short_id
            '6e2d2190'
            >>> foo3 = Foo('I am unique!')
            >>> foo1.short_id == foo3.short_id
            True

        """
        string = self.to_yaml_s(include_version=False)
        return audeer.uid(from_string=string, short=True)

    def to_dict(
            self,
            *,
            include_version: bool = True,
            flatten: bool = False,
            root: str = None,
    ) -> typing.Dict[str, resolver.DefaultValueType]:
        r"""Converts object to a dictionary.

        Includes items from :attr:`audobject.Object.arguments`.
        If an argument has a resolver, its value is encoded.
        Usually, the object can be re-instantiated using
        :meth:`audobject.Object.from_dict`.
        However, if ``flatten=True``, this is not possible.

        Args:
            include_version: add version to class name
            flatten: flatten the dictionary
            root: if file is written to disk, set to target directory

        Returns:
            dictionary that represent the object

        Examples:
            >>> import audobject.testing
            >>> o = audobject.testing.TestObject('test', point=(1, 1))
            >>> o.to_dict(include_version=False)
            {'$audobject.core.testing.TestObject': {'name': 'test', 'point': [1, 1]}}
            >>> o.to_dict(flatten=True)
            {'name': 'test', 'point.0': 1, 'point.1': 1}

        """  # noqa: E501
        name = utils.create_class_key(self.__class__, include_version)

        d = {
            key: self._encode_variable(
                key,
                value,
                include_version,
                root,
            )
            for key, value in self.arguments.items()
        }

        return Object._flatten(d) if flatten else {name: d}

    def to_yaml(
            self,
            path_or_stream: typing.Union[str, typing.IO],
            *,
            include_version: bool = True,
    ):
        r"""Save object to YAML file.

        Args:
            path_or_stream: file path or stream
            include_version: add version to class name

        """
        if isinstance(path_or_stream, str):
            path_or_stream = audeer.safe_path(path_or_stream)
            root = os.path.dirname(path_or_stream)
            audeer.mkdir(root)
            with open(path_or_stream, 'w') as fp:
                return self.to_yaml(fp, include_version=include_version)
        else:
            return yaml.dump(
                self.to_dict(
                    include_version=include_version,
                    root=os.path.dirname(path_or_stream.name),
                ),
                path_or_stream,
            )

    def to_yaml_s(
            self,
            *,
            include_version: bool = True,
    ) -> str:
        r"""Convert object to YAML string.

        Args:
            include_version: add version to class name

        Returns:
            YAML string

        Examples:
            >>> import audobject.testing
            >>> o = audobject.testing.TestObject('test', point=(1, 1))
            >>> print(o.to_yaml_s(include_version=False))
            $audobject.core.testing.TestObject:
              name: test
              point:
              - 1
              - 1

        """  # noqa: E501
        return yaml.dump(self.to_dict(include_version=include_version))

    def _encode_variable(
            self,
            name: str,
            value: typing.Any,
            include_version: bool,
            root: typing.Optional[str],
    ):
        r"""Encode value.

        Encode a value
        by first looking for a custom resolver,
        otherwise switch to default encoder.

        """
        value = self._resolve_value(name, value, root)
        return Object._encode_value(value, include_version)

    @staticmethod
    def _encode_value(
            value: typing.Any,
            include_version: bool,
    ):
        r"""Default value encoder."""
        if value is None:
            return None
        elif isinstance(value, Object) or utils.is_class(value):
            return value.to_dict(include_version=include_version)
        elif isinstance(value, define.DEFAULT_VALUE_TYPES):
            return value
        elif isinstance(value, list):
            return [
                Object._encode_value(
                    item, include_version
                ) for item in value
            ]
        elif isinstance(value, dict):
            return {
                Object._encode_value(key, include_version):
                    Object._encode_value(val, include_version)
                for key, val in value.items()
            }
        elif callable(value):
            raise RuntimeError(
                f"Cannot encode type '{type(value)}'."
            )
        else:
            warnings.warn(
                f"No default encoding exists for type '{type(value)}'. "
                f"Consider to register a custom resolver.",
                RuntimeWarning,
            )
            return value

    @staticmethod
    def _flatten(
        d: typing.Dict[str, resolver.DefaultValueType],
    ):
        r"""Flattens a dictionary."""
        def helper(dict_in: dict, dict_out: dict, prefix: str):
            for key, value in dict_out.items():
                if utils.is_class(key):
                    helper(dict_in, value, prefix)
                elif isinstance(value, (list, tuple)):
                    helper(
                        dict_in,
                        {idx: item for idx, item in enumerate(value)},
                        f'{prefix}.{key}' if prefix else key,
                    )
                elif isinstance(value, dict):
                    helper(
                        dict_in,
                        value,
                        f'{prefix}.{key}' if prefix else key,
                    )
                else:
                    if prefix:
                        key = f'{prefix}.{key}'
                    dict_in[key] = value

        ret = {}
        helper(ret, d, '')
        return ret

    def _resolve_value(
            self,
            name: str,
            value: typing.Any,
            root: typing.Optional[str],
    ) -> resolver.DefaultValueType:
        if value is not None and name in self.resolvers:
            # let resolver know if we write to a stream
            self.resolvers[name].__dict__[define.ROOT_ATTRIBUTE] = root
            value = self.resolvers[name].encode(value)
        return value

    def __hash__(self) -> int:
        r"""Hash based on object ID."""
        return hash(self.id)

    def __eq__(self, other: 'Object') -> bool:
        r"""Check if two objects are equal."""
        if isinstance(other, type(self)):
            return self.id == other.id
        else:
            # it might happen that we need to compare
            # to other non-supported types, compare
            # https://github.com/audeering/audinterface/issues/68
            return False

    def __repr__(self) -> str:  # noqa: D105
        return str(self.to_dict(include_version=False))

    def __str__(self) -> str:  # noqa: D105
        return self.to_yaml_s(include_version=False)
