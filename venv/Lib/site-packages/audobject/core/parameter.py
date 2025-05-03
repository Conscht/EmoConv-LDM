import argparse
import os
import typing

import packaging.specifiers
import packaging.version

from audobject.core.decorator import init_decorator
import audobject.core.define as define
from audobject.core.dictionary import Dictionary
from audobject.core.object import Object
import audobject.core.resolver as resolver


class Parameter(Object):
    r"""Single parameter.

    A parameter steers the behaviour of a an object (e.g. tuning a model).
    It has a specific type and default value, possibly one of a set of choices.
    It is possible to bound a parameter to a specific version (range).

    Args:
        value_type: data type, one of
            (``None``, ``str``, ``int``, ``float``, ``bool``)
            or a ``list`` or ``dict`` of those
        description: description
        value: value, if ``None`` set to ``default_value``
        default_value: default value
        choices: list with choices
        version: string defining in which versions the parameter is used
            (e.g. ``'>=1.0.0,<2.0.0,!=1.5.0'``),
            ``None`` matches any version

    Raises:
        TypeError: if value has an invalid type
        ValueError: if value is not in choices

    Examples:
        >>> foo = Parameter(
        ...     value_type=str,
        ...     description='some parameter',
        ...     default_value='bar',
        ...     choices=['bar', 'Bar', 'BAR'],
        ...     version='>=1.0.0,<2.0.0',
        ... )
        >>> # check version
        >>> '1.5.0' in foo
        True
        >>> '2.0.0' in foo
        False
        >>> # get/set value
        >>> foo.value
        'bar'
        >>> foo.set_value('Bar')
        >>> foo.value
        'Bar'
        >>> # set invalid value
        >>> try:
        ...     foo.set_value('par')
        ... except ValueError as ex:
        ...     print(ex)
        Invalid value 'par', expected one of ['bar', 'Bar', 'BAR'].

    """  # noqa: E501

    @init_decorator(
        resolvers={
            'value_type': resolver.Type,
        }
    )
    def __init__(
            self,
            *,
            value_type: type = str,
            description: str = '',
            value: typing.Any = None,
            default_value: typing.Any = None,
            choices: typing.Sequence[typing.Any] = None,
            version: str = None,
    ):

        self.value_type = value_type
        r"""Data type of parameter"""
        self.description = description
        r"""Value of parameter, use 'set_value' for type checking"""
        self.value = None
        r"""Description of parameter"""
        self.default_value = default_value
        r"""Default value of parameter"""
        self.choices = choices
        r"""Possible choices for parameter"""
        self.version = version
        r"""Versions in which the parameter is used"""

        if default_value is not None:
            self._check_value(default_value)

        if value is not None:
            self.set_value(value)
        else:
            self.set_value(default_value)

    def __contains__(self, version: typing.Optional[str]) -> bool:
        r"""Check if parameter is in parameter object."""
        if version is None or self.version is None:
            return True

        version = packaging.version.parse(version)
        version_range = packaging.specifiers.SpecifierSet(self.version)

        return version in version_range

    def set_value(self, value: typing.Any):
        r"""Sets a new value.

        Applies additional checks, e.g. if value is of the expected type.

        Args:
            value: new value

        Raises:
            TypeError: if value has an invalid type
            ValueError: if value is not in choices

        """
        self._check_value(value)
        self.value = value

    def _check_value(self, value: typing.Any):
        r"""Check if value matches expected type."""
        if value is not None and not isinstance(value, self.value_type):
            raise TypeError(
                f"Invalid type '{type(value)}', "
                f"expected {self.value_type}."
            )
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Invalid value '{value}', "
                f"expected one of {self.choices}."
            )


class Parameters(Dictionary):
    r"""List of parameters.

    Args:
        **kwargs: :class:`audobject.Parameter` objects

    Examples:
        >>> # create parameter
        >>> foo = Parameter(
        ...     value_type=str,
        ...     description='foo',
        ... )
        >>> # create list of parameters
        >>> params = Parameters(foo=foo)
        >>> # get / set parameter value
        >>> params.foo = 'bar'
        >>> params.foo
        'bar'
        >>> # add another parameter to list
        >>> pi = Parameter(
        ...     value_type=float,
        ...     description='mathematical constant',
        ...     value=3.14159265359,
        ... )
        >>> params['pi'] = pi
        >>> print(params)
        Name  Value          Default  Choices  Description            Version
        ----  -----          -------  -------  -----------            -------
        foo   bar            None     None     foo                    None
        pi    3.14159265359  None     None     mathematical constant  None
        >>> # convert to dictionary
        >>> params()
        {'foo': 'bar', 'pi': 3.14159265359}

    """  # noqa: E501
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

    def filter_by_version(
            self,
            version: str,
    ) -> 'Parameters':
        r"""Filter parameters by version.

        Returns a subset including only those parameters that match a given
        version.

        Args:
            version: version string

        Returns:
            List with matching parameters

        """
        params = Parameters()
        for name, param in self.items():
            if version in param:
                params[name] = param
        return params

    def from_command_line(
            self,
            args: argparse.Namespace,
    ) -> 'Parameters':
        r"""Parse parameters from command line parser.

        Args:
            args: command line arguments

        """
        for key, value in args.__dict__.items():
            if key in self.keys():
                self.__setattr__(key, value)
        return self

    def to_command_line(
            self,
            parser: argparse.ArgumentParser,
    ):
        r"""Add parameters to command line parser.

        .. note:: Command line arguments are named --<name>.

        Args:
            parser: command line parser

        """
        for name, param in self.items():

            if param.version is not None:
                help = f'{param.description} (version: {param.version})'
            else:
                help = param.description

            if param.value_type == bool:
                parser.add_argument(
                    f'--{name}',
                    action='store_true',
                    help=help,
                )
            else:
                parser.add_argument(
                    f'--{name}',
                    type=param.value_type,
                    default=param.default_value,
                    choices=param.choices,
                    help=help,
                )

    def to_path(
            self,
            *,
            delimiter: str = os.path.sep,
            include: typing.Sequence[str] = None,
            exclude: typing.Sequence[str] = None,
            sort: bool = False,
    ):
        r"""Creates path from parameters.

        Args:
            delimiter: delimiter character
            include: list of parameters to include
            exclude: list of parameters to exclude
            sort: sort parameters by name

        """
        names = self.keys()
        if sort:
            names = sorted(names)
        d = {
            name: self[name].value for name in names
        }
        exclude = set(exclude or [])
        if include is not None:
            for key in d:
                if key not in include:
                    exclude.add(key)
        for key in exclude:
            d.pop(key)
        parts = ['{}[{}]'.format(key, value) for key, value in d.items()]
        return delimiter.join(parts)

    def __call__(self):
        r"""Return parameters as dictionary."""
        return {
            name: param.value for name, param in self.items()
        }

    def __getattribute__(self, name) -> typing.Any:  # noqa: D105
        if not name == '__dict__' and name in self.__dict__:
            p = self.__dict__[name]
            return p.value
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: typing.Any):  # noqa: D105
        p = self.__dict__[name]
        p.set_value(value)

    def __str__(self):  # pragma: no cover  # noqa: D105
        table = [
            [
                'Name', 'Value', 'Default', 'Choices',
                'Description', 'Version',
            ],
            [
                '----', '-----', '-------', '-------',
                '-----------', '-------',
            ]
        ]
        for name, p in self.items():
            if name != define.ROOT_ATTRIBUTE:
                table.append(
                    [
                        name, p.value, p.default_value, p.choices,
                        p.description, p.version,
                    ]
                )
        padding = 2
        # Longest string in each column
        transposed_table = [list(x) for x in zip(*table)]
        col_width = [
            len(max([str(word) for word in row], key=len)) + padding
            for row in transposed_table
        ]
        # Don't pad the last column
        col_width[-1] -= padding
        row = [
            ''.join(
                str(word).ljust(width) for word, width in zip(row, col_width)
            )
            for row in table
        ]
        return '\n'.join(row)
