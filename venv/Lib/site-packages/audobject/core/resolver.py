import ast
import datetime
import inspect
import os
import textwrap
import types
import typing
import warnings

import audeer
import audobject.core.define as define


DefaultValueType = typing.Union[
    bool,
    datetime.datetime,
    dict,
    float,
    int,
    list,
    None,
    str,
]


class Base:
    r"""Abstract resolver class.

    Implement for arguments that are not one of:

    * ``bool``
    * ``datetime.datetime``
    * ``dict``
    * ``float``
    * ``int``
    * ``list``
    * ``None``
    * ``Object``
    * ``str``

    """
    def __init__(self):
        self.__dict__[define.ROOT_ATTRIBUTE] = None

    @property
    def root(self) -> typing.Optional[str]:
        r"""Root folder.

        Returns root folder when object is serialized to or from a file,
        otherwise ``None`` is returned.

        Returns:
            root directory

        """
        return self.__dict__[define.ROOT_ATTRIBUTE]

    def decode(self, value: DefaultValueType) -> typing.Any:
        r"""Decode value.

        Takes the encoded value and converts it back to its original type.

        Args:
            value: value to decode

        Returns:
            decoded value

        """
        raise NotImplementedError  # pragma: no cover

    def encode(self, value: typing.Any) -> DefaultValueType:
        r"""Encode value.

        The type of the returned value must be one of:

        * ``bool``
        * ``datetime.datetime``
        * ``dict``
        * ``float``
        * ``int``
        * ``list``
        * ``None``
        * ``Object``
        * ``str``

        Args:
            value: value to encode

        Returns:
            encoded value

        """
        raise NotImplementedError  # pragma: no cover

    def encode_type(self) -> type:
        r"""Return encoded type.

        Returns:
            encoded type

        """
        raise NotImplementedError  # pragma: no cover


class FilePath(Base):
    r"""File path resolver.

    Turns file path to a relative path
    when object is serialized to a file
    and expands it again during reading.

    Examples:
        >>> resolver = FilePath()
        >>> resolver._object_root_ = '/some/root'  # usually set by object
        >>> value = '/some/where/else'
        >>> encoded_value = resolver.encode(value)
        >>> encoded_value  # doctest: +SKIP
        '../where/else'
        >>> resolver.decode(encoded_value)  # doctest: +SKIP
        '/some/where/else'

    """

    def decode(self, value: str) -> str:
        r"""Decode file path.

        If object is read from a file,
        this will convert a relative file path
        to an absolute path by expanding it
        with the source directory.

        Args:
            value: relative file path

        Returns:
            expanded file path

        """
        if self.root is not None:
            root = self.root
            value = os.path.join(root, value)
            value = audeer.safe_path(value)
        return value

    def encode(self, value: str) -> str:
        r"""Encode file path.

        If object is written to a file,
        this will convert a file path
        to a path that is relative to the
        target directory.

        Args:
            value: original file path

        Returns:
            relative file path

        """
        if self.root is not None:
            root = self.root
            value = os.path.relpath(value, root)
        return value

    def encode_type(self) -> type:
        r"""Return encoded type.

        Returns:
            encoded type

        """
        return str


class Function(Base):
    r"""Function resolver.

    Encodes source code of function and
    dynamically evaluates it when the value is decoded again.

    Note that a decoded function
    might raise a :class:`NameError`,
    if it relies on objects or functions
    that are not defined or imported
    inside the function.
    For instance,
    the following example will raise an error
    since ``plus_1()`` relies on ``_plus_1()``,
    which is defined outside the function:

    .. code-block:: python

        def _plus_1(x):
            return x + 1

        def plus_1(x):
            return _plus_1(x)  # calls local function -> not serializable

        resolver = Function()
        encoded_value = resolver.encode(plus_1)
        del _plus_1
        decoded_value = resolver.decode(encoded_value)
        decoded_value(1)

    Examples:
        >>> resolver = Function()
        >>> def func(x):
        ...     return x * 2
        >>> func(5)
        10
        >>> encoded_value = resolver.encode(func)
        >>> encoded_value
        'def func(x):\n    return x * 2\n'
        >>> decoded_value = resolver.decode(encoded_value)
        >>> decoded_value(5)
        10

    """

    def decode(self, value: str) -> typing.Callable:
        r"""Decode (lambda) function.

        Args:
            value: source code

        Returns:
            function object

        """
        func = None

        # We must dynamically create the function
        # from the original source code we stored in YAML.
        # For a regular function we can do this
        # by calling ``exec()`` with a local namespace directory.
        # This will create the function in the namespace
        # from where we can return it.
        # This preserve defaults and keyword-only arguments.
        # For lambda expression this is not possible,
        # as we would end up with an empty namespace
        # (a lambda has no name!).
        # Therefore we first compile the code
        # and then use ``types.FunctionType()``
        # to create the function object.
        # This does not preserve defaults and keyword-only arguments,
        # but fortunately this is not relevant for lambda expressions.

        if value.startswith('lambda'):
            code = compile(value, '<string>', 'exec')
            for var in code.co_consts:
                if isinstance(var, types.CodeType):
                    func = types.FunctionType(var, globals())
        else:
            namespace = {}
            exec(value, globals(), namespace)
            func_name = next(iter(namespace))
            func = namespace[func_name]

        # we cannot inspect the source code of
        # dynamically defined functions so we attach it
        func.__source__ = value

        return func

    def encode(
            self,
            value: typing.Callable,
    ) -> typing.Union[str, object]:
        r"""Encode (lambda) function.

        Args:
            value: function object

        Returns:
            source code

        """
        from audobject.core.object import Object

        if isinstance(value, types.FunctionType):
            return self.get_source(value)
        elif isinstance(value, Object):
            return value
        else:
            raise ValueError(
                "Cannot decode object "
                "if it does not derive from "
                "'audobject.Object'."
            )

    def encode_type(self) -> type:
        r"""Returns encoded type.

        Returns:
            encoded type

        """
        return str

    def get_source(self, func: typing.Callable) -> str:
        r"""Obtain source code of (lambda) function.

        Retrieving the source of a lambda function can become tricky,
        see the following link for detailed discussion:

        http://xion.io/post/code/python-get-lambda-code.html

        Args:
            func: function object

        Returns:
            source code

        """
        # check if source code is attached
        # otherwise use inspect to get it
        if hasattr(func, '__source__'):
            return func.__source__
        else:
            if func.__name__ == "<lambda>":
                source = self._get_short_lambda_source(func)
            else:
                source = inspect.getsource(func)
            return textwrap.dedent(source)

    @staticmethod
    def _get_short_lambda_source(
            lambda_func: typing.Callable,
    ):  # pragma: no cover
        """Return the source of a (short) lambda function.

        If it's impossible to obtain, returns None.

        Original code:
        https://gist.github.com/Xion/617c1496ff45f3673a5692c3b0e3f75a

        """
        try:
            source_lines, _ = inspect.getsourcelines(lambda_func)
        except (IOError, TypeError):
            return None

        # skip `def`-ed functions and long lambdas
        if len(source_lines) != 1:
            return None

        source_text = os.linesep.join(source_lines).strip()

        # find the AST node of a lambda definition
        # so we can locate it in the source code
        source_ast = ast.parse(source_text)
        lambda_node = next((node for node in ast.walk(source_ast)
                            if isinstance(node, ast.Lambda)), None)
        if lambda_node is None:  # could be a single line `def fn(x): ...`
            return None

        # HACK: Since we can (and most likely will) get source lines
        # where lambdas are just a part of bigger expressions, they will have
        # some trailing junk after their definition.
        #
        # Unfortunately, AST nodes only keep their _starting_ offsets
        # from the original source, so we have to determine the end ourselves.
        # We do that by gradually shaving extra junk from after the definition.
        lambda_text = source_text[lambda_node.col_offset:]
        lambda_body_text = source_text[lambda_node.body.col_offset:]
        min_length = len('lambda:_')  # shortest possible lambda expression
        while len(lambda_text) > min_length:
            try:
                # What's annoying is that sometimes the junk even parses,
                # but results in a *different* lambda. You'd probably have to
                # be deliberately malicious to exploit it but here's one way:
                #
                #     bloop = lambda x: False, lambda x: True
                #     get_short_lamnda_source(bloop[0])
                #
                # Ideally, we'd just keep shaving until we get the same code,
                # but that most likely won't happen because we can't replicate
                # the exact closure environment.
                code = compile(lambda_body_text, '<unused filename>', 'eval')

                # Thus the next best thing is to assume some divergence due
                # to e.g. LOAD_GLOBAL in original code being LOAD_FAST in
                # the one compiled above, or vice versa.
                # But the resulting code should at least be the same *length*
                # if otherwise the same operations are performed in it.
                if len(code.co_code) == len(lambda_func.__code__.co_code):
                    return lambda_text
            except SyntaxError:
                pass
            lambda_text = lambda_text[:-1]
            lambda_body_text = lambda_body_text[:-1]

        return None


class Tuple(Base):
    r"""Tuple resolver.

    Encodes tuple as a list.

    Examples:
        >>> resolver = Tuple()
        >>> value = (1, 'a')
        >>> value
        (1, 'a')
        >>> encoded_value = resolver.encode(value)
        >>> encoded_value
        [1, 'a']
        >>> decoded_value = resolver.decode(encoded_value)
        >>> decoded_value
        (1, 'a')

    """

    def decode(self, value: list) -> tuple:
        r"""Decodes ``list`` as ``tuple``.

        Args:
            value: list

        Returns:
            tuple

        """
        return tuple(value)

    def encode(self, value: tuple) -> list:
        r"""Encodes ``tuple`` as ``list``.

        Args:
            value: tuple

        Returns:
            list

        """
        return list(value)

    def encode_type(self) -> type:
        r"""Return encoded type.

        Returns:
            encoded type

        """
        return list


class Type(Base):
    r"""Type resolver.

    Encodes type as a string.

    Examples:
        >>> resolver = Type()
        >>> value = str
        >>> value
        <class 'str'>
        >>> encoded_value = resolver.encode(value)
        >>> encoded_value
        'str'
        >>> decoded_value = resolver.decode(encoded_value)
        >>> decoded_value
        <class 'str'>

    """

    def decode(self, value: str) -> type:
        r"""Decodes ``str`` as ``type``.

        Args:
            value: type string

        Returns:
            type

        """
        return eval(value)

    def encode(self, value: type) -> str:
        r"""Encodes ``type`` as ``str``.

        Args:
            value: type class

        Returns:
            string

        """
        return str(value)[len("<class '"):-len("'>")]

    def encode_type(self) -> type:
        r"""Return encoded type.

        Returns:
            encoded type

        """
        return str


# deprecated classes


# @audeer.deprecated(
#     removal_version='1.0.0',
#     alternative='resolver.Base',
# )
# ->
# TypeError: function() argument 1 must be code, not str
# ->
# as a workaround we raise the deprecation warning in __init__
class ValueResolver:  # pragma: no cover  # noqa: D101

    def __init__(self):
        message = (
            'ValueResolver is deprecated and will be removed '
            'with version 1.0.0. Use resolver.Base instead.'
        )
        warnings.warn(message, category=UserWarning, stacklevel=2)
        self.__dict__[define.ROOT_ATTRIBUTE] = None

    @property
    def root(self) -> typing.Optional[str]:  # noqa: D102
        return self.__dict__[define.ROOT_ATTRIBUTE]

    def decode(self, value: DefaultValueType) -> typing.Any:  # noqa: D102
        raise NotImplementedError

    def encode(self, value: typing.Any) -> DefaultValueType:  # noqa: D102
        raise NotImplementedError

    def encode_type(self) -> type:  # noqa: D102
        raise NotImplementedError


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='resolver.FilePath',
)
class FilePathResolver(FilePath):  # pragma: no cover  # noqa: D101
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='resolver.Function',
)
class FunctionResolver(Function):  # pragma: no cover  # noqa: D101
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='resolver.Tuple',
)
class TupleResolver(Tuple):  # pragma: no cover  # noqa: D101
    pass


@audeer.deprecated(
    removal_version='1.0.0',
    alternative='resolver.Type',
)
class TypeResolver(Type):  # pragma: no cover  # noqa: D101
    pass
