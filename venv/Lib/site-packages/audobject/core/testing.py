from audobject.core.decorator import init_decorator
from audobject.core.object import Object
import audobject.core.resolver as resolver


class TestObject(Object):
    r"""Test object.

    Args:
        name: object name
        point: point with two dimensions
        date: date, if ``None`` initialized with current date
        **kwargs: additional variables

    Examples:
        >>> from datetime import datetime
        >>> foo = TestObject('test', pi=3.1416)
        >>> print(foo)
        $audobject.core.testing.TestObject:
          name: test
          point:
          - 0
          - 0
          pi: 3.1416

    """
    @init_decorator(
        resolvers={
            'point': resolver.Tuple,
        },
    )
    def __init__(
            self,
            name: str,
            *,
            point: (int, int) = (0, 0),
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.name = name
        self.point = point
        for key, value in kwargs.items():
            self.__dict__[key] = value
