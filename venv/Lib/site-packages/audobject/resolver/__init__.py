r"""Resolver classes.

:class:`audobject.resolver.Base` can be used to control
how an object of certain type is represented in YAML.

By default,
the following types are supported:

* ``bool``
* ``datetime.datetime``
* ``dict``
* ``float``
* ``int``
* ``list``
* ``None``
* ``Object``
* ``str``

For any other type,
it is not ensured that an object can be properly reconstructed
and a warning will be raised.
It is recommended to use a special resolver in those cases.

To apply a resolver,
register it using the :func:`audobject.init_decorator`, e.g.:

.. code-block::

    import audobject


    class MyObjectWithTuple(audobject.Object):

        @audobject.init_decorator(
            resolvers={
                't': audobject.resolver.Tuple,
            }
        )

        def __init__(
                self,
                t: tuple,
        ):
            self.t = t


To write a custom resolver derive from
:class:`audobject.resolver.Base`, e.g.:

.. code-block::

    from datetime import timedelta


    class DeltaResolver(audobject.resolver.Base):

        def decode(self, value: dict) -> timedelta:
            return timedelta(
                days=value['days'],
                seconds=value['seconds'],
                microseconds=value['microseconds'],
            )

        def encode(self, value: timedelta) -> dict:
            return {
                'days': value.days,
                'seconds': value.seconds,
                'microseconds': value.microseconds,
            }

        def encode_type(self):
            return dict

"""
from audobject.core.resolver import Base
from audobject.core.resolver import FilePath
from audobject.core.resolver import Function
from audobject.core.resolver import Tuple
from audobject.core.resolver import Type
