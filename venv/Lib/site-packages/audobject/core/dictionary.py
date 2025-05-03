import typing

from audobject.core import define
from audobject.core.object import Object


class Dictionary(Object):
    r"""Dictionary that can be serialized to YAML.

    Args:
        **kwargs: items

    Examples:
        >>> # create dictionary
        >>> d = Dictionary(foo='foo', bar='bar')
        >>> # set item
        >>> d['foo'] = 'FOO'
        >>> # get item
        >>> d['foo']
        'FOO'
        >>> # add item
        >>> d['new'] = 1.234
        >>> print(d)
        $audobject.core.dictionary.Dictionary:
          foo: FOO
          bar: bar
          new: 1.234
        >>> d2 = Dictionary(foo='Foo', none=None)
        >>> d.update(d2)
        >>> print(d)
        $audobject.core.dictionary.Dictionary:
          foo: Foo
          bar: bar
          new: 1.234
          none: null

    """  # noqa: E501
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            self[key] = value

    def keys(self) -> typing.KeysView[str]:
        r"""Return the keys."""
        return self._dict_wo_special_attributes.keys()

    def items(self) -> typing.ItemsView[str, typing.Any]:
        r"""Return items view."""
        return self._dict_wo_special_attributes.items()

    def update(
            self,
            other: 'Dictionary',
    ):
        r"""Update the dictionary with the key/value pairs from other.

        Args:
            other: the other dictionary

        """
        for key, value in other.items():
            self[key] = value

    def values(self) -> typing.ValuesView[typing.Any]:
        r"""Return values."""
        return self._dict_wo_special_attributes.values()

    @property
    def _dict_wo_special_attributes(self):
        r"""Return self.__dict__ without special attributes."""
        d = {}
        for key, value in self.__dict__.items():
            if key not in [
                define.KEYWORD_ARGUMENTS,
                define.OBJECT_LOADED,
            ]:
                d[key] = value
        return d

    def __contains__(self, key):
        r"""Check if key is in dictionary."""
        return key in self._dict_wo_special_attributes

    def __getitem__(self, name: str) -> typing.Any:
        r"""Return value at key from dictionary."""
        return self._dict_wo_special_attributes[name]

    def __len__(self):
        r"""Number of keys in dictionary."""
        return len(self._dict_wo_special_attributes)

    def __setitem__(self, key: str, value: typing.Any):
        r"""Set value at key in dictionary."""
        if key not in self.__dict__[define.KEYWORD_ARGUMENTS]:
            self.__dict__[define.KEYWORD_ARGUMENTS].append(key)
        self.__dict__[key] = value
