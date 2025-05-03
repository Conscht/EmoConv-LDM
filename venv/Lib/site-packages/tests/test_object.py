import os

import pytest
import yaml

import audeer
import audobject
import audobject.core.utils as utils
import audobject.testing


@pytest.mark.parametrize(
    'obj',
    [
        audobject.testing.TestObject(
            'test',
            d={'a': 1, 'b': 2, 'c': 3},
            d_empty={},
            f=1.0,
            i=0,
            l=[1, 2, 3],
            n=None,
            o=audobject.testing.TestObject('another'),
        ),
    ]
)
def test(tmpdir, obj):

    # load from YAML string
    obj_from_yaml_s = audobject.from_yaml_s(obj.to_yaml_s())
    assert obj == obj_from_yaml_s

    # load from YAML file
    path = os.path.join(tmpdir, 'test.yaml')
    obj.to_yaml(path)
    obj_from_yaml = audobject.from_yaml(path)
    assert obj == obj_from_yaml

    # comparison to other objects,
    # see https://github.com/audeering/audinterface/issues/68
    assert not obj == type
    assert repr(obj) == repr(obj_from_yaml)
    assert str(obj) == str(obj_from_yaml)
    assert obj.to_yaml_s() == obj_from_yaml.to_yaml_s()
    for key, value in obj.__dict__.items():
        if not key.strip('_'):
            assert obj_from_yaml.__dict__[key] == value

    # test hashable collection
    assert hash(obj) == hash(obj.id)
    set().add(obj)

    # check if object was loaded
    assert not obj.is_loaded_from_dict
    assert obj_from_yaml_s.is_loaded_from_dict
    assert obj_from_yaml.is_loaded_from_dict

    # IDs
    string = obj.to_yaml_s(include_version=False)
    assert obj.id == audeer.uid(from_string=string, short=False)
    assert obj.short_id == audeer.uid(from_string=string, short=True)


class ObjectTestIsLoadedFromDict(audobject.Object):
    def __init__(self):
        self.loaded = self.is_loaded_from_dict


def test_is_loaded_from_dict():

    obj = ObjectTestIsLoadedFromDict()
    obj_from_dict = audobject.from_dict(
        obj.to_dict(include_version=False),
    )

    assert not obj.loaded
    assert obj_from_dict.loaded


class Point:
    def __init__(
            self,
            x: int,
            y: int,
    ):
        self.x = x
        self.y = y


class ObjectWithBorrowedArguments(audobject.Object):

    @audobject.init_decorator(
        borrow={
            'x': 'point',
            'y': 'point',
            'z': 'd',
        },
    )
    def __init__(
            self,
            x: int,
            y: int,
            z: int,
    ):
        self.point = Point(x, y)
        self.d = {'z': z}


def test_borrowed():
    x = 0
    y = 1
    z = 2
    o = ObjectWithBorrowedArguments(x, y, z)
    assert x not in o.__dict__
    assert y not in o.__dict__
    assert z not in o.__dict__
    assert o.point.x == x
    assert o.point.y == y
    assert o.d['z'] == z
    o2 = audobject.from_yaml_s(o.to_yaml_s(include_version=False))
    assert isinstance(o2, ObjectWithBorrowedArguments)
    assert x not in o2.__dict__
    assert y not in o2.__dict__
    assert z not in o2.__dict__
    assert o2.point.x == x
    assert o2.point.y == y
    assert o.d['z'] == z


@pytest.mark.parametrize(
    'cls, package, include_version, expected',
    [
        (
            audeer.LooseVersion,
            'audeer',
            False,
            '$audeer.core.version.LooseVersion',
        ),
        (
            audeer.LooseVersion,
            'audeer',
            True,
            f'$audeer.core.version.LooseVersion=={audeer.__version__}',
        ),
        (
            yaml.Loader,
            'PyYAML',
            False,
            '$PyYAML:yaml.loader.Loader',
        ),
        (
            yaml.Loader,
            'PyYAML',
            True,
            f'$PyYAML:yaml.loader.Loader=={yaml.__version__}',
        ),
    ]
)
def test_class_key(cls, package, include_version, expected):
    key = utils.create_class_key(cls, include_version)
    assert key == expected
    p, m, c, v = utils.split_class_key(key)
    if include_version:
        assert expected.endswith(f'{m}.{c}=={v}')
    else:
        assert v is None
        assert expected.endswith(f'{m}.{c}')
    assert p == package


@pytest.mark.parametrize(
    'obj, expected',
    [
        (
            audobject.testing.TestObject(
                'test',
                point=(1, 1),
                foo={},
                bar=[],
            ),
            {
                'name': 'test',
                'point.0': 1,
                'point.1': 1,
            }
        ),
        (
            audobject.testing.TestObject(
                'test',
                point=(1, 1),
                object=audobject.testing.TestObject(
                    'test',
                    foo='foo',
                    point=(2, 2),
                    list=[
                        audobject.testing.TestObject(
                            'test',
                            point=(3, 3),
                        ),
                        'foo',
                    ],
                    dict={
                        'object': audobject.testing.TestObject(
                            'test',
                            bar='bar',
                            point=(4, 4),
                        ),
                        'foo': 'foo',
                    }
                )
            ),
            {
                'name': 'test',
                'point.0': 1,
                'point.1': 1,
                'object.name': 'test',
                'object.point.0': 2,
                'object.point.1': 2,
                'object.foo': 'foo',
                'object.list.0.name': 'test',
                'object.list.0.point.0': 3,
                'object.list.0.point.1': 3,
                'object.list.1': 'foo',
                'object.dict.object.name': 'test',
                'object.dict.object.point.0': 4,
                'object.dict.object.point.1': 4,
                'object.dict.object.bar': 'bar',
                'object.dict.foo': 'foo'
            },
        ),
    ]
)
def test_flatten(obj, expected):
    assert obj.to_dict(flatten=True) == expected


class ParentWithHiddenArguments(audobject.Object):

    @audobject.init_decorator(
        hide=['hidden_parent']
    )
    def __init__(
            self,
            hidden_parent: str = None,
    ):
        self.hidden_parent = hidden_parent


class ChildWithHiddenArguments(ParentWithHiddenArguments):

    @audobject.init_decorator(
        hide=['hidden_child']
    )
    def __init__(
            self,
            string: str,
            *,
            hidden_child: str = None,
            hidden_parent: str = None,
    ):
        super().__init__(hidden_parent)
        self.string = string
        self.hidden_child = hidden_child


def test_hidden_attributes(tmpdir):

    path = os.path.join(tmpdir, 'test.yaml')

    o = ChildWithHiddenArguments(
        'test',
        hidden_child='hidden_child',
        hidden_parent='hidden_parent',
    )
    assert o.hidden_child == 'hidden_child'
    assert o.hidden_parent == 'hidden_parent'

    o2 = audobject.from_yaml_s(o.to_yaml_s(include_version=False))
    assert isinstance(o2, ChildWithHiddenArguments)
    assert o2.hidden_child is None
    assert o2.hidden_parent is None

    o.to_yaml(path, include_version=False)
    o2 = audobject.from_yaml(path)
    assert isinstance(o2, ChildWithHiddenArguments)
    assert o2.hidden_child is None

    o2 = audobject.from_yaml_s(
        o.to_yaml_s(include_version=False),
        override_args={
            'hidden_child': 'hidden_child',
            'hidden_parent': 'hidden_parent'
        },
    )
    assert isinstance(o2, ChildWithHiddenArguments)
    assert o2.hidden_child == 'hidden_child'
    assert o2.hidden_parent == 'hidden_parent'

    o.to_yaml(path, include_version=False)
    o2 = audobject.from_yaml(
        path,
        override_args={
            'hidden_child': 'hidden_child',
            'hidden_parent': 'hidden_parent',
        }
    )
    assert isinstance(o2, ChildWithHiddenArguments)
    assert o2.hidden_child == 'hidden_child'
    assert o2.hidden_parent == 'hidden_parent'


class ArgumentWithNameRoot(audobject.Object):

    def __init__(
            self,
            root: str = None,
    ):
        self.root = root


def test_override_attributes(tmpdir):

    o = audobject.testing.TestObject(
        name='name',
    )
    assert o.name == 'name'

    o1 = audobject.from_dict(
        o.to_dict(include_version=False),
        override_args={
            'name': 'override',
        }
    )
    assert isinstance(o1, audobject.testing.TestObject)
    assert o1.name == 'override'

    o2 = audobject.from_yaml_s(
        o.to_yaml_s(include_version=False),
        override_args={
            'name': 'override',
        }
    )
    assert isinstance(o2, audobject.testing.TestObject)
    assert o2.name == 'override'

    # override argument named root

    o3 = ArgumentWithNameRoot()
    assert o3.root is None

    o4 = audobject.from_yaml_s(
        o3.to_yaml_s(include_version=False),
        override_args={
            'root': 'override',
        }
    )
    assert isinstance(o4, ArgumentWithNameRoot)
    assert o4.root == 'override'

    path = os.path.join(tmpdir, 'object.yaml')
    o3.to_yaml(path, include_version=False)
    o5 = audobject.from_yaml(
        path,
        override_args={
            'root': 'override',
        }
    )
    assert isinstance(o5, ArgumentWithNameRoot)
    assert o5.root == 'override'

    o6 = audobject.from_dict(
        o3.to_dict(include_version=False),
        override_args={
            'root': 'override',
        }
    )
    assert isinstance(o6, ArgumentWithNameRoot)
    assert o6.root == 'override'


def test_override_attributes_deprecated(tmpdir):

    o = audobject.testing.TestObject(
        name='name',
    )
    assert o.name == 'name'

    with pytest.warns(UserWarning):
        o1 = audobject.from_dict(
            o.to_dict(include_version=False),
            name='override',
        )
    assert isinstance(o1, audobject.testing.TestObject)
    assert o1.name == 'override'

    with pytest.warns(UserWarning):
        o2 = audobject.from_yaml_s(
            o.to_yaml_s(include_version=False),
            name='override',
        )
    assert isinstance(o2, audobject.testing.TestObject)
    assert o2.name == 'override'

    # override argument named root

    o3 = ArgumentWithNameRoot()
    assert o3.root is None

    with pytest.warns(UserWarning):
        o4 = audobject.from_yaml_s(
            o3.to_yaml_s(include_version=False),
            root='override',
        )
    assert isinstance(o4, ArgumentWithNameRoot)
    assert o4.root == 'override'

    path = os.path.join(tmpdir, 'object.yaml')
    o3.to_yaml(path, include_version=False)
    with pytest.warns(UserWarning):
        o5 = audobject.from_yaml(
            path,
            root='override',
        )
    assert isinstance(o5, ArgumentWithNameRoot)
    assert o5.root == 'override'


def test_no_resolver():

    obj = audobject.testing.TestObject(
        'test',
        no_encoder=(1, 2, 3),
    )
    with pytest.warns(RuntimeWarning):
        obj.to_yaml_s()

    obj = audobject.testing.TestObject(
        'test',
        no_encoder=lambda x: x,
    )
    with pytest.raises(RuntimeError):
        obj.to_yaml_s()

    def func():
        pass

    obj = audobject.testing.TestObject(
        'test',
        no_encoder=func,
    )
    with pytest.raises(RuntimeError):
        obj.to_yaml_s()

    obj = audobject.testing.TestObject(
        'test',
        no_encoder=[func],
    )
    with pytest.raises(RuntimeError):
        obj.to_yaml_s()

    obj = audobject.testing.TestObject(
        'test',
        no_encoder={'func': func},
    )
    with pytest.raises(RuntimeError):
        obj.to_yaml_s()


def test_bad_object():

    class BadObject(audobject.Object):
        def __init__(
                self,
                foo: str = None,
        ):
            self.bar = foo

    o = BadObject('foo')
    with pytest.raises(RuntimeError):
        with pytest.warns(RuntimeWarning):
            o.to_yaml_s()  # argument foo not assigned to an attribute

    class BadObject(audobject.Object):
        @audobject.init_decorator(
            hide=['foo'],
        )
        def __init__(
                self,
                foo: str,
        ):
            self.bar = foo

    with pytest.raises(RuntimeError):
        with pytest.warns(RuntimeWarning):
            BadObject('foo')  # cannot hide argument without default value

    class BadObject(audobject.Object):
        @audobject.init_decorator(
            borrow={
                'x': 'point',
                'y': 'point',
            },
        )
        def __init__(
                self,
                x: int,
                y: int,
        ):
            pass

    with pytest.raises(RuntimeError):
        with pytest.warns(RuntimeWarning):
            BadObject(0, 1).to_yaml_s()  # cannot borrow from missing attribute

    class BadObject(audobject.Object):
        @audobject.init_decorator(
            borrow={
                'x': 'point',
                'y': 'point',
            },
        )
        def __init__(
                self,
                x: int,
                y: int,
        ):
            self.point = 0

    with pytest.raises(RuntimeError):
        with pytest.warns(RuntimeWarning):
            BadObject(0, 1).to_yaml_s()  # cannot borrow missing attribute

    class BadObject(audobject.Object):
        @audobject.init_decorator(
            borrow={
                'x': 'point',
                'y': 'point',
                'z': 'point',
            },
        )
        def __init__(
                self,
                x: int,
                y: int,
        ):
            self.point = Point(x, y)

    with pytest.raises(RuntimeError):
        with pytest.warns(RuntimeWarning):
            BadObject(0, 1).to_yaml_s()  # cannot borrow missing attribute


class MyObjectWithKwargs(audobject.Object):

    @audobject.init_decorator(
        borrow={
            'borrow_arg': 'dict',
        },
        hide=['hide_arg', 'unassign_arg'],
        resolvers={
            'resolve_arg': audobject.resolver.Tuple,
        }
    )
    @audeer.deprecated_keyword_argument(
        deprecated_argument='deprecate_arg',
        removal_version='999.9.9',
    )
    def __init__(
            self,
            arg: str,
            *,
            kwarg: str,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.arg = arg
        self.kwarg = kwarg
        self.dict = {
            'borrow_arg': kwargs['borrow_arg']
        }
        self.no_arg = None
        self.hide_arg = kwargs['hide_arg'] if 'hide_arg' in kwargs else None
        self.resolve_arg = kwargs['resolve_arg']


def test_kwargs_object():

    with pytest.warns(UserWarning, match='deprecate'):
        o = MyObjectWithKwargs(
            'arg',
            borrow_arg='borrow',
            deprecate_arg='deprecate',
            hide_arg='hide',
            kwarg='kwarg',
            resolve_arg=('foo', 'bar'),
            unassign_arg='unassign',
        )

    assert 'borrow_arg' in o.arguments
    assert 'deprecate_arg' not in o.arguments
    assert 'hide_arg' not in o.arguments
    assert 'kwarg' in o.arguments
    assert 'no_arg' not in o.arguments
    assert 'resolve_arg' in o.arguments

    assert o.hide_arg is not None

    o2 = audobject.from_yaml_s(o.to_yaml_s(include_version=False))

    assert o == o2
    assert o2.hide_arg is None
