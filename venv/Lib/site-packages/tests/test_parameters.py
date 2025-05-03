import argparse
import os

import parse
import pytest

import audobject


@pytest.mark.parametrize(
    'value_type, default_value, choices, value',
    [
        (str, None, None, None),
        (str, 'default', None, 'value'),
        (str, 'default', ['default', 'value'], 'value'),
        (int, 0, [0, 1], 1),
        (str, None, [None, 'value'], 'value'),
        pytest.param(  # wrong value type
            str, 'default', None, 5,
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        pytest.param(  # value not in choices
            str, 'default', ['default'], 'value',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # default not in choices
            str, None, ['value'], 'value',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_parameter_type(value_type, default_value, choices, value):
    p = audobject.Parameter(
        value_type=value_type,
        description='bar',
        default_value=default_value,
        choices=choices,
    )
    assert p.value == default_value
    p.set_value(value)
    assert p.value == value


@pytest.mark.parametrize(
    'param_version,check_version,result',
    [
        (None, None, True),
        (None, '1.0.0', True),
        ('==1.0.0', None, True),
        ('==1.0.0', '1.0.0', True),
        ('==1.0.0', '2.0.0', False),
        ('>=1.0.0', '2.0.0', True),
        ('>=2.0.0', '2.0.0', True),
        ('>2.0.0', '2.0.0', False),
        ('>=1.0.0,<2.0.0', '2.0.0', False),
        ('>=1.0.0,!=2.0.0', '2.0.0', False),
        ('>=1.0.0,<3.0.0', '2.0.0', True),
        ('>=1.0.0,<3.0.0,!=2.0.0', '2.0.0', False),
    ]
)
def test_parameter_version(param_version, check_version, result):
    p = audobject.Parameter(
        value_type=str,
        description='bar',
        version=param_version,
    )
    assert result == (check_version in p)


@pytest.mark.parametrize(
    'params',
    [
        pytest.PARAMETERS
    ]
)
def test_parameters_call(params):
    d = {name: param.value for name, param in params.items()}
    assert params() == d


def test_parameters_command_line():

    p = pytest.PARAMETERS
    old_value = p.bar
    new_value = old_value + 1

    parser = argparse.ArgumentParser()
    p.to_command_line(parser)

    args = parser.parse_args(args=[f'--bar={new_value}', '--toggle'])

    assert p.foo == args.foo
    assert p.bar != args.bar
    assert p.toggle is False
    p.from_command_line(args)
    assert p.bar == args.bar
    assert p.toggle is True

    parser = argparse.ArgumentParser()
    p_filter = p.filter_by_version('1.0.0')
    p_filter.to_command_line(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(args=[f'--bar={new_value}'])


@pytest.mark.parametrize(
    'params',
    [
        (
            {}
        ),
        (
            {
                'foo':
                    audobject.Parameter(
                        value_type=str,
                        description='bar',
                    ),
                'idx':
                    audobject.Parameter(
                        value_type=int,
                        description='int',
                    )
            }
        ),
    ]
)
def test_parameters_init(params):
    pp = audobject.Parameters(**params)
    assert len(pp) == len(params)
    for name, p in params.items():
        assert name in pp
        assert name in pp.keys()
        assert p in pp.values()


@pytest.mark.parametrize(
    'delimiter, sort',
    [
        (',', True),
        (';', False),
    ]
)
def test_parameters_path(delimiter, sort):

    p = pytest.PARAMETERS

    path = p.to_path(delimiter=delimiter, sort=sort)
    for item in path.split(delimiter):
        key, value = parse.parse('{}[{}]', item)
        assert str(p[key].value) == value

    path = p.to_path(delimiter=delimiter, include=['bar'])
    assert 'bar' in path
    assert 'foo' not in path
    path = p.to_path(delimiter=delimiter, exclude=['bar'])
    assert 'bar' not in path
    assert 'foo' in path


def test_parameters_value():

    p = pytest.PARAMETERS
    old_value = p['bar'].value
    new_value = old_value + 1

    assert p.bar == old_value
    p.bar = new_value
    assert p.bar == new_value


@pytest.mark.parametrize(
    'other',
    [
        (
            {}
        ),
        (
            {
                'foo':
                    audobject.Parameter(
                        value_type=str,
                        description='bar',
                    ),
                'idx':
                    audobject.Parameter(
                        value_type=int,
                        description='int',
                    )
            }
        ),
    ]
)
def test_parameters_update(other):

    p = pytest.PARAMETERS
    p.update(other)
    for key, value in other.items():
        assert key in p
        assert p[key] == value


def test_parameters_yaml(tmpdir):

    p = pytest.PARAMETERS
    file = os.path.join(tmpdir, 'params.yaml')
    p.to_yaml(file)
    p2 = audobject.from_yaml(file)
    assert p == p2
