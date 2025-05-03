import warnings

import pytest

import audobject


# Signature mismatch

audobject.config.SIGNATURE_MISMATCH_WARN_LEVEL = \
    audobject.define.SignatureMismatchWarnLevel.VERBOSE


class MyObject(audobject.Object):
    def __init__(
            self,
            p: str,
            *,
            kw: int = 0,
    ):
        self.p = p
        self.kw = kw


# no version

with pytest.warns(RuntimeWarning):
    o_yaml = MyObject('test').to_yaml_s()


# an optional argument is added

class MyObject(audobject.Object):
    def __init__(
            self,
            p: str,
            new: float = 0.0,
            *,
            kw: int = 0,
    ):
        self.p = p
        self.kw = kw
        self.new = new


with pytest.warns(RuntimeWarning):
    audobject.Object.from_yaml_s(o_yaml)


# an argument is removed

class MyObject(audobject.Object):
    def __init__(
            self,
            *,
            kw: int = 0,
    ):
        self.kw = kw


with pytest.warns(RuntimeWarning):
    audobject.Object.from_yaml_s(o_yaml)


# a mandatory argument is added

class MyObject(audobject.Object):
    def __init__(
            self,
            p: str,
            new: float,
            *,
            kw: int = 0,
    ):
        self.p = p
        self.kw = kw
        self.new = new


with pytest.raises(RuntimeError):
    audobject.from_yaml_s(o_yaml)


audobject.config.SIGNATURE_MISMATCH_WARN_LEVEL = \
    audobject.define.SignatureMismatchWarnLevel.STANDARD


# Package mismatch

SMALLER_VERSION = '0.0.0'
INSTALLED_VERSION = audobject.__version__
GREATER_VERSION = '999.9.9'


@pytest.mark.parametrize(
    'level',
    [
        audobject.define.PackageMismatchWarnLevel.SILENT,
        audobject.define.PackageMismatchWarnLevel.VERBOSE,
        audobject.define.PackageMismatchWarnLevel.STANDARD,
    ],
)
def test_package_mismatch_no_version(level):

    warnings.simplefilter('error')

    # no version given -> never warn
    yaml_s = '''
    $audobject.core.testing.TestObject:
      name: test
    '''

    audobject.config.PACKAGE_MISMATCH_WARN_LEVEL = level

    with warnings.catch_warnings():
        audobject.from_yaml_s(yaml_s)

    audobject.config.PACKAGE_MISMATCH_WARN_LEVEL = \
        audobject.define.PackageMismatchWarnLevel.STANDARD

    warnings.simplefilter('default')


@pytest.mark.parametrize(
    'level, version, expected',
    [
        # silent -> never warn
        (
            audobject.define.PackageMismatchWarnLevel.SILENT,
            SMALLER_VERSION,
            False,
        ),
        (
            audobject.define.PackageMismatchWarnLevel.SILENT,
            INSTALLED_VERSION,
            False,
        ),
        (
            audobject.define.PackageMismatchWarnLevel.SILENT,
            GREATER_VERSION,
            False,
        ),
        # standard -> warn if installed version is smaller
        (
            audobject.define.PackageMismatchWarnLevel.STANDARD,
            SMALLER_VERSION,
            False,
        ),
        (
            audobject.define.PackageMismatchWarnLevel.STANDARD,
            INSTALLED_VERSION,
            False,
        ),
        (
            audobject.define.PackageMismatchWarnLevel.STANDARD,
            GREATER_VERSION,
            True,
        ),
        # verbose -> warn unless versions match
        (
            audobject.define.PackageMismatchWarnLevel.VERBOSE,
            SMALLER_VERSION,
            True,
        ),
        (
            audobject.define.PackageMismatchWarnLevel.VERBOSE,
            INSTALLED_VERSION,
            False,
        ),
        (
            audobject.define.PackageMismatchWarnLevel.VERBOSE,
            GREATER_VERSION,
            True,
        ),
    ],
)
def test_package_mismatch_with_version(level, version, expected):

    warnings.simplefilter('error')

    yaml_s = f'''
    $audobject.core.testing.TestObject=={version}:
      name: test
    '''

    audobject.config.PACKAGE_MISMATCH_WARN_LEVEL = level

    if expected:
        with pytest.warns(RuntimeWarning):
            audobject.from_yaml_s(yaml_s)
    else:
        with warnings.catch_warnings():
            audobject.from_yaml_s(yaml_s)

    audobject.config.PACKAGE_MISMATCH_WARN_LEVEL = \
        audobject.define.PackageMismatchWarnLevel.STANDARD

    warnings.simplefilter('default')
