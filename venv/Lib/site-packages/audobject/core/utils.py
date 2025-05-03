import importlib
import inspect
import operator
import types
import typing
import warnings

from importlib_metadata import packages_distributions

import audeer
from audobject.core import define
from audobject.core.config import config


# Cache result of packages_distributions()
PACKAGES_DISTRIBUTIONS = {}


def create_class_key(cls: type, include_version: bool) -> str:
    r"""Create class key.

    Convert class into a string that encodes
    package, module, class name and possibly version.
    Package name is omitted if it matches the module.

    Examples:
    - $audeer.core.version.LooseVersion
    - $audeer.core.version.LooseVersion==1.18.0
    - $PyYAML:yaml.loader.Loader

    """
    global PACKAGES_DISTRIBUTIONS
    key = define.OBJECT_TAG

    # add package name (if different from module name)
    module_name = cls.__module__.split('.')[0]
    if module_name not in PACKAGES_DISTRIBUTIONS:
        PACKAGES_DISTRIBUTIONS = packages_distributions()
    if module_name in PACKAGES_DISTRIBUTIONS:
        package_name = PACKAGES_DISTRIBUTIONS[module_name][0]
        if package_name != module_name:
            key += f'{package_name}{define.PACKAGE_TAG}'

    # add module and class name
    key += f'{cls.__module__}.{cls.__name__}'

    # add version (if requested)
    if include_version:
        version = get_version(cls.__module__)
        if version is not None:
            key += f'{define.VERSION_TAG}{version}'
        else:
            warnings.warn(
                f"Could not determine a version for "
                f"module '{cls.__module__}'.",
                RuntimeWarning,
            )

    return key


def get_class(
        key: str,
        auto_install: bool,
) -> (type, str, str):
    r"""Load class."""
    package_name, module_name, class_name, version = split_class_key(key)
    module = get_module(
        package_name,
        module_name,
        version,
        auto_install,
    )
    installed_version = get_version(module_name)

    # possibly raise warning if installed package does not match version
    level = config.PACKAGE_MISMATCH_WARN_LEVEL
    if version is not None and level != define.PackageMismatchWarnLevel.SILENT:
        if installed_version is not None:
            if level == define.PackageMismatchWarnLevel.STANDARD:
                # only warn if installed package is older
                op = operator.ge
            else:
                # warn if package do not match
                op = operator.eq
            if not op(
                    audeer.LooseVersion(installed_version),
                    audeer.LooseVersion(version),
            ):
                warnings.warn(
                    f"Instantiating {module_name}.{class_name} from "
                    f"version '{version}' when using "
                    f"version '{installed_version}'.",
                    RuntimeWarning,
                )

    return getattr(module, class_name), version, installed_version


def get_module(
        package_name: str,
        module_name: str,
        version: typing.Optional[str],
        auto_install: bool,
) -> types.ModuleType:
    r"""Load module."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as ex:
        if auto_install:
            audeer.install_package(
                package_name,
                version=version,
            )
            return get_module(
                package_name,
                module_name,
                version,
                False,
            )
        else:
            raise ex

    return module


def get_object(
        cls: type,
        version: str,
        installed_version: str,
        params: dict,
        root: typing.Optional[str],
        override_args: typing.Dict[str, typing.Any],
) -> (typing.Any, dict):
    r"""Create object from arguments without calling `__init__()`."""
    signature = inspect.signature(cls.__init__)
    supports_kwargs = 'kwargs' in signature.parameters
    supported_params = set([
        p.name for p in signature.parameters.values()
        if p.name not in ['self', 'kwargs']
    ])

    # check for missing mandatory arguments
    required_params = set([
        p.name for p in signature.parameters.values()
        if p.default == inspect.Parameter.empty and p.name not in [
            'self', 'args', 'kwargs',
        ]
    ])
    missing_required_params = list(required_params - set(params))
    if len(missing_required_params) > 0:
        raise RuntimeError(
            f"Missing mandatory arguments "
            f"{missing_required_params} "
            f"while instantiating {cls} from "
            f"version '{version}' when using "
            f"version '{installed_version}'."
        )

    # check for missing optional arguments
    optional_params = set([
        p.name for p in signature.parameters.values()
        if p.default != inspect.Parameter.empty and p.name not in [
            'self', 'args', 'kwargs',
        ]
    ])
    missing_optional_params = list(optional_params - set(params))
    if len(missing_optional_params) > 0:
        if config.SIGNATURE_MISMATCH_WARN_LEVEL > \
                define.SignatureMismatchWarnLevel.STANDARD:
            warnings.warn(
                f"Missing optional arguments "
                f"{missing_optional_params} "
                f"while instantiating {cls} from "
                f"version '{version}' when using "
                f"version '{installed_version}'.",
                RuntimeWarning,
            )

    # unless kwargs are supported check for additional arguments
    if not supports_kwargs:
        additional_params = list(set(params) - supported_params)
        if len(additional_params):
            if config.SIGNATURE_MISMATCH_WARN_LEVEL > \
                    define.SignatureMismatchWarnLevel.SILENT:
                warnings.warn(
                    f"Ignoring arguments "
                    f"{additional_params} "
                    f"while instantiating {cls} from "
                    f"version '{version}' when using "
                    f"version '{installed_version}'.",
                    RuntimeWarning,
                )
            params = {
                key: value for key, value in params.items()
                if key in supported_params
            }

    # select supported params from kwargs
    for key, value in override_args.items():
        if key in supported_params:
            params[key] = value

    # create object, but do not call __init__() yet
    object = cls.__new__(cls, **params)

    return object, params


def get_version(module_name: str) -> typing.Optional[str]:
    module = importlib.import_module(module_name.split('.')[0])
    if '__version__' in module.__dict__:
        return module.__version__
    else:
        return None


def is_class(value: typing.Any):
    r"""Check if value is a class."""
    if isinstance(value, str):
        if value.startswith(define.OBJECT_TAG):
            return True
        # only for backward compatibility with `auglib` and `audbenchmark`
        if value.startswith('auglib.core.') or \
                value.startswith('audbenchmark.core.'):
            return True  # pragma: no cover
    return False


def split_class_key(key: str) -> [str, str, str, typing.Optional[str]]:
    r"""Split class key into package, module, class and version.

    Expects a key in the format output by create_class_key().
    If package is not encoded,
    the module name is returned.
    If version is not encoded,
    None is returned.
    Leading $ can be omitted.

    """
    version = None

    # possibly remove leading $
    if key.startswith(define.OBJECT_TAG):
        key = key[len(define.OBJECT_TAG):]

    # split off version (if available)
    if define.VERSION_TAG in key:
        key, version = key.split(define.VERSION_TAG)

    # split off package (if available)
    package_name = None
    if define.PACKAGE_TAG in key:
        package_name, key = key.split(define.PACKAGE_TAG)

    # split off module and class name
    tokens = key.split('.')
    module_name = '.'.join(tokens[:-1])
    class_name = tokens[-1]

    # if package name not given, set to module name
    if package_name is None:
        package_name = tokens[0]

    return package_name, module_name, class_name, version
