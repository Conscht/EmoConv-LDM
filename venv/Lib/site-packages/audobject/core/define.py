import datetime


BORROWED_ATTRIBUTES = '_object_borrowed_'
CUSTOM_VALUE_RESOLVERS = '_object_resolvers_'
DEFAULT_VALUE_TYPES = (str, int, float, bool, datetime.datetime)
HIDDEN_ATTRIBUTES = '_object_hidden_'
KEYWORD_ARGUMENTS = '_object_kwargs_'
OBJECT_TAG = '$'
OBJECT_LOADED = '_object_loaded_'
PACKAGE_TAG = ':'
ROOT_ATTRIBUTE = '_object_root_'
VERSION_TAG = '=='


class PackageMismatchWarnLevel:
    r"""Controls verbosity for package mismatch.

    A package mismatch might occur
    if the current installed version of a package
    does not match the version an object
    was generated with.

    Set via :class:`audobject.config`.

    """
    SILENT = 0
    r"""no warnings"""
    STANDARD = 1
    r"""only warn when installed version is older"""
    VERBOSE = 2
    r"""warn when versions do not match"""


class SignatureMismatchWarnLevel:
    r"""Controls verbosity for signature mismatch.

    A signature mismatch might occur
    if an object is loaded from a different package version
    it was generated with.

    Set via :class:`audobject.config`.

    """
    SILENT = 0
    r"""no warnings"""
    STANDARD = 1
    r"""warn when arguments are removed"""
    VERBOSE = 2
    r"""show all warnings"""
