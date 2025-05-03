import audobject.core.define as define


class config:
    r"""Get/set configuration values for the :mod:`audobject` module.

    Examples:
        >>> config.PACKAGE_MISMATCH_WARN_LEVEL
        1
        >>> config.PACKAGE_MISMATCH_WARN_LEVEL = 0
        >>> config.PACKAGE_MISMATCH_WARN_LEVEL
        0

    """

    PACKAGE_MISMATCH_WARN_LEVEL = define.PackageMismatchWarnLevel.STANDARD
    """Controls when a warning is shown in case of a package mismatch,
    see :class:`audobject.define.PackageMismatchWarnLevel`."""

    SIGNATURE_MISMATCH_WARN_LEVEL = define.SignatureMismatchWarnLevel.STANDARD
    """Controls when a warning is shown in case of a signature mismatch,
    see :class:`audobject.define.SignatureMismatchWarnLevel`."""
