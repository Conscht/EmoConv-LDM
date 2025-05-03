from audobject import define
from audobject import resolver
from audobject.core.api import from_dict
from audobject.core.api import from_yaml
from audobject.core.api import from_yaml_s
from audobject.core.config import config
from audobject.core.decorator import init_decorator
from audobject.core.dictionary import Dictionary
from audobject.core.object import Object
from audobject.core.parameter import Parameter
from audobject.core.parameter import Parameters
from audobject.core.resolver import FilePathResolver
from audobject.core.resolver import FunctionResolver
from audobject.core.resolver import TupleResolver
from audobject.core.resolver import TypeResolver
from audobject.core.resolver import ValueResolver


# Disencourage from audobject import *
__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata
    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
