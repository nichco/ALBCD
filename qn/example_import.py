"""Import an example's models.py (shared by airliner_blocks.py and ranger_blocks.py)."""

import importlib
import sys


def import_example_models(example, alias):
    """Import ``example``/models.py through sys.path, registered as ``alias``: both examples name their model
    file models.py, so a plain ``import models`` would return whichever example was imported first."""
    if alias in sys.modules:
        return sys.modules[alias]
    if str(example) not in sys.path:
        sys.path.insert(0, str(example))
    saved = sys.modules.pop("models", None)
    try:
        module = importlib.import_module("models")
    finally:
        sys.modules.pop("models", None)
        if saved is not None:
            sys.modules["models"] = saved
    sys.modules[alias] = module
    return module
