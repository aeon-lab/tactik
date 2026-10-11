
"""TACTIK: text analysis, clustering, tuning, and keyword extraction."""

from importlib import import_module

# Keep package import lightweight. Python loads submodules on demand for
# imports such as `from tactik import utilities`.
__version__ = "1.1.0"

_PUBLIC_MODULES = {
    "clustering_pipeline",
    "clustering_tuning",
    "embeddings",
    "preprocessing",
    "topic_extraction",
    "utilities",
    "visualization",
}


def __getattr__(name):
    """Load public submodules lazily while retaining package-level access."""
    if name not in _PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


__all__ = ["__version__", *sorted(_PUBLIC_MODULES)]


