"""General-purpose utilities."""

import importlib.util
import os
from pathlib import Path

from . import __project__  # Keep as relative for templating reasons.


def find_package_location(package: str = __project__) -> Path:
    """Return path to this package."""
    pkg_spec = importlib.util.find_spec(package)
    if pkg_spec is None:
        raise RuntimeError(f"Could not find module spec for {package}")
    search_locs = pkg_spec.submodule_search_locations
    if search_locs is None:
        raise RuntimeError(f"Could not find submodule search locations for {package}")
    return Path(search_locs[0])


def find_repo_location(package: str = __project__) -> Path:
    """Return path to the text-detection-baselines repository."""
    return Path(find_package_location(package) / os.pardir)
