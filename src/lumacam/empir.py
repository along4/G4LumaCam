"""
EMPIR binary discovery utilities.

Resolves the EMPIR installation directory and locates binaries
regardless of their subfolder layout.

Priority for resolving the EMPIR path:
    1. Explicit ``empir_dirpath`` argument passed to the constructor
    2. ``EMPIR_PATH`` environment variable
    3. Fallback to ``./empir``
"""

import os
from pathlib import Path
from typing import Dict, Optional, List


# Common subdirectories where EMPIR binaries may be located
_SEARCH_SUBDIRS = [
    ".",
    "bin",
    "empir_export",
    "empir_binning",
]


def resolve_empir_dir(empir_dirpath: Optional[str] = None) -> Path:
    """Resolve the EMPIR installation directory.

    Args:
        empir_dirpath: Explicit path. Takes highest priority.

    Returns:
        Path to the EMPIR directory.

    Raises:
        FileNotFoundError: If the resolved directory does not exist.
    """
    if empir_dirpath is not None:
        path = Path(empir_dirpath)
    elif "EMPIR_PATH" in os.environ:
        path = Path(os.environ["EMPIR_PATH"])
    else:
        path = Path("./empir")

    if not path.exists():
        raise FileNotFoundError(
            f"EMPIR directory not found: {path}\n"
            f"Set the EMPIR_PATH environment variable or pass empir_dirpath explicitly."
        )

    return path


def find_binary(empir_dir: Path, name: str) -> Path:
    """Locate a single EMPIR binary by name within the EMPIR directory.

    Searches the root directory and common subdirectories.

    Args:
        empir_dir: Root EMPIR directory to search in.
        name: Binary filename to find (e.g. ``empir_export_events``).

    Returns:
        Absolute path to the binary.

    Raises:
        FileNotFoundError: If the binary cannot be found.
    """
    for subdir in _SEARCH_SUBDIRS:
        candidate = empir_dir / subdir / name
        if candidate.exists():
            return candidate.resolve()

    # Searched locations for error message
    searched = [str(empir_dir / s / name) for s in _SEARCH_SUBDIRS]
    raise FileNotFoundError(
        f"EMPIR binary '{name}' not found. Searched:\n  " + "\n  ".join(searched)
    )


def find_binaries(empir_dir: Path, names: List[str]) -> Dict[str, Path]:
    """Locate multiple EMPIR binaries by name.

    Args:
        empir_dir: Root EMPIR directory to search in.
        names: List of binary filenames to find.

    Returns:
        Dict mapping binary name to its resolved path.

    Raises:
        FileNotFoundError: If any binary cannot be found.
    """
    result = {}
    missing = []

    for name in names:
        try:
            result[name] = find_binary(empir_dir, name)
        except FileNotFoundError:
            missing.append(name)

    if missing:
        searched_dirs = [str(empir_dir / s) for s in _SEARCH_SUBDIRS]
        raise FileNotFoundError(
            f"EMPIR binaries not found: {missing}\n"
            f"Searched directories:\n  " + "\n  ".join(searched_dirs) + "\n"
            f"EMPIR directory: {empir_dir}\n"
            f"Set the EMPIR_PATH environment variable to the correct location."
        )

    return result
