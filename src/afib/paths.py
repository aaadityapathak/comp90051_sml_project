"""Resolve local datasets independently of the working directory.

The submission contains code and experiment results. Raw and prepared datasets
live outside it; environment variables allow a different local storage layout.
"""

from __future__ import annotations

import os
from pathlib import Path


def _external_directory(project, variable, subdirectory):
    project = Path(project).resolve()
    configured = os.environ.get(variable)
    directory = Path(configured).expanduser() if configured else project.parent / 'afib_data' / subdirectory
    # A relative override has the same meaning in notebooks and CLI runners.
    return (project / directory).resolve() if not directory.is_absolute() else directory.resolve()


def data_directory(project):
    """Raw sources: AFIB_DATA_DIR, otherwise ../afib_data/raw."""
    return _external_directory(project, 'AFIB_DATA_DIR', 'raw')


def processed_directory(project):
    """Prepared inputs: AFIB_PROCESSED_DIR, otherwise ../afib_data/processed."""
    return _external_directory(project, 'AFIB_PROCESSED_DIR', 'processed')


def source_path(project, recorded_path, *, data_dir=None):
    """Resolve a manifest's portable ``data/...`` source identifier.

    The prefix is retained in historical manifests so moving the input folder
    does not alter the experiment's recorded source identities or file hashes.
    """
    relative = Path(recorded_path)
    if relative.is_absolute() or not relative.parts or relative.parts[0] != 'data' or '..' in relative.parts:
        raise ValueError(f'Expected a relative data/... source identifier: {recorded_path}')
    root = Path(data_dir).resolve() if data_dir is not None else data_directory(project)
    return root.joinpath(*relative.parts[1:])
