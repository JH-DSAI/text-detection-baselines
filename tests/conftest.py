"""Shared fixtures.

The CLI tests invoke the click commands in-process with :class:`click.testing.CliRunner`,
which is fast but shares module-level state across invocations in a way subprocesses do
not. The fixtures here contain that state.
"""

from __future__ import annotations

import logging

import pytest
from click.testing import CliRunner

from text_detection_baselines.datasets import DATASET_REGISTRY


@pytest.fixture
def runner() -> CliRunner:
    """A click test runner.

    Tests pass ``tmp_path`` rather than using ``isolated_filesystem``, so the runner needs
    no configuration.
    """
    return CliRunner()


@pytest.fixture
def clean_registry():
    """Restore the global dataset registry, so runtime registrations do not leak.

    Needed by any test that registers a dataset, whether directly or via the CLI's
    ``--register-file-dataset``.
    """
    snapshot = dict(DATASET_REGISTRY)
    try:
        yield
    finally:
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(snapshot)


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Restore root logger handlers around every test.

    ``logging.basicConfig`` binds a handler to the ``sys.stderr`` in effect when it first
    runs and is a no-op thereafter, so a handler installed under one ``CliRunner``
    invocation would swallow the logs of every later one. Nothing in the package
    configures logging outside its ``__main__`` guard; this keeps it that way.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
