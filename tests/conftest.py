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
    """Restore root logger handlers and level around every test.

    Autouse, and load-bearing rather than precautionary: keep it that way.
    ``test_cli_can_run_twice_in_one_process`` clears the root handlers to check that the
    CLI's command body does not call ``logging.basicConfig`` -- which would bind a handler
    to the ``sys.stderr`` of the first ``CliRunner`` invocation and, being a no-op once the
    root logger has handlers, silently swallow every later invocation's logs. That test
    relies on this fixture to put pytest's own logging handlers back, and pytest reports
    nothing about the run without them.

    Clearing is also why the check has to be written that way: pytest installs those
    handlers before any test body runs, so a stray ``basicConfig`` call would do nothing
    under the test suite and go unnoticed wherever it lived.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
