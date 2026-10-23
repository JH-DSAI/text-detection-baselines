"""Shared fixtures.

The CLI tests invoke the click commands in-process with :class:`click.testing.CliRunner`,
which is fast but shares module-level state across invocations in a way subprocesses do
not. The fixtures here contain that state.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import pytest
from click.testing import CliRunner

from text_detection_baselines.datasets import DATASET_REGISTRY
from text_detection_baselines.models.azure_batch import AzureBatchConfig
from text_detection_baselines.settings import ENV_PREFIX, Settings

#: Every settings model, all of which read ``TDB_*`` variables and ``.env``.
_SETTINGS_MODELS = (Settings, AzureBatchConfig)


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

    Autouse, and load-bearing.
    ``test_cli_can_run_twice_in_one_process`` clears the root handlers to check that the
    CLI's command body does not call ``logging.basicConfig`` -- which would bind a handler
    to the ``sys.stderr`` of the first ``CliRunner`` invocation and, being a no-op once the
    root logger has handlers, silently swallow every later invocation's logs. That test
    relies on this fixture to restore pytest's own logging handlers back on teardown.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Hide the developer's own ``TDB_*`` configuration from every test.

    The settings models read the process environment and a ``.env`` file in the working
    directory, which for a normal pytest run is the checkout root, where a developer's real ``.env``
    lives. Without this, a machine configured to reach a real endpoint would silently supply
    values the tests expect to be absent, or override the defaults they assert on. Clearing
    the environment alone is not enough, as ``.env`` fills in whatever it lacks.
    """
    for name in list(os.environ):
        if name.startswith(ENV_PREFIX):
            monkeypatch.delenv(name)
    for settings_cls in _SETTINGS_MODELS:
        monkeypatch.setitem(settings_cls.model_config, "env_file", None)


@pytest.fixture
def dotenv(monkeypatch, tmp_path) -> Callable[..., None]:
    """Return a function that writes a ``.env`` file and points every settings model at it.

    Undoes :func:`_isolated_settings` for ``.env`` only: the environment stays cleared.
    """

    def write(*lines: str) -> None:
        path = tmp_path / ".env"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for settings_cls in _SETTINGS_MODELS:
            monkeypatch.setitem(settings_cls.model_config, "env_file", path)

    return write
