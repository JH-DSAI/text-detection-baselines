"""Package-wide settings, read from ``TDB_*`` environment variables.

Each setting is read from the process environment, falling back to a ``.env`` file
in the working directory. An optional component keeps its settings in a model of
its own under a longer prefix -- :class:`~.models.azure_batch.AzureBatchConfig`
reads ``TDB_AZURE_BATCH_*`` -- so that its required fields are validated only when
the component is used, and a checkout without it configured still loads these.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Prefix shared by every environment variable this package reads.
ENV_PREFIX = "TDB_"


class Settings(BaseSettings):
    """Package-wide settings."""

    # ``env_ignore_empty`` so a blank variable reads as unset.
    # ``only_existing`` so the ``.env`` file shared with the component
    # models is read field by field; by default every key it does not declare is
    # rejected. The stricter ``match_prefix`` would catch typos, but would also
    # claim the components' keys, which all start with this model's prefix.
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_ignore_empty=True,
        dotenv_filtering="only_existing",
        frozen=True,
    )

    # Where the prepared GEDE dataset is written and looked up, overriding the
    # search in ``datasets.resolve_gede_path``.
    gede_path: Path | None = None
