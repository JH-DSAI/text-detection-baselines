"""Command-line entry point for converting the upstream GEDE database.

Run as ``pixi run prepare-gede`` or ``python -m text_detection_baselines.prepare_gede``.
The corpus itself is obtained from the upstream project by the user; see
``datasets/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import click

from .datasets import default_gede_output_path
from .datasets.gede import ATTRIBUTION_NOTICE, GedePrepReport, GedeSourceError, metadata_path, prepare_gede


def _format_counts(title: str, counts: dict[str, int]) -> str:
    rows = "\n".join(f"  {name}: {count}" for name, count in counts.items())
    return f"{title}\n{rows}"


def _summarize(report: GedePrepReport) -> str:
    sections = [
        f"Wrote {report.n_records} records to {report.output}",
        f"Provenance: {metadata_path(report.output)}",
        _format_counts("Labels:", report.counts_by_label),
        _format_counts("Source corpora:", report.counts_by_dataset),
        _format_counts("Contribution levels:", report.counts_by_category),
    ]
    if report.unmapped_categories:
        sections.append(
            _format_counts(
                "Contribution levels with no known normalization (passed through unchanged):",
                report.unmapped_categories,
            ),
        )
    if report.dropped_blank_answers:
        sections.append(
            _format_counts(
                f"Dropped {report.dropped_blank_answers} records with empty text:",
                report.dropped_by_dataset,
            ),
        )
    if report.label_disagreements:
        sections.append(
            f"WARNING: {report.label_disagreements} records where is_human disagrees with text_author. "
            "The is_human column was used. Please report this upstream.",
        )
    return "\n".join(sections)


@click.command()
@click.option(
    "--source",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Upstream GEDE database.db, with add_aae_to_database.py already run against it.",
)
@click.option(
    "--out",
    "out",
    default=None,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Destination JSONL file. Defaults to the location the 'gede' dataset resolves to.",
)
@click.option(
    "--allow-missing-aae",
    is_flag=True,
    default=False,
    help="Drop records with empty text instead of failing. Yields an incomplete corpus.",
)
def main(source: Path, out: Path | None, allow_missing_aae: bool) -> None:
    """Convert the upstream GEDE SQLite database into a loadable JSONL dataset."""
    destination = out if out is not None else default_gede_output_path()
    try:
        report = prepare_gede(source, destination, allow_missing_aae=allow_missing_aae)
    except GedeSourceError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_summarize(report))
    click.echo("")
    click.echo(ATTRIBUTION_NOTICE)


if __name__ == "__main__":
    main()
