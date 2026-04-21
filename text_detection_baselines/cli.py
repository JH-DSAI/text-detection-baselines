"""Command-line interface for text detection baseline evaluation.

This module wires together the core evaluation logic, rich console output, and
file export.  Run it with::

    python -m text_detection_baselines.cli [OPTIONS]

or via the pixi task::

    pixi run main
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from .datasets import (
    get_dataset_spec,
    get_default_dataset_names,
    list_registered_datasets,
    register_file_dataset,
)
from .evaluate import build_results_tree, evaluate_model_on_dataset
from .models import build_model, get_default_model_names, list_registered_models

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    """Format a metric value for display in a rich table."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_console_tables(tree: dict[str, Any]) -> None:
    """Render overall and per-category metrics as rich tables.

    Args:
        tree: Nested results dict produced by :func:`build_results_tree`.

    Returns:
        None. Tables are printed to the active console.
    """
    console = Console(record=True)

    # ── Overall metrics ──────────────────────────────────────────────
    overall_rows = _flatten_overall(tree)
    summary = Table(title="Text Detection Metrics", show_lines=False)
    for col in ("dataset", "model", "AUROC", "FPR@tau", "TPR@tau", "CalGap", "OOD%", "tau"):
        summary.add_column(col, justify="left" if col in {"dataset", "model"} else "right")

    for row in sorted(overall_rows, key=lambda r: (r["dataset"], r["model"])):
        summary.add_row(
            row["dataset"],
            row["model"],
            _fmt(row.get("auroc")),
            _fmt(row.get("fpr_at_tau")),
            _fmt(row.get("tpr_at_tau")),
            _fmt(row.get("calibration_gap")),
            _fmt(row.get("ood_percent")),
            _fmt(row.get("tau")),
        )
    console.print(summary)

    # ── Calibration metrics (normalized models only) ─────────────────
    if any(r.get("normalized_scores") for r in overall_rows):
        cal = Table(title="Normalized Score Calibration Metrics", show_lines=False)
        for col in ("dataset", "model", "Brier", "ECE"):
            cal.add_column(col, justify="left" if col in {"dataset", "model"} else "right")
        for row in sorted(overall_rows, key=lambda r: (r["dataset"], r["model"])):
            if not row.get("normalized_scores"):
                continue
            cal.add_row(row["dataset"], row["model"], _fmt(row.get("brier")), _fmt(row.get("ece")))
        console.print(cal)

    # ── Per-category metrics ─────────────────────────────────────────
    per_cat_rows = _flatten_per_category(tree)

    has_normalized = any(r.get("normalized_scores") for r in per_cat_rows)
    cat_cols = ["dataset", "model", "category", "AUROC", "FPR@tau", "TPR@tau", "CalGap", "OOD%", "n"]
    if has_normalized:
        cat_cols += ["Brier", "ECE"]

    per_cat_table = Table(title="Per-Category Metrics (contribution_level)", show_lines=False)
    for col in cat_cols:
        per_cat_table.add_column(col, justify="left" if col in {"dataset", "model", "category"} else "right")

    for row in sorted(per_cat_rows, key=lambda r: (r["dataset"], r["model"], r["category"])):
        cells = [
            row["dataset"],
            row["model"],
            row["category"],
            _fmt(row.get("auroc")),
            _fmt(row.get("fpr_at_tau")),
            _fmt(row.get("tpr_at_tau")),
            _fmt(row.get("calibration_gap")),
            _fmt(row.get("ood_percent")),
            str(row.get("n_samples", "-")),
        ]
        if has_normalized:
            cells += [_fmt(row.get("brier")), _fmt(row.get("ece"))]
        per_cat_table.add_row(*cells)

    console.print(per_cat_table)

    return None


# ---------------------------------------------------------------------------
# Flat-row helpers for CSV export
# ---------------------------------------------------------------------------


def _flatten_overall(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a flat list of overall-metric rows."""
    rows = []
    for dataset, models in tree.get("overall", {}).items():
        for model, metrics in models.items():
            rows.append({"dataset": dataset, "model": model, **metrics})
    return rows


def _flatten_per_category(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a flat list of per-category-metric rows."""
    rows = []
    for dataset, cats in tree.get("per-category", {}).items():
        for category, models in cats.items():
            for model, metrics in models.items():
                rows.append({"dataset": dataset, "category": category, "model": model, **metrics})
    return rows


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------


def export_results(
    output_dir: Path,
    export_formats: tuple[str, ...],
    tree: dict[str, Any],
) -> None:
    """Write evaluation results in the requested formats.

    Files created:
    - ``metrics.json`` / ``metrics.yaml`` — consolidated nested structure.
    - ``overall-metrics.csv`` — one row per (dataset, model).
    - ``per-category-metrics.csv`` — one row per (dataset, category, model).
    """
    overall_rows = _flatten_overall(tree)
    per_cat_rows = _flatten_per_category(tree)

    for fmt in export_formats:
        if fmt == "json":
            (output_dir / "metrics.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
        elif fmt == "yaml":
            (output_dir / "metrics.yaml").write_text(yaml.safe_dump(tree, sort_keys=False), encoding="utf-8")
        elif fmt == "csv":
            _write_csv(output_dir / "overall-metrics.csv", overall_rows)
            _write_csv(output_dir / "per-category-metrics.csv", per_cat_rows)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat dicts to a CSV file."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--dataset",
    "datasets",
    type=str,
    multiple=True,
    help="Registered dataset name to evaluate. Repeat to evaluate multiple datasets.",
)
@click.option(
    "--model",
    "models",
    type=str,
    multiple=True,
    help="Registered model name to evaluate. Repeat to evaluate multiple models.",
)
@click.option(
    "--register-file-dataset",
    "runtime_file_datasets",
    type=str,
    multiple=True,
    help="Register file dataset at runtime as NAME=PATH. Repeatable.",
)
@click.option("--target-alpha", type=float, default=0.05, show_default=True, help="Target FPR for learning tau.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("evaluation_results"),
    show_default=True,
    help="Directory for optional structured exports.",
)
@click.option(
    "--export",
    "export_formats",
    type=click.Choice(["csv", "json", "yaml"], case_sensitive=False),
    multiple=True,
    help="Write structured output files in selected format(s).",
)
@click.option("--text-key", type=str, default="answer", show_default=True, help="Dataset field containing the text.")
@click.option("--label-key", type=str, default="label", show_default=True, help="Dataset field containing the label.")
@click.option(
    "--category-key",
    type=str,
    default="contribution_level",
    show_default=True,
    help="Dataset field used for per-category grouping.",
)
@click.option("--seed", type=int, default=7, show_default=True, help="Random seed for model stubs.")
@click.option(
    "--ood-margin",
    type=float,
    default=0.08,
    show_default=True,
    help="Uncertainty margin for OOD flagging.",
)
def main(
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    runtime_file_datasets: tuple[str, ...],
    target_alpha: float,
    output_dir: Path,
    export_formats: tuple[str, ...],
    text_key: str,
    label_key: str,
    category_key: str,
    seed: int,
    ood_margin: float,
) -> None:
    """Evaluate registered text detection models on one or more datasets."""
    if not 0.0 <= target_alpha <= 1.0:
        raise click.BadParameter("target-alpha must be in [0, 1]")

    for item in runtime_file_datasets:
        if "=" not in item:
            raise click.BadParameter(
                "register-file-dataset entries must look like NAME=PATH",
                param_hint="--register-file-dataset",
            )
        name, raw_path = item.split("=", 1)
        dataset_path = Path(raw_path).expanduser()
        if not dataset_path.exists():
            raise click.BadParameter(
                f"Dataset path does not exist: {dataset_path}",
                param_hint="--register-file-dataset",
            )
        register_file_dataset(name=name, path=dataset_path)

    selected_datasets = tuple(datasets) if datasets else get_default_dataset_names()
    selected_models = tuple(models) if models else get_default_model_names()

    unknown_datasets = [name for name in selected_datasets if name.lower() not in set(list_registered_datasets())]
    if unknown_datasets:
        valid = ", ".join(list_registered_datasets())
        raise click.BadParameter(f"Unknown dataset(s): {', '.join(unknown_datasets)}. Valid options: {valid}")

    unknown_models = [name for name in selected_models if name.lower() not in set(list_registered_models())]
    if unknown_models:
        valid = ", ".join(list_registered_models())
        raise click.BadParameter(f"Unknown model(s): {', '.join(unknown_models)}. Valid options: {valid}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_objs = [build_model(name, ood_margin=ood_margin, seed=seed) for name in selected_models]

    run_results: list[tuple[str, str, dict, dict]] = []

    for dataset_name in selected_datasets:
        dataset = get_dataset_spec(dataset_name)
        for model in model_objs:
            LOGGER.info("Evaluating model=%s on dataset=%s", model.model_name, dataset.name)
            overall, per_cat = evaluate_model_on_dataset(
                dataset_path=dataset.path,
                model=model,
                target_alpha=target_alpha,
                text_key=dataset.text_key or text_key,
                label_key=dataset.label_key or label_key,
                category_key=dataset.category_key or category_key,
            )
            run_results.append((dataset.name, model.model_name, overall, per_cat))

    tree = build_results_tree(run_results)
    render_console_tables(tree)

    if export_formats:
        export_results(
            output_dir=output_dir,
            export_formats=tuple(fmt.lower() for fmt in export_formats),
            tree=tree,
        )
        LOGGER.info("Results saved to %s", output_dir)

    LOGGER.info("Evaluation complete")


if __name__ == "__main__":
    main()
