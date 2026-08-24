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
    DEFAULT_CATEGORY_KEY,
    DEFAULT_LABEL_KEY,
    DEFAULT_TEXT_KEY,
    GEDE_PREPARE_HINT,
    DatasetSpec,
    dataset_available,
    get_dataset_spec,
    get_default_dataset_names,
    list_registered_datasets,
    register_file_dataset,
)
from .evaluate import build_results_tree, evaluate_model_on_dataset
from .models import build_model, get_default_model_names, list_registered_models

LOGGER = logging.getLogger(__name__)


def _describe_registered_datasets() -> str:
    """List registered datasets, marking any whose data file is not present."""
    described = []
    for name in list_registered_datasets():
        available = dataset_available(get_dataset_spec(name))
        described.append(name if available else f"{name} (not prepared)")
    return ", ".join(described)


_REGISTERED_DATASETS = _describe_registered_datasets()
_DEFAULT_DATASETS = ", ".join(get_default_dataset_names())
_REGISTERED_MODELS = ", ".join(list_registered_models())
_DEFAULT_MODELS = ", ".join(get_default_model_names())


def _raise_for_unavailable_dataset(spec: DatasetSpec) -> None:
    """Fail with an actionable message when a dataset's file is missing.

    Without this the failure surfaces from ``read_text`` as a ``FileNotFoundError``
    naming a path the user never chose, which does not say that the dataset has to
    be prepared first.
    """
    if dataset_available(spec):
        return

    lines = [
        f"Dataset '{spec.name}' is registered but its data file is not present:",
        f"  {spec.path}",
    ]
    if spec.name == "gede":
        lines += [
            "",
            "The GEDE corpus is not redistributed with this package. Obtain it from",
            "the upstream project and convert it with:",
            f"  {GEDE_PREPARE_HINT}",
            "",
            "See datasets/README.md for the acquisition steps and licence terms.",
        ]
    else:
        lines += [
            "",
            "Point --register-file-dataset at an existing file, or choose another dataset.",
        ]
    raise click.ClickException("\n".join(lines))


class NamePathParamType(click.ParamType):
    """Parse and validate a ``NAME=PATH`` CLI value."""

    name = "name=path"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[str, Path]:
        if not isinstance(value, str) or "=" not in value:
            self.fail("entries must look like NAME=PATH", param, ctx)

        name, raw_path = value.split("=", 1)
        dataset_path = Path(raw_path).expanduser()
        if not dataset_path.exists():
            self.fail(f"Dataset path does not exist: {dataset_path}", param, ctx)

        return name, dataset_path


NAME_PATH = NamePathParamType()


def _raise_for_unknown_names(
    *,
    names: tuple[str, ...],
    valid_names: tuple[str, ...],
    kind: str,
    param_hint: str,
) -> None:
    """Raise ``click.BadParameter`` when one or more names are unknown."""
    valid_lookup = {name.lower() for name in valid_names}
    unknown = [name for name in names if name.lower() not in valid_lookup]
    if unknown:
        valid = ", ".join(valid_names)
        raise click.BadParameter(
            f"Unknown {kind}(s): {', '.join(unknown)}. Valid options: {valid}",
            param_hint=param_hint,
        )


def _unique_preserve_order(names: list[str]) -> tuple[str, ...]:
    """Return de-duplicated names while preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
    return tuple(ordered)


def _resolve_selection(
    *,
    explicit_names: tuple[str, ...],
    default_names: tuple[str, ...],
    all_names: tuple[str, ...],
    runtime_names: tuple[str, ...],
    include_all: bool,
    excluded_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the final ordered selection for a registry-backed CLI option."""
    selected: list[str] = []
    if include_all:
        selected.extend(all_names)
    else:
        selected.extend(default_names)
    selected.extend(runtime_names)
    selected.extend(explicit_names)
    excluded = {name.lower() for name in excluded_names}
    return tuple(name for name in _unique_preserve_order(selected) if name.lower() not in excluded)


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    """Format a metric value for display in a rich table."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
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
    summary = Table(
        title="Text Detection Metrics",
        caption="All models are stubs. dummy-* use fixed, arbitrary weights fit to nothing.",
        show_lines=False,
    )
    for col in ("dataset", "model", "AUROC", "AUROC@1%", "AP", "FPR@tau", "TPR@tau", "CalGap", "OOD%", "tau"):
        summary.add_column(col, justify="left" if col in {"dataset", "model"} else "right")

    for row in sorted(overall_rows, key=lambda r: (r["dataset"], r["model"])):
        summary.add_row(
            row["dataset"],
            row["model"],
            _fmt(row.get("auroc")),
            _fmt(row.get("auroc_at_1pct")),
            _fmt(row.get("average_precision")),
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
    cat_cols = [
        "dataset",
        "model",
        "category",
        "AUROC",
        "AUROC@1%",
        "AP",
        "FPR@tau",
        "TPR@tau",
        "CalGap",
        "OOD%",
        "n",
    ]
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
            _fmt(row.get("auroc_at_1pct")),
            _fmt(row.get("average_precision")),
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
    "-d",
    type=str,
    multiple=True,
    help=(
        "Registered dataset name to evaluate. Repeat to evaluate multiple datasets. "
        f"Available: {_REGISTERED_DATASETS}. Defaults: {_DEFAULT_DATASETS}."
    ),
)
@click.option(
    "--model",
    "models",
    "-m",
    type=str,
    multiple=True,
    help=(
        "Registered model name to evaluate. Repeat to evaluate multiple models. "
        f"Available: {_REGISTERED_MODELS}. Defaults: {_DEFAULT_MODELS}."
    ),
)
@click.option(
    "--register-file-dataset",
    "runtime_file_datasets",
    type=NAME_PATH,
    multiple=True,
    help="Register file dataset at runtime as NAME=PATH and include it in evaluation. Repeatable.",
)
@click.option(
    "--exclude-dataset",
    "exclude_datasets",
    "-xd",
    type=str,
    multiple=True,
    help="Exclude dataset(s) by registered name. Repeatable.",
)
@click.option(
    "--exclude-model",
    "exclude_models",
    "-xm",
    type=str,
    multiple=True,
    help="Exclude model(s) by registered name. Repeatable.",
)
@click.option(
    "--all-datasets",
    is_flag=True,
    help="Evaluate on all registered datasets, including any registered at runtime.",
)
@click.option(
    "--all-models",
    is_flag=True,
    help="Evaluate on all registered models, including non-default models.",
)
@click.option(
    "--all",
    "-a",
    "select_all_flag",
    is_flag=True,
    help="Equivalent to specifying both --all-datasets and --all-models.",
)
@click.option("--target-alpha", type=float, default=0.05, show_default=True, help="Target FPR for learning tau.")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("evaluation_results"),
    show_default=True,
    help="Directory for optional structured exports.",
)
@click.option(
    "--export",
    "export_formats",
    "-e",
    type=click.Choice(["csv", "json", "yaml"], case_sensitive=False),
    multiple=True,
    help="Write structured output files in selected format(s).",
)
@click.option(
    "--text-key",
    type=str,
    default=DEFAULT_TEXT_KEY,
    show_default=True,
    help=(
        "Field name for text in datasets registered via --register-file-dataset. "
        "Applies to every such entry in the run; built-in datasets keep their own schema."
    ),
)
@click.option(
    "--label-key",
    type=str,
    default=DEFAULT_LABEL_KEY,
    show_default=True,
    help=(
        "Field name for the label in datasets registered via --register-file-dataset. "
        "Applies to every such entry in the run; built-in datasets keep their own schema."
    ),
)
@click.option(
    "--category-key",
    type=str,
    default=DEFAULT_CATEGORY_KEY,
    show_default=True,
    help=(
        "Field name for per-category grouping in datasets registered via "
        "--register-file-dataset. Applies to every such entry in the run; built-in "
        "datasets keep their own schema."
    ),
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
    runtime_file_datasets: tuple[tuple[str, Path], ...],
    exclude_datasets: tuple[str, ...],
    exclude_models: tuple[str, ...],
    all_datasets: bool,
    all_models: bool,
    select_all_flag: bool,
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

    all_datasets = all_datasets or select_all_flag
    all_models = all_models or select_all_flag

    runtime_dataset_names: list[str] = []
    for name, dataset_path in runtime_file_datasets:
        register_file_dataset(
            name=name,
            path=dataset_path,
            text_key=text_key,
            label_key=label_key,
            category_key=category_key,
        )
        runtime_dataset_names.append(name)

    registered_datasets = tuple(list_registered_datasets())
    registered_models = tuple(list_registered_models())

    _raise_for_unknown_names(
        names=datasets,
        valid_names=registered_datasets,
        kind="dataset",
        param_hint="--dataset",
    )
    _raise_for_unknown_names(
        names=models,
        valid_names=registered_models,
        kind="model",
        param_hint="--model",
    )
    _raise_for_unknown_names(
        names=exclude_datasets,
        valid_names=registered_datasets,
        kind="dataset",
        param_hint="--exclude-dataset",
    )
    _raise_for_unknown_names(
        names=exclude_models,
        valid_names=registered_models,
        kind="model",
        param_hint="--exclude-model",
    )

    selected_datasets = _resolve_selection(
        explicit_names=datasets,
        default_names=get_default_dataset_names(),
        all_names=registered_datasets,
        runtime_names=tuple(runtime_dataset_names),
        include_all=all_datasets,
        excluded_names=exclude_datasets,
    )
    selected_models = _resolve_selection(
        explicit_names=models,
        default_names=get_default_model_names(),
        all_names=registered_models,
        runtime_names=(),
        include_all=all_models,
        excluded_names=exclude_models,
    )

    if not selected_datasets:
        raise click.BadParameter(
            "No datasets selected. Use --dataset and/or --register-file-dataset, or avoid excluding every dataset.",
        )

    if not selected_models:
        raise click.BadParameter(
            "No models selected. Use --model, or avoid excluding every model.",
        )

    # Resolved and checked before any model is built: constructing ``smollm2``
    # downloads weights, which should not happen only to fail on a missing dataset.
    specs = [get_dataset_spec(name) for name in selected_datasets]
    for spec in specs:
        _raise_for_unavailable_dataset(spec)

    model_objs = [build_model(name, ood_margin=ood_margin, seed=seed) for name in selected_models]

    run_results: list[tuple[str, str, dict, dict]] = []

    for dataset in specs:
        for model in model_objs:
            LOGGER.info("Evaluating model=%s on dataset=%s", model.model_name, dataset.name)
            overall, per_cat = evaluate_model_on_dataset(
                dataset_path=dataset.path,
                model=model,
                target_alpha=target_alpha,
                text_key=dataset.text_key,
                label_key=dataset.label_key,
                category_key=dataset.category_key,
            )
            run_results.append((dataset.name, model.model_name, overall, per_cat))

    tree = build_results_tree(run_results)
    render_console_tables(tree)

    if export_formats:
        output_dir.mkdir(parents=True, exist_ok=True)
        export_results(
            output_dir=output_dir,
            export_formats=tuple(fmt.lower() for fmt in export_formats),
            tree=tree,
        )
        LOGGER.info("Results saved to %s", output_dir)

    LOGGER.info("Evaluation complete")


if __name__ == "__main__":
    # Configured at the process entry point rather than inside ``main``: ``basicConfig``
    # binds a handler to the ``sys.stderr`` in effect at the first call and is a no-op
    # afterwards, so calling it from the command body silently sends the logs of every
    # later in-process invocation to the first caller's stream, which causes problems in
    # testing.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
