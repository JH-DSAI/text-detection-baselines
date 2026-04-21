"""evaluate.py — Evaluation pipeline for a machine text detection model.

Reports AUROC, FPR, TPR, calibration gap.

Outputs (in output_dir):
    results.jsonl       JSON record with evaluation metrics.
    summary.txt         Formatted metrics table with per-category breakdown.

Usage:
    python evaluate.py --config eval_config.yaml
    python evaluate.py --config eval_config.yaml --output-dir ./eval_results/
    python evaluate.py --config eval_config.yaml --features-dir ./cached_features/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np


def run_experiment(
    config: Any,
    experiment_id: int,
    val_categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run a single evaluation.

    Returns a dict of metrics, including per-category TPR if
    *val_categories* is provided.
    """
    # Binary predictions: flag=1, no_action=0, ood=excluded
    is_flag = np.array([d == "flag" for d in decisions])
    is_ood = np.array([d == "ood" for d in decisions])

    human_mask = val_labels == 0
    machine_mask = val_labels == 1

    # AUROC (on all non-OOD documents)
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

    non_ood = ~is_ood
    auroc = float("nan")
    if non_ood.sum() > 0 and len(np.unique(val_labels[non_ood])) == 2:
        auroc = roc_auc_score(val_labels[non_ood], agg_scores[non_ood])

    # FPR at threshold (false positives among human documents)
    n_human = human_mask.sum()
    n_machine = machine_mask.sum()
    fpr_at_tau = (is_flag & human_mask).sum() / max(n_human, 1)
    tpr_at_tau = (is_flag & machine_mask).sum() / max(n_machine, 1)
    ood_rate = is_ood.sum() / len(decisions)

    calibration_gap = abs(fpr_at_tau - config.calibration.alpha)

    metrics = {
        "experiment_id": experiment_id,
        "kernel": config.model.kernel,
        "use_ard": config.model.use_ard,
        "alpha": config.calibration.alpha,
        "aggregation_method": config.calibration.aggregation_method,
        "num_epochs": config.model.num_epochs,
        "train_fraction": config.calibration.train_fraction,
        "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
        "fpr_at_tau": round(float(fpr_at_tau), 4),
        "tpr_at_tau": round(float(tpr_at_tau), 4),
        "calibration_gap": round(float(calibration_gap), 4),
        "ood_rate": round(float(ood_rate), 4),
        "score_threshold": round(model.score_threshold, 4),
        "uncertainty_threshold": round(model.uncertainty_threshold, 6),
        "train_loss": round(float(losses[-1]), 6),
        "train_time_s": round(train_time, 1),
        "n_human_val": int(n_human),
        "n_machine_val": int(n_machine),
        "n_ood": int(is_ood.sum()),
    }

    # ── Per-category breakdown ───────────────────────────────────────
    if val_categories and any(val_categories):
        cat_arr = np.array(val_categories)
        per_category: dict[str, dict[str, Any]] = {}
        unique_cats = sorted(set(c for c in val_categories if c))
        for cat in unique_cats:
            cat_mask = cat_arr == cat
            cat_labels = val_labels[cat_mask]
            cat_flags = is_flag[cat_mask]
            cat_ood = is_ood[cat_mask]
            cat_scores = agg_scores[cat_mask]
            cat_human = cat_labels == 0
            cat_machine = cat_labels == 1
            n_cat = int(cat_mask.sum())
            n_cat_human = int(cat_human.sum())
            n_cat_machine = int(cat_machine.sum())

            # Per-category TPR (detection rate for machine text)
            cat_tpr = float(cat_flags[cat_machine].sum()) / max(n_cat_machine, 1)
            # Per-category FPR (false alarms for human text)
            cat_fpr = float(cat_flags[cat_human].sum()) / max(n_cat_human, 1)
            # Per-category AUROC
            cat_non_ood = ~cat_ood
            cat_auroc = float("nan")
            if cat_non_ood.sum() > 0 and len(np.unique(cat_labels[cat_non_ood])) == 2:
                cat_auroc = roc_auc_score(cat_labels[cat_non_ood], cat_scores[cat_non_ood])

            per_category[cat] = {
                "n": n_cat,
                "n_human": n_cat_human,
                "n_machine": n_cat_machine,
                "tpr": round(cat_tpr, 4),
                "fpr": round(cat_fpr, 4),
                "auroc": round(cat_auroc, 4) if not np.isnan(cat_auroc) else None,
            }

        metrics["per_category"] = per_category

    return metrics


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def write_summary(
    all_metrics: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write results.jsonl and summary.txt."""

    # ── results.jsonl ────────────────────────────────────────────────
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as fh:
        for m in all_metrics:
            fh.write(json.dumps(m) + "\n")
    logging.info("Results written to %s", results_path)

    # ── summary.txt ──────────────────────────────────────────────────
    # Sort by AUROC descending.
    sorted_metrics = sorted(
        all_metrics,
        key=lambda m: m.get("auroc") or 0.0,
        reverse=True,
    )

    columns = [
        ("ID", "experiment_id", "3d"),
        ("Kernel", "kernel", "10s"),
        ("ARD", "use_ard", "5s"),
        ("α", "alpha", ".2f"),
        ("Agg", "aggregation_method", "8s"),
        ("AUROC", "auroc", ".4f"),
        ("FPR@τ", "fpr_at_tau", ".4f"),
        ("TPR@τ", "tpr_at_tau", ".4f"),
        ("CalGap", "calibration_gap", ".4f"),
        ("OOD%", "ood_rate", ".3f"),
        ("τ", "score_threshold", ".4f"),
        ("Loss", "train_loss", ".4f"),
        ("Time", "train_time_s", ".0f"),
    ]

    header = "  ".join(f"{name:>{len(name) + 2}}" for name, _, _ in columns)
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for m in sorted_metrics:
        row_parts = []
        for name, key, fmt in columns:
            val = m.get(key) if m.get(key) is not None else "—"
            if val == "—":
                row_parts.append(f"{'—':>{len(name) + 2}}")
            elif fmt.endswith("s"):
                row_parts.append(f"{str(val):>{len(name) + 2}}")
            else:
                row_parts.append(f"{val:>{len(name) + 2}{fmt}}")
        lines.append("  ".join(row_parts))
    lines.append(sep)

    summary_text = "\n".join(lines) + "\n"

    # ── Per-category breakdown table ────────────────────────────────
    # Collect all categories across experiments.
    all_cats: set[str] = set()
    for m in sorted_metrics:
        pc = m.get("per_category", {})
        all_cats.update(pc.keys())

    if all_cats:
        cat_lines: list[str] = []
        cat_lines.append("")
        cat_lines.append("Detection Rate (TPR) by Category")
        cat_lines.append("================================")
        cat_lines.append("")

        # Separate human and machine categories.
        human_cats = sorted(c for c in all_cats if c.lower() == "human")
        machine_cats = sorted(c for c in all_cats if c.lower() != "human")
        ordered_cats = machine_cats + human_cats  # human last

        # Column width: enough for experiment labels.
        id_width = 6  # "Exp 01"
        cat_col_width = max(len(c) for c in ordered_cats) + 2
        exp_col_width = max(id_width, 8)  # room for "0.1234"

        # Header row: category | Exp 01 | Exp 02 | ...
        header_parts = [f"{'Category':<{cat_col_width}}"]
        for m in sorted_metrics:
            eid = m.get("experiment_id", "?")
            label = f"Exp {eid:>2}"
            header_parts.append(f"{label:>{exp_col_width}}")
        cat_header = "  ".join(header_parts)
        cat_sep = "-" * len(cat_header)

        # Sub-header: show kernel + agg for each experiment.
        config_parts = [f"{'':<{cat_col_width}}"]
        for m in sorted_metrics:
            kernel_short = m.get("kernel", "?")[:6]
            agg_short = m.get("aggregation_method", "?")[:3]
            config_label = f"{kernel_short}/{agg_short}"
            config_parts.append(f"{config_label:>{exp_col_width}}")
        config_line = "  ".join(config_parts)

        cat_lines.append(cat_sep)
        cat_lines.append(cat_header)
        cat_lines.append(config_line)
        cat_lines.append(cat_sep)

        # Machine categories: show TPR.
        for cat in machine_cats:
            parts = [f"{cat:<{cat_col_width}}"]
            for m in sorted_metrics:
                pc = m.get("per_category", {})
                if cat in pc:
                    tpr = pc[cat].get("tpr")
                    n = pc[cat].get("n_machine", pc[cat].get("n", 0))
                    if tpr is not None:
                        parts.append(f"{tpr:>{exp_col_width}.3f}")
                    else:
                        parts.append(f"{'—':>{exp_col_width}}")
                else:
                    parts.append(f"{'—':>{exp_col_width}}")
            cat_lines.append("  ".join(parts))

        # Separator before human row.
        if human_cats:
            cat_lines.append(cat_sep)
            # Human categories: show FPR (false alarm rate).
            for cat in human_cats:
                label = f"{cat} (FPR)"
                parts = [f"{label:<{cat_col_width}}"]
                for m in sorted_metrics:
                    pc = m.get("per_category", {})
                    if cat in pc:
                        fpr = pc[cat].get("fpr")
                        if fpr is not None:
                            parts.append(f"{fpr:>{exp_col_width}.3f}")
                        else:
                            parts.append(f"{'—':>{exp_col_width}}")
                    else:
                        parts.append(f"{'—':>{exp_col_width}}")
                cat_lines.append("  ".join(parts))

        cat_lines.append(cat_sep)

        # Category counts row.
        cat_lines.append("")
        cat_lines.append("Category sample counts:")
        # Use first experiment's per_category for counts (same validation set).
        ref = sorted_metrics[0].get("per_category", {})
        for cat in ordered_cats:
            info = ref.get(cat, {})
            n = info.get("n", 0)
            n_h = info.get("n_human", 0)
            n_m = info.get("n_machine", 0)
            cat_lines.append(f"  {cat}: {n} documents ({n_h} human, {n_m} machine)")

        category_text = "\n".join(cat_lines) + "\n"
        summary_text += category_text

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(summary_text)
    logging.info("Summary written to %s", summary_path)

    # Print to console.
    print("\n" + summary_text)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to evaluation YAML config (e.g., eval_config.yaml).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Override output directory from YAML config.",
)
@click.option(
    "--features-dir",
    type=click.Path(),
    default=None,
    help=(
        "Directory of pre-extracted features (output of fit_features.py). "
        "If provided, skips feature extraction."
    ),
)
def main(
    config: str,
    output_dir: str | None,
    features_dir: str | None,
) -> None:
    """Evaluate a machine-generated text detector on validation data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(output_dir or cfg.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Validation set: %d human + %d machine",
        int((val_y == 0).sum()),
        int((val_y == 1).sum()),
    )

    logging.info("Running evaluation")

    metrics = run_experiment(
        config=cfg,
        x_dict=x_dict,
        labels=labels,
        val_x=val_x,
        val_y=val_y,
        metadata=metadata,
        experiment_id=1,
        val_categories=val_categories,
    )

    logging.info(
        "AUROC=%.4f  FPR@τ=%.4f  TPR@τ=%.4f  OOD=%.1f%%",
        metrics.get("auroc") or 0.0,
        metrics["fpr_at_tau"],
        metrics["tpr_at_tau"],
        100 * metrics["ood_rate"],
    )

    # ── Write results ────────────────────────────────────────────────
    write_summary([metrics], out_dir)

    logging.info("\nEvaluation complete.")
    logging.info("Results: %s", out_dir)


if __name__ == "__main__":
    main()
