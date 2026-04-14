#!/usr/bin/env python3
"""
evaluate.py — YAML-driven evaluation pipeline for the multi-view detector.

Runs the full fit → train → calibrate → score pipeline against held-out
validation data, reporting classification metrics for each configuration.

When the YAML includes an ``evaluation.grid`` section, this script expands
the grid into individual experiment configs and evaluates each combination.

Outputs (in ``evaluation.output_dir``):
    results.jsonl       One JSON record per configuration with all metrics.
    summary.txt         Pretty-printed comparison table.
    best_config.yaml    Production-ready YAML for the best configuration
                        (evaluation section stripped).

Usage:
    uv run python evaluate.py --config eval_config.yaml
    uv run python evaluate.py --config eval_config.yaml --quick
    uv run python evaluate.py --config eval_config.yaml --output-dir eval_v2/
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

# Maps grid parameter names to the (section, field) they override in the
# PipelineConfig dataclass hierarchy.
_GRID_PARAM_MAP: dict[str, tuple[str, str]] = {
    "kernel": ("model", "kernel"),
    "use_ard": ("model", "use_ard"),
    "num_epochs": ("model", "num_epochs"),
    "lr": ("model", "lr"),
    "alpha": ("calibration", "alpha"),
    "num_folds": ("calibration", "num_folds"),
    "train_fraction": ("calibration", "train_fraction"),
    "aggregation_method": ("calibration", "aggregation_method"),
    "use_mahalanobis": ("features", "use_mahalanobis"),
}

# Reduced grid for --quick mode.
_QUICK_GRID: dict[str, list] = {
    "kernel": ["matern52"],
    "alpha": [0.05],
    "aggregation_method": ["max", "weighted"],
}


def expand_grid(
    config: Any,
    grid: dict[str, list] | None,
) -> list[tuple[dict[str, Any], Any]]:
    """Expand a parameter grid into individual experiment configs.

    Args:
        config: Base PipelineConfig.
        grid:   Dict mapping parameter names to lists of values.

    Returns:
        List of (param_dict, config_copy) tuples.
    """
    if not grid:
        return [({}, config)]

    param_names = list(grid.keys())
    param_values = [grid[k] for k in param_names]
    combos = list(itertools.product(*param_values))

    results: list[tuple[dict[str, Any], Any]] = []
    for combo in combos:
        cfg = deepcopy(config)
        params: dict[str, Any] = {}
        for name, value in zip(param_names, combo):
            params[name] = value
            if name in _GRID_PARAM_MAP:
                section, field = _GRID_PARAM_MAP[name]
                setattr(getattr(cfg, section), field, value)
            else:
                logging.warning("Unknown grid parameter: %s", name)
        results.append((params, cfg))

    return results


# ---------------------------------------------------------------------------
# Feature extraction (Step 1)
# ---------------------------------------------------------------------------


def extract_features(
    config: Any,
    cache_dir: Path,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """Run feature extraction or load from cache.

    Features are cached to ``cache_dir`` so that grid sweeps over GP
    hyperparameters do not re-extract features.

    Returns:
        (x_dict, labels, metadata)
    """
    meta_path = cache_dir / "metadata.json"

    if meta_path.exists():
        logging.info("Loading cached features from %s", cache_dir)
        from train_and_calibrate import load_features

        return load_features(str(cache_dir))

    # --- Extract fresh features ---
    logging.info("Extracting features (this may take a while on GPU)...")
    from fit_features import extract_all_features, load_texts, load_texts_multi
    from multiview_detector.aggregation.distance_features import (
        DistanceFeatureConstructor,
    )

    text_key = config.data.text_key
    human_texts = load_texts(config.data.human_support, text_key)
    machine_texts = load_texts_multi(config.data.machine_support, text_key)

    logging.info(
        "Support set: %d human + %d machine documents",
        len(human_texts),
        len(machine_texts),
    )

    all_texts = human_texts + machine_texts
    labels = np.array([0] * len(human_texts) + [1] * len(machine_texts), dtype=np.int32)

    view_names = [v.name for v in config.views]
    raw_features = extract_all_features(all_texts, view_names)

    # Save raw features.
    cache_dir.mkdir(parents=True, exist_ok=True)
    for view_name, arr in raw_features.items():
        np.save(cache_dir / f"{view_name}.npy", arr)
    np.save(cache_dir / "labels.npy", labels)

    # Fit distance constructor.
    train_x = {name: torch.tensor(arr) for name, arr in raw_features.items()}
    train_y = torch.tensor(labels, dtype=torch.float32)

    constructor = DistanceFeatureConstructor(
        use_robust_standardization=config.features.use_robust_standardization,
        use_mahalanobis=config.features.use_mahalanobis,
    )
    constructor.fit(train_x, train_y)
    constructor.save(str(cache_dir / "distance_constructor.pt"))

    metadata = {
        "views": view_names,
        "text_key": text_key,
        "use_mahalanobis": config.features.use_mahalanobis,
        "num_human": len(human_texts),
        "num_machine": len(machine_texts),
        "feature_shapes": {k: list(v.shape) for k, v in raw_features.items()},
    }
    with open(cache_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    from train_and_calibrate import load_features

    return load_features(str(cache_dir))


def extract_validation_features(
    config: Any,
    view_names: list[str],
    cache_dir: Path,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str]]:
    """Extract features for the held-out validation set.

    Returns:
        (x_dict, labels, categories) where categories[i] is the
        contribution_level (or empty string) for each document.
    """
    val_cache = cache_dir / "validation"
    val_meta = val_cache / "metadata.json"

    if val_meta.exists():
        logging.info("Loading cached validation features from %s", val_cache)
        x_dict: dict[str, torch.Tensor] = {}
        for vn in view_names:
            arr = np.load(val_cache / f"{vn}.npy").astype(np.float32)
            x_dict[vn] = torch.tensor(arr)
        labels = torch.tensor(np.load(val_cache / "labels.npy"), dtype=torch.float32)
        # Load cached categories.
        cats_path = val_cache / "categories.json"
        categories: list[str] = []
        if cats_path.exists():
            categories = json.loads(cats_path.read_text())
        else:
            categories = [""] * len(labels)
        return x_dict, labels, categories

    logging.info("Extracting validation features...")
    from fit_features import extract_all_features

    text_key = config.data.text_key
    label_key = getattr(config.data, "label_key", "")

    # Load validation data — expects JSONL with a label field
    #   label=0 → human, label=1 → machine
    val_texts: list[str] = []
    val_labels: list[int] = []
    val_categories: list[str] = []
    with open(config.data.validation) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get(text_key, "")
            if text:
                val_texts.append(str(text).strip())
                # Binary label: explicit 'label' field, or infer from
                # contribution_level (Human=0, everything else=1).
                if "label" in rec:
                    val_labels.append(int(rec["label"]))
                elif label_key and rec.get(label_key):
                    val_labels.append(0 if rec[label_key] == "Human" else 1)
                else:
                    val_labels.append(0)
                val_categories.append(str(rec.get(label_key, "")) if label_key else "")

    logging.info("Validation set: %d documents", len(val_texts))
    if label_key and val_categories:
        from collections import Counter

        cat_counts = Counter(val_categories)
        for cat, cnt in sorted(cat_counts.items()):
            logging.info("  %s: %d", cat or "(unlabeled)", cnt)

    raw_features = extract_all_features(val_texts, view_names)

    val_cache.mkdir(parents=True, exist_ok=True)
    for vn, arr in raw_features.items():
        np.save(val_cache / f"{vn}.npy", arr)
    labels_arr = np.array(val_labels, dtype=np.int32)
    np.save(val_cache / "labels.npy", labels_arr)
    # Cache categories for reuse.
    with open(val_cache / "categories.json", "w") as fh:
        json.dump(val_categories, fh)
    with open(val_meta, "w") as fh:
        json.dump({"views": view_names, "n_docs": len(val_texts)}, fh)

    x_dict = {k: torch.tensor(v) for k, v in raw_features.items()}
    return x_dict, torch.tensor(labels_arr, dtype=torch.float32), val_categories


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------


def run_experiment(
    config: Any,
    x_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    val_x: dict[str, torch.Tensor],
    val_y: torch.Tensor,
    metadata: dict,
    experiment_id: int,
    val_categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run a single train → calibrate → evaluate experiment.

    Returns a dict of metrics, including per-category TPR if
    *val_categories* is provided.
    """
    from multiview_detector.aggregation.distance_gp import DistanceGP
    from multiview_detector.configs.config import get_default_distance_gp_config
    from train_and_calibrate import split_features

    # ── Train / calibration split ────────────────────────────────────
    train_x, train_y, cal_x, cal_y = split_features(
        x_dict,
        labels,
        train_fraction=config.calibration.train_fraction,
        seed=config.calibration.seed,
    )

    # ── Build GP config ──────────────────────────────────────────────
    gp_config = get_default_distance_gp_config()
    gp_config.views = metadata["views"]
    gp_config.kernel_type = config.model.kernel
    gp_config.use_ard = config.model.use_ard
    gp_config.num_epochs = config.model.num_epochs
    gp_config.lr = config.model.lr
    gp_config.alpha = config.calibration.alpha
    gp_config.num_folds = config.calibration.num_folds
    gp_config.use_mahalanobis = metadata.get("use_mahalanobis", True)
    gp_config.aggregation_method = config.calibration.aggregation_method

    # Auto-select variational vs exact GP.
    n_train_per_class = min(int((train_y == 0).sum()), int((train_y == 1).sum()))
    if config.model.use_variational is None:
        gp_config.use_variational = n_train_per_class > 500
    else:
        gp_config.use_variational = config.model.use_variational

    if gp_config.use_variational:
        gp_config.inducing_point_method = config.model.inducing_point_method
        gp_config.num_inducing = config.model.num_inducing

    # ── Train ────────────────────────────────────────────────────────
    t0 = time.time()
    model = DistanceGP(config=gp_config)
    losses = model.fit(train_x, train_y, verbose=False)
    train_time = time.time() - t0

    # ── Calibrate ────────────────────────────────────────────────────
    model.calibrate(cal_x, cal_y)

    # ── Evaluate on validation set ───────────────────────────────────
    result = model.decide(val_x)

    if result is None:
        logging.warning("Experiment %d: model returned no predictions", experiment_id)
        return {"experiment_id": experiment_id, "error": "no predictions"}

    pred = result.prediction
    agg_scores = pred.aggregate_score.numpy()
    decisions = result.decisions
    val_labels = val_y.numpy()

    # Binary predictions: flag=1, no_action=0, ood=excluded
    is_flag = np.array([d == "flag" for d in decisions])
    is_ood = np.array([d == "ood" for d in decisions])

    human_mask = val_labels == 0
    machine_mask = val_labels == 1

    # AUROC (on all non-OOD documents)
    from sklearn.metrics import roc_auc_score

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


def write_best_config(
    all_metrics: list[dict[str, Any]],
    base_config: Any,
    output_dir: Path,
) -> None:
    """Write a production-ready YAML for the best configuration."""
    import yaml

    # Find best by AUROC.
    valid = [m for m in all_metrics if m.get("auroc") is not None]
    if not valid:
        logging.warning("No valid results — skipping best_config.yaml")
        return

    best = max(valid, key=lambda m: m["auroc"])

    # Build a clean config dict (no evaluation section).
    out: dict[str, Any] = {}

    # Data.
    out["data"] = {
        "human_support": base_config.data.human_support,
        "machine_support": base_config.data.machine_support,
        "text_key": base_config.data.text_key,
    }

    # Views.
    out["views"] = [{"name": v.name} for v in base_config.views]

    # Features.
    out["features"] = {
        "use_mahalanobis": base_config.features.use_mahalanobis,
        "use_robust_standardization": (base_config.features.use_robust_standardization),
    }

    # Model — from the best experiment.
    out["model"] = {
        "kernel": best["kernel"],
        "use_ard": best["use_ard"],
        "num_epochs": best["num_epochs"],
    }

    # Calibration — from the best experiment.
    out["calibration"] = {
        "alpha": best["alpha"],
        "num_folds": base_config.calibration.num_folds,
        "train_fraction": best.get("train_fraction", 0.5),
        "seed": base_config.calibration.seed,
        "aggregation_method": best["aggregation_method"],
    }

    best_path = output_dir / "best_config.yaml"
    header = (
        f"# Auto-generated by evaluate.py — best config from grid sweep\n"
        f"# AUROC: {best['auroc']}  "
        f"FPR@τ: {best['fpr_at_tau']}  "
        f"TPR@τ: {best['tpr_at_tau']}\n"
        f"#\n"
        f"# This YAML is ready for production deployment.\n\n"
    )
    with open(best_path, "w") as fh:
        fh.write(header)
        yaml.dump(out, fh, default_flow_style=False, sort_keys=False)
    logging.info("Best config written to %s", best_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YAML-driven evaluation pipeline for multi-view detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to evaluation YAML config (e.g. eval_config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override evaluation.output_dir from YAML.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced hyperparameter grid for fast iteration.",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help=(
            "Directory of pre-extracted features (output of fit_features.py). "
            "If provided, skips feature extraction."
        ),
    )
    parser.add_argument(
        "--machine-support",
        nargs="+",
        default=None,
        metavar="JSONL",
        help=(
            "Override data.machine_support from the YAML config. "
            "Accepts one or more JSONL paths. Use this to switch between "
            "zero-shot (static corpus) and few-shot (generated JSONL) modes "
            "without editing the config file."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load config ──────────────────────────────────────────────────
    from pipeline_config import load_pipeline_config

    config = load_pipeline_config(Path(args.config))

    # Apply CLI overrides.
    if args.machine_support:
        if len(args.machine_support) == 1:
            config.data.machine_support = args.machine_support[0]
        else:
            config.data.machine_support = args.machine_support
        logging.info(
            "Using CLI-provided machine_support: %s",
            config.data.machine_support,
        )

    output_dir = Path(args.output_dir or config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine grid ───────────────────────────────────────────────
    grid = config.evaluation.grid
    if args.quick:
        grid = _QUICK_GRID
        logging.info("Quick mode: using reduced grid %s", grid)

    experiments = expand_grid(config, grid)
    logging.info("Evaluation plan: %d experiment(s)", len(experiments))

    # ── Step 1: Extract features ─────────────────────────────────────
    features_dir = Path(args.features_dir) if args.features_dir else (output_dir / "features")
    x_dict, labels, metadata = extract_features(config, features_dir)

    view_names = metadata["views"]

    # ── Extract validation features ──────────────────────────────────
    if not config.data.validation:
        logging.error(
            "No validation data path set in config (data.validation). "
            "Cannot evaluate without held-out data."
        )
        sys.exit(1)

    val_x, val_y, val_categories = extract_validation_features(
        config,
        view_names,
        features_dir,
    )
    logging.info(
        "Validation set: %d human + %d machine",
        int((val_y == 0).sum()),
        int((val_y == 1).sum()),
    )

    # ── Run experiments ──────────────────────────────────────────────
    all_metrics: list[dict[str, Any]] = []

    for i, (params, exp_config) in enumerate(experiments):
        logging.info(
            "\n" + "=" * 60 + "\nExperiment %d/%d: %s" + "\n" + "=" * 60,
            i + 1,
            len(experiments),
            params or "(base config)",
        )

        metrics = run_experiment(
            config=exp_config,
            x_dict=x_dict,
            labels=labels,
            val_x=val_x,
            val_y=val_y,
            metadata=metadata,
            experiment_id=i + 1,
            val_categories=val_categories,
        )
        all_metrics.append(metrics)

        logging.info(
            "  → AUROC=%.4f  FPR@τ=%.4f  TPR@τ=%.4f  OOD=%.1f%%",
            metrics.get("auroc") or 0.0,
            metrics["fpr_at_tau"],
            metrics["tpr_at_tau"],
            100 * metrics["ood_rate"],
        )

    # ── Write results ────────────────────────────────────────────────
    write_summary(all_metrics, output_dir)
    write_best_config(all_metrics, config, output_dir)

    logging.info("\nEvaluation complete. %d experiments run.", len(all_metrics))
    logging.info("Results: %s", output_dir)


if __name__ == "__main__":
    main()
