"""Model evaluation entry point for text detection baselines.

This module evaluates one or more detector stubs on one or more datasets.
Datasets are expected to be JSON-lines files (one JSON object per line).

Main metrics written to summary and exports:
- AUROC
- FPR@tau
- TPR@tau
- CalGap = |FPR - target_alpha|
- OOD%

Per-category metrics use the contribution_level field:
- TPR by category
- FPR by category
- AUROC by category

For models with normalized scores in [0, 1], additional calibration metrics are
reported:
- Brier score
- Expected calibration error (ECE, 10 bins)
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - fallback when torch is unavailable
    torch = None
    nn = None


LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetRecord:
    """Single evaluation sample."""

    text: str
    label: int
    category: str


@dataclass
class StubModelOutput:
    """Model outputs used for metric computation."""

    scores: np.ndarray
    predictions: np.ndarray
    ood_flags: np.ndarray


class StubTextDetector:
    """Simple detector interface for stubbed model baselines."""

    def __init__(self, model_name: str, normalized_scores: bool, ood_margin: float, seed: int) -> None:
        self.model_name = model_name
        self.normalized_scores = normalized_scores
        self.ood_margin = ood_margin
        self.seed = seed

    def predict(self, texts: list[str]) -> StubModelOutput:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _feature_matrix(texts: list[str]) -> np.ndarray:
        """Compute deterministic numeric features from raw text."""
        lengths = np.array([len(t) for t in texts], dtype=float)
        token_counts = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
        punct = np.array([sum(c in ".,!?:;" for c in t) for t in texts], dtype=float)
        unique_ratio = np.array(
            [len(set(t.split())) / max(len(t.split()), 1) for t in texts],
            dtype=float,
        )
        return np.column_stack((lengths, token_counts, punct, unique_ratio))


class TorchLinearStubDetector(StubTextDetector):
    """Torch-based linear detector stub.

    Uses fixed weights to produce repeatable scores without training.
    """

    def __init__(self, model_name: str, normalized_scores: bool, ood_margin: float, seed: int) -> None:
        super().__init__(model_name, normalized_scores, ood_margin, seed)
        weights = np.array([0.015, 0.09, -0.03, 1.3], dtype=np.float32)
        bias = np.float32(-2.1)

        self._np_weights = weights.astype(float)
        self._np_bias = float(bias)

        if torch is not None and nn is not None:
            torch.manual_seed(seed)
            self._layer: nn.Module | None = nn.Linear(4, 1, bias=True)
            with torch.no_grad():
                self._layer.weight[:] = torch.tensor(weights.reshape(1, -1))
                self._layer.bias[:] = torch.tensor([bias])
        else:
            self._layer = None

    def predict(self, texts: list[str]) -> StubModelOutput:
        feats = self._feature_matrix(texts)
        raw = self._forward(feats)

        if self.normalized_scores:
            scores = 1.0 / (1.0 + np.exp(-raw))
            preds = (scores >= 0.5).astype(int)
            uncertainty = np.abs(scores - 0.5)
        else:
            scores = raw
            preds = (scores >= 0.0).astype(int)
            scale = max(np.std(scores), 1e-6)
            uncertainty = np.abs(scores) / scale

        ood = (uncertainty < self.ood_margin) | (feats[:, 0] < 40)
        return StubModelOutput(scores=scores, predictions=preds, ood_flags=ood)

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        if self._layer is None:
            return feats @ self._np_weights + self._np_bias

        x = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            return self._layer(x).squeeze(1).cpu().numpy()


class LengthHeuristicStubDetector(StubTextDetector):
    """Normalized heuristic model that can be used without torch."""

    def predict(self, texts: list[str]) -> StubModelOutput:
        feats = self._feature_matrix(texts)
        length = feats[:, 0]
        token_count = feats[:, 1]
        punct = feats[:, 2]
        unique_ratio = feats[:, 3]

        raw = (length / np.maximum(token_count, 1)) * 0.45 + (1 - unique_ratio) * 1.4 - punct * 0.02 - 2.0
        scores = 1.0 / (1.0 + np.exp(-raw))
        preds = (scores >= 0.5).astype(int)
        ood = (np.abs(scores - 0.5) < self.ood_margin) | (length < 40)
        return StubModelOutput(scores=scores, predictions=preds, ood_flags=ood)


def load_dataset(path: Path, text_key: str, label_key: str, category_key: str) -> list[DatasetRecord]:
    """Load records from JSONL; accepts JSON arrays for convenience."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset is empty: {path}")

    records: list[dict[str, Any]]
    if raw[0] == "[":
        loaded = json.loads(raw)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected JSON array in {path}")
        records = [r for r in loaded if isinstance(r, dict)]
    else:
        records = []
        for idx, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}") from exc
            if isinstance(row, dict):
                records.append(row)

    parsed: list[DatasetRecord] = []
    for row in records:
        if text_key not in row or label_key not in row:
            continue
        parsed.append(
            DatasetRecord(
                text=str(row[text_key]),
                label=normalize_label(row[label_key]),
                category=str(row.get(category_key, "unknown")),
            ),
        )

    if not parsed:
        raise ValueError(f"No valid samples with required keys in {path}")
    return parsed


def normalize_label(raw_label: Any) -> int:
    """Map common label representations to 0/1 where 1 = machine text."""
    if isinstance(raw_label, bool):
        return int(raw_label)
    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)

    val = str(raw_label).strip().lower()
    if val in {"1", "machine", "fake", "ai", "generated"}:
        return 1
    if val in {"0", "human", "real", "organic"}:
        return 0
    raise ValueError(f"Unsupported label value: {raw_label}")


def build_stub_model(model_name: str, ood_margin: float, seed: int) -> StubTextDetector:
    """Factory for named detector stubs."""
    if model_name == "torch-normalized":
        return TorchLinearStubDetector(model_name=model_name, normalized_scores=True, ood_margin=ood_margin, seed=seed)
    if model_name == "torch-raw":
        return TorchLinearStubDetector(model_name=model_name, normalized_scores=False, ood_margin=ood_margin, seed=seed)
    if model_name == "length-normalized":
        return LengthHeuristicStubDetector(model_name=model_name, normalized_scores=True, ood_margin=ood_margin, seed=seed)
    raise ValueError(
        f"Unknown model '{model_name}'. Valid options: torch-normalized, torch-raw, length-normalized",
    )


def evaluate_predictions(
    labels: np.ndarray,
    categories: np.ndarray,
    output: StubModelOutput,
    target_alpha: float,
    normalized_scores: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute summary and per-category metrics."""
    scores = output.scores
    ood_flags = output.ood_flags

    human_mask = labels == 0
    machine_mask = labels == 1
    non_ood = ~ood_flags

    if human_mask.sum() == 0 or machine_mask.sum() == 0:
        raise ValueError("Dataset must contain both human and machine labels")

    human_scores_non_ood = scores[human_mask & non_ood]
    if human_scores_non_ood.size == 0:
        tau = float(np.quantile(scores[human_mask], 1 - target_alpha))
    else:
        tau = float(np.quantile(human_scores_non_ood, 1 - target_alpha))

    flags = (scores >= tau) & non_ood

    fpr = float((flags & human_mask).sum() / max(int(human_mask.sum()), 1))
    tpr = float((flags & machine_mask).sum() / max(int(machine_mask.sum()), 1))
    cal_gap = abs(fpr - target_alpha)
    ood_pct = float(ood_flags.mean() * 100.0)

    auroc: float | None
    if non_ood.sum() > 0 and len(np.unique(labels[non_ood])) == 2:
        auroc = float(roc_auc_score(labels[non_ood], scores[non_ood]))
    else:
        auroc = None

    main_metrics: dict[str, Any] = {
        "auroc": safe_round(auroc),
        "fpr_at_tau": safe_round(fpr),
        "tpr_at_tau": safe_round(tpr),
        "calibration_gap": safe_round(cal_gap),
        "ood_percent": safe_round(ood_pct),
        "tau": safe_round(tau),
        "target_alpha": target_alpha,
    }

    if normalized_scores:
        main_metrics["brier"] = safe_round(float(np.mean((scores - labels) ** 2)))
        main_metrics["ece"] = safe_round(expected_calibration_error(scores=scores, labels=labels, bins=10))

    per_category: list[dict[str, Any]] = []
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        cat_labels = labels[mask]
        cat_scores = scores[mask]
        cat_non_ood = non_ood[mask]
        cat_flags = flags[mask]

        cat_human = cat_labels == 0
        cat_machine = cat_labels == 1

        cat_tpr = float((cat_flags & cat_machine).sum() / max(int(cat_machine.sum()), 1))
        cat_fpr = float((cat_flags & cat_human).sum() / max(int(cat_human.sum()), 1))

        cat_auroc: float | None
        if cat_non_ood.sum() > 0 and len(np.unique(cat_labels[cat_non_ood])) == 2:
            cat_auroc = float(roc_auc_score(cat_labels[cat_non_ood], cat_scores[cat_non_ood]))
        else:
            cat_auroc = None

        per_category.append(
            {
                "category": category,
                "n_samples": int(mask.sum()),
                "n_human": int(cat_human.sum()),
                "n_machine": int(cat_machine.sum()),
                "tpr": safe_round(cat_tpr),
                "fpr": safe_round(cat_fpr),
                "auroc": safe_round(cat_auroc),
            },
        )

    return main_metrics, per_category


def evaluate_model_on_dataset(
    dataset_path: Path,
    model: StubTextDetector,
    target_alpha: float,
    text_key: str,
    label_key: str,
    category_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one model against one dataset."""
    records = load_dataset(dataset_path, text_key=text_key, label_key=label_key, category_key=category_key)

    texts = [r.text for r in records]
    labels = np.array([r.label for r in records], dtype=int)
    categories = np.array([r.category for r in records], dtype=object)

    output = model.predict(texts)
    main_metrics, per_category = evaluate_predictions(
        labels=labels,
        categories=categories,
        output=output,
        target_alpha=target_alpha,
        normalized_scores=model.normalized_scores,
    )

    main_row = {
        "dataset": dataset_path.stem,
        "dataset_path": str(dataset_path),
        "model": model.model_name,
        "normalized_scores": model.normalized_scores,
        "n_samples": len(records),
        **main_metrics,
    }

    per_category_rows = [
        {
            "dataset": dataset_path.stem,
            "model": model.model_name,
            **cat,
        }
        for cat in per_category
    ]

    return main_row, per_category_rows


def safe_round(value: float | None, ndigits: int = 4) -> float | None:
    """Round numeric values while preserving None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), ndigits)


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Compute ECE for normalized probabilities."""
    scores = np.clip(scores, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)

        if not np.any(mask):
            continue

        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(labels[mask]))
        frac = float(np.mean(mask))
        ece += abs(acc - conf) * frac

    return ece


def render_console_tables(main_rows: list[dict[str, Any]], per_category_rows: list[dict[str, Any]]) -> str:
    """Render summary tables and return rendered plain text."""
    console = Console(record=True)

    summary = Table(title="Text Detection Metrics", show_lines=False)
    for col in [
        "dataset",
        "model",
        "AUROC",
        "FPR@tau",
        "TPR@tau",
        "CalGap",
        "OOD%",
        "tau",
    ]:
        summary.add_column(col, justify="right" if col not in {"dataset", "model"} else "left")

    for row in sorted(main_rows, key=lambda r: (r["dataset"], r["model"])):
        summary.add_row(
            row["dataset"],
            row["model"],
            fmt(row.get("auroc")),
            fmt(row.get("fpr_at_tau")),
            fmt(row.get("tpr_at_tau")),
            fmt(row.get("calibration_gap")),
            fmt(row.get("ood_percent")),
            fmt(row.get("tau")),
        )

    console.print(summary)

    if any(r.get("normalized_scores") for r in main_rows):
        cal_table = Table(title="Normalized Score Calibration Metrics", show_lines=False)
        for col in ["dataset", "model", "Brier", "ECE"]:
            cal_table.add_column(col, justify="right" if col not in {"dataset", "model"} else "left")

        for row in sorted(main_rows, key=lambda r: (r["dataset"], r["model"])):
            if not row.get("normalized_scores"):
                continue
            cal_table.add_row(
                row["dataset"],
                row["model"],
                fmt(row.get("brier")),
                fmt(row.get("ece")),
            )

        console.print(cal_table)

    per_cat = Table(title="Per-Category Metrics (contribution_level)", show_lines=False)
    for col in ["dataset", "model", "category", "TPR", "FPR", "AUROC", "n"]:
        per_cat.add_column(col, justify="right" if col not in {"dataset", "model", "category"} else "left")

    for row in sorted(per_category_rows, key=lambda r: (r["dataset"], r["model"], r["category"])):
        per_cat.add_row(
            row["dataset"],
            row["model"],
            row["category"],
            fmt(row.get("tpr")),
            fmt(row.get("fpr")),
            fmt(row.get("auroc")),
            str(row.get("n_samples", "-")),
        )

    console.print(per_cat)

    return console.export_text(clear=False)


def fmt(value: Any) -> str:
    """Format numbers in rich tables."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def export_results(
    output_dir: Path,
    export_formats: tuple[str, ...],
    main_rows: list[dict[str, Any]],
    per_category_rows: list[dict[str, Any]],
) -> None:
    """Write metrics in selected structured formats."""
    for fmt_name in export_formats:
        if fmt_name == "json":
            (output_dir / "metrics.json").write_text(json.dumps(main_rows, indent=2), encoding="utf-8")
            (output_dir / "per_category.json").write_text(
                json.dumps(per_category_rows, indent=2),
                encoding="utf-8",
            )
        elif fmt_name == "yaml":
            (output_dir / "metrics.yaml").write_text(yaml.safe_dump(main_rows, sort_keys=False), encoding="utf-8")
            (output_dir / "per_category.yaml").write_text(
                yaml.safe_dump(per_category_rows, sort_keys=False),
                encoding="utf-8",
            )
        elif fmt_name == "csv":
            write_csv(output_dir / "metrics.csv", main_rows)
            write_csv(output_dir / "per_category.csv", per_category_rows)
        else:
            raise ValueError(f"Unsupported export format: {fmt_name}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows to CSV."""
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


@click.command()
@click.option(
    "--dataset",
    "datasets",
    type=click.Path(path_type=Path, exists=True),
    multiple=True,
    required=True,
    help="Path to JSONL dataset. Repeat this option to evaluate multiple datasets.",
)
@click.option(
    "--model",
    "models",
    type=click.Choice(["torch-normalized", "torch-raw", "length-normalized"], case_sensitive=False),
    multiple=True,
    required=True,
    help="Model stub to evaluate. Repeat this option to evaluate multiple models.",
)
@click.option(
    "--target-alpha",
    type=float,
    default=0.05,
    show_default=True,
    help="Target false positive rate used to learn tau.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("evaluation_results"),
    show_default=True,
    help="Directory for summary and optional exports.",
)
@click.option(
    "--export",
    "export_formats",
    type=click.Choice(["csv", "json", "yaml"], case_sensitive=False),
    multiple=True,
    help="Write structured output files in selected format(s).",
)
@click.option("--text-key", type=str, default="answer", show_default=True, help="Text field key in dataset rows.")
@click.option("--label-key", type=str, default="label", show_default=True, help="Label field key in dataset rows.")
@click.option(
    "--category-key",
    type=str,
    default="contribution_level",
    show_default=True,
    help="Category field used for per-category metrics.",
)
@click.option("--seed", type=int, default=7, show_default=True, help="Seed for deterministic model behavior.")
@click.option(
    "--ood-margin",
    type=float,
    default=0.08,
    show_default=True,
    help="Uncertainty margin used by stubs when flagging OOD samples.",
)
def main(
    datasets: tuple[Path, ...],
    models: tuple[str, ...],
    target_alpha: float,
    output_dir: Path,
    export_formats: tuple[str, ...],
    text_key: str,
    label_key: str,
    category_key: str,
    seed: int,
    ood_margin: float,
) -> None:
    """Evaluate text detection model stubs on one or more datasets."""
    if not 0.0 <= target_alpha <= 1.0:
        raise click.BadParameter("target-alpha must be in [0, 1]")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_objs = [build_stub_model(name.lower(), ood_margin=ood_margin, seed=seed) for name in models]

    main_rows: list[dict[str, Any]] = []
    per_category_rows: list[dict[str, Any]] = []

    for dataset_path in datasets:
        for model in model_objs:
            LOGGER.info("Evaluating model=%s on dataset=%s", model.model_name, dataset_path)
            main_row, cat_rows = evaluate_model_on_dataset(
                dataset_path=dataset_path,
                model=model,
                target_alpha=target_alpha,
                text_key=text_key,
                label_key=label_key,
                category_key=category_key,
            )
            main_rows.append(main_row)
            per_category_rows.extend(cat_rows)

    summary_text = render_console_tables(main_rows, per_category_rows)
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    if export_formats:
        export_results(
            output_dir=output_dir,
            export_formats=tuple(fmt.lower() for fmt in export_formats),
            main_rows=main_rows,
            per_category_rows=per_category_rows,
        )

    LOGGER.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
