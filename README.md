# text-detection-baselines

[![CI](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/ci.yml/badge.svg)](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/text-detection-baselines/badge/?version=latest)](https://text-detection-baselines.readthedocs.io/en/latest/?badge=latest)
[![Security](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/security.yml/badge.svg)](https://github.com/JH-DSAI/text-detection-baselines/actions/workflows/security.yml)
<!-- [![DOI](https://zenodo.org/badge/DOI/REPLACE/ME.svg)](https://doi.org/REPLACE/ME) -->

Benchmarking suite for machine text detection.

To do:

* Uncomment and update the DOI above in this README.
* Import package into [readthedocs](https://readthedocs.org/).
* Update [.zenodo.json](.zenodo.json). For more details see [zenodo.json docs](https://developers.zenodo.org/#representation) and [zenodo docs on contributors vs creators](https://help.zenodo.org/docs/deposit/describe-records/contributors/).
* Update quickstart guide below.

## Quickstart (pixi)

1. Install pixi 0.75 or later from <https://pixi.sh/latest/>.
1. Clone this repository.
1. Run evaluation on default datasets and models (automatically installs dependencies in a virtual environment):

```bash
pixi run main
```

The first run resolves a ~800 MB environment, most of it PyTorch.

## Datasets

Two datasets
are registered; see [datasets/README.md](datasets/README.md) for provenance,
licences, and citations.

| name | default | what it is |
| --- | --- | --- |
| `demo` | yes | 200 records of **synthetic** text bundled inside the package. Both classes are separated by deliberately planted surface statistics, so its numbers describe the pipeline, not detection quality. |
| `gede` | no | The GEDE corpus of student essays and LLM rewrites. **Licensed CC BY-NC-SA 4.0 and not bundled** — you obtain it from [upstream](https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts) and convert it locally. |

To use `gede`,
follow the acquisition steps in [datasets/README.md](datasets/README.md), then:

```bash
pixi run prepare-gede --source /path/to/database.db
pixi run main -- --dataset gede
```

Every `gede` figure quoted in the Metrics section below was measured on the full
published corpus. Three of its properties are load-bearing for reading those
figures — it is 93.27% machine, it has only 916 human samples, and eight of its nine
categories are single-label — and all three are documented in
[datasets/README.md](datasets/README.md).

To evaluate your own file:

```bash
pixi run main -- --register-file-dataset mydata=/path/to/mydata.jsonl
```

## Models

Every model currently registered in [models/](text_detection_baselines/models/) is a **stub**.
None of them are trained, and none should be treated as a working detector — they
exist to exercise the evaluation pipeline end to end with realistic-looking outputs.

| name | what it does |
| --- | --- |
| `dummy-norm` | Linear layer over four surface features (char length, token count, punctuation count, type-token ratio) with **hard-coded, arbitrary weights**. Logit passed through a sigmoid, so scores lie in `[0, 1]`. |
| `dummy-raw` | Same arbitrary weights, raw logit reported as an unnormalized score. |
| `length` | Hand-written heuristic: longer texts with lower type-token ratio and less punctuation score as more machine-like. An actual (weak, unvalidated) hypothesis, unlike the `dummy-*` pair. |
| `smollm2` | Prompts a small local LLM. Not a default; opt in with `--model smollm2`. |

The `dummy-*` weights were picked by hand and fit to nothing. Their metrics measure
the harness, not detection quality, and any apparent skill they show on a dataset is
an artifact of that dataset's length distribution.

## Metrics

Metrics are computed per (dataset, model) pair and again per `contribution_level`
category.

Scores follow one convention throughout: **higher means more likely
machine-generated**, and labels are `0` = human, `1` = machine.

### Detection metrics

Defined in [metrics/detection.py](text_detection_baselines/metrics/detection.py).
All three ranking metrics are computed on **non-OOD samples only** and return `null`
when that subset is empty or contains just one class.

| Registry key | Table | Description |
| --- | --- | --- |
| `auroc` | AUROC | Area under the ROC curve. |
| `auroc_at_1pct` | AUROC@1% | Partial AUROC over the FPR ≤ 1% region — the regime a deployed detector actually operates in. See the note on its scaling below. |
| `average_precision` | AP | Average precision — the precision-recall summary. Sums over achievable operating points rather than interpolating between them, as trapezoidal PR-AUC would. |
| `fpr_at_tau` | FPR@tau | False positive rate at the learned threshold, among human samples. |
| `tpr_at_tau` | TPR@tau | True positive rate at the learned threshold, among machine samples. |
| `calibration_gap` | CalGap | `abs(fpr_at_tau - target_alpha)`, i.e. how far threshold learning missed its FPR budget. |
| `ood_percent` | OOD% | Percentage of samples flagged out-of-distribution. |

The threshold `tau` is learned per run as the `1 - target_alpha` quantile of
non-OOD human scores, and a sample is flagged machine when `score >= tau`
(inclusive). `--target-alpha` defaults to 0.05.

### Calibration metrics

Defined in [metrics/calibration.py](text_detection_baselines/metrics/calibration.py).
Both are registered with `requires_normalized_scores=True`, so they are computed
only for models whose scores lie in `[0, 1]` and are omitted entirely — not set to
`null` — for raw-score models. Both are computed on **non-OOD samples only**, like
the ranking metrics, and return `null` when every sample is flagged OOD. Unlike the
ranking metrics they do not require both classes to be present.

| Registry key | Table | Description |
| --- | --- | --- |
| `brier` | Brier | Mean squared error between score and binary label. |
| `ece` | ECE | Expected calibration error over 10 equal-width bins. |

### Selective prediction

`compute_abstention_curve` in
[metrics/selective.py](text_detection_baselines/metrics/selective.py) sweeps an
uncertainty threshold from tight to full coverage and reports coverage, accuracy,
AUROC, and partial AUROC on the retained samples at each point.

It returns parallel curves rather than a scalar, so it is **not** in the registry
and does not appear in the console tables or exports. Call it
directly:

```python
from text_detection_baselines.metrics import compute_abstention_curve

curve = compute_abstention_curve(
    scores=scores, labels=labels, variances=variances, tau=tau,
)
```

It needs a per-sample `variances` array that no detector in this package currently
exposes, so the caller must supply one.

### Implementation notes and discrepancies

Details that are easy to misread, and points where this implementation
deliberately diverges from the researchers' implementation:

* **`auroc_at_1pct` is not a raw partial area.** It is the McClish-standardized
  partial AUC that `sklearn.metrics.roc_auc_score(..., max_fpr=0.01)` returns: the
  raw area over FPR ≤ 0.01, rescaled so a random ranker scores 0.5 and a perfect
  ranker 1.0. The rescaling is **not clamped** — a ranker worse than chance in the
  low-FPR region scores below 0.5 (`dummy-raw` scores 0.4976 on `gede`).

* **Registered metrics exclude OOD-flagged samples; the abstention curve does not.**
  `auroc`, `auroc_at_1pct`, `average_precision`, `brier`, and `ece` all restrict to
  `~ood_flags`, so they will not reproduce values computed over the whole
  population, and every column of a results row describes the same sample set.
  `compute_abstention_curve` currently takes no `ood_flags` argument and so is a
  whole-population measure — its AUROC is not comparable to `auroc` from the same
  run.

  The reference implementation had no OOD notion at all. Its calibration numbers
  therefore correspond to the whole population: on `gede`, `length` reports Brier
  0.1126 / ECE 0.2300 over all samples against 0.1113 / 0.2309 over the 98.9% that
  are non-OOD.

* **`average_precision` is the only PR summary reported.** A trapezoidal PR-AUC
  was previously exported alongside it and has been removed: linear interpolation
  between adjacent PR points describes operating points that cannot be achieved,
  and scikit-learn documents that estimator as misleading. It disagreed with AP by
  up to 3.4 points on `dummy-norm`, whose degenerate scores leave sparse PR points
  for the trapezoid to inflate across. Earlier numbers reported under a `pr_auc`
  column are not comparable to `average_precision`.

  How far apart the two estimators land is governed by score granularity, not by the
  data. Measured at n=4000 with a fixed signal, quantizing the scores to a given
  number of distinct values:

  | distinct scores | trapezoidal PR-AUC | AP | difference |
  | --- | --- | --- | --- |
  | 2 | 0.788125 | 0.656056 | +0.132069 |
  | 5 | 0.798885 | 0.745213 | +0.053672 |
  | 25 | 0.797125 | 0.786356 | +0.010768 |
  | 100 | 0.795676 | 0.793066 | +0.002610 |
  | 4000 (continuous) | 0.794817 | 0.794931 | −0.000114 |

  On the continuous-score detectors the two agree to within 1e-4 (`length` differs
  by 2e-6 on `gede`, `dummy-raw` by 6.5e-5), which is why historical numbers from
  those models remain readable. The inflation is worst when a score vector collapses
  entirely: the trapezoid then returns `prevalence + (1 - prevalence)/2` exactly,
  which is how `dummy-norm` read 0.9664 against its true AP of 0.9327.

* **Average precision is floored by class prevalence.** `gede` is 93.27% machine,
  so every model scores AP above 0.93 regardless of ranking quality.
  `dummy-norm` has AUROC exactly 0.5 — no ranking signal at all — and
  scores AP 0.9327, which *is* the prevalence to four decimal places. Read AP only
  against that floor, and never compare it across slices with different base rates.

* **ECE clips scores into `[0, 1]` before binning.** The reference implementation
  did not, which silently dropped out-of-range probabilities from the numerator
  while still counting them in the denominator.

* **Threshold comparisons are inclusive (`score >= tau`) everywhere.** This matches
  the detectors' own prediction rule (`preds = scores >= 0.0` for raw scores,
  `>= 0.5` for normalized) and keeps the abstention curve's full-coverage point
  consistent with `fpr_at_tau` / `tpr_at_tau`. Because `tau` is an empirical
  quantile, the convention is decidable rather than measure-zero: at
  `target_alpha = 1.0` only the inclusive form yields FPR = 1.0 as intended, while
  at `target_alpha = 0` it flags at least one human and so yields FPR > 0.

* **Exported values carry full precision; only the console rounds.** The
  CSV/JSON/YAML exports hold the unrounded metric, so they remain usable for
  paired significance testing between two close models.

* **Non-finite metric values are exported as `null`.** `normalize_metric_value`
  maps NaN and ±inf to `None` on the way out of `run_all_metrics`, because none of
  the three has a JSON representation. A metric that goes non-finite is therefore
  indistinguishable in the exports from one that returned `null` deliberately; no
  metric currently does, but a new one should return `null` explicitly rather than
  rely on this.

* **Per-category ranking metrics are `null` on `gede`.** Every
  `contribution_level` category in that dataset contains exactly one label, so all
  three ranking metrics are undefined for every category row.

## Common commands

```bash
# lint/format
pixi run -e dev lint
pixi run -e dev format

# tests + coverage outputs
pixi run -e dev test

# docs
pixi run -e docs build-docs

# distribution artifacts (wheel + sdist)
pixi run -e dist build-dist
```

## Docker

Build:

```bash
docker build -t text-detection-baselines .
```

Run:

```bash
docker run --rm text-detection-baselines
```

## Git hook (optional)

Install the pre-push hook to run style checks before pushing:

```bash
cp ./githooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Notes

* CI, security, docs, and distribution workflows use pixi tasks.
* Read the Docs installs documentation dependencies from project extras.
