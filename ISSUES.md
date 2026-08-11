# Open issues and follow-ups

Known limitations and follow-up work for the evaluation metrics. Each entry notes
what is wrong, why it matters, and a suggested direction. Nothing here is a
blocker for prototyping; they are recorded so they are not rediscovered later.

Metric implementations live in [text_detection_baselines/metrics/](text_detection_baselines/metrics/).
See the [Metrics section of the README](README.md#metrics) for what each metric
currently computes.

---

## Blocking the abstention curve

### 1. No model produces the `variances` input, so the curve is unreachable

`compute_abstention_curve` in [selective.py](text_detection_baselines/metrics/selective.py)
requires a per-sample uncertainty array. `StubModelOutput`
([models/base.py](text_detection_baselines/models/base.py)) carries only `scores`,
`predictions`, and `ood_flags`. All three stub detectors compute a continuous
uncertainty and then throw it away by binarizing it into `ood_flags`:
[torch_linear.py:56-63](text_detection_baselines/models/torch_linear.py#L56-L63)
binds it to an `uncertainty` variable and never returns it, while
[length_heuristic.py:30](text_detection_baselines/models/length_heuristic.py#L30) and
[prompting_smol.py:116](text_detection_baselines/models/prompting_smol.py#L116)
compute `np.abs(scores - 0.5)` inline inside the `ood` expression.

The function is therefore tested but never called by the pipeline.

**Direction:** add an optional `uncertainties` field to `StubModelOutput`, have the
detectors populate it from the quantity they already compute, and call the curve
from `evaluate_predictions`. Requires resolving issue 3 first, since the curve
cannot travel through the current metric registry.

### 2. The curve ignores `ood_flags`

Every registered ranking metric restricts itself to non-OOD samples via
`_non_ood_binary`. `compute_abstention_curve` takes no `ood_flags` argument at all,
so its AUROC values are computed over the full sample set and are **not**
comparable to the `auroc` reported for the same run.

**Direction:** accept `ood_flags` and apply the same restriction, or document
loudly that this curve is a whole-population measure. The former is more
consistent.

### 3. Curves cannot enter the metric registry, and there is no scalar summary

`MetricFunc` is typed `-> float | None` and `run_all_metrics` pipes every result
through `safe_round`, which raises on a list. So the curve cannot be registered,
cannot appear in the console tables, and cannot appear in the CSV/JSON/YAML
exports. It is reachable only by importing the function directly.

**Direction:** register a scalar reduction of the curve — area under the
coverage-accuracy curve (AURC), or accuracy at a fixed coverage such as 80% — so
selective-prediction quality shows up in the standard outputs. Keep the full curve
available for plotting.

### 4. `tau` is held fixed across the sweep rather than recalibrated

At each abstention threshold, accuracy is computed against the `tau` learned on the
*full* sample. After abstaining, the FPR at that `tau` drifts, so the reported
`accuracy` conflates two effects: the retained subset genuinely being easier to
separate, and the operating point sliding away from `target_alpha`.

**Direction:** decide deliberately between the two readings. Recalibrating `tau`
per retained subset isolates separability; holding it fixed measures what a
deployed system with a frozen threshold would actually do. Consider reporting both.

### 5. Dropped sweep points are silent

Threshold points are skipped when fewer than two samples survive, when only one
label survives, or when tied variances duplicate the preceding subset. The
returned lists are simply shorter than `num_points` with no record of how many
points were dropped or why.

**Direction:** return a `dropped` count or a per-point reason, so a caller can tell
a 12-point curve that requested 20 from a genuinely 12-point sweep.

### 6. The `min_quantile=0.1` default is arbitrary

The reference implementation hard-coded a floor of 0.5, which capped minimum
coverage near 50% and hid the high-confidence regime entirely. The current default
of 0.1 is a better guess but is still a guess.

**Direction:** pick the floor from the smallest subset that still yields a stable
AUROC estimate, or sweep coverage on a log scale so the high-confidence tail is
sampled more densely.

---

## `auroc_at_1pct`

### 7. Unstable when the negative class is small

The 1% FPR ceiling can only be resolved to `1 / n_human`. On the bundled `gede`
dataset there are 916 human samples in total, so FPR ≤ 0.01 covers roughly nine
negatives — the metric is estimated from a handful of samples, and any per-category
or subsampled slice is worse.

**Direction:** return `None` below a minimum negative count (100 gives 1%
resolution), or report a bootstrap interval alongside the point estimate.

### 8. Values below 0.5 are possible and easy to misread

The McClish standardization maps a random ranker to 0.5 and a perfect ranker to
1.0, but it does **not** clamp at 0.5: a ranker worse than chance in the low-FPR
region scores below it. `dummy-raw` scores 0.4976 on `gede`. Anyone reading
this as a normalized AUROC-like quantity bounded below by 0.5 will be confused.

**Direction:** note the actual range wherever the metric is surfaced. Already
documented in the docstring and README.

### 9. The 1% ceiling is hard-coded and unrelated to `target_alpha`

`--target-alpha` defaults to 0.05, so the threshold `tau` is learned for a 5% FPR
budget while the partial AUROC summarizes the 1% regime. The two do not describe
the same operating point.

**Direction:** make the ceiling a parameter that defaults to `target_alpha`, or
register both a 1% and a `target_alpha` variant. Requires the registry to support
parameterized metrics, which it currently does not.

---

## PR-curve summaries

### 10. Two PR summaries are reported, and they disagree

`average_precision` and `pr_auc` both summarize the same curve.
`pr_auc` uses trapezoidal integration, which interpolates between operating points
that are not achievable; scikit-learn documents it as misleading. Observed
divergence on `gede`:

| model | `average_precision` | `pr_auc` (trapezoidal) |
| --- | --- | --- |
| dummy-norm | 0.9327 | 0.9664 |
| dummy-raw | 0.9300 | 0.9299 |
| length | 0.9710 | 0.9710 |

The 3.4-point gap on `dummy-norm` arises because that model's scores
are degenerate (AUROC exactly 0.5), leaving sparse PR points for the trapezoid to
inflate across.

**Direction:** retire `pr_auc` once `average_precision` is confirmed as the
reported summary. It was retained only for continuity with earlier numbers, and
two similarly named columns side by side invite misreading.

### 11. Average precision is not comparable across slices with different prevalence

AP has the positive-class base rate as its floor. `gede` is 93.27% machine
(12703/13619), which is why every model scores above 0.93 regardless of AUROC.
`dummy-norm` scores AUROC exactly 0.5 — no ranking signal whatsoever —
and AP 0.9327, i.e. exactly the prevalence. A reader scanning the AP column would
conclude that model is excellent. AP values from slices with different prevalence
also cannot be compared to each other.

**Direction:** report prevalence next to AP, or normalize (e.g. AP minus
prevalence, over 1 minus prevalence) so slices become comparable.

---

## Dataset and reporting

### 12. Every `gede` category is single-label, so per-category ranking metrics are all `None`

`auroc`, `auroc_at_1pct`, `average_precision`, and `pr_auc` all require both classes
present. On `gede`, every `contribution_level` category contains exactly one label:

| category | n | n_human | n_machine |
| --- | --- | --- | --- |
| Human | 916 | 916 | 0 |
| Humanize | 1783 | 0 | 1783 |
| Improved-Human | 1832 | 0 | 1832 |
| Rewrite-Human | 1832 | 0 | 1832 |
| Rewrite-LLM | 1832 | 0 | 1832 |
| Summary | 1776 | 0 | 1776 |
| Task | 1832 | 0 | 1832 |
| Task+Summary | 1776 | 0 | 1776 |
| task+resource | 40 | 0 | 40 |

So four of the eleven per-category columns are `-` for every row. This is a
property of the dataset, not a metric bug, but it makes the per-category table
substantially less useful than it appears.

**Direction:** decide what per-category evaluation should mean here. Options:
score each machine category against the shared human pool (which makes AUROC
defined and answers "which generation style is hardest to detect"); or suppress
ranking columns in the per-category table when no category has both labels.

### 13. `task+resource` has 40 samples and is reported with the same precision as categories 45× larger

No sample-count floor and no uncertainty estimate, so a 40-sample category renders
identically to a 1832-sample one.

**Direction:** report bootstrap confidence intervals, or flag categories below a
minimum size.

### 14. All exported metrics are rounded to 4 decimal places

`run_all_metrics` applies `safe_round(value)` with the default `ndigits=4`, so the
JSON/YAML/CSV exports are lossy. Fine for reading a table, lossy for paired
significance testing between two close models.

**Direction:** keep full precision in the structured exports and round only at the
console-rendering layer.

### 15. Console table columns are hard-coded positionally

The registry is fully dynamic on the producer side, but `render_console_tables`
([cli.py](text_detection_baselines/cli.py)) hard-codes each column header and cell
in matching positions across two tables. Adding one metric requires four
coordinated edits, and a mismatch silently shifts values into the wrong columns.
CSV/JSON/YAML export needs no change, which makes the asymmetry easy to forget.

**Direction:** give `MetricSpec` a display-name and ordering field and derive the
table columns from the registry.

### 16. The per-category table overflows an 80-column terminal

Now at 11 columns (13 for normalized-score models), headers truncate to `AURO…`
and dataset names to `dat…`.

**Direction:** select a metric subset for console display, transpose the table, or
gate columns behind a `--verbose` flag. The full set remains in the exports.
