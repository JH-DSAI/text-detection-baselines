# Package review

A review of the package as it stands, aimed at the question "what would bite an external NLP
researcher who picked this up?" rather than at internal-use readiness.

This document **complements [ISSUES.md](ISSUES.md)** and does not restate it. ISSUES.md records
14 metric-semantics items and remains the source of truth for those. Where a finding here
extends or reprioritizes one of them, it says so explicitly. Findings touching ISSUES.md are
R2 (#11), R8 (#4), and R25 (#1/#3/#9).

Severity is calibrated to the stated goal — usable and portable for outsiders, not high-polish.
**High** means an external user hits it and is either blocked or silently misled. **Medium**
means it costs them real time or credibility. **Low** means it is worth doing but should not
displace anything above it.

---

## Summary

The evaluation core is in better shape than its "stub" framing suggests. Three things stand out
as genuinely good and worth preserving through the coming changes:

* **The registries are the right abstraction.** Models, datasets, and metrics each register
  through a small, uniform interface, and the CLI derives its help text and selection logic from
  them. Adding a real baseline model is a factory function and one `register_model` call.
* **The metric documentation is unusually honest.** The README's "Implementation notes and
  discrepancies" section pre-empts exactly the misreadings these metrics invite — the McClish
  standardization not being clamped at 0.5, AP being floored by prevalence, why trapezoidal
  PR-AUC was removed, why thresholds are inclusive. Most benchmark suites do not do this, and
  it is the single strongest signal that the numbers can be trusted.
* **OOD restriction and score-normalization gating are applied consistently.** Every registered
  metric restricts to `~ood_flags`, so a results row describes one sample set; calibration
  metrics are omitted rather than nulled for raw-score models. Both invariants are tested.

The infrastructure (pixi, multi-Python CI matrix, bandit, mypy, Docker, Zenodo/RTD scaffolding)
is well beyond what internal research code usually carries.

What would actually bite an outside user, in order: the bundled corpus has no licence or
provenance (**R1**, mostly resolved); most of the per-category table reported fabricated numbers
(**R2**, resolved); three CLI flags silently did nothing (**R3**, resolved); and `pip install`
yields an unimportable package (**R11**). Beyond those, the largest structural item is that the public API bakes `Stub`
into its type names right before real models land (**R31**).

---

## Blockers for public release

| ID | Finding | Why it blocks |
| --- | --- | --- |
| **R1 (mostly resolved)** | Bundled corpus has no licence, attribution, or provenance | Mostly resolved: corpus no longer redistributed. History purge still outstanding and dataset reproduction test failed with reconciliation deferred to a follow-up issue |
| **R2 (resolved)** | Per-category FPR@tau / TPR@tau / CalGap are fabricated for single-label categories | Resolved: the three metrics return `None` on slices whose denominator class is empty, so the per-category table no longer publishes numbers that describe nothing |
| **R3 (resolved)** | `--text-key` / `--label-key` / `--category-key` are silently inert | Resolved: the flags now feed runtime registration, so the documented path for using your own data works |

R11 (broken `pip install`) is a blocker for PyPI specifically, not for the currently supported
clone-and-pixi path.

---

## Blocking findings

### R1. The bundled corpus ships with no licence, attribution, or provenance — High

**Status: mostly resolved.** The corpus is no longer redistributed. `datasets/gede_essays.json`
has been deleted, the two `tests/data/` fixtures (which contained the same third-party essay
text) have been deleted along with the whole `tests/data/` directory — the tests that used them
now build their input inline under `tmp_path` — `datasets/README.md` records provenance, the
licence chain, and composition, and users now obtain and convert the corpus themselves via
`pixi run prepare-gede`. Two items remain, recorded under "Remaining" below.

The upstream licence question is now answered rather than open: **the GEDE data is CC BY-NC-SA
4.0**, stated in the upstream README, and the upstream repository carries no `LICENSE` file of
its own. Redistribution under this repository's BSD-3 licence was therefore not permissible,
which is why the strip-and-prepare route was taken rather than seeking permission.

**Remaining:**

* **Purge the blob and the old fixtures from git history.** Deleting them from `HEAD` does not
  remove them from clones or from GitHub's archive downloads. This is a prerequisite for the
  repository going public, and needs coordinating with anyone holding a clone.
* **Reconcile the reproduced corpus with that from the research team.** The adapter does not exactly reproduce the JSON corpus we had, as seen in `2b2ec6b` (to be purged).  Sixty entries have empty questions, and other discrepancies are present as well.  Reconciling these discrepancies appears non-trivial and is deferred to a follow-up issue, R40.

---

The original finding, retained for context:

`datasets/gede_essays.json` is 13,619 records / 32 MB, tracked in git and redistributed under
the repository's BSD-3 licence. Its own `dataset` field records that it is derived from three
third-party corpora:

| source corpus | records |
| --- | --- |
| BAWE | 6,531 |
| argument-annotated-essays | 5,938 |
| persuade | 1,150 |

These are corpora of student academic writing, and several are distributed under their own
terms — some restrictive or non-commercial — which are not compatible with silent
redistribution of the full texts under BSD-3. The repository documents no source, no citation,
no data licence, and no record of redistribution permission. The `text_author` field also shows
916 human-authored student essays alongside machine rewrites, so this is third-party human
subject text, not synthetic data.

This is not something to resolve by reading the code. **Direction:** get the redistribution
terms for all three sources reviewed before the repository goes public. Independently of the
outcome, add attribution, a citation for each source corpus, a `datasets/README.md` recording
provenance and construction, and the generation parameters for the machine-authored portion
(the records carry `temperature` and `max_tokens`, so this is recoverable). If redistribution
is not clearly permitted, replace the checked-in blob with a fetch-and-build script so licence
acceptance stays with the user — which also resolves R12.

### R2. Per-category FPR@tau, TPR@tau, and CalGap are fabricated for single-label categories — High

**Status: resolved.** All three metrics now return `None` instead of dividing by a guarded
denominator. `fpr_at_tau_metric` and `tpr_at_tau_metric` delegate to a shared `_rate_at_tau`
helper in [metrics/detection.py](text_detection_baselines/metrics/detection.py) that builds the
denominator as `(labels == label_value) & ~ood_flags` and returns `None` when it is empty;
`calibration_gap_metric` calls the same helper and propagates the `None` rather than computing
`abs(0.0 - target_alpha)`. The return annotations widened from `float` to `float | None`, which
`MetricFunc` already admitted, so nothing downstream changed — `normalize_metric_value` passes
`None` through and the console renders `-`.

The denominator excludes OOD samples, not just absent classes. `flags` is already
`(scores >= tau) & ~ood_flags`, so an OOD sample could only ever land in the denominator; a
slice whose entire denominator class was OOD would otherwise have kept reporting a measured
rate of zero by a second route. This also makes the README's `target_alpha = 1.0` → `FPR = 1.0`
claim hold on datasets with OOD humans. The one behaviour change outside the empty case is that
`tpr_at_tau` no longer counts OOD machine samples as misses; the abstention curve remains the
place where coverage is measured. README's metric table, its non-OOD paragraph, and the
"Registered metrics exclude OOD-flagged samples" note now name all three metrics, and
ISSUES.md #11 records that six per-category columns rather than three are now `-` on `gede`.

Three regression tests cover it, each parameterized over all three metrics: denominator class
absent from the slice, denominator class present but entirely OOD, and — for `fpr_at_tau` /
`tpr_at_tau` — a mixed slice asserting the rate is `1/1` rather than `1/2`. The existing
`test_detection_metrics` expectation for `tpr_at_tau` moved from `0.5` to `1.0` for that same
reason. On the bundled `demo` dataset the per-category FPR@tau and CalGap columns now read `-`
for `Summary` and `Task`, and TPR@tau reads `-` for `Human`. The diagnosis below is kept as the
record of why the fix took this shape; its line references point at the pre-fix code.

`fpr_at_tau_metric` and `tpr_at_tau_metric`
([metrics/detection.py:113-159](text_detection_baselines/metrics/detection.py#L113-L159))
divide by `max(n_human, 1)` and `max(n_machine, 1)`. When the denominator class is absent from
a slice, they return `0.0` rather than `None`, and `calibration_gap` then reports
`abs(0.0 - target_alpha)`, i.e. exactly `target_alpha`.

Reproduced on a synthetic dataset with two all-machine categories:

| category | n_human | n_machine | fpr_at_tau | tpr_at_tau | calibration_gap | auroc |
| --- | --- | --- | --- | --- | --- | --- |
| Human | 2 | 0 | 0.5 | **0.0** | 0.45 | `None` |
| Summary | 0 | 2 | **0.0** | 1.0 | **0.05** | `None` |
| Task | 0 | 2 | **0.0** | 1.0 | **0.05** | `None` |

Eight of the nine `gede` categories contain only machine samples (per the table in ISSUES.md
#11), so on the bundled dataset the FPR@tau column reads `0.000` and CalGap reads `0.050` for
almost every per-category row — and both are artifacts of the guard, not measurements. The
`Human` category symmetrically fabricates `tpr_at_tau = 0.000`.

The damage is that these render as ordinary values. ISSUES.md #11 already notes that the three
ranking metrics go `None` on these slices, and a reader seeing `-` correctly infers "undefined."
A reader seeing `0.000` next to `-` reasonably infers "defined, and zero." This finding extends
#11 to the metrics it does not cover, and is more serious than #11 for exactly that reason.

**Direction (implemented):** return `None` when the denominator class is empty, in all three
metrics. The `max(n, 1)` guard was presumably added to avoid a divide-by-zero; `None` is the
correct answer to the same question. Add a regression test asserting `None` for both
single-label directions.

### R3. `--text-key`, `--label-key`, and `--category-key` are silently inert — High

**Status: resolved.** The three key literals now live as `DEFAULT_TEXT_KEY`, `DEFAULT_LABEL_KEY`,
and `DEFAULT_CATEGORY_KEY` in
[datasets/__init__.py](text_detection_baselines/datasets/__init__.py) and are the single source
of the `DatasetSpec` field defaults, `register_file_dataset`'s keyword defaults, and the three
`click.option` defaults. The flags are passed into `register_file_dataset` for every
`--register-file-dataset` entry, the `or` fallback in the evaluation loop is gone in favour of
`text_key=dataset.text_key`, and the help text now says the flags describe runtime-registered
files. `DatasetSpec`'s fields stayed required `str`. Two smoke tests cover it: a runtime dataset
whose text field is `text` loads under `--text-key text`, and `demo` still loads from `answer` in
the same run. The diagnosis and reasoning below are kept as the record of why the fix took this
shape; its line references point at the pre-fix code.

[cli.py:501-503](text_detection_baselines/cli.py#L501-L503) resolves each key as
`dataset.text_key or text_key`. But `DatasetSpec` defaults those fields to non-empty strings
([datasets/__init__.py:22-24](text_detection_baselines/datasets/__init__.py#L22-L24)), so the
spec value is always truthy and the CLI value is never reached. Verified: for the `gede` spec,
`spec.text_key or "MY_CLI_KEY"` evaluates to `"answer"`.

The `or` fallback reads as "spec overrides, CLI is the default," which would be defensible — but
because the spec fields can never be empty, the CLI half is dead code for every dataset,
including ones the user registers at runtime with `--register-file-dataset` (which also takes
the defaults).

The user-visible consequence: a researcher points the tool at their own JSON whose text field
is `"text"` rather than `"answer"`, passes `--text-key text`, and gets
`ValueError: No valid samples with required keys in <path>` — an error that names the file but
not the reason, for a flag that was accepted without complaint. That message is its own finding
(**R39**), because it will stay the observable symptom of any key mismatch after this one is
fixed.

**Direction (implemented):** keep all three `DatasetSpec` fields as required `str` and make the CLI flags feed
*registration* rather than evaluation. Evaluation cannot proceed without the keys, so every
registered dataset should carry a usable triple by construction; widening the fields to
`str | None` would relocate that guarantee into every consumer instead of resolving it. The `or`
fallback is also the wrong shape independent of its typing — `--text-key` is a single global flag,
so even a working fallback would apply one schema to every selected dataset at once. Concretely:

1. Hoist the default literals into named constants in
   [datasets/__init__.py](text_detection_baselines/datasets/__init__.py) —
   `DEFAULT_TEXT_KEY = "answer"`, `DEFAULT_LABEL_KEY = "label"`,
   `DEFAULT_CATEGORY_KEY = "contribution_level"` — and use them for the `DatasetSpec` field
   defaults, for `register_file_dataset`'s keyword defaults, and for the three `click.option`
   defaults. Those literals are currently spelled out three times across two files with nothing
   tying them together, so they can drift silently.
2. Pass the flags into registration at [cli.py:471-473](text_detection_baselines/cli.py#L471-L473):
   `register_file_dataset(name=name, path=dataset_path, text_key=text_key, label_key=label_key,
   category_key=category_key)`.
3. Drop the `or` at [cli.py:501-503](text_detection_baselines/cli.py#L501-L503) in favour of
   `text_key=dataset.text_key`. The spec becomes the single source of truth, and there is no
   longer a fallback branch that can be dead.
4. Retarget the help text, which is what currently implies the flags are global: "Field name for
   text in datasets registered via `--register-file-dataset`." Likewise for the other two.

Built-in specs such as `gede` are then explicitly unaffected by these flags — which is already
the actual behaviour; this change stops the help text advertising otherwise. The flags end up
doing exactly one legible thing: describing the schema of files registered at runtime.

One limitation worth stating in the help text: the flags apply uniformly to every
`--register-file-dataset` entry in a run, so two runtime files with different schemas cannot be
expressed. See **R38**.

Tests: register a runtime dataset over a JSON whose text field is `text`, pass `--text-key text`,
and assert it loads; and assert that `--text-key` leaves the `gede` spec's `answer` intact.

---

## Correctness and data handling

### R4. `normalize_label` passes arbitrary integers through unchecked — Medium

[datasets/file.py:29-41](text_detection_baselines/datasets/file.py#L29-L41) returns
`int(raw_label)` for any integer input. Verified: `2 → 2`, `-1 → -1`, `7 → 7`. String labels are
validated against an allow-list; integers are not.

A record with label `2` then matches neither `labels == 0` nor `labels == 1`. It inflates
`n_samples`, is counted in neither `n_human` nor `n_machine`, is excluded from the FPR and TPR
denominators, and still reaches sklearn — which raises on a multiclass target with a message
that does not name the file or the offending value. A three-class label column (a plausible
mistake when adapting a dataset with a `neutral` or `unknown` class) therefore produces either
a confusing sklearn error or, if it survives, silently wrong denominators.

**Direction:** raise `ValueError` for any integer outside `{0, 1}`, matching how string labels
are already handled, and include the record index in the message.

### R5. Malformed rows are dropped silently — Medium

[datasets/file.py:77-85](text_detection_baselines/datasets/file.py#L77-L85) skips any row
missing the text or label key, and the JSON-array branch (line 53) additionally drops any
non-dict element. Nothing is counted or logged. A dataset that is half-malformed evaluates
without complaint on whatever survived, and the reported `n_samples` looks authoritative.

The `if not texts: raise` guard catches only the total-failure case — which is also the case
that would be caught anyway.

**Direction:** count skipped rows and log at WARNING with the count and the first few offending
indices. Consider a `--strict` flag that makes any skip fatal.

### R39. The "no valid samples" error names the file but not the reason — Medium

When the key check at [datasets/file.py:78](text_detection_baselines/datasets/file.py#L78) skips
every record, [file.py:84-85](text_detection_baselines/datasets/file.py#L84-L85) raises
`ValueError(f"No valid samples with required keys in {path}")`. The message states neither which
keys were required nor which keys the file actually contains, so the one fact needed to fix the
problem is the one fact withheld. It also reaches the user as a bare traceback rather than a CLI
diagnostic.

This is the first error a researcher hits when pointing the tool at their own data. It is the
observable symptom of **R3** today, and it stays the observable symptom of every future key
mismatch — a typo'd `--text-key`, a renamed column, a JSON export that nests records one level
deeper — once R3 is fixed. Distinct from **R5**, which is about partial drops that are never
reported at all; this is the total-failure case that *is* reported, unhelpfully.

**Direction:** name the required keys and the keys present on the first record, and surface it
through `click.ClickException` at the CLI boundary so it does not print a traceback:

```
No samples in data/my_corpus.jsonl had required keys 'text' and 'label'
(first record has: answer, label, contribution_level)
```

Listing the fields that are actually there turns a dead end into a one-line diagnosis, and it
distinguishes the two failure modes the current message conflates: wrong key names versus a file
whose records are not the shape the loader expects at all. Fold this into whatever skip
bookkeeping R5 introduces so both findings share a single counting pass.

### R6. A single-class dataset is an unhandled `ValueError` — Low

[evaluate.py:118-119](text_detection_baselines/evaluate.py#L118-L119) raises when either class
is absent. This is a plausible user situation (evaluating on a machine-only generation set), and
the error surfaces as a bare traceback out of the CLI rather than a diagnostic. The check is
correct; its presentation is not.

**Direction:** catch it in `main` and exit via `click.ClickException` with the dataset name, or
skip the dataset with a warning and continue with the others.

### R7. `tau` is learned and evaluated on the same samples — Medium (methodology)

[evaluate.py:122-128](text_detection_baselines/evaluate.py#L122-L128) learns `tau` as an
empirical quantile of the evaluation set's own human scores. `FPR@tau` is therefore an in-sample
fit to `target_alpha` by construction, and `CalGap` measures quantile granularity — how close
the empirical quantile can land to the target given `n_human` — rather than anything about the
detector. On `gede`'s 916 human samples, CalGap is floored around `1/916`.

For a stub harness this is a reasonable simplification. For a suite whose numbers researchers
may cite, it needs to be stated plainly, and revisited when real models arrive: a held-out
calibration split is the standard alternative, and the difference will matter more for models
that actually separate the classes.

**Direction:** document the in-sample calibration in the README's implementation-notes section
now. Track a calibration/evaluation split as a prerequisite for the first real baseline model.

### R8. Per-category rows reuse the global `tau` but print a per-category `target_alpha` — Low

`evaluate_predictions` computes `flags` once at the dataset level and slices them per category
([evaluate.py:128](text_detection_baselines/evaluate.py#L128),
[evaluate.py:146-157](text_detection_baselines/evaluate.py#L146-L157)), which is the right
default — a per-category threshold would not correspond to any deployable system. But
`_base_counts` writes `target_alpha` into every per-category row, implying each category has its
own operating point, and per-category CalGap is computed against a target the category was never
calibrated to.

This is the same fixed-versus-recalibrated question ISSUES.md #4 raises for the abstention
curve, and the two should be decided together rather than diverging.

**Direction:** document that per-category rows share the run-level `tau`, and either drop
`target_alpha` from per-category rows or rename the per-category CalGap to make clear it is
measured against the global target.

### R9. Misplaced bandit suppressions — Low

`# nosec: B615[huggingface_unsafe_download]` appears on two calls that load a local JSON file
and never touch Hugging Face ([evaluate.py:54](text_detection_baselines/evaluate.py#L54),
[evaluate.py:187](text_detection_baselines/evaluate.py#L187)). The genuine `from_pretrained`
calls in [models/prompting_smol.py:58-59](text_detection_baselines/models/prompting_smol.py#L58-L59)
already pin `revision` and need no suppression at all.

Suppressions on code that cannot trigger the rule train reviewers to skim past them.

**Direction:** remove both, and re-run `pixi run -e dev check-security` to confirm nothing was
actually being suppressed.

### R10. `predict()` does not validate its own output shape — Low

`StubModelOutput` carries three arrays that every downstream metric assumes are parallel to
`labels`, but nothing checks it. A model returning a wrong-length `ood_flags` produces a
broadcasting error deep inside a metric rather than at the model boundary. Cheap to fix in
`__post_init__`, and worth doing before third parties write detectors.

### R40. GEDE data is not exactly reproduced — High

The GEDE adapter does not exactly reproduce the JSON corpus we had, as provided by the research team.  Sixty entries have empty questions, and other discrepancies are present as well.  These discrepancies might be due to additional postprocessing performed by the research team and/or changes in the upstream data.

---

## Packaging and portability

These are must-fix-before-PyPI, not breakage on the currently supported clone-and-pixi path.

### R11. `pip install` produces an unimportable package — High (for PyPI)

`[project] dependencies` lists only click, pyyaml, and rich
([pyproject.toml:14](pyproject.toml#L14)). The package imports numpy and scikit-learn
unconditionally, and torch optionally. The pixi environments supply these as conda dependencies,
which is why nothing surfaces on the supported path.

Verified against the repository's own pip-created `.venv`:

```
$ ./.venv/bin/python -c "import text_detection_baselines.cli"
  File ".../text_detection_baselines/datasets/file.py", line 10, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

**Direction:** add `numpy` and `scikit-learn` to `[project] dependencies`. Put `torch` and
`transformers` behind a `[project.optional-dependencies] models` extra — `torch_linear` already
degrades to a numpy path when torch is absent, so only `smollm2` hard-requires them, and it is
already opt-in.

### R12. The bundled dataset is unreachable from an installed package — High (for PyPI)

`_DEFAULT_GEDE_PATH` walks `parents[2]` up from the package directory
([datasets/__init__.py:102](text_detection_baselines/datasets/__init__.py#L102)). In a source
checkout that lands on `<repo>/datasets/`. In an installed package it lands on
`<site-packages>/datasets/`, and the wheel ships only `packages = ["text_detection_baselines"]`
([pyproject.toml:50-51](pyproject.toml#L50-L51)) — so the data directory is never installed.

Registration does not check existence, so `gede` still advertises itself in `--help` and in
`get_default_dataset_names()`; the failure arrives later, from `read_text`, as a
`FileNotFoundError` pointing at a path inside site-packages.

**Direction:** decide the data story before more datasets land, since checking corpora into git
does not scale past one and interacts with R1. Either move data inside the package and load via
`importlib.resources`, or fetch on demand into a cache directory. Either way, have
`register_file_dataset` record whether the path resolves, and fail with a message that says the
dataset was not installed rather than surfacing a raw path.

### R13. No `py.typed` marker — Low

The package is fully annotated and mypy runs over it in CI, but without `py.typed` no downstream
user gets any of that. One empty file plus a wheel-inclusion line.

### R14. Distribution metadata is unfinished — Medium

`description = ""` ([pyproject.toml:8](pyproject.toml#L8)); no `keywords`, no `classifiers`, no
bug-tracker URL, and `documentation` pointing at the GitHub repository rather than Read the Docs
([pyproject.toml:35-38](pyproject.toml#L35-L38)). The PyPI page would render essentially blank
above the README. `license-files` also lists `.zenodo.json`
([pyproject.toml:11](pyproject.toml#L11)), which is citation metadata rather than a licence, and
ships it into `dist-info/licenses/`.

### R15. Dependencies are declared twice and have already drifted — Medium

`[project.optional-dependencies]` and the pixi features list overlapping development
dependencies with different contents: `types-pyyaml` appears only in the pixi feature, while
`bandit`, `build`, `hatchling`, and `hatch-vcs` appear only in the extras. Read the Docs installs
from the extras ([.readthedocs.yaml](.readthedocs.yaml)) while CI installs from pixi, so the two
paths can diverge without anything failing.

**Direction:** pick one as authoritative. Keeping `[project]` minimal and treating pixi as the
development source of truth is the smaller change; then the extras exist only for RTD and should
say so in a comment.

### R16. `.coveragerc` silently excludes every `__init__.py` — Medium

**Status: resolved.** `omit = **/_*.py` ([.coveragerc](.coveragerc)) matches `__init__.py`. Confirmed against the
current `coverage.json`, whose file list contains none of them:

```
text_detection_baselines/cli.py, datasets/file.py, evaluate.py, util.py,
metrics/{calibration,detection,selective}.py, models/{base,length_heuristic,prompting_smol,torch_linear}.py
```

The omitted files are exactly where the extension points live — `build_model` and
`register_model` in `models/__init__.py`, `run_all_metrics`, `register_metric`, and
`normalize_metric_value` in `metrics/__init__.py`, and the whole dataset registry in
`datasets/__init__.py`. Those functions *are* exercised by the test suite; they are just not
measured, so the reported 82.7% describes a smaller package than the one that ships, and CI's
`PATCH_MIN_THRESHOLD: 80` diff-cover gate ([.github/workflows/ci.yml](.github/workflows/ci.yml))
cannot see changes to them at all.

**Direction:** replace the pattern with explicit entries (`_version.py`, `setup.py`,
`*/tests/*`). Expect the headline number to move; that movement is the point.


## Testing

### R17. `main()` is entirely untested — Medium

`cli.py` sits at 77.7% with all 39 missed lines in the command body: flag resolution,
`--all` handling, the unknown-name validation calls, model construction, the dataset/model
evaluation loop, and the export branch. Everything `main` delegates to is tested; the wiring
between them is not, which is where R3 lives — a test that ran the CLI against a fixture with
non-default field names would have caught it.

`AGENTS.md` forbids `CliRunner`, not subprocess execution. A `subprocess.run` smoke test over a
tiny fixture with `--export json`, asserting exit code 0 and the presence of expected keys in
`metrics.json`, covers the whole path without asserting on console formatting — which is the
distinction `AGENTS.md` is drawing.

### R18. Tests resolve fixtures relative to the working directory — Low

**Status: resolved.**  [tests/test_datasets.py:21](tests/test_datasets.py#L21) and its neighbours use
`Path("tests/data/…")`, so the suite passes only when invoked from the repository root. Anchor
on `Path(__file__).parent / "data"`, ideally via a fixture.

### R19. `test_determinism_same_seed` asserts nothing — Low

[tests/test_models.py:81-87](tests/test_models.py#L81-L87) builds two models with the same seed
and compares outputs, but no model consumes `seed` and all three are deterministic. The test
passes for reasons unrelated to its name and would keep passing if seeding were broken. Pairs
with R20 — resolve both together.

### R20. No pytest configuration — Low

No `[tool.pytest.ini_options]`: no `testpaths`, no `--strict-markers`, no warning filters. A
`filterwarnings = ["error"]` setting in particular would catch sklearn's undefined-metric
warnings, which are load-bearing signals in a metrics package.

---

## CLI and usability

### R21. `--seed` is accepted, threaded through every factory, and used by nothing — Medium

The flag is documented as "Random seed for model stubs," is stored on `StubTextDetector`, and is
passed through all four factories. No model reads it. In a benchmarking tool, an accepted
`--seed` is a reproducibility claim, and this one is empty.

**Direction:** either remove it until something needs it, or wire it to a `np.random.Generator`
on the base class so the first stochastic model gets it for free. Removing it is honest now;
keeping it means R19 should become a real test.

### R22. No way to run on a subset, and no progress output — Medium

`smollm2` scores one text at a time with two forward passes each
([models/prompting_smol.py:111-112](text_detection_baselines/models/prompting_smol.py#L111-L112),
[models/prompting_smol.py:102-103](text_detection_baselines/models/prompting_smol.py#L102-L103)) —
roughly 27,000 sequential passes over `gede`, unbatched, uncached, with no progress indication
and no way to interrupt and resume. There is no `--limit` or `--sample` flag, so the smallest
possible trial of the one model that resembles a real detector is the full dataset. `rich` is
already a dependency, so a progress bar is nearly free.

**Direction:** add `--limit N` (with deterministic sampling) and a progress bar. Batching and an
on-disk score cache are the follow-ups, and are prerequisites for the Azure model — see R27.

### R23. Runs carry no provenance, and no per-sample outputs — Medium

Exports record `tau`, `target_alpha`, `n_samples`, and `dataset_path`, but not the package
version, git SHA, seed, `ood_margin`, or the CLI invocation. A `metrics.json` found six months
later cannot be traced to the run that produced it — which matters more than usual here, because
the metric definitions have already changed once (`pr_auc` → `average_precision`) in a way that
makes old and new files silently incomparable, as the README itself notes.

Separately, only aggregates are exported. There is no way to obtain per-sample scores, so error
analysis, threshold re-sweeps, and significance testing all require re-running the model.

This makes the README claim at [README.md:176-178](README.md#L176-L178) incorrect: full-precision
*summary* metrics do not enable "paired significance testing between two close models." A paired
test needs per-sample scores from both models. The precision claim is right; the justification
attached to it is not.

**Direction:** write a `run-metadata.json` alongside the metrics (version, git SHA, full argv,
resolved options, UTC timestamp); add `--export-predictions` writing per-sample
score/label/category/ood rows; and fix the README sentence to justify full precision on its
actual merits.

### R24. Minor CLI gaps — Low

* `Console(record=True)` ([cli.py:143](text_detection_baselines/cli.py#L143)) accumulates every
  rendered table in memory and the recording is never exported. Either add `--save-console` using
  `console.export_text()`, or drop the flag.
* No `--version` flag, which is the first thing a user reaches for when filing a bug.
* `--output-dir` defaults to `evaluation_results` and is advertised in `--help`, but is only
  created when `--export` is also passed. Harmless, mildly confusing.

### R38. Text, label, category key overrides cannot be specified per-dataset

Specifying a dataset's text key, label key, and/or category key in the CLI sets those keys uniformly across all datasets; those keys cannot be specified differently per-dataset.

**Direction:** Extend `NamePathParamType`
([cli.py:87](text_detection_baselines/cli.py#L87)) to accept optional per-entry key overrides,
rather than adding more global flags.

---

## Documentation

### R25. The Sphinx documentation is unedited template text — Medium

`docs/source/index.rst` still reads "**text_detection_baselines** is a Python library for...";
`usage.rst` ends mid-sentence at "first install it using...:"; `api.rst` is a bare
`autosummary` over the top-level package with no `:recursive:`, so nothing below
`text_detection_baselines/__init__.py` is documented — none of the metrics, models, or dataset
APIs. Meanwhile the README carries a Read the Docs badge for a project its own to-do list says
has not been imported yet, so the badge currently resolves to nothing.

The docstrings themselves are good and would carry a real API reference with little work.

**Direction:** add `:recursive:` to the autosummary, write the two prose pages, and move the
"Metrics" section of the README into the docs (leaving a pointer). Until the RTD project exists,
remove the badge rather than shipping a broken one.

### R26. The README opens with a maintenance to-do list — Medium

Lines 10-15 are four internal chores (DOI, RTD import, `.zenodo.json`, "Update quickstart guide
below") sitting above the first description of what the package does. On a public landing page
this reads as abandoned rather than in-progress. The same applies to `.zenodo.json`, whose
`creators`, `description`, and `references` are all literal `TODO`, and which is the file
`CONTRIBUTING.md` directs contributors to add themselves to.

**Direction:** move all four into the tracker. Fill in `.zenodo.json` before the first tagged
release, since Zenodo captures it at tag time.

### R27. The README does not describe the bundled dataset — Medium

`gede` is the default dataset and the source of every number in the documentation, but the README
never says where it came from, that it is 93.27% machine, that it contains 916 human samples
against 12,703 machine ones, or that every `contribution_level` category is single-label. All
three facts are needed before any reported number can be read correctly, and the first is also
part of R1. The prevalence and single-label properties are recorded in ISSUES.md — which is not
shipped, so that information disappears at release.

**Direction:** add a "Datasets" section to the README with the composition table, sources, and
citations. Migrate the parts of ISSUES.md #10 and #11 that are dataset *facts* rather than
open questions.

### R28. `CONTRIBUTING.md` has a broken code fence — Low

An unmatched ``` at [CONTRIBUTING.md:32](CONTRIBUTING.md#L32) opens a fence that never closes,
swallowing the rest of "Getting Started" into a code block when rendered on GitHub. The document
also never mentions `pixi run -e dev test` or `pixi run -e dev ci`, so a contributor following it
has no way to know how to run the tests.

### R29. Missing `CHANGELOG.md` and `CITATION.cff` — Low

Both are conventions research users look for, and a DOI is already planned. `CITATION.cff`
renders as a "Cite this repository" button on GitHub and is cheap. A changelog matters more than
usual given metric definitions have already changed incompatibly once.

### R30. `CODEOWNERS` names no one — Low

`*    JHU DSAI` is neither a GitHub username nor an `@org/team` reference, so the rule matches
everything and assigns nobody. GitHub reports this as invalid in the repository settings but
does not fail anything.

---

## Larger and structural concerns

These are not one-PR fixes, but each gets materially more expensive after the package is public
or after the next feature lands on top of it.

### R31. `Stub` is baked into the public API, immediately before real models arrive — High

`StubTextDetector`, `StubModelOutput`, and `build_stub_model` are all exported from
[models/__init__.py:13-26](text_detection_baselines/models/__init__.py#L13-L26). The base class
is the interface every future detector implements, including real baselines and the Azure model —
none of which are stubs. The naming is accurate today and will be wrong within one milestone,
and by then external users will have written `class MyDetector(StubTextDetector)`.

**Direction:** rename to `TextDetector` and `ModelOutput` now, while the package is unpublished
and the cost is a sed. Drop `build_stub_model` at the same time — it is documented as a
"backward-compatible alias" ([models/__init__.py:105-107](text_detection_baselines/models/__init__.py#L105-L107))
in a package with no released version to be compatible with, and it is what the test suite
happens to call, which keeps it alive.

### R32. `MetricFunc`'s signature is the common root of three ISSUES.md items — Medium

`MetricFunc` is a fixed six-positional-argument callable returning `float | None`
([metrics/__init__.py:15-18](text_detection_baselines/metrics/__init__.py#L15-L18)). Every
registered metric therefore takes every argument whether it uses it or not — hence the `del`
statements at the top of all nine — and three separate ISSUES.md items are downstream of this
one shape:

* **#1** needs an `uncertainties` array threaded to a metric. No slot for it.
* **#3** needs a metric to return a curve. Return type forbids it, and `normalize_metric_value`
  raises on a list.
* **#9** needs a metric parameterized by an FPR ceiling. No mechanism, as the issue notes.

Adding an argument today means editing all nine metrics and every test that calls them
positionally.

**Direction:** one change unblocks all three — pass a single frozen `MetricContext` dataclass
(labels, scores, ood_flags, flags, target_alpha, tau, plus optional fields as they arrive) and
widen the return type to `float | Sequence[float] | None`, with `normalize_metric_value`
dispatching on it. Worth doing before three separate PRs each work around the same constraint.

### R33. No plugin mechanism for external or proprietary models — Medium

`register_model` is import-time only, so a model can only enter the registry by being imported
by this package. For the planned Azure integration that leaves two options: vendor proprietary
code into a public BSD-3 repository, or maintain a fork. Both are bad, and the same constraint
applies to any outside researcher who wants to evaluate their own detector against these
baselines without a PR.

**Direction:** declare an entry-point group (`text_detection_baselines.models`, and eventually
`.datasets`) and discover it at import. A separate private package then registers itself on
install, the public repo stays clean, and third-party detectors become a `pip install` rather
than a fork. This is a small change now and a migration later.

### R34. The detector interface will not carry an API-backed model — Medium

`predict(texts: list[str]) -> StubModelOutput` is synchronous, all-at-once, and returns only
when every text is scored. Against a hosted endpoint on 13,619 documents that means no batching,
no concurrency, no retry or rate-limit handling, no partial-progress persistence — one failure at
document 13,000 discards the whole run — and no caching, so every rerun re-spends. `smollm2`
already demonstrates the shape of the problem locally (R22); a network endpoint with quotas and
per-call cost turns it from slow into unusable.

**Direction:** settle the interface before the Azure work rather than during it. The pieces are a
batch-iterator protocol (`predict_batch(texts) -> Iterator[...]`) so results stream and can be
checkpointed, an on-disk score cache keyed by (model id, model version, text hash), an explicit
retry/backoff policy, and credentials read from the environment rather than passed as CLI
options. The cache is the highest-value piece and benefits `smollm2` immediately.

### R35. Dataset loaders have no public registration function — Low

Models get `register_model` and dataset instances get `register_dataset`, but `DATASET_LOADERS`
([datasets/__init__.py:12](text_detection_baselines/datasets/__init__.py#L12)) is a bare dict
that must be mutated directly. Adding the planned Hugging Face loader means reaching into module
state. A `register_dataset_loader` function would make the three registries symmetric.

### R36. Torch is a mandatory dependency for a stub — Low

The default pixi environment resolves to ~823 MB, dominated by PyTorch, to support a four-feature
linear layer with hard-coded weights that already has a working numpy fallback
([models/torch_linear.py:73-79](text_detection_baselines/models/torch_linear.py#L73-L79)). This
becomes reasonable the moment a real model lands, so it is not worth undoing — but today it is
the dominant cost of a first `pixi run main`, and it is worth noting in the README so the first
run is not a surprise.

### R37. The Docker image is larger than it needs to be — Low

`COPY . .` in the [Dockerfile](Dockerfile) includes the 32 MB corpus, `.dockerignore` does not
exclude `datasets/`, and `pixi install --locked` builds the full torch environment — for an image
whose `CMD` reruns the entire benchmark on every start. A multi-stage build, or at minimum
excluding the corpus and defaulting to `--help`, would make it usable as a tool rather than only
as a demo.

---

## Checked and deliberately not raised

Recorded so these read as decisions rather than oversights, given the stated polish target:

* **Console table width and hard-coded columns** — already ISSUES.md #13 and #14, and I agree
  with both the diagnosis and the suggested directions. Nothing to add.
* **`assert` for type narrowing** in [models/prompting_smol.py:76](text_detection_baselines/models/prompting_smol.py#L76)
  and [:92](text_detection_baselines/models/prompting_smol.py#L92) — stripped under `python -O`,
  but they narrow state the surrounding code already guarantees, so the failure mode is
  theoretical.
* **`find_repo_location`** ([util.py:21-23](text_detection_baselines/util.py#L21-L23)) returns a
  meaningless path for an installed package and is used only by its own test. Template cruft;
  harmless. Worth deleting if `util.py` is touched for another reason.
* **No dependency scanning** (`pip-audit` / Dependabot alerts) behind the Security badge, which
  covers only bandit. Reasonable scope for this project's size.
* **The docs CI job builds latex and epub** (`make clean html latex epub`) on every push, which
  is slower than needed. Seconds, not minutes.
* **Registry mutation makes test ordering matter in principle** — the tests that register into
  global state already clean up in `finally`, so this is handled.

---

## Suggested order of work

1. **R1** — done except purging the corpus from git history, which is a prerequisite for
   going public and needs coordinating across clones.
2. **R4** — correctness. Small, well-scoped, and a silent-wrongness bug. **R2** and **R3** are
   done; **R39** was to ride along with R3 and still stands on its own: it is the message a user
   actually sees when the keys are wrong.
3. **R31**, **R32** — the two renames/reshapes that get more expensive with every week of use.
4. **R17** — make the test signal trustworthy before building on it.
5. **R11**, **R12**, **R13**, **R14** — the PyPI gate, as one batch when a release is in view.
6. **R25**, **R26**, **R27** — documentation, before announcing to anyone outside.
7. **R33**, **R34** — settle before the Azure integration starts, not during.
