"""Tests for CLI option selection helpers and for the assembled command itself."""

import json
from pathlib import Path

import click
import pytest

from text_detection_baselines import cli as cli_module
from text_detection_baselines.cli import (
    NAME_PATH,
    _flatten_overall,
    _flatten_per_category,
    _raise_for_unavailable_dataset,
    _raise_for_unknown_names,
    _resolve_selection,
    _unique_preserve_order,
    _write_csv,
    export_results,
    main,
    render_console_tables,
)
from text_detection_baselines.datasets import DatasetSpec, register_file_dataset


def test_resolve_selection_uses_defaults_when_not_all():
    selected = _resolve_selection(
        explicit_names=(),
        default_names=("gede",),
        all_names=("gede", "runtime"),
        runtime_names=(),
        include_all=False,
        excluded_names=(),
    )

    assert selected == ("gede",)


def test_resolve_selection_includes_runtime_registration():
    selected = _resolve_selection(
        explicit_names=(),
        default_names=("gede",),
        all_names=("gede", "runtime"),
        runtime_names=("runtime",),
        include_all=False,
        excluded_names=(),
    )

    assert selected == ("gede", "runtime")


def test_resolve_selection_excludes_by_name_case_insensitive():
    selected = _resolve_selection(
        explicit_names=("runtime",),
        default_names=("gede",),
        all_names=("gede", "runtime"),
        runtime_names=(),
        include_all=False,
        excluded_names=("GeDe",),
    )

    assert selected == ("runtime",)


def test_resolve_selection_can_remove_all_items():
    selected = _resolve_selection(
        explicit_names=(),
        default_names=("gede",),
        all_names=("gede",),
        runtime_names=(),
        include_all=False,
        excluded_names=("gede",),
    )

    assert selected == ()


def test_unique_preserve_order_deduplicates_case_insensitively():
    result = _unique_preserve_order(["A", "b", "a", "B", "c"])
    assert result == ("A", "b", "c")


def test_unique_preserve_order_empty():
    assert _unique_preserve_order([]) == ()


def test_resolve_selection_defaults_when_no_explicit():
    result = _resolve_selection(
        explicit_names=(),
        default_names=("d1", "d2"),
        all_names=("d1", "d2", "d3"),
        runtime_names=(),
        include_all=False,
        excluded_names=(),
    )
    assert result == ("d1", "d2")


def test_resolve_selection_all_overrides_defaults():
    result = _resolve_selection(
        explicit_names=(),
        default_names=("d1",),
        all_names=("d1", "d2", "d3"),
        runtime_names=(),
        include_all=True,
        excluded_names=(),
    )
    assert result == ("d1", "d2", "d3")


def test_resolve_selection_explicit_added_to_defaults():
    result = _resolve_selection(
        explicit_names=("d3",),
        default_names=("d1", "d2"),
        all_names=("d1", "d2", "d3"),
        runtime_names=(),
        include_all=False,
        excluded_names=(),
    )
    assert result == ("d1", "d2", "d3")


def test_resolve_selection_runtime_names_included_in_all():
    # runtime names are registered before _resolve_selection is called;
    # when include_all=True, all_names includes them.
    result = _resolve_selection(
        explicit_names=(),
        default_names=("d1",),
        all_names=("d1", "runtime-ds"),
        runtime_names=("runtime-ds",),
        include_all=True,
        excluded_names=(),
    )
    assert result == ("d1", "runtime-ds")


def test_resolve_selection_runtime_names_appended_when_not_in_all():
    # runtime_names added even when not in all_names (edge case where
    # all_names snapshot was taken before registration).
    result = _resolve_selection(
        explicit_names=(),
        default_names=(),
        all_names=(),
        runtime_names=("runtime-ds",),
        include_all=False,
        excluded_names=(),
    )
    assert result == ("runtime-ds",)


def test_resolve_selection_deduplicates_across_sources():
    result = _resolve_selection(
        explicit_names=("d1",),
        default_names=("d1", "d2"),
        all_names=("d1", "d2"),
        runtime_names=("d2",),
        include_all=False,
        excluded_names=(),
    )
    # d1 from defaults, d2 from defaults, runtime d2 deduped, explicit d1 deduped
    assert result == ("d1", "d2")


def test_resolve_selection_preserves_insertion_order():
    result = _resolve_selection(
        explicit_names=("z",),
        default_names=("a", "b"),
        all_names=("a", "b", "z"),
        runtime_names=(),
        include_all=False,
        excluded_names=(),
    )
    assert result == ("a", "b", "z")


def test_flatten_helpers():
    tree = {
        "overall": {"ds1": {"model-a": {"auroc": 0.8}}},
        "per-category": {"ds1": {"Cat1": {"model-a": {"tpr_at_tau": 0.9}}}},
    }
    overall_rows = _flatten_overall(tree)
    assert len(overall_rows) == 1
    assert overall_rows[0]["dataset"] == "ds1"
    assert overall_rows[0]["model"] == "model-a"
    assert overall_rows[0]["auroc"] == 0.8

    per_cat_rows = _flatten_per_category(tree)
    assert len(per_cat_rows) == 1
    assert per_cat_rows[0]["category"] == "Cat1"
    assert per_cat_rows[0]["tpr_at_tau"] == 0.9


def test_write_csv_writes_header_and_rows(tmp_path):
    path = tmp_path / "rows.csv"
    rows = [
        {"dataset": "ds1", "model": "m1", "auroc": 0.9},
        {"dataset": "ds2", "model": "m2", "auroc": 0.8},
    ]

    _write_csv(path, rows)

    content = path.read_text(encoding="utf-8")
    assert "dataset,model,auroc" in content
    assert "ds1,m1,0.9" in content
    assert "ds2,m2,0.8" in content


def test_write_csv_empty_rows_creates_empty_file(tmp_path):
    path = tmp_path / "rows.csv"

    _write_csv(path, [])

    assert path.read_text(encoding="utf-8") == ""


def test_export_results_writes_json_yaml_and_csv(tmp_path):
    tree = {
        "overall": {
            "ds1": {
                "m1": {
                    "auroc": 0.91,
                    "fpr_at_tau": 0.1,
                    "tpr_at_tau": 0.9,
                }
            }
        },
        "per-category": {
            "ds1": {
                "catA": {
                    "m1": {
                        "auroc": 0.95,
                        "fpr_at_tau": 0.05,
                        "tpr_at_tau": 0.94,
                    }
                }
            }
        },
    }

    export_results(output_dir=tmp_path, export_formats=("json", "yaml", "csv"), tree=tree)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.yaml").exists()
    assert (tmp_path / "overall-metrics.csv").exists()
    assert (tmp_path / "per-category-metrics.csv").exists()

    assert '"overall"' in (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "per-category:" in (tmp_path / "metrics.yaml").read_text(encoding="utf-8")


def _table_rows(table):
    return list(zip(*[col._cells for col in table.columns], strict=False))


def _rows_by_header(table):
    """Map each rendered row to {column header: cell}.

    Keyed on headers rather than position so that adding a metric column does not
    invalidate assertions about unrelated columns.
    """
    headers = [col.header for col in table.columns]
    return [dict(zip(headers, cells, strict=True)) for cells in _table_rows(table)]


def test_render_console_tables_structure_and_metrics(monkeypatch):
    captured = []

    class DummyConsole:
        def __init__(self, record: bool = True):
            self.record = record

        def print(self, obj):
            captured.append(obj)

    monkeypatch.setattr(cli_module, "Console", DummyConsole)

    tree = {
        "overall": {
            "ds1": {
                "m1": {
                    "auroc": 0.91,
                    "auroc_at_1pct": 0.87,
                    "average_precision": 0.93,
                    "fpr_at_tau": 0.1,
                    "tpr_at_tau": 0.9,
                    "calibration_gap": 0.02,
                    "ood_percent": 0.03,
                    "tau": 0.5,
                    "normalized_scores": True,
                    "brier": 0.12,
                    "ece": 0.08,
                },
                "m2": {
                    "auroc": 0.75,
                    "auroc_at_1pct": None,
                    "average_precision": 0.81,
                    "fpr_at_tau": 0.2,
                    "tpr_at_tau": 0.7,
                    "calibration_gap": 0.05,
                    "ood_percent": 0.06,
                    "tau": 0.45,
                    "normalized_scores": False,
                },
            },
            "ds2": {
                "m1": {
                    "auroc": 0.88,
                    "auroc_at_1pct": 0.84,
                    "average_precision": 0.9,
                    "fpr_at_tau": 0.12,
                    "tpr_at_tau": 0.86,
                    "calibration_gap": 0.03,
                    "ood_percent": 0.04,
                    "tau": 0.52,
                    "normalized_scores": True,
                    "brier": 0.14,
                    "ece": 0.09,
                }
            },
        },
        "per-category": {
            "ds1": {
                "catA": {
                    "m1": {
                        "auroc": 0.92,
                        "auroc_at_1pct": 0.89,
                        "average_precision": 0.94,
                        "fpr_at_tau": 0.09,
                        "tpr_at_tau": 0.9,
                        "calibration_gap": 0.03,
                        "ood_percent": 0.02,
                        "n_samples": 10,
                        "normalized_scores": True,
                        "brier": 0.11,
                        "ece": 0.07,
                    },
                    "m2": {
                        "auroc": 0.74,
                        "auroc_at_1pct": 0.7,
                        "average_precision": 0.8,
                        "fpr_at_tau": 0.21,
                        "tpr_at_tau": 0.68,
                        "calibration_gap": 0.06,
                        "ood_percent": 0.07,
                        "n_samples": 10,
                        "normalized_scores": False,
                    },
                },
                "catB": {
                    "m1": {
                        "auroc": 0.9,
                        "auroc_at_1pct": 0.86,
                        "average_precision": 0.92,
                        "fpr_at_tau": 0.1,
                        "tpr_at_tau": 0.89,
                        "calibration_gap": 0.03,
                        "ood_percent": 0.03,
                        "n_samples": 8,
                        "normalized_scores": True,
                        "brier": 0.12,
                        "ece": 0.08,
                    }
                },
            }
        },
    }

    render_console_tables(tree)

    assert len(captured) == 3
    summary_rows = _rows_by_header(captured[0])
    calibration_rows = _rows_by_header(captured[1])
    per_category_rows = _rows_by_header(captured[2])

    # One row per (dataset, model) overall.
    assert len(summary_rows) == 3
    summary_by_key = {(row["dataset"], row["model"]): row for row in summary_rows}
    assert set(summary_by_key) == {("ds1", "m1"), ("ds1", "m2"), ("ds2", "m1")}

    assert summary_by_key[("ds1", "m1")] == {
        "dataset": "ds1",
        "model": "m1",
        "AUROC": "0.910",
        "AUROC@1%": "0.870",
        "AP": "0.930",
        "FPR@tau": "0.100",
        "TPR@tau": "0.900",
        "CalGap": "0.020",
        "OOD%": "0.030",
        "tau": "0.500",
    }
    # A model that reports no partial AUROC renders as a placeholder, not a crash.
    assert summary_by_key[("ds1", "m2")]["AUROC"] == "0.750"
    assert summary_by_key[("ds1", "m2")]["AUROC@1%"] == "-"
    assert summary_by_key[("ds1", "m2")]["AP"] == "0.810"
    assert summary_by_key[("ds2", "m1")]["AUROC"] == "0.880"
    assert summary_by_key[("ds2", "m1")]["tau"] == "0.520"

    # Calibration table has only normalized model rows.
    assert len(calibration_rows) == 2
    calibration_by_key = {(row["dataset"], row["model"]): row for row in calibration_rows}
    assert set(calibration_by_key) == {("ds1", "m1"), ("ds2", "m1")}
    assert calibration_by_key[("ds1", "m1")]["Brier"] == "0.120"
    assert calibration_by_key[("ds1", "m1")]["ECE"] == "0.080"
    assert calibration_by_key[("ds2", "m1")]["Brier"] == "0.140"
    assert calibration_by_key[("ds2", "m1")]["ECE"] == "0.090"

    # One row per (dataset, category, model).
    assert len(per_category_rows) == 3
    assert per_category_rows[0] == {
        "dataset": "ds1",
        "model": "m1",
        "category": "catA",
        "AUROC": "0.920",
        "AUROC@1%": "0.890",
        "AP": "0.940",
        "FPR@tau": "0.090",
        "TPR@tau": "0.900",
        "CalGap": "0.030",
        "OOD%": "0.020",
        "n": "10",
        "Brier": "0.110",
        "ECE": "0.070",
    }


def test_raise_for_unknown_names_accepts_known_names_case_insensitive():
    _raise_for_unknown_names(
        names=("GeDe",),
        valid_names=("gede", "other"),
        kind="dataset",
        param_hint="--dataset",
    )


def test_raise_for_unknown_names_raises_for_unknown_name():
    with pytest.raises(click.BadParameter) as exc_info:
        _raise_for_unknown_names(
            names=("missing",),
            valid_names=("gede", "other"),
            kind="dataset",
            param_hint="--dataset",
        )

    assert "Unknown dataset(s): missing. Valid options: gede, other" in str(exc_info.value)


def test_name_path_param_type_rejects_non_name_path():
    with pytest.raises(click.BadParameter) as exc_info:
        NAME_PATH.convert("invalid", param=None, ctx=None)

    assert "entries must look like NAME=PATH" in str(exc_info.value)


def test_name_path_param_type_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(click.BadParameter) as exc_info:
        NAME_PATH.convert(f"runtime={missing}", param=None, ctx=None)

    assert f"Dataset path does not exist: {missing}" in str(exc_info.value)


def test_name_path_param_type_parses_name_and_existing_path(tmp_path):
    data_path = tmp_path / "data.json"
    data_path.write_text("{}", encoding="utf-8")

    name, parsed_path = NAME_PATH.convert(f"runtime={data_path}", param=None, ctx=None)

    assert name == "runtime"
    assert parsed_path == Path(data_path)


def test_raise_for_unavailable_dataset_passes_for_a_present_file(tmp_path):
    path = tmp_path / "present.jsonl"
    path.write_text('{"answer":"a","label":"real"}\n', encoding="utf-8")
    spec = DatasetSpec(name="present", dataset_type="file", path=path)

    _raise_for_unavailable_dataset(spec)


def test_raise_for_unavailable_dataset_points_gede_at_the_prepare_command(tmp_path):
    spec = DatasetSpec(name="gede", dataset_type="file", path=tmp_path / "absent.jsonl")

    with pytest.raises(click.ClickException) as exc_info:
        _raise_for_unavailable_dataset(spec)

    message = str(exc_info.value)
    assert "prepare-gede" in message
    assert "datasets/README.md" in message
    assert str(tmp_path / "absent.jsonl") in message


def test_raise_for_unavailable_dataset_reports_other_datasets_plainly(tmp_path):
    spec = DatasetSpec(name="mine", dataset_type="file", path=tmp_path / "absent.jsonl")

    with pytest.raises(click.ClickException) as exc_info:
        _raise_for_unavailable_dataset(spec)

    assert "--register-file-dataset" in str(exc_info.value)


# ---------------------------------------------------------------------------
# The assembled command
#
# These invoke ``main`` in-process, which covers the wiring the helper tests above
# cannot: that each option is attached to the right validator and param type, and
# that failures surface as exit codes rather than tracebacks.
#
# Keep them cheap. Never pass ``-a``/``--all-models`` here: that selects ``smollm2``,
# which downloads model weights when built.
# ---------------------------------------------------------------------------

# Enough rows for both labels to appear in a single category, so ranking metrics are
# defined and the evaluation exercises a real path rather than a degenerate one.
_TINY_ROWS = [
    {"answer": "A short human answer.", "label": "real", "contribution_level": "Mixed"},
    {
        "answer": "Another human answer, a little longer than the first one.",
        "label": "real",
        "contribution_level": "Mixed",
    },
    {
        "answer": "In conclusion, a balanced approach is essential for all stakeholders.",
        "label": "fake",
        "contribution_level": "Mixed",
    },
    {
        "answer": "Furthermore it is important to consider the wider implications here.",
        "label": "fake",
        "contribution_level": "Mixed",
    },
]

# Narrows the run to a single model without relying on the default model list, which
# always contributes to the selection alongside any explicit --model.
_ONE_MODEL = ["--model", "dummy-norm", "--exclude-model", "dummy-raw", "--exclude-model", "length"]


@pytest.fixture
def tiny_dataset(tmp_path):
    """A four-row dataset file in the loader's default key layout."""
    path = tmp_path / "tiny.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _TINY_ROWS), encoding="utf-8")
    return path


def test_cli_help_lists_registered_datasets_and_models(runner):
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "demo" in result.output
    assert "length" in result.output


def test_cli_rejects_a_target_alpha_outside_the_unit_interval(runner):
    result = runner.invoke(main, ["--target-alpha", "1.5"])

    assert result.exit_code == 2
    assert "target-alpha" in result.stderr


@pytest.mark.parametrize(
    ("option", "kind"),
    [
        ("--dataset", "dataset"),
        ("--exclude-dataset", "dataset"),
        ("--model", "model"),
        ("--exclude-model", "model"),
    ],
)
def test_cli_rejects_unknown_names_on_every_selection_option(runner, option, kind):
    result = runner.invoke(main, [option, "nope"])

    assert result.exit_code == 2
    assert f"Unknown {kind}(s): nope" in result.stderr


def test_cli_rejects_excluding_every_dataset(runner):
    result = runner.invoke(main, ["--exclude-dataset", "demo"])

    assert result.exit_code == 2
    assert "No datasets selected" in result.stderr


def test_cli_rejects_excluding_every_model(runner):
    result = runner.invoke(
        main,
        ["--exclude-model", "dummy-norm", "--exclude-model", "dummy-raw", "--exclude-model", "length"],
    )

    assert result.exit_code == 2
    assert "No models selected" in result.stderr


def test_cli_evaluates_a_runtime_registered_dataset(runner, tmp_path, tiny_dataset, clean_registry):
    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"tiny={tiny_dataset}",
            "--exclude-dataset",
            "demo",
            *_ONE_MODEL,
            "--export",
            "json",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["overall"]) == {"tiny"}
    assert metrics["overall"]["tiny"]["dummy-norm"]["n_samples"] == 4


def test_cli_rejects_a_malformed_runtime_dataset_registration(runner):
    result = runner.invoke(main, ["--register-file-dataset", "no-equals-sign"])

    assert result.exit_code == 2
    assert "NAME=PATH" in result.stderr


def test_cli_rejects_a_runtime_dataset_whose_file_is_absent(runner, tmp_path):
    missing = tmp_path / "absent.jsonl"

    result = runner.invoke(main, ["--register-file-dataset", f"tiny={missing}"])

    assert result.exit_code == 2
    assert str(missing) in result.stderr


def test_cli_writes_every_requested_export_format(runner, tmp_path, tiny_dataset, clean_registry):
    output_dir = tmp_path / "out"

    result = runner.invoke(
        main,
        [
            "--register-file-dataset",
            f"tiny={tiny_dataset}",
            "--exclude-dataset",
            "demo",
            *_ONE_MODEL,
            "--export",
            "json",
            "--export",
            "yaml",
            "--export",
            "csv",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    written = {path.name for path in output_dir.iterdir()}
    assert written == {"metrics.json", "metrics.yaml", "overall-metrics.csv", "per-category-metrics.csv"}


def test_cli_reports_an_unavailable_dataset_without_a_traceback(runner, tmp_path, clean_registry):
    # Re-registering 'gede' at a path that does not exist is the in-process equivalent
    # of never having run prepare-gede; the registry is otherwise populated at import.
    register_file_dataset(name="gede", path=tmp_path / "never-prepared.jsonl")

    result = runner.invoke(main, ["--dataset", "gede", "--exclude-dataset", "demo", *_ONE_MODEL])

    # A ClickException, which click reports as exit 1, not an unhandled FileNotFoundError.
    assert result.exit_code == 1
    assert "prepare-gede" in result.stderr
    assert "Traceback" not in result.output


def test_cli_can_run_twice_in_one_process(runner, tmp_path, tiny_dataset, clean_registry):
    # Regression guard: main() used to call logging.basicConfig, which bound a handler
    # to the first invocation's stream and silently swallowed later runs' log output.
    for run in ("first", "second"):
        result = runner.invoke(
            main,
            [
                "--register-file-dataset",
                f"tiny={tiny_dataset}",
                "--exclude-dataset",
                "demo",
                *_ONE_MODEL,
                "--export",
                "json",
                "--output-dir",
                str(tmp_path / run),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / run / "metrics.json").is_file()
