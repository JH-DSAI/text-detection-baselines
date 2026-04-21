"""Tests for CLI option selection helpers."""

from pathlib import Path

import click
import pytest

from text_detection_baselines import cli as cli_module
from text_detection_baselines.cli import (
    NAME_PATH,
    _flatten_overall,
    _flatten_per_category,
    _raise_for_unknown_names,
    _resolve_selection,
    _unique_preserve_order,
    _write_csv,
    export_results,
    render_console_tables,
)


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
    summary_rows = _table_rows(captured[0])
    calibration_rows = _table_rows(captured[1])
    per_category_rows = _table_rows(captured[2])

    # One row per (dataset, model) overall.
    assert len(summary_rows) == 3
    assert ("ds1", "m1", "0.9100", "0.1000", "0.9000", "0.0200", "0.0300", "0.5000") in summary_rows
    assert ("ds1", "m2", "0.7500", "0.2000", "0.7000", "0.0500", "0.0600", "0.4500") in summary_rows
    assert ("ds2", "m1", "0.8800", "0.1200", "0.8600", "0.0300", "0.0400", "0.5200") in summary_rows

    # Calibration table has only normalized model rows.
    assert len(calibration_rows) == 2
    assert ("ds1", "m1", "0.1200", "0.0800") in calibration_rows
    assert ("ds2", "m1", "0.1400", "0.0900") in calibration_rows

    # One row per (dataset, category, model).
    assert len(per_category_rows) == 3
    assert (
        "ds1",
        "m1",
        "catA",
        "0.9200",
        "0.0900",
        "0.9000",
        "0.0300",
        "0.0200",
        "10",
        "0.1100",
        "0.0700",
    ) in per_category_rows


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
