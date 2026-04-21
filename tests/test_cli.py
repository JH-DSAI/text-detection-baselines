"""Tests for CLI option selection helpers."""

from text_detection_baselines.cli import (
    _flatten_overall,
    _flatten_per_category,
    _resolve_selection,
    _unique_preserve_order,
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


def test_resolve_selection_defaults_still_selected_without_explicit_names():
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


def test_resolve_selection_explicit_keeps_defaults_when_not_all():
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
