"""Tests for the on-disk score cache that the expensive detectors keep."""

from __future__ import annotations

from text_detection_baselines.models.cache import (
    SCORE_CACHE_FILENAME,
    SCORE_CACHE_PATH_ENV_VAR,
    ScoreCache,
    resolve_score_cache_path,
)

_KEY = {"model": "m", "revision": "r1", "text": "An answer."}


def test_a_value_survives_reopening_the_cache(tmp_path):
    path = tmp_path / "scores.sqlite3"
    first = ScoreCache(path)
    first.put("ns", _KEY, {"score": 0.25, "flags": [True, False]})
    first.close()

    assert ScoreCache(path).get("ns", _KEY) == {"score": 0.25, "flags": [True, False]}


def test_a_float_reads_back_exactly(tmp_path):
    # A cached score must reproduce the metrics of the run that computed it.
    cache = ScoreCache(tmp_path / "scores.sqlite3")
    cache.put("ns", _KEY, 0.1 + 0.2)
    assert cache.get("ns", _KEY) == 0.1 + 0.2


def test_a_lookup_misses_on_any_difference_in_key_or_namespace(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite3")
    cache.put("ns", _KEY, 1.0)

    assert cache.get("ns", dict(_KEY, revision="r2")) is None
    assert cache.get("ns", {name: value for name, value in _KEY.items() if name != "text"}) is None
    assert cache.get("other", _KEY) is None


def test_the_order_of_a_key_does_not_matter(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite3")
    cache.put("ns", _KEY, 1.0)
    assert cache.get("ns", dict(reversed(_KEY.items()))) == 1.0


def test_refresh_ignores_what_is_stored_and_overwrites_it(tmp_path):
    path = tmp_path / "scores.sqlite3"
    ScoreCache(path).put("ns", _KEY, "old")

    refreshing = ScoreCache(path, refresh=True)
    assert refreshing.get("ns", _KEY) is None
    refreshing.put("ns", _KEY, "new")

    assert ScoreCache(path).get("ns", _KEY) == "new"


def test_nothing_is_created_until_the_cache_is_used(tmp_path):
    path = tmp_path / "nested" / "scores.sqlite3"
    cache = ScoreCache(path)
    assert not path.parent.exists()

    cache.put("ns", _KEY, 1.0)
    assert path.is_file()


def test_the_file_holds_no_text_from_a_key(tmp_path):
    path = tmp_path / "scores.sqlite3"
    cache = ScoreCache(path)
    cache.put("ns", {"text": "a distinctive dataset answer"}, 1.0)
    cache.close()

    assert b"distinctive" not in path.read_bytes()


def test_hits_and_writes_are_counted(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite3")
    cache.get("ns", _KEY)
    cache.put("ns", _KEY, 1.0)
    cache.get("ns", _KEY)
    cache.get("ns", _KEY)

    assert (cache.hits, cache.writes) == (2, 1)


def test_the_path_is_read_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(SCORE_CACHE_PATH_ENV_VAR, str(tmp_path / "mine.sqlite3"))
    assert resolve_score_cache_path() == tmp_path / "mine.sqlite3"


def test_the_path_is_read_from_dotenv(dotenv, monkeypatch, tmp_path):
    monkeypatch.delenv(SCORE_CACHE_PATH_ENV_VAR)
    dotenv(f"{SCORE_CACHE_PATH_ENV_VAR}={tmp_path / 'from-dotenv.sqlite3'}")
    assert resolve_score_cache_path() == tmp_path / "from-dotenv.sqlite3"


def test_the_path_defaults_to_the_user_cache_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(SCORE_CACHE_PATH_ENV_VAR)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert resolve_score_cache_path() == tmp_path / "text-detection-baselines" / SCORE_CACHE_FILENAME
