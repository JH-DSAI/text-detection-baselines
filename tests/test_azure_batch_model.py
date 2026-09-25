"""Tests for the Azure ML batch endpoint detector.

Every test drives the detector through a fake :class:`BatchEndpointClient`: the
real one needs credentials and a multi-minute remote job, and what is worth
testing here is the mapping between the endpoint's verdict records and
``StubModelOutput``, not the Azure SDK.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from text_detection_baselines.models import build_model
from text_detection_baselines.models.azure_batch import (
    AzureBatchConfig,
    AzureBatchDetector,
    MissingConfigurationError,
    parse_analysis_reports,
)

_ENV = {
    "AZURE_STORAGE_ACCOUNT_URL": "https://example.blob.core.windows.net",
    "AZURE_DATASTORE_NAME": "workspaceblobstore",
    "AZURE_ML_SUBSCRIPTION_ID": "sub-1",
    "AZURE_ML_RESOURCE_GROUP": "rg-1",
    "AZURE_ML_WORKSPACE_NAME": "ws-1",
}

_QUESTION = "Should children be taught to compete or to co-operate?"


def _record(submission_id, decision, *, score=0.9, tau=0.7, is_flagged=None, inconclusive_reason=None):
    """One ``analysis_reports/<doc_id>.json`` payload."""
    return {
        "schema_version": "1.2",
        "submission_id": submission_id,
        "decision": decision,
        "is_flagged": (decision == "Flag for review") if is_flagged is None else is_flagged,
        "score": score,
        "tau": tau,
        "n_windows": 4,
        "inconclusive_reason": inconclusive_reason,
    }


class FakeClient:
    """Records what was submitted and replays canned verdicts."""

    def __init__(self, records, statuses=("Completed",)):
        self.records = records
        self.statuses = list(statuses)
        self.submissions = []
        self.status_calls = 0

    def submit(self, *, run_prefix, question, word_count, answers):
        self.submissions.append(
            {"run_prefix": run_prefix, "question": question, "word_count": word_count, "answers": answers},
        )
        return f"job-{len(self.submissions)}"

    def job_status(self, job_name):
        self.status_calls += 1
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def download_results(self, job_name):
        return self.records


def _detector(client, **kwargs):
    return AzureBatchDetector(
        model_name="azure-batch",
        normalized_scores=False,
        ood_margin=0.08,
        seed=7,
        config=AzureBatchConfig.from_env(_ENV),
        client=client,
        poll_interval_seconds=0.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_from_env_applies_application_defaults():
    config = AzureBatchConfig.from_env(_ENV)
    # Defaults are the application backend's, so a working deployment's
    # environment configures this detector unchanged.
    assert config.endpoint_name == "text-detection-batch-processing"
    assert config.storage_container == "text-detect-uploads-staging"
    assert config.default_word_count == 300


def test_config_from_env_reports_every_missing_variable_at_once():
    with pytest.raises(MissingConfigurationError) as excinfo:
        AzureBatchConfig.from_env({"AZURE_STORAGE_ACCOUNT_URL": "https://example.blob.core.windows.net"})
    message = str(excinfo.value)
    assert "AZURE_ML_SUBSCRIPTION_ID" in message
    assert "AZURE_DATASTORE_NAME" in message
    assert "AZURE_STORAGE_ACCOUNT_URL" not in message


def test_config_rejects_non_positive_word_count():
    with pytest.raises(MissingConfigurationError, match="must be positive"):
        AzureBatchConfig.from_env({**_ENV, "AZURE_ASSIGNMENT_DEFAULT_WORD_COUNT": "0"})


def test_detector_is_registered_but_not_a_default():
    from text_detection_baselines.models import get_default_model_names, list_registered_models

    assert "azure-batch" in list_registered_models()
    assert "azure-batch" not in get_default_model_names()


def test_build_model_does_not_touch_the_environment(monkeypatch):
    # Construction must stay cheap and credential-free: the CLI builds every
    # selected model before any dataset is scored.
    for name in AzureBatchConfig.REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    model = build_model("azure-batch", ood_margin=0.05, seed=1)
    assert isinstance(model, AzureBatchDetector)
    assert model.normalized_scores is False


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


def test_predict_maps_decisions_to_predictions_and_ood_flags():
    client = FakeClient(
        {
            "000000.txt": _record("000000.txt", "No action", score=0.40, tau=0.70),
            "000001.txt": _record("000001.txt", "Flag for review", score=0.95, tau=0.70),
            "000002.txt": _record(
                "000002.txt",
                "Inconclusive",
                score=0.0,
                tau=0.0,
                inconclusive_reason={"code": "too_short", "explanation": "…"},
            ),
        },
    )
    output = _detector(client).predict(_QUESTION, ["a", "b", "c"])

    np.testing.assert_array_equal(output.predictions, [0, 1, 0])
    np.testing.assert_array_equal(output.ood_flags, [False, False, True])
    # score - tau: the per-document threshold is what makes raw scores
    # incomparable across submissions.
    np.testing.assert_allclose(output.scores, [-0.30, 0.25, 0.0])


def test_predict_prefers_the_explicit_is_flagged_field():
    # ``is_flagged`` is the authoritative field; a decision string this package
    # does not recognize must not silently read as "not flagged".
    client = FakeClient({"000000.txt": _record("000000.txt", "Raise alert", is_flagged=True)})
    output = _detector(client).predict(_QUESTION, ["a"])
    np.testing.assert_array_equal(output.predictions, [1])


def test_predict_sends_the_question_as_the_assignment():
    client = FakeClient({"000000.txt": _record("000000.txt", "No action")})
    _detector(client).predict(_QUESTION, ["a"])

    submission = client.submissions[0]
    assert submission["question"] == _QUESTION
    assert submission["word_count"] == 300
    assert list(submission["answers"]) == ["000000.txt"]


def test_predict_requires_a_question():
    client = FakeClient({})
    with pytest.raises(ValueError, match="requires an assignment question"):
        _detector(client).predict("   ", ["a"])


def test_predict_with_no_answers_makes_no_remote_call():
    client = FakeClient({})
    output = _detector(client).predict(_QUESTION, [])
    assert output.scores.shape == (0,)
    assert client.submissions == []


def test_predict_raises_when_a_verdict_is_missing():
    # The pipeline emits one record per input file, including files it declined
    # to score, so a gap means results cannot be aligned with inputs.
    client = FakeClient({"000000.txt": _record("000000.txt", "No action")})
    with pytest.raises(RuntimeError, match="no verdict for 1 of 2"):
        _detector(client).predict(_QUESTION, ["a", "b"])


def test_predict_chunks_large_batches_and_keeps_row_order():
    records = {f"{i:06d}.txt": _record(f"{i:06d}.txt", "No action", score=i / 10, tau=0.0) for i in range(5)}
    client = FakeClient(records)
    output = _detector(client, max_batch_size=2).predict(_QUESTION, ["a", "b", "c", "d", "e"])

    assert len(client.submissions) == 3
    # Filenames stay globally indexed across chunks, so chunk two is 2 and 3.
    assert list(client.submissions[1]["answers"]) == ["000002.txt", "000003.txt"]
    np.testing.assert_allclose(output.scores, [0.0, 0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------


def test_wait_polls_until_the_job_completes():
    client = FakeClient(
        {"000000.txt": _record("000000.txt", "No action")}, statuses=["Running", "Running", "Completed"]
    )
    _detector(client).predict(_QUESTION, ["a"])
    assert client.status_calls == 3


def test_failed_job_raises():
    client = FakeClient({}, statuses=["Failed"])
    with pytest.raises(RuntimeError, match="ended in state 'Failed'"):
        _detector(client).predict(_QUESTION, ["a"])


def test_job_that_never_finishes_times_out():
    client = FakeClient({}, statuses=["Running"])
    with pytest.raises(RuntimeError, match="did not finish within"):
        _detector(client, timeout_seconds=0.0).predict(_QUESTION, ["a"])


# ---------------------------------------------------------------------------
# Downloaded output parsing
# ---------------------------------------------------------------------------


def test_parse_analysis_reports_reads_records_and_skips_sidecars(tmp_path):
    (tmp_path / "000000.txt.json").write_text(json.dumps(_record("000000.txt", "No action")), encoding="utf-8")
    (tmp_path / "000000.txt.md").write_text("# report", encoding="utf-8")
    # The batch sidecar has no ``decision`` and must not be read as a verdict.
    (tmp_path / "batch_verdict.json").write_text(json.dumps({"alpha": 0.05, "docs": []}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    records = parse_analysis_reports(tmp_path)

    assert list(records) == ["000000.txt"]
    assert records["000000.txt"]["decision"] == "No action"


def test_parse_analysis_reports_falls_back_to_the_filename(tmp_path):
    # ``doc_id`` carries the submission's own extension, so the fallback strips
    # only the ``.json`` the render step appended.
    payload = {"decision": "Flag for review", "is_flagged": True, "score": 0.9, "tau": 0.7}
    (tmp_path / "essay-01.docx.json").write_text(json.dumps(payload), encoding="utf-8")

    records = parse_analysis_reports(tmp_path)

    assert list(records) == ["essay-01.docx"]
