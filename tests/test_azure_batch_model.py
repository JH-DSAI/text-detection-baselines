"""Tests for the Azure ML batch endpoint detector.

The detector tests drive it through a fake :class:`BatchEndpointClient`: the
real one needs credentials and a multi-minute remote job, and what is worth
testing there is the mapping between the endpoint's verdict records and
``ModelOutput``, not the Azure SDK. The real client's own tests stub the SDK
handles and check only the request it builds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from text_detection_baselines.models import build_model
from text_detection_baselines.models.azure_batch import (
    AzureBatchConfig,
    AzureBatchDetector,
    AzureMLBatchClient,
    MissingConfigurationError,
    MissingDependencyError,
    TransientEndpointError,
    parse_analysis_reports,
)

_ENV = {
    "TDB_AZURE_BATCH_STORAGE_ACCOUNT_URL": "https://example.blob.core.windows.net",
    "TDB_AZURE_BATCH_DATASTORE_NAME": "workspaceblobstore",
    "TDB_AZURE_BATCH_ML_SUBSCRIPTION_ID": "sub-1",
    "TDB_AZURE_BATCH_ML_RESOURCE_GROUP": "rg-1",
    "TDB_AZURE_BATCH_ML_WORKSPACE_NAME": "ws-1",
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
    """Records what was submitted, cancelled, and deleted, and replays canned statuses and verdicts.

    A status that is an exception is raised rather than returned. The last status
    repeats for as long as the job is polled. A *submit_error* is raised after the
    submission is recorded, as if the upload had failed partway.
    """

    def __init__(self, records, statuses=("Completed",), cancel_error=None, submit_error=None, delete_error=None):
        self.records = records
        self.statuses = list(statuses)
        self.cancel_error = cancel_error
        self.submit_error = submit_error
        self.delete_error = delete_error
        self.submissions = []
        self.status_calls = 0
        self.cancelled = []
        self.deleted = []

    def submit(self, *, run_prefix, question, word_count, answers):
        self.submissions.append(
            {"run_prefix": run_prefix, "question": question, "word_count": word_count, "answers": answers},
        )
        if self.submit_error is not None:
            raise self.submit_error
        return f"job-{len(self.submissions)}"

    def job_status(self, job_name):
        self.status_calls += 1
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, BaseException):
            raise status
        return status

    def cancel(self, job_name):
        self.cancelled.append(job_name)
        if self.cancel_error is not None:
            raise self.cancel_error

    def download_results(self, job_name):
        return self.records

    def delete_inputs(self, run_prefix):
        self.deleted.append(run_prefix)
        if self.delete_error is not None:
            raise self.delete_error


def _config(**overrides):
    """A config built directly, so no environment is involved."""
    return AzureBatchConfig(
        storage_account_url="https://example.blob.core.windows.net",
        datastore_name="workspaceblobstore",
        ml_subscription_id="sub-1",
        ml_resource_group="rg-1",
        ml_workspace_name="ws-1",
        **overrides,
    )


def _detector(client, **config_overrides):
    return AzureBatchDetector(
        model_name="azure-batch",
        normalized_scores=False,
        ood_margin=0.08,
        seed=7,
        config=_config(**config_overrides),
        client=client,
    )


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    """Stand in for the module's clock, so job polling never really sleeps.

    ``sleep`` advances ``monotonic`` instead, which lets the polling tests use
    real intervals and timeouts and still finish at once.
    """
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        "text_detection_baselines.models.azure_batch.time",
        SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep),
    )
    return clock


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_from_env_reads_the_prefixed_variables(monkeypatch):
    for name, value in _ENV.items():
        monkeypatch.setenv(name, value)

    config = AzureBatchConfig.from_env()

    assert config.ml_subscription_id == "sub-1"
    assert config.endpoint_name == "text-detection-batch-processing"
    assert config.storage_container == "text-detect-uploads-staging"
    assert config.assignment_default_word_count == 300
    assert config.max_batch_size is None


def test_config_from_env_reads_the_job_settings(monkeypatch):
    for name, value in _ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TDB_AZURE_BATCH_POLL_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("TDB_AZURE_BATCH_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("TDB_AZURE_BATCH_MAX_BATCH_SIZE", "50")

    config = AzureBatchConfig.from_env()

    assert (config.poll_interval_seconds, config.timeout_seconds, config.max_batch_size) == (5.0, 600.0, 50)


def test_config_from_env_reports_every_missing_variable_at_once(monkeypatch):
    monkeypatch.setenv("TDB_AZURE_BATCH_STORAGE_ACCOUNT_URL", "https://example.blob.core.windows.net")

    with pytest.raises(MissingConfigurationError) as excinfo:
        AzureBatchConfig.from_env()

    message = str(excinfo.value)
    assert "TDB_AZURE_BATCH_ML_SUBSCRIPTION_ID" in message
    assert "TDB_AZURE_BATCH_DATASTORE_NAME" in message
    assert "TDB_AZURE_BATCH_STORAGE_ACCOUNT_URL" not in message


def test_config_from_env_treats_a_blank_variable_as_unset(monkeypatch):
    for name, value in _ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TDB_AZURE_BATCH_ML_WORKSPACE_NAME", "")

    with pytest.raises(MissingConfigurationError, match="TDB_AZURE_BATCH_ML_WORKSPACE_NAME"):
        AzureBatchConfig.from_env()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("TDB_AZURE_BATCH_ASSIGNMENT_DEFAULT_WORD_COUNT", "0"),
        ("TDB_AZURE_BATCH_POLL_INTERVAL_SECONDS", "0"),
        ("TDB_AZURE_BATCH_TIMEOUT_SECONDS", "-60"),
        # Zero would otherwise read as "no limit", and a negative size would
        # submit nothing at all.
        ("TDB_AZURE_BATCH_MAX_BATCH_SIZE", "0"),
        ("TDB_AZURE_BATCH_MAX_BATCH_SIZE", "-2"),
    ],
)
def test_config_rejects_non_positive_numbers(monkeypatch, variable, value):
    for name, env_value in _ENV.items():
        monkeypatch.setenv(name, env_value)
    monkeypatch.setenv(variable, value)

    with pytest.raises(MissingConfigurationError, match=rf"{variable} \(input should be greater than 0\)"):
        AzureBatchConfig.from_env()


def test_config_rejects_a_non_numeric_word_count(monkeypatch):
    for name, value in _ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TDB_AZURE_BATCH_ASSIGNMENT_DEFAULT_WORD_COUNT", "many")

    with pytest.raises(MissingConfigurationError, match="valid integer"):
        AzureBatchConfig.from_env()


def test_config_from_env_reads_a_dotenv_file_shared_with_other_settings(dotenv):
    # The package-wide settings and unrelated tooling share the file; neither
    # their keys nor a blank placeholder may be reported as a problem here.
    dotenv(
        *(f"{name}={value}" for name, value in _ENV.items()),
        "TDB_AZURE_BATCH_STORAGE_CONTAINER=",
        "TDB_GEDE_PATH=/data/gede_essays.jsonl",
        "UNRELATED_TOOL_TOKEN=abc",
    )

    config = AzureBatchConfig.from_env()

    assert config.ml_workspace_name == "ws-1"
    assert config.storage_container == "text-detect-uploads-staging"


def test_config_from_env_reports_an_unrecognised_dotenv_variable(dotenv):
    # A misspelled key would otherwise leave its field at the default without a word.
    dotenv(*(f"{name}={value}" for name, value in _ENV.items()), "TDB_AZURE_BATCH_STORAGE_CONTANER=mine")

    with pytest.raises(MissingConfigurationError, match="TDB_AZURE_BATCH_STORAGE_CONTANER"):
        AzureBatchConfig.from_env()


def test_detector_is_registered_but_not_a_default():
    from text_detection_baselines.models import get_default_model_names, list_registered_models

    assert "azure-batch" in list_registered_models()
    assert "azure-batch" not in get_default_model_names()


def _parent_package_missing(name):
    raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")


@pytest.mark.parametrize("find_spec", [lambda name: None, _parent_package_missing], ids=["module", "parent"])
def test_missing_sdks_are_reported_with_how_to_install_them(monkeypatch, find_spec):
    monkeypatch.setattr("text_detection_baselines.models.azure_batch.find_spec", find_spec)
    detector = AzureBatchDetector(
        model_name="azure-batch", normalized_scores=False, ood_margin=0.08, seed=7, config=_config()
    )

    with pytest.raises(MissingDependencyError, match=r"pixi run -e azure main"):
        detector.predict(_QUESTION, ["a"])


def test_build_model_does_not_touch_the_environment():
    # Construction must stay cheap and credential-free: the CLI builds every
    # selected model before any dataset is scored. The autouse fixture has
    # cleared the environment, so a config read here would raise.
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


@pytest.mark.parametrize("field", ["score", "tau"])
def test_predict_raises_when_a_scored_verdict_has_no_margin(field):
    # A stand-in margin would go into AUROC and AP as if it were real.
    records = {f"{i:06d}.txt": _record(f"{i:06d}.txt", "No action") for i in range(2)}
    records["000001.txt"][field] = None
    client = FakeClient(records)

    with pytest.raises(
        RuntimeError, match=r"no score or tau for 1 of 2 submission\(s\) not marked Inconclusive: 000001.txt"
    ):
        _detector(client).predict(_QUESTION, ["a", "b"])


def test_predict_accepts_an_inconclusive_verdict_without_a_margin():
    # The pipeline leaves the score out for a submission it declined to score.
    record = _record("000000.txt", "Inconclusive", inconclusive_reason={"code": "too_short", "explanation": "…"})
    record["score"] = record["tau"] = None
    client = FakeClient({"000000.txt": record})

    output = _detector(client).predict(_QUESTION, ["a"])

    np.testing.assert_array_equal(output.ood_flags, [True])
    np.testing.assert_array_equal(output.scores, [0.0])


def test_predict_chunks_large_batches_and_keeps_row_order():
    records = {f"{i:06d}.txt": _record(f"{i:06d}.txt", "No action", score=i / 10, tau=0.0) for i in range(5)}
    client = FakeClient(records)
    output = _detector(client, max_batch_size=2).predict(_QUESTION, ["a", "b", "c", "d", "e"])

    assert len(client.submissions) == 3
    # Filenames stay globally indexed across chunks, so chunk two is 2 and 3.
    assert list(client.submissions[1]["answers"]) == ["000002.txt", "000003.txt"]
    np.testing.assert_allclose(output.scores, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert client.deleted == [submission["run_prefix"] for submission in client.submissions]


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------


def test_wait_polls_until_the_job_completes():
    client = FakeClient(
        {"000000.txt": _record("000000.txt", "No action")}, statuses=["Running", "Running", "Completed"]
    )
    _detector(client).predict(_QUESTION, ["a"])
    assert client.status_calls == 3
    assert client.cancelled == []


def test_failed_job_raises_without_being_cancelled():
    # The job is already over, so there is nothing to cancel.
    client = FakeClient({}, statuses=["Failed"])
    with pytest.raises(RuntimeError, match="ended in state 'Failed'"):
        _detector(client).predict(_QUESTION, ["a"])
    assert client.cancelled == []


def test_job_that_never_finishes_times_out_and_is_cancelled(fake_clock):
    # Otherwise it would keep running, and billing, with nothing waiting for it.
    client = FakeClient({}, statuses=["Running"])
    with pytest.raises(RuntimeError, match="did not finish within 90s"):
        _detector(client, poll_interval_seconds=30, timeout_seconds=90).predict(_QUESTION, ["a"])
    # Checked at 0, 30, 60, and 90 seconds.
    assert client.status_calls == 4
    assert fake_clock.now == 90
    assert client.cancelled == ["job-1"]


def test_interrupted_wait_cancels_the_job():
    client = FakeClient({}, statuses=[KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        _detector(client).predict(_QUESTION, ["a"])
    assert client.cancelled == ["job-1"]


def test_transient_status_errors_are_polled_through():
    client = FakeClient(
        {"000000.txt": _record("000000.txt", "No action")},
        statuses=[TransientEndpointError("connection reset"), "Running", "Completed"],
    )
    _detector(client).predict(_QUESTION, ["a"])
    assert client.status_calls == 3
    assert client.cancelled == []


def test_transient_status_errors_still_time_out():
    client = FakeClient({}, statuses=[TransientEndpointError("host unreachable")])
    with pytest.raises(RuntimeError, match="did not finish within"):
        _detector(client).predict(_QUESTION, ["a"])
    assert client.cancelled == ["job-1"]


def test_other_status_errors_end_the_wait_and_cancel_the_job():
    # Not known to clear up on its own, but the job itself may still be running.
    client = FakeClient({}, statuses=[PermissionError("token expired")])
    with pytest.raises(PermissionError):
        _detector(client).predict(_QUESTION, ["a"])
    assert client.cancelled == ["job-1"]


def test_failed_cancel_is_logged_and_keeps_the_original_error(caplog):
    client = FakeClient({}, statuses=["Running"], cancel_error=RuntimeError("network down"))
    with pytest.raises(RuntimeError, match="did not finish within"):
        _detector(client).predict(_QUESTION, ["a"])
    # Named in the log so the job can be cancelled by hand.
    assert "Could not cancel batch job job-1" in caplog.text


# ---------------------------------------------------------------------------
# Uploaded input cleanup
# ---------------------------------------------------------------------------


def test_inputs_are_deleted_once_the_results_are_in():
    client = FakeClient({"000000.txt": _record("000000.txt", "No action")})
    _detector(client).predict(_QUESTION, ["a"])
    assert client.deleted == [client.submissions[0]["run_prefix"]]


@pytest.mark.parametrize(
    "client_kwargs",
    [
        {"statuses": ["Failed"]},
        {"statuses": [KeyboardInterrupt()]},
        {"submit_error": ConnectionError("upload failed partway")},
    ],
    ids=["failed-job", "interrupted", "failed-submit"],
)
def test_inputs_are_deleted_when_the_job_is_abandoned(client_kwargs):
    client = FakeClient({}, **client_kwargs)
    with pytest.raises((RuntimeError, KeyboardInterrupt, ConnectionError)):
        _detector(client).predict(_QUESTION, ["a"])
    assert client.deleted == [client.submissions[0]["run_prefix"]]


def test_failed_delete_is_logged_and_does_not_fail_the_run(caplog):
    client = FakeClient({"000000.txt": _record("000000.txt", "No action")}, delete_error=RuntimeError("forbidden"))

    output = _detector(client).predict(_QUESTION, ["a"])

    assert output.predictions.tolist() == [0]
    # Named in the log so the inputs can be deleted by hand.
    assert f"Could not delete uploaded inputs under {client.submissions[0]['run_prefix']}" in caplog.text


# ---------------------------------------------------------------------------
# Endpoint request
# ---------------------------------------------------------------------------


def _submit(monkeypatch, config):
    """Run :meth:`AzureMLBatchClient.submit` against stub SDK handles.

    Returns:
        The job name, the uploaded blobs as path to bytes, and the keyword
        arguments the endpoint was invoked with.
    """
    # The azure extra is not installed in the dev environment. ``Input`` is a
    # plain record here, so the test sees exactly what each input was given.
    azure_ml = ModuleType("azure.ai.ml")
    azure_ml.Input = SimpleNamespace
    monkeypatch.setitem(sys.modules, "azure.ai.ml", azure_ml)

    container = MagicMock()
    blob_service = MagicMock()
    blob_service.get_container_client.return_value = container
    ml_client = MagicMock()
    ml_client.batch_endpoints.invoke.return_value = SimpleNamespace(name="job-1")

    client = AzureMLBatchClient(config)
    monkeypatch.setattr(client, "_blob", lambda: blob_service)
    monkeypatch.setattr(client, "_ml", lambda: ml_client)

    job_name = client.submit(
        run_prefix="run-1",
        question=_QUESTION,
        word_count=300,
        answers={"000000.txt": "first answer", "000001.txt": "second answer"},
    )

    uploads = {call.kwargs["name"]: call.kwargs["data"] for call in container.upload_blob.call_args_list}
    return job_name, uploads, ml_client.batch_endpoints.invoke.call_args.kwargs


def test_submit_uploads_the_assignment_json_and_answers(monkeypatch):
    _, uploads, _ = _submit(monkeypatch, _config())

    assert json.loads(uploads.pop("run-1/assignment.json")) == {
        "essay_question": _QUESTION,
        "word_count": 300,
        "class_context_block": "",
    }
    assert uploads == {
        "run-1/submissions/000000.txt": b"first answer",
        "run-1/submissions/000001.txt": b"second answer",
    }


def test_submit_invokes_the_endpoint_with_every_required_input(monkeypatch):
    # Non-default endpoint and asset versions, so they are seen to come from
    # the config. The data assets are pinned on every invoke because AzureML
    # disallows defaults on pipeline data inputs.
    config = _config(
        endpoint_name="custom-endpoint",
        pipeline_config_asset="azureml:pipeline_config_yaml:9",
        detection_pool_asset="azureml:detection_pool:2",
    )

    job_name, _, invocation = _submit(monkeypatch, config)

    assert job_name == "job-1"
    # The two run inputs point at the paths the answers and assignment were
    # uploaded to, through the datastore the workspace can authenticate to.
    datastore = "azureml://datastores/workspaceblobstore/paths"
    assert invocation == {
        "endpoint_name": "custom-endpoint",
        "inputs": {
            "student_assignments_dir": SimpleNamespace(type="uri_folder", path=f"{datastore}/run-1/submissions"),
            "assignment_json_file": SimpleNamespace(type="uri_file", path=f"{datastore}/run-1/assignment.json"),
            "pipeline_config_file": SimpleNamespace(type="uri_file", path="azureml:pipeline_config_yaml:9"),
            "detection_pool": SimpleNamespace(type="uri_folder", path="azureml:detection_pool:2"),
        },
    }


def _stub_azure_core_exceptions(monkeypatch):
    """Install stand-ins for the azure-core exceptions, which the dev environment lacks."""
    module = ModuleType("azure.core.exceptions")

    class AzureError(Exception):
        pass

    class HttpResponseError(AzureError):
        def __init__(self, message="", status_code=None):
            super().__init__(message)
            self.status_code = status_code

    module.ServiceRequestError = type("ServiceRequestError", (AzureError,), {})
    module.ServiceResponseError = type("ServiceResponseError", (AzureError,), {})
    module.HttpResponseError = HttpResponseError
    monkeypatch.setitem(sys.modules, "azure.core.exceptions", module)
    return module


@pytest.mark.parametrize(
    ("error_name", "status_code", "transient"),
    [
        ("ServiceRequestError", None, True),
        ("ServiceResponseError", None, True),
        ("HttpResponseError", 503, True),
        ("HttpResponseError", 429, True),
        ("HttpResponseError", 404, False),
        ("HttpResponseError", None, False),
    ],
)
def test_job_status_marks_only_retryable_sdk_errors_as_transient(monkeypatch, error_name, status_code, transient):
    exceptions = _stub_azure_core_exceptions(monkeypatch)
    error_cls = getattr(exceptions, error_name)
    error = error_cls("boom", status_code=status_code) if status_code is not None else error_cls("boom")

    ml_client = MagicMock()
    ml_client.jobs.get.side_effect = error
    client = AzureMLBatchClient(_config())
    monkeypatch.setattr(client, "_ml", lambda: ml_client)

    with pytest.raises(TransientEndpointError if transient else error_cls):
        client.job_status("job-1")


def test_cancel_requests_cancellation_of_the_named_job(monkeypatch):
    ml_client = MagicMock()
    client = AzureMLBatchClient(_config())
    monkeypatch.setattr(client, "_ml", lambda: ml_client)

    client.cancel("job-1")

    ml_client.jobs.begin_cancel.assert_called_once_with("job-1")


def test_delete_inputs_deletes_every_blob_under_the_run_prefix(monkeypatch):
    container = MagicMock()
    container.list_blobs.return_value = [
        SimpleNamespace(name="run-1/assignment.json"),
        SimpleNamespace(name="run-1/submissions/000000.txt"),
    ]
    blob_service = MagicMock()
    blob_service.get_container_client.return_value = container
    client = AzureMLBatchClient(_config())
    monkeypatch.setattr(client, "_blob", lambda: blob_service)

    client.delete_inputs("run-1")

    # The trailing slash keeps the prefix from also matching a longer one.
    container.list_blobs.assert_called_once_with(name_starts_with="run-1/")
    deleted = [call.args[0] for call in container.delete_blob.call_args_list]
    assert deleted == ["run-1/assignment.json", "run-1/submissions/000000.txt"]


def test_sdk_clients_share_one_credential_with_the_full_default_chain(monkeypatch):
    sdk_classes = {
        ("azure.identity", "DefaultAzureCredential"): MagicMock(),
        ("azure.ai.ml", "MLClient"): MagicMock(),
        ("azure.storage.blob", "BlobServiceClient"): MagicMock(),
    }
    for (module_name, class_name), cls in sdk_classes.items():
        module = ModuleType(module_name)
        setattr(module, class_name, cls)
        monkeypatch.setitem(sys.modules, module_name, module)
    credential_cls, ml_client_cls, blob_service_cls = sdk_classes.values()

    client = AzureMLBatchClient(_config())
    client._ml()
    client._blob()

    # No exclusions: the environment credential is how a service principal
    # authenticates a headless run.
    credential_cls.assert_called_once_with()
    credential = credential_cls.return_value
    assert ml_client_cls.call_args.args[0] is credential
    assert blob_service_cls.call_args.kwargs["credential"] is credential


# ---------------------------------------------------------------------------
# Downloaded output parsing
# ---------------------------------------------------------------------------


def _write_report(folder, submission_id, payload=None):
    """Write one submission's report and verdict record, as the pipeline names them."""
    folder.mkdir(parents=True, exist_ok=True)
    record = json.dumps(_record(submission_id, "No action")) if payload is None else payload
    (folder / f"{submission_id}.json").write_text(record, encoding="utf-8")
    (folder / f"{submission_id}.md").write_text("# report", encoding="utf-8")


def test_parse_analysis_reports_reads_the_record_beside_each_report(tmp_path):
    _write_report(tmp_path, "000000.txt")
    # The batch sidecar has no report beside it, so it is not read as a verdict.
    (tmp_path / "batch_verdict.json").write_text(json.dumps({"alpha": 0.05, "docs": []}), encoding="utf-8")

    records = parse_analysis_reports(tmp_path)

    assert list(records) == ["000000.txt"]
    assert records["000000.txt"]["decision"] == "No action"


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        json.dumps(["not", "an", "object"]),
        json.dumps({"submission_id": "000000.txt", "score": 0.9, "tau": 0.7}),
        json.dumps({"decision": "Flag for review", "is_flagged": True, "score": 0.9, "tau": 0.7}),
    ],
    ids=["unreadable", "not-an-object", "no-decision", "no-submission-id"],
)
def test_parse_analysis_reports_skips_an_unusable_record_with_a_warning(tmp_path, caplog, payload):
    _write_report(tmp_path, "000000.txt", payload)

    records = parse_analysis_reports(tmp_path)

    assert records == {}
    assert "Skipping" in caplog.text


def _download(monkeypatch, write_outputs):
    """Run :meth:`AzureMLBatchClient.download_results` against a stub ``jobs.download``.

    *write_outputs* is called with the download folder, standing in for the SDK
    writing the job's output into it.

    Returns:
        The parsed records and the stub ML client.
    """
    ml_client = MagicMock()
    ml_client.jobs.download.side_effect = lambda **kwargs: write_outputs(Path(kwargs["download_path"]))
    client = AzureMLBatchClient(_config())
    monkeypatch.setattr(client, "_ml", lambda: ml_client)

    return client.download_results("job-1"), ml_client


def test_download_results_reads_the_analysis_reports_output(monkeypatch):
    def write_outputs(root):
        _write_report(root / "named-outputs" / "analysis_reports", "000000.txt")
        # Not the analysis_reports output, so not read.
        _write_report(root / "named-outputs" / "other_output", "000001.txt")

    records, ml_client = _download(monkeypatch, write_outputs)

    assert list(records) == ["000000.txt"]
    ml_client.jobs.download.assert_called_once()
    assert ml_client.jobs.download.call_args.kwargs["name"] == "job-1"
    assert ml_client.jobs.download.call_args.kwargs["output_name"] == "analysis_reports"


def test_download_results_falls_back_to_the_download_folder(monkeypatch):
    records, _ = _download(monkeypatch, lambda root: _write_report(root, "000000.txt"))

    assert list(records) == ["000000.txt"]
