"""Detector backed by the HopDetect Azure ML batch inference endpoint.

Unlike the other detectors in this package, this one runs remotely and
asynchronously: a single :meth:`AzureBatchDetector.predict` call uploads the
answers to blob storage, invokes the batch endpoint, polls until the job
finishes, and downloads the per-submission verdicts. A call therefore takes
minutes, not milliseconds, and needs Azure credentials.

The wire contract is the one the application backend consumes (dsai_detection
at ``1c87c29``, ``SubmissionResult`` schema 1.2):

* Submissions are uploaded as individual files; the pipeline's ``doc_id`` is the
  **full filename including its extension**, which is how results are matched
  back to input rows here.
* The job's ``analysis_reports`` output holds ``<doc_id>.json`` (the structured
  verdict -- the ``SubmissionResult`` record minus ``report_markdown``, plus an
  explicit ``is_flagged`` boolean).
* ``decision`` is one of ``Flag for review``, ``No action``, or ``Inconclusive``.

Mapping onto :class:`~.base.StubModelOutput`:

* ``predictions`` -- the record's ``is_flagged``.
* ``ood_flags`` -- an ``Inconclusive`` decision, i.e. a submission the detector
  declined to assess (unreadable, too short, extraction quality too low).
* ``scores`` -- ``score - tau``. The raw ``score`` is a window-max cosine judged
  against ``tau``, a *per-document* length-matched conformal threshold, so raw
  scores are not comparable across submissions and ranking metrics computed on
  them would be meaningless. The margin is the comparable quantity, and it is
  unbounded, hence ``normalized_scores=False``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .base import StubModelOutput, StubTextDetector

LOGGER = logging.getLogger(__name__)

#: Decision strings emitted by the scoring step (``deploy/score.py``
#: ``DECISION_MAPPING`` at ``1c87c29``).
FLAG_DECISION = "Flag for review"
NO_ACTION_DECISION = "No action"
INCONCLUSIVE_DECISION = "Inconclusive"

#: Azure ML job states mapped to the three outcomes this detector cares about.
#: Mirrors the application backend's ``_STATUS_MAP``; anything unrecognized is
#: treated as still pending rather than as a failure.
_TERMINAL_SUCCESS = frozenset({"Completed"})
_TERMINAL_FAILURE = frozenset({"Failed", "Canceled", "CancelRequested"})

#: Name of the batch job output holding the per-submission reports.
ANALYSIS_REPORTS_OUTPUT = "analysis_reports"


class MissingConfigurationError(RuntimeError):
    """Raised when required Azure environment variables are unset."""


@dataclass(frozen=True)
class AzureBatchConfig:
    """Connection settings for the Azure ML batch endpoint.

    Field names and defaults mirror the application backend's ``Settings``
    (JH-DSAI/text-detect-batch ``backend/app/config.py``), so a working
    deployment's environment configures this detector unchanged.
    """

    storage_account_url: str
    storage_container: str
    datastore_name: str
    subscription_id: str
    resource_group: str
    workspace_name: str
    endpoint_name: str
    pipeline_config_asset: str
    detection_pool_asset: str
    default_word_count: int

    #: Blob prefix this package writes under, kept separate from the
    #: application's ``class-<id>/assignment-<id>`` tree.
    blob_prefix: str = "text-detection-baselines"

    #: Environment variables without a usable default. An endpoint cannot be
    #: reached without them, so they are checked up front and reported together
    #: rather than surfacing one at a time as an Azure SDK error.
    REQUIRED_ENV_VARS = (
        "AZURE_STORAGE_ACCOUNT_URL",
        "AZURE_DATASTORE_NAME",
        "AZURE_ML_SUBSCRIPTION_ID",
        "AZURE_ML_RESOURCE_GROUP",
        "AZURE_ML_WORKSPACE_NAME",
    )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AzureBatchConfig:
        """Build a config from environment variables.

        Args:
            env: Mapping to read instead of :data:`os.environ`, for testing.

        Returns:
            A populated :class:`AzureBatchConfig`.

        Raises:
            MissingConfigurationError: If any of :data:`REQUIRED_ENV_VARS` is
                unset or empty.
        """
        source = os.environ if env is None else env

        missing = [name for name in cls.REQUIRED_ENV_VARS if not source.get(name)]
        if missing:
            raise MissingConfigurationError(
                "Azure batch endpoint is not configured. Set: " + ", ".join(missing),
            )

        raw_word_count = source.get("AZURE_ASSIGNMENT_DEFAULT_WORD_COUNT", "300")
        try:
            word_count = int(raw_word_count)
        except ValueError as exc:
            raise MissingConfigurationError(
                f"AZURE_ASSIGNMENT_DEFAULT_WORD_COUNT must be an integer, got {raw_word_count!r}",
            ) from exc
        if word_count <= 0:
            raise MissingConfigurationError(
                f"AZURE_ASSIGNMENT_DEFAULT_WORD_COUNT must be positive, got {word_count}",
            )

        return cls(
            storage_account_url=source["AZURE_STORAGE_ACCOUNT_URL"],
            storage_container=source.get("AZURE_STORAGE_CONTAINER", "text-detect-uploads-staging"),
            datastore_name=source["AZURE_DATASTORE_NAME"],
            subscription_id=source["AZURE_ML_SUBSCRIPTION_ID"],
            resource_group=source["AZURE_ML_RESOURCE_GROUP"],
            workspace_name=source["AZURE_ML_WORKSPACE_NAME"],
            endpoint_name=source.get("AZURE_BATCH_ENDPOINT_NAME", "text-detection-batch-processing"),
            pipeline_config_asset=source.get("AZURE_PIPELINE_CONFIG_ASSET", "azureml:pipeline_config_yaml:6"),
            detection_pool_asset=source.get("AZURE_DETECTION_POOL_ASSET", "azureml:detection_pool:1"),
            default_word_count=word_count,
        )


class BatchEndpointClient(Protocol):
    """The remote operations :class:`AzureBatchDetector` needs.

    Extracted as a protocol so the detector can be tested without Azure
    credentials or network access.
    """

    def submit(self, *, run_prefix: str, question: str, word_count: int, answers: dict[str, str]) -> str:
        """Upload one assignment's answers and invoke the endpoint.

        Args:
            run_prefix:  Blob path prefix unique to this invocation.
            question:    The assignment prompt.
            word_count:  Expected answer length, for the assignment JSON.
            answers:     Submission filename to answer text.

        Returns:
            The Azure ML job name, for polling.
        """
        ...

    def job_status(self, job_name: str) -> str:
        """Return the raw Azure ML job status string."""
        ...

    def download_results(self, job_name: str) -> dict[str, dict[str, Any]]:
        """Return the finished job's verdict records, keyed by submission id."""
        ...


class AzureMLBatchClient:
    """:class:`BatchEndpointClient` backed by the real Azure ML SDK."""

    def __init__(self, config: AzureBatchConfig) -> None:
        self.config = config
        self._ml_client: Any | None = None
        self._blob_service: Any | None = None

    # -- lazily constructed SDK handles ------------------------------------
    # The Azure SDKs are an optional dependency and cost seconds to import, so
    # they are imported on first use rather than at module scope: importing
    # ``models`` must not require them.

    def _ml(self) -> Any:
        if self._ml_client is None:
            from azure.ai.ml import MLClient

            self._ml_client = MLClient(
                self._credential(),
                subscription_id=self.config.subscription_id,
                resource_group_name=self.config.resource_group,
                workspace_name=self.config.workspace_name,
            )
        return self._ml_client

    def _blob(self) -> Any:
        if self._blob_service is None:
            from azure.storage.blob import BlobServiceClient

            self._blob_service = BlobServiceClient(
                account_url=self.config.storage_account_url,
                credential=self._credential(),
            )
        return self._blob_service

    @staticmethod
    def _credential() -> Any:
        from azure.identity import DefaultAzureCredential

        # Environment credential excluded to match the application backend: on
        # Azure the user-assigned identity is the one that works, and the
        # environment attempt fails first and pollutes the log stream.
        return DefaultAzureCredential(exclude_environment_credential=True)

    def _input_uri(self, blob_path: str) -> str:
        # A datastore URI rather than a raw blob URL: the pipeline runs inside
        # the Azure ML workspace, which cannot authenticate against the blob
        # API directly. Same reasoning as the application backend's _blob_uri.
        return f"azureml://datastores/{self.config.datastore_name}/paths/{blob_path}"

    # -- BatchEndpointClient ------------------------------------------------

    def submit(self, *, run_prefix: str, question: str, word_count: int, answers: dict[str, str]) -> str:
        """Upload the assignment and its answers, then invoke the endpoint."""
        from azure.ai.ml import Input

        container = self._blob().get_container_client(self.config.storage_container)

        submissions_prefix = f"{run_prefix}/submissions"
        for filename, text in answers.items():
            container.upload_blob(
                name=f"{submissions_prefix}/{filename}",
                data=text.encode("utf-8"),
                overwrite=True,
            )

        assignment_path = f"{run_prefix}/assignment.json"
        assignment = {
            "essay_question": question,
            "word_count": word_count,
            "class_context_block": "",
        }
        container.upload_blob(
            name=assignment_path,
            data=json.dumps(assignment).encode("utf-8"),
            overwrite=True,
        )

        job = self._ml().batch_endpoints.invoke(
            endpoint_name=self.config.endpoint_name,
            inputs={
                "student_assignments_dir": Input(type="uri_folder", path=self._input_uri(submissions_prefix)),
                "assignment_json_file": Input(type="uri_file", path=self._input_uri(assignment_path)),
                "pipeline_config_file": Input(type="uri_file", path=self.config.pipeline_config_asset),
                "detection_pool": Input(type="uri_folder", path=self.config.detection_pool_asset),
            },
        )
        return str(job.name)

    def job_status(self, job_name: str) -> str:
        """Return the raw Azure ML job status string."""
        return str(self._ml().jobs.get(job_name).status)

    def download_results(self, job_name: str) -> dict[str, dict[str, Any]]:
        """Download ``analysis_reports`` and parse the per-submission verdicts."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            self._ml().jobs.download(
                name=job_name,
                download_path=tmpdir,
                output_name=ANALYSIS_REPORTS_OUTPUT,
            )

            output_dir = Path(tmpdir) / "named-outputs" / ANALYSIS_REPORTS_OUTPUT
            if not output_dir.exists():
                output_dir = Path(tmpdir)

            return parse_analysis_reports(output_dir)


def parse_analysis_reports(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Read ``<doc_id>.json`` verdict records from a downloaded output folder.

    The record's own ``submission_id`` is authoritative; the filename is only a
    fallback, since ``doc_id`` includes the submission's file extension and so
    survives :meth:`~pathlib.Path.stem` badly (``0001.txt.json`` stems to
    ``0001.txt`` only by accident of there being exactly one extension left).

    Args:
        output_dir: Directory holding the job's ``analysis_reports`` output.

    Returns:
        Verdict records keyed by submission id.
    """
    records: dict[str, dict[str, Any]] = {}
    for json_file in sorted(output_dir.rglob("*.json")):
        try:
            record = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Skipping unreadable verdict file %s", json_file)
            continue

        if not isinstance(record, dict) or "decision" not in record:
            # batch_verdict.json and any other sidecar living in the same folder.
            continue

        submission_id = record.get("submission_id") or json_file.name[: -len(".json")]
        records[str(submission_id)] = record

    return records


class AzureBatchDetector(StubTextDetector):
    """Detector that scores submissions via the Azure ML batch endpoint.

    One :meth:`predict` call is one endpoint invocation (or one per chunk, when
    *max_batch_size* is set): the pipeline builds a support set from the
    assignment question and calibrates thresholds across the answers it is
    given, so the composition of the batch is part of the input, not an
    implementation detail.
    """

    def __init__(
        self,
        model_name: str,
        normalized_scores: bool,
        ood_margin: float,
        seed: int,
        config: AzureBatchConfig | None = None,
        client: BatchEndpointClient | None = None,
        poll_interval_seconds: float = 30.0,
        timeout_seconds: float = 3600.0,
        max_batch_size: int | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            model_name:       Registry name, reported in results.
            normalized_scores: Must be False; the score margin is unbounded.
            ood_margin:       Unused. OOD comes from the endpoint's own
                ``Inconclusive`` decision, not from a score-confidence band.
            seed:             Unused. Scoring happens remotely.
            config:           Connection settings. Read from the environment on
                first use when omitted.
            client:           Remote-operations implementation. A real
                :class:`AzureMLBatchClient` is built on first use when omitted.
            poll_interval_seconds: Delay between job status checks.
            timeout_seconds:  Give up on a job after this long.
            max_batch_size:   Split larger requests across several jobs. None
                sends every answer for a question in a single job, which is the
                regime the endpoint's batch-level calibration assumes.
        """
        super().__init__(model_name, normalized_scores, ood_margin, seed)
        self._config = config
        self._client = client
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_batch_size = max_batch_size

    @property
    def config(self) -> AzureBatchConfig:
        """The connection settings, read from the environment on first access."""
        if self._config is None:
            self._config = AzureBatchConfig.from_env()
        return self._config

    def _ensure_client(self) -> BatchEndpointClient:
        if self._client is None:
            self._client = AzureMLBatchClient(self.config)
        return self._client

    def predict(self, question: str, answers: list[str]) -> StubModelOutput:
        """Score one assignment's answers on the batch endpoint.

        Args:
            question: The assignment prompt. Required: the pipeline generates
                its per-assignment support set from it.
            answers:  The submissions to score.

        Returns:
            Output arrays parallel to *answers*.

        Raises:
            ValueError: If *question* is empty.
        """
        if not question.strip():
            raise ValueError(
                f"Model '{self.model_name}' requires an assignment question; "
                "the dataset supplied an empty one (check --question-key).",
            )

        if not answers:
            return StubModelOutput(
                scores=np.empty(0, dtype=float),
                predictions=np.empty(0, dtype=int),
                ood_flags=np.empty(0, dtype=bool),
            )

        chunk_size = self.max_batch_size or len(answers)
        records: dict[str, dict[str, Any]] = {}
        for start in range(0, len(answers), chunk_size):
            chunk = answers[start : start + chunk_size]
            records.update(self._score_chunk(question, chunk, first_index=start))

        return self._to_output(records, n_answers=len(answers))

    def _score_chunk(self, question: str, answers: list[str], first_index: int) -> dict[str, dict[str, Any]]:
        """Run one endpoint invocation and return its verdict records."""
        client = self._ensure_client()

        # Globally unique per invocation: concurrent evaluation runs share the
        # container, and a reused prefix would have one run's job read another
        # run's submissions folder.
        stamp = datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_prefix = f"{self.config.blob_prefix}/{stamp}-{uuid.uuid4().hex[:12]}"

        submissions = {self._submission_filename(first_index + offset): text for offset, text in enumerate(answers)}

        LOGGER.info("Submitting %d answer(s) to endpoint %s", len(submissions), self.config.endpoint_name)
        job_name = client.submit(
            run_prefix=run_prefix,
            question=question,
            word_count=self.config.default_word_count,
            answers=submissions,
        )
        LOGGER.info("Batch job %s submitted; polling every %.0fs", job_name, self.poll_interval_seconds)

        self._wait_for_job(client, job_name)
        return client.download_results(job_name)

    @staticmethod
    def _submission_filename(index: int) -> str:
        """Blob filename for one answer.

        Zero-padded so the endpoint's alphabetical file ordering matches dataset
        order, which makes the job's own logs readable next to these results.
        """
        return f"{index:06d}.txt"

    def _wait_for_job(self, client: BatchEndpointClient, job_name: str) -> None:
        """Block until the job succeeds, fails, or the timeout elapses.

        Raises:
            RuntimeError: If the job reaches a failure state or does not finish
                within :attr:`timeout_seconds`.
        """
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            status = client.job_status(job_name)
            if status in _TERMINAL_SUCCESS:
                LOGGER.info("Batch job %s completed", job_name)
                return
            if status in _TERMINAL_FAILURE:
                raise RuntimeError(f"Azure ML batch job {job_name} ended in state {status!r}")

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Azure ML batch job {job_name} did not finish within "
                    f"{self.timeout_seconds:.0f}s (last state {status!r})",
                )
            time.sleep(self.poll_interval_seconds)

    def _to_output(self, records: dict[str, dict[str, Any]], n_answers: int) -> StubModelOutput:
        """Map verdict records back onto dataset-order arrays.

        Raises:
            RuntimeError: If any submission has no verdict record. The pipeline
                emits one record per input file -- including files it declined
                to score -- so a gap means results and inputs cannot be aligned,
                and silently filling it would attribute one submission's verdict
                to another.
        """
        scores = np.zeros(n_answers, dtype=float)
        predictions = np.zeros(n_answers, dtype=int)
        ood_flags = np.zeros(n_answers, dtype=bool)

        missing: list[str] = []
        for index in range(n_answers):
            key = self._submission_filename(index)
            record = records.get(key)
            if record is None:
                missing.append(key)
                continue

            decision = record.get("decision")
            # ``is_flagged`` is written precisely so the flag need not be
            # inferred from the decision string; the comparison is the fallback
            # for a record that predates it.
            flagged = record.get("is_flagged")
            predictions[index] = int(bool(flagged) if flagged is not None else decision == FLAG_DECISION)
            ood_flags[index] = decision == INCONCLUSIVE_DECISION or record.get("inconclusive_reason") is not None

            score = record.get("score")
            tau = record.get("tau")
            # Zeroed by the pipeline for a submission it declined to score; the
            # resulting 0.0 margin is excluded from the ranking metrics anyway,
            # because the row is flagged OOD above.
            scores[index] = 0.0 if score is None or tau is None else float(score) - float(tau)

        if missing:
            shown = ", ".join(missing[:5])
            suffix = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            raise RuntimeError(
                f"Endpoint returned no verdict for {len(missing)} of {n_answers} submission(s): {shown}{suffix}",
            )

        return StubModelOutput(scores=scores, predictions=predictions, ood_flags=ood_flags)
