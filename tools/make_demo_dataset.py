"""Generate the fully synthetic ``demo`` dataset and the test fixtures.

Every word of text this script emits is composed from the hand-written template
pools below, so the output carries no third-party content and is redistributable
under this repository's licence.  See ``datasets/README.md`` for why that matters.

The two writing styles are partly separable by surface statistics -- machine texts
use longer words and less punctuation on average -- because the registered stub
detectors score exactly those features, and a dataset whose classes were
statistically identical would make every stub report AUROC 0.5 and so hide
pipeline regressions.

The separation is deliberately imperfect in both directions: no single feature
fully discriminates, and :data:`BORDERLINE_RATE` of records are written in the
other class's style.  Perfect separation would be just as useless as none, since
every metric would read 1.0 or 0.0 regardless of whether it was computed
correctly.  Response lengths are capped for a related reason, documented at
:data:`MAX_TOKENS`.

Run from the repository root::

    python tools/make_demo_dataset.py

Outputs (all overwritten in place):

* ``text_detection_baselines/datasets/data/demo.jsonl``
* ``tests/data/test_single_doc_per_author.json``
* ``tests/data/test_multi_docs_per_author.json``
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = REPO_ROOT / "text_detection_baselines" / "datasets" / "data" / "demo.jsonl"
SINGLE_DOC_FIXTURE = REPO_ROOT / "tests" / "data" / "test_single_doc_per_author.json"
MULTI_DOC_FIXTURE = REPO_ROOT / "tests" / "data" / "test_multi_docs_per_author.json"

SEED = 20260811
DATASET_NAME = "demo-synthetic"
GENERATOR_ID = "template-generator-v1"

TOPICS = [
    "public transport funding",
    "school uniform policies",
    "remote laboratory work",
    "open access publishing",
    "urban cycling networks",
    "seasonal water rationing",
    "municipal recycling targets",
    "night bus coverage",
]

PROMPTS = [
    "Explain whether {topic} deserves a larger share of the local budget, and justify your position.",
    "Some readers argue that {topic} has been oversold. Assess that claim using your own reasoning.",
    "Describe one benefit and one drawback of {topic}, then say which weighs more heavily.",
    "A committee must decide the future of {topic}. Write the argument you would put to them.",
]

# Human-style pools: many distinct clauses, comma- and semicolon-heavy, and
# willing to hedge or ask a question. Drawing from a wide pool keeps the
# type-token ratio high.
HUMAN_OPENERS = [
    "My honest view on {topic} shifted the week I actually had to rely on it.",
    "I used to shrug at {topic}, and then a very ordinary Tuesday changed my mind.",
    "Whenever {topic} comes up, someone reaches for a statistic before asking who is affected.",
    "There is a tidy version of the {topic} argument, and then there is the version I live with.",
    "I want to defend {topic}, but not with the slogans it usually attracts.",
    "Ask me about {topic} on a good day and I sound reasonable; ask me in February and I do not.",
]

HUMAN_MIDDLES = [
    "The cost sits with people who never appear in the committee minutes.",
    "What convinced me was small: a neighbour who stopped apologising for being late.",
    "Numbers help, though they flatten the part that stings.",
    "I keep returning to the question of who gets to be inconvenienced.",
    "Two of my colleagues disagree with me, and both have better data than I do.",
    "It is not obvious that the cheaper option is the kinder one.",
    "The pilot scheme worked, which is precisely why nobody trusts it yet.",
    "I would rather admit the trade-off than pretend it dissolves under scrutiny.",
    "Somebody has to absorb the delay; the argument is really about who.",
    "There is a version of this that is merely bookkeeping, and a version that is about dignity.",
    "My grandmother would call this common sense, and she would be half right.",
    "Nothing here is settled, so I hold the position loosely.",
]

# Long-word human clauses. Average word length is the dominant term in the
# ``length`` heuristic, so without these the two classes separate perfectly on
# that one feature and the heuristic reports a meaningless AUROC of 1.0.
HUMAN_FORMAL_MIDDLES = [
    "Institutional inertia, unfortunately, outlasts every enthusiastic consultation exercise.",
    "The distributional consequences disproportionately accumulate among households already experiencing precarity.",
    "Administrators habitually conflate measurable throughput with substantive improvement.",
    "Consultation documents systematically underrepresent inconvenient methodological uncertainties.",
    "Infrastructure commitments outlive the political arrangements that authorised them.",
    "Accountability mechanisms deteriorate approximately as fast as enthusiasm does.",
]

HUMAN_CLOSERS = [
    "So I land on yes, with reservations I would want written down.",
    "On balance I would fund it, and I would say plainly what it displaces.",
    "I am persuaded, though I would not call the case overwhelming.",
    "Given all that, my answer is a qualified no.",
    "I think the honest conclusion is that we do not yet know.",
]

# Machine-style pools: a deliberately narrow set of formulaic connectives and
# filler, reused across sentences, with very little punctuation. The repetition
# drives the type-token ratio down.
MACHINE_OPENERS = [
    "In today's rapidly evolving society the topic of {topic} has become increasingly important.",
    "It is widely acknowledged that {topic} plays a crucial role in modern communities.",
    "The question of {topic} is one of the most significant issues facing society today.",
]

# Some of these carry commas on purpose. Without them every machine text lands on
# an identical punctuation count, which makes that one feature a perfect
# discriminator and would hand an AUROC of 1.0 to any model that used it.
MACHINE_MIDDLES = [
    "Furthermore it is important to consider the various stakeholders who are involved in this important issue.",
    "Moreover, the available evidence clearly demonstrates that a balanced approach is essential for success.",
    "In addition it should be noted that careful planning plays a crucial role in achieving positive outcomes.",
    "Additionally, there are many important factors that must be considered, when evaluating this topic.",
    "It is also essential to recognise that sustainable, long term planning delivers measurable benefits.",
]

# Short-word machine clauses, the mirror image of HUMAN_FORMAL_MIDDLES.
MACHINE_PLAIN_MIDDLES = [
    "Also we can see that this is a very big topic for a lot of people today.",
    "Next it is clear that we all have to do our part to make this work well.",
    "So it is easy to see why so many of us care so much about this topic now.",
    "Then we must ask what is best for all of us in the years to come.",
]

MACHINE_CLOSERS = [
    "In conclusion, a balanced and carefully considered approach is essential for the best possible outcome.",
    "To summarise it is clear that this issue requires careful consideration from all relevant stakeholders.",
    "Overall, the evidence strongly suggests that a thoughtful, sustainable approach delivers the greatest benefit.",
]

# Deliberately under 40 characters, which is the threshold at which every stub
# flags a sample out-of-distribution. Without a few of these, OOD% reads 0.000 on
# every run and the restriction of metrics to non-OOD samples -- an invariant the
# whole results table depends on -- never actually runs outside its unit tests.
SHORT_ANSWERS = [
    "Not sure. Skipped.",
    "I ran out of time.",
    "See my other answer.",
    "No opinion either way.",
    "Yes, mostly agree.",
    "Fund it.",
    "Too complicated.",
    "It depends entirely.",
]


# Fraction of records written in the *other* class's surface style. Without this
# the two classes are perfectly separable and every stub reports AUROC 1.0, which
# tells you nothing about whether the metric code works. The overlap keeps the
# reported numbers in a range where FPR@tau, TPR@tau, and CalGap are meaningful.
BORDERLINE_RATE = 0.3


# Token budget. Both bounds exist to keep the registered stubs informative on
# this dataset, and both are proxies for properties asserted in
# tests/test_demo_dataset.py.
#
# Upper bound: ``dummy-norm`` runs its linear layer in float32 and keeps the
# sigmoid in float32, so scores lose all resolution as the logit grows -- past
# roughly 16 they tie at exactly 1.0, and well before that the spacing near 1.0
# (about 6e-8) already collapses distinct texts onto identical scores. Under the
# stub's hard-coded weights the logit runs about 0.18 per token, so a ~55-token
# cap keeps it near 10 and leaves the scores separable.
#
# Lower bound: both stubs flag any text under 40 characters as
# out-of-distribution, and OOD samples are excluded from every metric.
#
# Note that these weights put the logit above zero for *any* text long enough to
# avoid the OOD floor, so ``dummy-norm`` scores are structurally confined to
# (0.5, 1). That is a property of the stub, not something the dataset can fix.
MAX_TOKENS = 55
MIN_TOKENS = 30


def _assemble(opener: str, middles: list[str], closer: str) -> str:
    """Join clauses into one response, trimming to fit :data:`MAX_TOKENS`.

    Trimming from the middle rather than capping the clause counts keeps the
    sampled variety wide while making the budget a guarantee instead of
    something re-tuned by hand every time a clause pool changes.
    """
    kept = list(middles)
    while kept and len(" ".join([opener, *kept, closer]).split()) > MAX_TOKENS:
        kept.pop()
    return " ".join([opener, *kept, closer])


def _human_text(rng: random.Random, topic: str, *, borderline: bool) -> str:
    """Compose one human-style response, within the token budget."""
    if borderline:
        # Drawn with replacement and from the long-word pool, so both the
        # type-token ratio and the average word length land in the machine range.
        middles = [rng.choice(HUMAN_FORMAL_MIDDLES) for _ in range(rng.randint(4, 6))]
    else:
        middles = rng.sample(HUMAN_MIDDLES, rng.randint(3, 5))
    return _assemble(rng.choice(HUMAN_OPENERS).format(topic=topic), middles, rng.choice(HUMAN_CLOSERS))


def _machine_text(rng: random.Random, topic: str, *, borderline: bool) -> str:
    """Compose one machine-style response, within the token budget."""
    if borderline:
        # Distinct short-word clauses and a shorter body, so the text looks more varied.
        middles = rng.sample(MACHINE_PLAIN_MIDDLES, rng.randint(2, 3))
    else:
        middles = [rng.choice(MACHINE_MIDDLES) for _ in range(rng.randint(2, 4))]
    return _assemble(rng.choice(MACHINE_OPENERS).format(topic=topic), middles, rng.choice(MACHINE_CLOSERS))


def _record(rng: random.Random, *, is_human: bool, category: str, short: bool = False) -> dict[str, Any]:
    topic = rng.choice(TOPICS)
    question = rng.choice(PROMPTS).format(topic=topic)
    borderline = rng.random() < BORDERLINE_RATE
    if short:
        text = rng.choice(SHORT_ANSWERS)
    elif is_human:
        text = _human_text(rng, topic, borderline=borderline)
    else:
        text = _machine_text(rng, topic, borderline=borderline)
    return {
        "contribution_level": category,
        "text_author": "human" if is_human else GENERATOR_ID,
        "question": question,
        "answer": text,
        "label": "real" if is_human else "fake",
    }


# (category, n_human, n_machine). ``Human`` and the two machine-only categories
# are single-label on purpose: that is the slice shape the per-category ranking
# and rate metrics have to return ``None`` for. ``Mixed`` is the two-class slice
# where those metrics are defined.
DEMO_CATEGORIES = [
    ("Human", 60, 0),
    ("Task", 0, 45),
    ("Summary", 0, 45),
    ("Mixed", 21, 21),
]

# (category, n_human, n_machine) for the short, OOD-triggering records. Confined
# to ``Mixed`` so no category becomes entirely OOD, which would give it an empty
# non-OOD subset and make every one of its metrics undefined for an unrelated reason.
DEMO_SHORT_CATEGORIES = [("Mixed", 4, 4)]


def build_demo() -> list[dict[str, Any]]:
    """Build the ``demo`` dataset records."""
    rng = random.Random(SEED)
    rows: list[dict[str, Any]] = []
    for category, n_human, n_machine in DEMO_CATEGORIES:
        for is_human in [True] * n_human + [False] * n_machine:
            rows.append(_record(rng, is_human=is_human, category=category))
    for category, n_human, n_machine in DEMO_SHORT_CATEGORIES:
        for is_human in [True] * n_human + [False] * n_machine:
            rows.append(_record(rng, is_human=is_human, category=category, short=True))

    rng.shuffle(rows)
    for idx, row in enumerate(rows, start=1):
        row["id"] = idx
        row["question_id"] = 1 + (idx % len(PROMPTS))
        row["dataset"] = DATASET_NAME
    return rows


# The fixtures reproduce the record counts, label mixes, category sets, and
# per-author grouping of the corpus-derived files they replace, so the existing
# assertions in tests/test_datasets.py continue to describe the same shapes.
SINGLE_DOC_PLAN = [("Human", 9, 0), ("Task", 0, 7), ("Summary", 0, 10)]
MULTI_DOC_PLAN = [("Human", 6, 0), ("Task", 0, 10), ("Summary", 0, 10)]
MULTI_DOC_AUTHOR_SIZES = [3, 3, 10, 10]


def _fixture_rows(rng: random.Random, plan: list[tuple[str, int, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, n_human, n_machine in plan:
        for is_human in [True] * n_human + [False] * n_machine:
            rows.append(_record(rng, is_human=is_human, category=category))
    rng.shuffle(rows)
    for idx, row in enumerate(rows, start=1):
        row["question_id"] = idx
        row["dataset"] = DATASET_NAME
        row["temperature"] = "n/a" if row["label"] == "real" else 1
        row["max_tokens"] = "n/a" if row["label"] == "real" else 512
    return rows


def build_single_doc_fixture() -> list[dict[str, Any]]:
    """Build the 26-record fixture with one document per author."""
    rng = random.Random(SEED + 1)
    rows = _fixture_rows(rng, SINGLE_DOC_PLAN)
    for idx, row in enumerate(rows, start=1):
        row["author_id"] = 1000 + idx
    return rows


def build_multi_doc_fixture() -> list[dict[str, Any]]:
    """Build the 26-record fixture with several documents per author."""
    rng = random.Random(SEED + 2)
    rows = _fixture_rows(rng, MULTI_DOC_PLAN)
    author_ids = [author_id for author_id, size in enumerate(MULTI_DOC_AUTHOR_SIZES, start=1) for _ in range(size)]
    if len(author_ids) != len(rows):
        raise AssertionError(f"author plan covers {len(author_ids)} rows but built {len(rows)}")
    for row, author_id in zip(rows, author_ids):
        row["author_id"] = author_id
    return rows


_FIELD_ORDER = [
    "id",
    "author_id",
    "dataset",
    "contribution_level",
    "text_author",
    "question_id",
    "question",
    "answer",
    "temperature",
    "max_tokens",
    "label",
]


def _ordered(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _FIELD_ORDER if key in row}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_ordered(row), ensure_ascii=False) + "\n")


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records as an indented JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_ordered(row) for row in rows]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


#: Character length below which every stub flags a sample out-of-distribution.
OOD_CHAR_THRESHOLD = 40


def check_token_budget(rows: list[dict[str, Any]], label: str) -> None:
    """Fail loudly if any full-length text falls outside the token budget."""
    scored = [row for row in rows if len(row["answer"]) >= OOD_CHAR_THRESHOLD]
    n_short = len(rows) - len(scored)
    counts = [len(row["answer"].split()) for row in scored]
    low, high = min(counts), max(counts)
    if low < MIN_TOKENS or high > MAX_TOKENS:
        raise AssertionError(
            f"{label}: token counts {low}-{high} fall outside the "
            f"{MIN_TOKENS}-{MAX_TOKENS} budget; see the MAX_TOKENS comment.",
        )
    print(f"{label}: {len(rows)} records, {low}-{high} tokens, {n_short} short (OOD)")


def main() -> None:
    """Regenerate every synthetic data file."""
    demo = build_demo()
    check_token_budget(demo, "demo")
    single_doc = build_single_doc_fixture()
    multi_doc = build_multi_doc_fixture()
    check_token_budget(single_doc, "single-doc fixture")
    check_token_budget(multi_doc, "multi-doc fixture")

    write_jsonl(DEMO_PATH, demo)
    write_json_array(SINGLE_DOC_FIXTURE, single_doc)
    write_json_array(MULTI_DOC_FIXTURE, multi_doc)
    print(f"wrote {len(demo)} records to {DEMO_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {len(single_doc)} records to {SINGLE_DOC_FIXTURE.relative_to(REPO_ROOT)}")
    print(f"wrote {len(multi_doc)} records to {MULTI_DOC_FIXTURE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
