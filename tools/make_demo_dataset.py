"""Generate the fully synthetic ``demo`` dataset.

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
correctly.  Response lengths are bounded on both sides, for reasons documented at
:data:`MAX_TOKENS`.

Run from the repository root::

    python tools/make_demo_dataset.py

Outputs (overwritten in place), both in ``text_detection_baselines/datasets/data/``:

* ``demo.jsonl`` -- the full ``demo`` dataset.
* ``demo-one-question.jsonl`` -- the ``demo`` records answering its most common
  question, unchanged, for detectors that are invoked once per question at real cost.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "text_detection_baselines" / "datasets" / "data"
DEMO_PATH = DATA_DIR / "demo.jsonl"
ONE_QUESTION_PATH = DATA_DIR / "demo-one-question.jsonl"

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
    "Last winter the service ran late so often that people simply stopped planning around it.",
    "Is it fair to ask the people with the least slack to carry the most risk?",
    "I have changed my mind on this twice, which probably tells you something.",
    "My sister runs a small shop; she sees the effects long before any report does.",
    "The loudest voices at the meeting were, unsurprisingly, the ones least affected.",
    "None of this fits neatly into a spreadsheet, and that is part of the problem.",
    "If the scheme fails, it will fail quietly, in places nobody bothers to measure.",
    "Honestly, the strongest objection is one I cannot fully answer.",
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
    "Implementation timelines invariably accommodate administrative convenience rather than community requirements.",
    "Comprehensive evaluations remain conspicuously unavailable, notwithstanding repeated parliamentary commitments.",
    "Procurement arrangements frequently prioritise demonstrable compliance over genuine effectiveness.",
    "Longitudinal monitoring, regrettably, receives negligible appropriations relative to promotional communications.",
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
    "We know that this is not a new idea and it has been on the list for a long time.",
    "It is fair to say that most of us want to see this done the right way.",
    "If we get this right then the whole town will feel the good it can do.",
    "That is why it is so key that we all work as one to see it through.",
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


# Token budget. Both bounds are proxies for properties asserted in
# tests/test_demo_dataset.py.
#
# Lower bound: some detectors may automatically flag too-short answers, which
# would reduce the dataset's utility. tests/test_demo_dataset.py asserts at least
# 50 words; 75 leaves margin. (The stubs' own floor, 40 characters, is far lower
# and is reached only by SHORT_ANSWERS.)
#
# Upper bound: under its hard-coded weights ``dummy-norm``'s logit runs about
# 0.18-0.25 per token, so 120 tokens puts it at 20-30. The stub's sigmoid is
# computed in float64, which hits 1.0 only past about 37, so the cap
# leaves the scores separable. Without a cap the logit grows with length until
# every score ties, as it does on ``gede``'s longer essays.
#
# Note that these weights put the logit above zero for *any* text long enough to
# avoid the OOD floor, so ``dummy-norm`` scores are structurally confined to
# (0.5, 1). That is a property of the stub, not something the dataset can fix.
MAX_TOKENS = 120
MIN_TOKENS = 75


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
        # Make this text look machine-like: formal clauses drawn with replacement
        # repeat words and use long ones. A few ordinary clauses are mixed in
        # because an all-formal body overshoots, giving human text longer words
        # than machine text and leaving the length heuristic no signal.
        middles = [rng.choice(HUMAN_FORMAL_MIDDLES) for _ in range(rng.randint(4, 6))]
        middles += rng.sample(HUMAN_MIDDLES, rng.randint(2, 4))
        rng.shuffle(middles)
    else:
        middles = rng.sample(HUMAN_MIDDLES, rng.randint(6, 7))
    return _assemble(rng.choice(HUMAN_OPENERS).format(topic=topic), middles, rng.choice(HUMAN_CLOSERS))


def _machine_text(rng: random.Random, topic: str, *, borderline: bool) -> str:
    """Compose one machine-style response, within the token budget."""
    if borderline:
        # Distinct short-word clauses and a shorter body, so the text looks more varied.
        middles = rng.sample(MACHINE_PLAIN_MIDDLES, rng.randint(3, 5))
    else:
        middles = [rng.choice(MACHINE_MIDDLES) for _ in range(rng.randint(4, 6))]
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


def build_one_question(demo: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the ``demo`` records that answer its most common question.

    The pipeline invokes a detector once per question, and for ``azure-batch`` each
    invocation is a remote batch job taking minutes and costing money, so a
    single-question subset is the cheapest real end-to-end run.  Records are kept
    as they are, ``id`` included, so each one traces back to its ``demo`` row.
    ``Counter.most_common`` breaks ties by first appearance, so the choice is
    deterministic.
    """
    question, _ = collections.Counter(row["question"] for row in demo).most_common(1)[0]
    return [row for row in demo if row["question"] == question]


_FIELD_ORDER = [
    "id",
    "dataset",
    "contribution_level",
    "text_author",
    "question_id",
    "question",
    "answer",
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
    """Regenerate the ``demo`` and ``demo-one-question`` datasets."""
    demo = build_demo()
    check_token_budget(demo, "demo")
    one_question = build_one_question(demo)
    for path, rows in [(DEMO_PATH, demo), (ONE_QUESTION_PATH, one_question)]:
        write_jsonl(path, rows)
        print(f"wrote {len(rows)} records to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
