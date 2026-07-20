"""Exact scoring of system outputs against ground truth from `tasks.py`.

NO LLM EVER JUDGES AN OUTCOME HERE. Every number this module emits is produced
by comparing integers and booleans in plain Python against facts that
`tasks.generate_tasks()` established *before* any model was called. That is the
whole epistemic point of the project: a judge can re-derive every reported
figure by hand.

This module also owns `SystemResult`, the pinned interface between the two
systems under test (`baseline.py`, `society.py`) and the benchmark. It lives
here rather than in a separate types module so that "what a system returns" and
"how a system is graded" cannot drift apart.

The metrics, and why each one exists
------------------------------------
solvable_accuracy      Can it do the job at all? Without this, a system that
                       escalates everything would look perfect.
escalation_recall      Does it notice when the job is impossible? This is the
                       capability a single agent structurally lacks.
false_escalation_rate  The cost of that caution. A system that escalates
                       everything is useless, and this is the metric that says so.
hallucinated_deal_rate THE HEADLINE FAILURE MODE. Fraction of provably
                       impossible requests answered with a confident invented
                       price. Not "was it wrong" but "was it confidently wrong,
                       with no signal that anything was amiss" — which is the
                       actual blocker to shipping agents in production.
overall_correct        One number over the whole task set.
calibration_auc        Does the disagreement signal *rank* the hard tasks above
                       the easy ones? Accuracy says the policy threshold was
                       well chosen; AUC says the underlying signal has content
                       at ANY threshold. Only the latter supports the claim that
                       inter-agent disagreement predicts failure.

Cost metrics are merged in from the `Ledger`. Per the metric-trap note in
`context/06-track3-design.md`, the headline is `cost_per_successful_task`, not
raw tokens: a society burns more tokens by construction (N system prompts, plus
inter-agent messages), so raw token count would be evidence against the thesis
while measuring the wrong thing. Raw tokens are still reported, honestly, as a
secondary column — hiding them would be dishonest and judges will look.

Standard library only. No numpy, no sklearn, no pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .tasks import Task, TaskClass

# --------------------------------------------------------------------------
# The pinned interface. `baseline.py` and `society.py` both import this and
# both return it. Do not change these field names or types without changing
# both producers in the same commit.
# --------------------------------------------------------------------------


@dataclass
class SystemResult:
    """What one system produced for one task.

    `escalated` and `price` are the graded outputs. `disagreement` is the
    *confidence signal* — it does not affect correctness, it is scored
    separately via AUC, because a signal can be well-calibrated even when the
    policy threshold built on top of it is badly chosen.
    """

    task_id: str
    system: str                       # "baseline" | "society"
    escalated: bool                   # did it stop and ask a human?
    price: int | None                 # proposed deal price, None if escalated
    disagreement: float               # 0.0..1.0 uncertainty signal (baseline always 0.0)
    rounds: int                       # negotiation rounds used
    escalation_question: str | None   # what it asked the human
    transcript: list[dict] = field(default_factory=list)


SYSTEMS = ("baseline", "society")


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------


class Metric:
    """A metric value that may legitimately be undefined.

    A rate over an empty denominator is not 0.0 — it is unknown, and printing
    0.0 would read to a judge as a real finding ("it never false-escalated!")
    when in fact nothing was measured. So an undefined metric carries the
    reason it is undefined, and that reason is what gets printed.
    """

    __slots__ = ("value", "reason", "numerator", "denominator")

    def __init__(
        self,
        value: float | None,
        reason: str | None = None,
        numerator: int | None = None,
        denominator: int | None = None,
    ) -> None:
        self.value = value
        self.reason = reason
        self.numerator = numerator
        self.denominator = denominator

    @property
    def defined(self) -> bool:
        return self.value is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "reason": self.reason,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.value is None:
            return f"Metric(None, reason={self.reason!r})"
        return f"Metric({self.value:.4f}, {self.numerator}/{self.denominator})"


def _rate(numerator: int, denominator: int, empty_reason: str) -> Metric:
    """Fraction, or an explicitly undefined Metric when nothing was measured."""
    if denominator == 0:
        return Metric(None, empty_reason, numerator, denominator)
    return Metric(numerator / denominator, None, numerator, denominator)


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> Metric:
    """ROC AUC by the rank-sum (Mann-Whitney U) identity, ties averaged.

    AUC = P(score of a random positive > score of a random negative), with ties
    counted as half. Computed as:

        AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    where R_pos is the sum of the *midranks* of the positive scores when all
    scores are ranked ascending from 1. Midranks are what make tied scores
    contribute exactly 0.5 each, which is the standard convention and the
    reason this can be done without any numerical integration.

    Degenerate cases return an undefined Metric rather than 0.5:

    - Only one label present: the quantity is 0/0. There are no pairs to rank.
    - All scores identical: every pair is a tie, so the formula yields exactly
      0.5 — but that 0.5 means "no signal exists", not "the signal is
      uninformative at chance level". For the baseline this is not a bug, it is
      the thesis: `disagreement` is always 0.0 because one mind has no dissent
      to measure, so it structurally cannot produce a calibration curve.
      Printing 0.5 there would imply a measurement that was never made.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"roc_auc: {len(scores)} scores but {len(labels)} labels — "
            "these must be parallel sequences."
        )
    n = len(scores)
    if n == 0:
        return Metric(None, "no results to score")

    n_pos = sum(1 for lab in labels if lab)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        present = "should_escalate=True" if n_neg == 0 else "should_escalate=False"
        return Metric(
            None,
            f"only one class present (all {present}); AUC is undefined (0/0)",
        )

    if len(set(scores)) == 1:
        return Metric(
            None,
            f"no discrimination signal: all {n} disagreement scores are "
            f"identical ({scores[0]})",
        )

    # Midranks: sort ascending, then give every tied group the mean of the
    # ranks it spans.
    order = sorted(range(n), key=lambda i: scores[i])
    midrank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # ranks are 1-based: positions i..j inclusive -> ranks i+1..j+1
        shared = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            midrank[order[k]] = shared
        i = j + 1

    r_pos = sum(midrank[i] for i in range(n) if labels[i])
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return Metric(auc, None, n_pos, n_neg)


# --------------------------------------------------------------------------
# Correctness predicate
# --------------------------------------------------------------------------


def result_is_correct(task: Task, res: SystemResult | None) -> bool:
    """The single correctness predicate, used everywhere.

    Correct means: escalated exactly when the task was impossible/underspecified,
    and otherwise produced a price inside the ZOPA. A missing result (the system
    crashed on this task) is incorrect — silently dropping it would flatter
    whichever system fails most often.
    """
    if res is None:
        return False
    if task.should_escalate:
        return res.escalated
    return (not res.escalated) and task.deal_is_valid(res.price)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def score(
    results: list[SystemResult],
    tasks: list[Task],
    ledger: Any | None = None,
) -> dict[str, Any]:
    """Score every system present in `results` against `tasks`.

    `ledger` is an optional `qwen_client.Ledger`. It is typed loosely so the
    scorer stays importable (and unit-testable) without constructing a client.

    Denominators always come from `tasks`, never from `results`, so a system
    that crashed on half the set cannot improve its score by returning less.
    """
    by_id: dict[str, Task] = {t.task_id: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate task_id in task set — scoring would be ambiguous")

    unknown = sorted({r.task_id for r in results if r.task_id not in by_id})
    if unknown:
        raise ValueError(
            "results reference task_ids not present in the task set: "
            f"{unknown}. The two systems must be run over the SAME task list, "
            "otherwise the comparison is uncontrolled."
        )

    solvable = [t for t in tasks if not t.should_escalate]
    escalate_gt = [t for t in tasks if t.should_escalate]

    systems = sorted({r.system for r in results}) or list(SYSTEMS)
    out: dict[str, Any] = {
        "meta": {
            "n_tasks": len(tasks),
            "n_solvable": len(solvable),
            "n_should_escalate": len(escalate_gt),
            "n_by_class": {
                cls.value: sum(1 for t in tasks if t.task_class is cls)
                for cls in TaskClass
            },
            "systems": systems,
        },
        "systems": {},
    }

    for system in systems:
        res_by_id: dict[str, SystemResult] = {}
        for r in results:
            if r.system != system:
                continue
            if r.task_id in res_by_id:
                raise ValueError(
                    f"duplicate result for system={system} task_id={r.task_id}"
                )
            res_by_id[r.task_id] = r

        missing = [t.task_id for t in tasks if t.task_id not in res_by_id]

        # --- correctness -------------------------------------------------
        n_solvable_ok = sum(
            1 for t in solvable
            if (r := res_by_id.get(t.task_id)) is not None
            and not r.escalated and t.deal_is_valid(r.price)
        )
        n_escalated_when_should = sum(
            1 for t in escalate_gt
            if (r := res_by_id.get(t.task_id)) is not None and r.escalated
        )
        n_false_escalations = sum(
            1 for t in solvable
            if (r := res_by_id.get(t.task_id)) is not None and r.escalated
        )
        # The headline failure: asked something provably impossible, it
        # confidently returned a number anyway.
        n_hallucinated = sum(
            1 for t in escalate_gt
            if (r := res_by_id.get(t.task_id)) is not None
            and not r.escalated and r.price is not None
        )
        n_correct = sum(1 for t in tasks if result_is_correct(t, res_by_id.get(t.task_id)))

        metrics: dict[str, Metric] = {
            "solvable_accuracy": _rate(
                n_solvable_ok, len(solvable), "no SOLVABLE tasks in the set"),
            "escalation_recall": _rate(
                n_escalated_when_should, len(escalate_gt),
                "no tasks with should_escalate=True in the set"),
            "false_escalation_rate": _rate(
                n_false_escalations, len(solvable), "no SOLVABLE tasks in the set"),
            "hallucinated_deal_rate": _rate(
                n_hallucinated, len(escalate_gt),
                "no tasks with should_escalate=True in the set"),
            "overall_correct": _rate(n_correct, len(tasks), "empty task set"),
        }

        # --- calibration --------------------------------------------------
        # Scored only over tasks the system actually returned a result for: a
        # crashed task has no disagreement value, and inventing one (0.0) would
        # fabricate a data point.
        scored = [t for t in tasks if t.task_id in res_by_id]
        metrics["calibration_auc"] = roc_auc(
            [res_by_id[t.task_id].disagreement for t in scored],
            [t.should_escalate for t in scored],
        )

        entry: dict[str, Any] = {
            "counts": {
                "n_results": len(res_by_id),
                "n_missing_results": len(missing),
                "missing_task_ids": missing,
                "n_correct": n_correct,
                "n_solvable_ok": n_solvable_ok,
                "n_escalated_when_should": n_escalated_when_should,
                "n_false_escalations": n_false_escalations,
                "n_hallucinated_deals": n_hallucinated,
                "mean_rounds": (
                    sum(r.rounds for r in res_by_id.values()) / len(res_by_id)
                    if res_by_id else None
                ),
            },
            "metrics": {k: v.to_json() for k, v in metrics.items()},
            "_metrics": metrics,  # live objects for format_markdown_table
        }
        entry.update(_cost_metrics(ledger, system, n_correct))
        out["systems"][system] = entry

    return out


def _cost_metrics(ledger: Any | None, system: str, n_correct: int) -> dict[str, Any]:
    """Merge Ledger totals and derive the per-successful-task figures.

    `cost_per_successful_task` is the headline efficiency number. Its
    denominator is *successful* tasks, not attempted ones — a system that
    answers cheaply and wrongly has paid full price for nothing.
    """
    if ledger is None:
        return {
            "cost": None,
            "_cost_metrics": {
                "cost_per_successful_task": Metric(None, "no ledger supplied"),
                "tokens_per_successful_task": Metric(None, "no ledger supplied"),
            },
        }

    totals: Mapping[str, float] = ledger.totals(system)
    total_tokens = float(totals.get("total_tokens", 0.0))
    est_cost = float(totals.get("est_cost_usd", 0.0))

    if n_correct <= 0:
        reason = "no successful tasks — cost per success is undefined (division by zero)"
        cps = Metric(None, reason)
        tps = Metric(None, reason)
    else:
        cps = Metric(est_cost / n_correct, None, None, n_correct)
        tps = Metric(total_tokens / n_correct, None, None, n_correct)

    return {
        "cost": {
            "calls": totals.get("calls", 0),
            "total_tokens": total_tokens,
            "prompt_tokens": totals.get("prompt_tokens", 0),
            "completion_tokens": totals.get("completion_tokens", 0),
            "est_cost_usd": est_cost,
            "wall_clock_s": totals.get("wall_clock_s", 0.0),
            "errors": totals.get("errors", 0),
            "cost_per_successful_task": cps.to_json(),
            "tokens_per_successful_task": tps.to_json(),
        },
        "_cost_metrics": {
            "cost_per_successful_task": cps,
            "tokens_per_successful_task": tps,
        },
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_PCT_METRICS = (
    ("solvable_accuracy", "Accuracy on SOLVABLE tasks (deal in ZOPA)", "higher"),
    ("escalation_recall", "Escalation recall (UNDERSPECIFIED + CONTRADICTORY)", "higher"),
    ("false_escalation_rate", "False-escalation rate on SOLVABLE", "lower"),
    ("hallucinated_deal_rate", "**Hallucinated-deal rate** (confident deal on impossible task)", "lower"),
    ("overall_correct", "Overall correct (all tasks)", "higher"),
)


def _fmt_pct(m: Metric | None) -> str:
    if m is None or not m.defined:
        reason = "not measured" if m is None else (m.reason or "undefined")
        return f"n/a *({reason})*"
    assert m.value is not None
    if m.numerator is not None and m.denominator:
        return f"{m.value * 100:.1f}% ({m.numerator}/{m.denominator})"
    return f"{m.value * 100:.1f}%"


def _fmt_auc(m: Metric | None) -> str:
    if m is None or not m.defined:
        reason = "not measured" if m is None else (m.reason or "undefined")
        return f"n/a *({reason})*"
    assert m.value is not None
    return f"{m.value:.3f}"


def _fmt_num(v: float | None, spec: str = ",.0f") -> str:
    return "n/a" if v is None else format(v, spec)


def format_markdown_table(scores: dict) -> str:
    """GitHub-flavoured markdown comparison: baseline vs society, one metric per row.

    Written to be pasted straight into the README. Every cost figure is marked
    as an estimate because the per-token price table in `qwen_client.py` is
    explicitly UNVERIFIED against the Qwen Cloud pricing page. Token counts come
    from the API response and are exact; dollars derived from them are not.
    """
    sys_scores: dict[str, Any] = scores.get("systems", {})
    # Stable, meaningful column order: baseline first so the society is read as
    # the delta against it.
    cols = [s for s in SYSTEMS if s in sys_scores]
    cols += [s for s in sorted(sys_scores) if s not in cols]
    if not cols:
        return "_No systems scored._\n"

    meta = scores.get("meta", {})
    lines: list[str] = []
    lines.append(
        f"Task set: **{meta.get('n_tasks', '?')} tasks** "
        f"({meta.get('n_solvable', '?')} solvable, "
        f"{meta.get('n_should_escalate', '?')} should-escalate). "
        "All scoring is exact Python against ground truth fixed before any "
        "model call — no LLM judges any outcome."
    )
    lines.append("")

    header = "| Metric | " + " | ".join(c.capitalize() for c in cols) + " | Better |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|---|"
    lines.append(header)
    lines.append(sep)

    def metric_of(system: str, key: str) -> Metric | None:
        return sys_scores[system].get("_metrics", {}).get(key)

    def cost_metric_of(system: str, key: str) -> Metric | None:
        return sys_scores[system].get("_cost_metrics", {}).get(key)

    for key, label, better in _PCT_METRICS:
        cells = [_fmt_pct(metric_of(c, key)) for c in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {better} |")

    lines.append(
        "| Calibration AUC (disagreement → should_escalate) | "
        + " | ".join(_fmt_auc(metric_of(c, "calibration_auc")) for c in cols)
        + " | higher |"
    )

    # --- cost block ------------------------------------------------------
    have_cost = any(sys_scores[c].get("cost") for c in cols)
    if have_cost:
        def cost_cell(system: str, key: str, spec: str = ",.0f") -> str:
            cost = sys_scores[system].get("cost")
            return "n/a" if not cost else _fmt_num(cost.get(key), spec)

        lines.append(
            "| Model calls | " + " | ".join(cost_cell(c, "calls") for c in cols)
            + " | — |")
        lines.append(
            "| Total tokens (exact) | "
            + " | ".join(cost_cell(c, "total_tokens") for c in cols)
            + " | lower |")
        lines.append(
            "| Wall clock (s) | "
            + " | ".join(cost_cell(c, "wall_clock_s", ",.1f") for c in cols)
            + " | lower |")
        lines.append(
            "| Est. cost (USD)¹ | "
            + " | ".join("$" + cost_cell(c, "est_cost_usd", ",.4f") for c in cols)
            + " | lower |")
        lines.append(
            "| **Tokens per SUCCESSFUL task** | "
            + " | ".join(
                _fmt_num(m.value, ",.0f") if (m := cost_metric_of(c, "tokens_per_successful_task"))
                and m.defined else f"n/a *({(m.reason if m else 'not measured')})*"
                for c in cols)
            + " | lower |")
        lines.append(
            "| **Est. cost per SUCCESSFUL task (USD)¹** | "
            + " | ".join(
                ("$" + format(m.value, ",.4f")) if (m := cost_metric_of(c, "cost_per_successful_task"))
                and m.defined else f"n/a *({(m.reason if m else 'not measured')})*"
                for c in cols)
            + " | lower |")

    lines.append("")
    lines.append(
        "¹ **Estimate.** Per-token prices in `src/qwen_client.py` are UNVERIFIED "
        "against the Qwen Cloud pricing page and are currently placeholders, so "
        "USD figures may read as $0.0000. Token counts are exact (reported by the "
        "API); only the conversion to dollars is unconfirmed. Treat "
        "**tokens per successful task** as the load-bearing efficiency number."
    )
    lines.append("")
    lines.append(
        "> Raw token totals are reported deliberately: a society burns more "
        "tokens than a single agent by construction (N system prompts plus "
        "inter-agent messages). The honest comparison is cost per *successful* "
        "task, and both are shown."
    )
    return "\n".join(lines) + "\n"


def strip_internals(scores: dict) -> dict:
    """Return a JSON-serialisable copy (drops the live `Metric` objects)."""
    clean = {"meta": scores.get("meta", {}), "systems": {}}
    for name, entry in scores.get("systems", {}).items():
        clean["systems"][name] = {
            k: v for k, v in entry.items() if not k.startswith("_")
        }
    return clean


# --------------------------------------------------------------------------
# Inline self-test. Run: python3 -m src.scorer
# --------------------------------------------------------------------------


def _selftest() -> int:  # pragma: no cover - this IS the test
    """Verify the metrics on hand-checkable fixtures.

    Cannot use the real systems (baseline.py / society.py are being written
    concurrently), so this constructs `SystemResult` objects directly. That is
    the right level anyway: it tests the scorer, not the agents.
    """
    from .tasks import generate_tasks

    failures: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        else:
            print(f"  ok  {name} = {got!r}")

    # ---- 1. hand-checked AUC -------------------------------------------
    # positives [0.6, 0.4], negatives [0.5, 0.3]
    # ascending: 0.3(neg) 0.4(pos) 0.5(neg) 0.6(pos) -> ranks 1,2,3,4
    # R_pos = 2 + 4 = 6 ; AUC = (6 - 2*3/2) / (2*2) = 3/4 = 0.75
    print("AUC unit cases")
    m = roc_auc([0.6, 0.4, 0.5, 0.3], [True, True, False, False])
    check("auc hand-checked case", m.value, 0.75)

    # Perfect separation.
    check("auc perfect", roc_auc([0.9, 0.8, 0.2, 0.1],
                                 [True, True, False, False]).value, 1.0)
    # Perfectly inverted.
    check("auc inverted", roc_auc([0.1, 0.2, 0.8, 0.9],
                                  [True, True, False, False]).value, 0.0)
    # Partial ties get 0.5 credit, NOT the degenerate path:
    # pos [0.5, 0.5], neg [0.5, 0.1] -> ties contribute 0.5 each.
    # ascending 0.1(neg), then three 0.5s share midrank (2+3+4)/3 = 3
    # R_pos = 3 + 3 = 6 ; AUC = (6 - 3)/4 = 0.75
    check("auc partial ties", roc_auc([0.5, 0.5, 0.5, 0.1],
                                      [True, True, False, False]).value, 0.75)

    # Degenerate: all scores identical -> None with a reason, never 0.5.
    m_flat = roc_auc([0.0, 0.0, 0.0, 0.0], [True, True, False, False])
    check("auc all-identical is None", m_flat.value, None)
    assert m_flat.reason and "no discrimination signal" in m_flat.reason, m_flat.reason
    print(f"  ok  reason = {m_flat.reason!r}")

    # Degenerate: one class only -> None with a reason, never a crash.
    m_one = roc_auc([0.1, 0.9], [True, True])
    check("auc single-class is None", m_one.value, None)
    assert m_one.reason and "only one class" in m_one.reason, m_one.reason
    print(f"  ok  reason = {m_one.reason!r}")

    # ---- 2. full score() over all three task classes --------------------
    print("\nscore() over a fabricated run")
    tasks = generate_tasks(n_per_class=2, seed=1)
    assert len(tasks) == 6, len(tasks)
    solvable = [t for t in tasks if not t.should_escalate]
    esc = [t for t in tasks if t.should_escalate]
    assert len(solvable) == 2 and len(esc) == 4, (len(solvable), len(esc))

    results: list[SystemResult] = []

    # BASELINE: the failure mode we claim. Nails both solvable tasks, never
    # escalates, hallucinates a price on all four impossible ones.
    for t in solvable:
        results.append(SystemResult(
            task_id=t.task_id, system="baseline", escalated=False,
            price=(t.zopa_low + t.zopa_high) // 2, disagreement=0.0,
            rounds=1, escalation_question=None))
    for t in esc:
        results.append(SystemResult(
            task_id=t.task_id, system="baseline", escalated=False,
            price=99_000, disagreement=0.0, rounds=1, escalation_question=None))

    # SOCIETY: one solvable task correct, one falsely escalated; three of four
    # impossible tasks escalated, one hallucinated. Disagreement ranks the
    # escalate-tasks above the solvable ones but imperfectly.
    s0, s1 = solvable
    results.append(SystemResult(
        task_id=s0.task_id, system="society", escalated=False,
        price=(s0.zopa_low + s0.zopa_high) // 2, disagreement=0.10,
        rounds=3, escalation_question=None))
    results.append(SystemResult(
        task_id=s1.task_id, system="society", escalated=True, price=None,
        disagreement=0.80, rounds=3,
        escalation_question="Is the stated ceiling firm?"))
    for i, t in enumerate(esc):
        hallucinate = (i == 0)
        results.append(SystemResult(
            task_id=t.task_id, system="society",
            escalated=not hallucinate,
            price=88_000 if hallucinate else None,
            disagreement=[0.05, 0.70, 0.90, 0.95][i],
            rounds=4,
            escalation_question=None if hallucinate else "Need the missing term.",
        ))

    sc = score(results, tasks, ledger=None)
    b = sc["systems"]["baseline"]["_metrics"]
    s = sc["systems"]["society"]["_metrics"]

    check("baseline solvable_accuracy", b["solvable_accuracy"].value, 1.0)
    check("baseline escalation_recall", b["escalation_recall"].value, 0.0)
    check("baseline false_escalation_rate", b["false_escalation_rate"].value, 0.0)
    check("baseline hallucinated_deal_rate", b["hallucinated_deal_rate"].value, 1.0)
    check("baseline overall_correct", b["overall_correct"].value, 2 / 6)
    check("baseline calibration_auc", b["calibration_auc"].value, None)
    assert "no discrimination signal" in (b["calibration_auc"].reason or "")
    print(f"  ok  baseline AUC reason = {b['calibration_auc'].reason!r}")

    check("society solvable_accuracy", s["solvable_accuracy"].value, 0.5)
    check("society escalation_recall", s["escalation_recall"].value, 0.75)
    check("society false_escalation_rate", s["false_escalation_rate"].value, 0.5)
    check("society hallucinated_deal_rate", s["hallucinated_deal_rate"].value, 0.25)
    # correct = s0 (valid deal) + 3 escalations = 4 of 6
    check("society overall_correct", s["overall_correct"].value, 4 / 6)

    # Society AUC, hand-checked.
    # scores/labels: s0 0.10/F, s1 0.80/F, esc 0.05/T, 0.70/T, 0.90/T, 0.95/T
    # ascending: 0.05(T)=1, 0.10(F)=2, 0.70(T)=3, 0.80(F)=4, 0.90(T)=5, 0.95(T)=6
    # R_pos = 1+3+5+6 = 15 ; n_pos=4, n_neg=2
    # AUC = (15 - 4*5/2) / (4*2) = (15-10)/8 = 0.625
    check("society calibration_auc", s["calibration_auc"].value, 0.625)

    # ---- 3. cost merge with a real Ledger -------------------------------
    print("\ncost metrics via Ledger")
    from .qwen_client import CallRecord, Ledger

    ledger = Ledger()
    for i in range(2):  # baseline: 2 calls, 300 tokens total
        ledger.add(CallRecord(run_id="t", task_id="x", system="baseline",
                              role="solo", model="m", prompt_tokens=100,
                              completion_tokens=50, latency_s=1.0, ok=True))
    for i in range(6):  # society: 6 calls, 1200 tokens total
        ledger.add(CallRecord(run_id="t", task_id="x", system="society",
                              role="buyer", model="m", prompt_tokens=150,
                              completion_tokens=50, latency_s=0.5, ok=True))

    sc2 = score(results, tasks, ledger=ledger)
    check("baseline total_tokens", sc2["systems"]["baseline"]["cost"]["total_tokens"], 300.0)
    check("society total_tokens", sc2["systems"]["society"]["cost"]["total_tokens"], 1200.0)
    # baseline got 2 right -> 150 tokens/success ; society got 4 -> 300.
    check("baseline tokens/success",
          sc2["systems"]["baseline"]["_cost_metrics"]["tokens_per_successful_task"].value,
          150.0)
    check("society tokens/success",
          sc2["systems"]["society"]["_cost_metrics"]["tokens_per_successful_task"].value,
          300.0)

    # Division-by-zero guard: a system that got nothing right.
    zero = [SystemResult(task_id=t.task_id, system="baseline", escalated=False,
                         price=None, disagreement=0.0, rounds=1,
                         escalation_question=None) for t in tasks]
    sc3 = score(zero, tasks, ledger=ledger)
    zm = sc3["systems"]["baseline"]["_cost_metrics"]["tokens_per_successful_task"]
    check("zero-success tokens/success is None", zm.value, None)
    print(f"  ok  reason = {zm.reason!r}")
    # price=None and escalated=False is a non-answer, not a hallucinated deal.
    check("zero-success hallucinated_deal_rate",
          sc3["systems"]["baseline"]["_metrics"]["hallucinated_deal_rate"].value, 0.0)

    # ---- 4. empty-denominator guard -------------------------------------
    only_solvable = [t for t in tasks if not t.should_escalate]
    sc4 = score([r for r in results
                 if r.task_id in {t.task_id for t in only_solvable}],
                only_solvable)
    er = sc4["systems"]["baseline"]["_metrics"]["escalation_recall"]
    check("escalation_recall with no escalate tasks is None", er.value, None)
    print(f"  ok  reason = {er.reason!r}")

    # ---- 5. table renders ----------------------------------------------
    print("\nformat_markdown_table(scores) output\n")
    table = format_markdown_table(sc2)
    print(table)
    for needle in ("Hallucinated-deal rate", "Calibration AUC",
                   "SUCCESSFUL task", "Estimate", "Baseline", "Society"):
        if needle not in table:
            failures.append(f"table missing {needle!r}")

    # JSON round-trip must not choke on Metric objects.
    import json
    json.dumps(strip_internals(sc2))
    print("  ok  strip_internals() is JSON-serialisable")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nALL SCORER SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
