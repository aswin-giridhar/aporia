"""The agent society — negotiation with measured dissent.

Three agents, each with STRICTLY LESS information than the baseline:

  BuyerAgent   sees only the buyer's brief
  SellerAgent  sees only the seller's brief
  Auditor      sees NEITHER brief — only the public exchange

They negotiate over rounds. Agents exchange offers and public rationales; they
never see each other's private reservation prices. This information asymmetry
is the point, not a limitation: it is what makes their disagreement *carry
information*. Two agents who share a context cannot meaningfully disagree —
they would be one agent with extra steps.

The disagreement signal
-----------------------
After each round we compute a scalar in [0, 1] from four observable quantities:

  gap        normalised distance between the standing offers
  stall      how little the gap moved this round (a converging negotiation
             closes; a doomed one oscillates or freezes)
  impossible fraction of agents asserting no deal exists
  missing    fraction of agents asserting a required term was never specified

None of these requires an LLM to judge an outcome, and none is available to a
single agent — a lone model has no second opinion to differ from. This is the
signal the benchmark tests: does it predict failure?

Escalation
----------
High disagreement that fails to resolve becomes a decision to STOP and ask a
human, rather than emit a confident answer. The escalation carries a targeted
question, so the human is asked for the specific missing thing rather than
being handed the whole problem back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .qwen_client import QwenClient, MODEL_STANDARD
from .scorer import SystemResult
from .tasks import Task

# ---------------------------------------------------------------------------
# Escalation policy — THE key tunable of this system.
#
# These two numbers encode a real trade-off with no objectively correct answer:
#   - Escalate too eagerly and the system is useless; it hands everything back
#     to the human it was supposed to help.
#   - Escalate too reluctantly and it does the exact thing this project exists
#     to prevent: confidently inventing an answer to an impossible question.
# The benchmark measures both sides of that trade (escalation_recall against
# false_escalation_rate), so these values are empirically checkable rather than
# matters of taste.
ESCALATION_THRESHOLD = 0.55   # disagreement above which we stop and ask
MAX_ROUNDS = 3                # negotiation rounds before forcing a decision
# ---------------------------------------------------------------------------

_AGENT_PROMPT = """You are the {party} agent in a procurement negotiation. You \
represent your principal's interests and you can see ONLY your own brief.

YOUR PRIVATE BRIEF:
{brief}
Your reservation price (never state this number directly): {reservation}

{history}

Respond with a single JSON object and nothing else:

{{
  "offer": <integer price you propose, or null if you believe no deal is possible>,
  "rationale": "<one public sentence — safe for the other party to read>",
  "deal_possible": true | false,
  "missing_information": "<the specific term required to evaluate this deal that was never specified, or null>",
  "confidence": <float 0.0-1.0>
}}

Be honest. If the other party's position cannot be reconciled with your \
reservation price, say deal_possible is false. If a term needed to judge the \
deal was never specified, name it in missing_information. Do not invent facts."""

_AUDITOR_PROMPT = """You are an impartial negotiation auditor. You CANNOT see \
either party's private brief — only their public exchange below.

{history}

Assess whether these parties are converging on a deal, deadlocked, or unable to \
proceed because information is missing.

Respond with a single JSON object and nothing else:

{{
  "assessment": "converging" | "deadlocked" | "underspecified",
  "should_escalate": true | false,
  "question_for_human": "<the single most useful question to ask a human to unblock this, or null>",
  "confidence": <float 0.0-1.0>
}}"""


@dataclass
class RoundState:
    buyer_offer: int | None
    seller_offer: int | None
    impossible_votes: float
    missing_votes: float
    gap: float


def _parse(raw: str) -> dict:
    """Shared tolerant JSON extraction (see baseline._parse for rationale)."""
    text = raw.strip()
    if "```" in text:
        for p in text.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    start, depth = text.find("{"), 0
    if start == -1:
        return {}
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _normalised_gap(buyer: int | None, seller: int | None) -> float:
    """Distance between standing offers, scaled to [0, 1].

    A null offer means an agent has declared no deal possible; that is maximal
    disagreement, not missing data.
    """
    if buyer is None or seller is None:
        return 1.0
    scale = max(abs(buyer), abs(seller), 1)
    return min(1.0, abs(buyer - seller) / scale)


def compute_disagreement(history: list[RoundState]) -> float:
    """Collapse the round history into one uncertainty score in [0, 1].

    Weighting rationale: explicit assertions by agents that a deal is
    impossible, or that information is missing, are strong direct evidence and
    are weighted highest. The offer gap is real but noisy — a wide gap early in
    a healthy negotiation is normal. Stall is what separates "still working" from
    "going nowhere", so it is only meaningful once we have two rounds.
    """
    if not history:
        return 0.0
    last = history[-1]

    stall = 0.0
    if len(history) >= 2:
        prev = history[-2]
        # Gap that fails to close (or widens) is evidence of a doomed process.
        stall = max(0.0, 1.0 - max(0.0, prev.gap - last.gap) / max(prev.gap, 1e-6))

    score = (
        0.35 * last.impossible_votes
        + 0.30 * last.missing_votes
        + 0.20 * last.gap
        + 0.15 * stall
    )
    return max(0.0, min(1.0, score))


def _format_history(transcript: list[dict]) -> str:
    if not transcript:
        return "This is the opening round. No offers have been made yet."
    lines = ["NEGOTIATION SO FAR (public):"]
    for t in transcript:
        offer = t.get("offer")
        offer_s = "no deal possible" if offer is None else str(offer)
        line = f"  [{t['role']}] offer={offer_s} — {t.get('rationale', '')}"
        if t.get("missing_information"):
            line += f" [FLAGGED MISSING: {t['missing_information']}]"
        lines.append(line)
    return "\n".join(lines)


def run_society(
    task: Task,
    client: QwenClient,
    run_id: str,
    model: str = MODEL_STANDARD,
    threshold: float = ESCALATION_THRESHOLD,
    max_rounds: int = MAX_ROUNDS,
) -> SystemResult:
    """Run the negotiation, measure dissent, decide whether to escalate."""
    public_transcript: list[dict] = []
    history: list[RoundState] = []
    full_transcript: list[dict] = []
    rounds_used = 0

    for rnd in range(max_rounds):
        rounds_used = rnd + 1
        round_replies: dict[str, dict] = {}

        for party, brief in (("buyer", task.buyer), ("seller", task.seller)):
            prompt = _AGENT_PROMPT.format(
                party=party,
                brief=brief.public_context,
                reservation=brief.reservation_price,
                history=_format_history(public_transcript),
            )
            raw = client.chat(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "Make your move."}],
                run_id=run_id, task_id=task.task_id,
                system="society", role=f"{party}_agent", model=model,
            )
            parsed = _parse(raw)
            round_replies[party] = parsed
            full_transcript.append({"role": f"{party}_agent", "round": rounds_used,
                                    "content": raw})

        # Publish only what is safe to share — offers and public rationales.
        # Reservation prices never enter the shared transcript. This is also the
        # hook the optional privacy upgrade (option A) would measure against.
        for party in ("buyer", "seller"):
            p = round_replies[party]
            public_transcript.append({
                "role": f"{party}_agent",
                "offer": p.get("offer"),
                "rationale": p.get("rationale", ""),
                # An agent's claim that a required term is MISSING is not
                # private information — it reveals nothing about that party's
                # position, only that the shared problem is malformed. It must
                # be published, or the auditor (which sees only this transcript)
                # is structurally blind to the very signal it exists to detect.
                "missing_information": p.get("missing_information"),
                "deal_possible": p.get("deal_possible"),
            })

        def _offer(p: dict) -> int | None:
            v = p.get("offer")
            return int(v) if isinstance(v, (int, float)) else None

        b_off, s_off = _offer(round_replies["buyer"]), _offer(round_replies["seller"])
        impossible = sum(
            1.0 for p in round_replies.values() if p.get("deal_possible") is False
        ) / 2.0
        missing = sum(
            1.0 for p in round_replies.values() if p.get("missing_information")
        ) / 2.0

        history.append(RoundState(
            buyer_offer=b_off, seller_offer=s_off,
            impossible_votes=impossible, missing_votes=missing,
            gap=_normalised_gap(b_off, s_off),
        ))

        # Early exit: both agents agree a deal is impossible or blocked. Burning
        # further rounds would spend tokens to re-confirm a settled conclusion.
        if impossible >= 1.0 or missing >= 1.0:
            break
        # Converged: offers have met or crossed.
        if b_off is not None and s_off is not None and b_off >= s_off:
            break

    disagreement = compute_disagreement(history)

    # The auditor sees only the public exchange — an independent third opinion
    # that cannot be anchored by either principal's private position.
    auditor_raw = client.chat(
        [{"role": "system", "content": _AUDITOR_PROMPT.format(
            history=_format_history(public_transcript))},
         {"role": "user", "content": "Give your assessment."}],
        run_id=run_id, task_id=task.task_id,
        system="society", role="auditor", model=model,
    )
    auditor = _parse(auditor_raw)
    full_transcript.append({"role": "auditor", "content": auditor_raw})

    # The auditor can force escalation even below threshold. It is the only
    # participant with no stake in closing a deal, so its judgement that
    # something is missing is treated as decisive rather than advisory.
    auditor_escalates = bool(auditor.get("should_escalate", False))
    if auditor.get("assessment") == "underspecified":
        auditor_escalates = True

    # UNANIMITY IS CATEGORICAL, NOT WEIGHTED.
    # If every agent independently asserts the deal is impossible, or that a
    # required term was never specified, that is a direct claim about the
    # problem's validity — not a soft signal to be averaged down against a
    # narrowing price gap. Folding it into the weighted score let unanimous
    # "information is missing" land at 0.30 and silently produce a deal, which
    # is precisely the failure this project exists to prevent.
    unanimous_block = any(
        r.impossible_votes >= 1.0 or r.missing_votes >= 1.0 for r in history
    )

    if unanimous_block:
        # Keep the reported score consistent with the decision. Calibration AUC
        # is computed from `disagreement`, so leaving it low while escalating
        # would measure the metric against evidence the policy already acted on.
        disagreement = max(disagreement, 0.85)

    escalate = disagreement >= threshold or auditor_escalates or unanimous_block

    last = history[-1] if history else None
    price: int | None = None
    if not escalate and last and last.buyer_offer is not None and last.seller_offer is not None:
        # Settle at the midpoint of the crossed offers — inside both parties'
        # accepted range by construction.
        price = (last.buyer_offer + last.seller_offer) // 2

    # A society that escalates without saying what it needs has only deferred
    # the problem. Always carry a concrete question.
    question = auditor.get("question_for_human") if escalate else None
    if escalate and not question:
        question = ("The parties could not reconcile their positions. Confirm "
                    "whether either side's stated limits are negotiable, or "
                    "supply the term that was left unspecified.")

    return SystemResult(
        task_id=task.task_id,
        system="society",
        escalated=escalate,
        price=price,
        disagreement=disagreement,
        rounds=rounds_used,
        escalation_question=question,
        transcript=full_transcript,
    )
