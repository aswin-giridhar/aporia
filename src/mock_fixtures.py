"""Deterministic mock agent behaviour — a TEST FIXTURE, not evidence.

READ THIS BEFORE USING ANY NUMBER THIS MODULE PRODUCES
------------------------------------------------------
This module simulates plausible agent responses so the orchestration and
scoring machinery can be exercised without an API key. It exists to answer one
question: *does the pipeline compute the right metrics when fed varied inputs?*

It CANNOT answer the project's actual research question. The behaviour here is
hand-written by us, so a mock run showing "society beats baseline" would be
measuring our own assumptions, not model behaviour. That would be circular.

**Numbers from `--mode mock` must never appear in the README results table or
be presented to judges as findings.** The benchmark labels them accordingly.

What the simulation does
------------------------
Agents parse their reservation price out of their own prompt (it is genuinely
there) and behave like a self-interested negotiator: anchor away from their
limit, concede across rounds, flag missing information when the brief says a
term was unspecified, and declare impossibility when the counterparty's
standing offer cannot be reconciled.

The single agent is modelled as answering whenever it can compute an overlap —
which is the documented LLM failure mode on malformed requests, but here it is
an assumption we encoded, not an observation.
"""

from __future__ import annotations

import hashlib
import json
import re


def _stable_jitter(seed_text: str, lo: float, hi: float) -> float:
    """Deterministic pseudo-randomness.

    Keyed off the prompt so a given task always produces the same run — a mock
    that varied between runs would make failures unreproducible.
    """
    h = hashlib.sha256(seed_text.encode()).hexdigest()
    return lo + (int(h[:8], 16) / 0xFFFFFFFF) * (hi - lo)


def _find_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _offers_in_history(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"offer=(\d+)", text)]


def mock_handler(role: str, messages: list[dict]) -> str:
    """Return a plausible JSON reply for the given agent role."""
    prompt = "\n".join(str(m.get("content", "")) for m in messages)
    underspecified = "NOTE:" in prompt and "never specified" in prompt

    # ---------------- single-agent baseline ----------------
    if role == "solo_analyst":
        ceiling = _find_int(r"Maximum the buyer can pay:\s*(\d+)", prompt)
        floor = _find_int(r"Minimum the seller will accept:\s*(\d+)", prompt)

        if underspecified:
            # ASSUMPTION (not observation): a single agent usually proceeds
            # anyway, because it was asked for a deal and a price is computable.
            proceeds = _stable_jitter(prompt, 0, 1) < 0.75
            if proceeds and ceiling and floor and floor <= ceiling:
                return json.dumps({
                    "escalate": False, "price": (ceiling + floor) // 2,
                    "confidence": 0.72,
                    "reason": "Ranges overlap; settling at the midpoint.",
                    "question_for_human": None})
            return json.dumps({
                "escalate": True, "price": None, "confidence": 0.4,
                "reason": "A required term was not specified.",
                "question_for_human": "Please supply the unspecified term."})

        if ceiling is not None and floor is not None and floor > ceiling:
            # Provably impossible. Modelled as usually-but-not-always caught.
            caught = _stable_jitter(prompt + "c", 0, 1) < 0.45
            if caught:
                return json.dumps({
                    "escalate": True, "price": None, "confidence": 0.55,
                    "reason": "No overlap between the parties' limits.",
                    "question_for_human": "Can either limit move?"})
            return json.dumps({
                "escalate": False, "price": (ceiling + floor) // 2,
                "confidence": 0.68,
                "reason": "Proposing a compromise between the two positions.",
                "question_for_human": None})

        if ceiling is not None and floor is not None:
            return json.dumps({
                "escalate": False, "price": (ceiling + floor) // 2,
                "confidence": 0.88,
                "reason": "Ranges overlap; midpoint is acceptable to both.",
                "question_for_human": None})
        return json.dumps({"escalate": True, "price": None, "confidence": 0.3,
                           "reason": "Insufficient information.",
                           "question_for_human": "Provide the party limits."})

    # ---------------- society: buyer / seller ----------------
    if role in ("buyer_agent", "seller_agent"):
        reservation = _find_int(r"reservation price[^:]*:\s*(\d+)", prompt)
        prior = _offers_in_history(prompt)
        rounds_done = len(prior) // 2

        if reservation is None:
            return json.dumps({"offer": None, "rationale": "No brief available.",
                               "deal_possible": False,
                               "missing_information": "party limits",
                               "confidence": 0.2})

        # Anchor away from the limit, then concede toward it each round.
        anchor = 0.82 if role == "buyer_agent" else 1.20
        concession = 0.06 * rounds_done
        offer = int(reservation * (anchor + concession if role == "buyer_agent"
                                   else anchor - concession))

        counter = prior[-1] if prior else None
        impossible = False
        if counter is not None:
            # Buyer cannot exceed its ceiling; seller cannot go below its floor.
            if role == "buyer_agent" and counter > reservation:
                impossible = rounds_done >= 1
            if role == "seller_agent" and counter < reservation:
                impossible = rounds_done >= 1

        return json.dumps({
            "offer": None if impossible else offer,
            "rationale": ("This exceeds what I am authorised to agree."
                          if impossible else
                          f"I can work toward {offer} given current terms."),
            "deal_possible": not impossible,
            "missing_information": ("the term noted as never specified"
                                    if underspecified else None),
            "confidence": 0.5 if impossible else 0.8,
        })

    # ---------------- society: auditor ----------------
    if role == "auditor":
        offers = _offers_in_history(prompt)
        if underspecified or "no deal possible" in prompt:
            return json.dumps({
                "assessment": "underspecified" if underspecified else "deadlocked",
                "should_escalate": True,
                "question_for_human": ("Which term applies? The parties cannot "
                                       "evaluate the deal without it."),
                "confidence": 0.75})
        if len(offers) >= 2 and abs(offers[-1] - offers[-2]) > 0.25 * max(offers):
            return json.dumps({"assessment": "deadlocked", "should_escalate": True,
                               "question_for_human": "Are either side's limits negotiable?",
                               "confidence": 0.6})
        return json.dumps({"assessment": "converging", "should_escalate": False,
                           "question_for_human": None, "confidence": 0.8})

    return json.dumps({"note": f"unhandled role {role}"})
