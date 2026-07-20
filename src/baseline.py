"""Single-agent baseline — the control in the experiment.

This is deliberately a *strong* baseline, not a strawman. It receives strictly
MORE information than any individual agent in the society: both private briefs,
pooled into one context. It is explicitly told it may refuse, and is shown the
same escalation vocabulary the society uses.

Making the baseline strong matters. The project's claim is that a society has a
reliability signal a single agent structurally cannot have. That claim is only
interesting if the single agent was given every fair advantage — otherwise a
judge rightly reads the result as a rigged comparison.

What the baseline cannot do, by construction, is *disagree with itself*. It
emits one opinion, so its confidence signal is degenerate. That is the gap the
benchmark measures.
"""

from __future__ import annotations

import json

from .privacy import measure_baseline_exposure
from .qwen_client import QwenClient, MODEL_STANDARD
from .scorer import SystemResult
from .tasks import Task

BASELINE_SYSTEM_PROMPT = """You are an expert procurement analyst. You are given \
the complete private briefs of BOTH parties in a negotiation and must decide the outcome.

You must respond with a single JSON object and nothing else:

{
  "escalate": true | false,
  "price": <integer or null>,
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>",
  "question_for_human": "<string or null>"
}

Set "escalate" to true if — and only if — no responsible agreement can be reached
from the information given. Reasons to escalate include: the parties' acceptable
ranges do not overlap at all, or a term required to evaluate the deal was never
specified. If you can identify a fair price both parties would accept, set
"escalate" to false and give that price.

Do not invent facts that are not in the briefs."""


def _render_task(task: Task) -> str:
    """Pool both private briefs — the baseline's structural advantage."""
    return f"""ITEM: {task.item}

BUYER BRIEF (private):
{task.buyer.public_context}
Maximum the buyer can pay: {task.buyer.reservation_price}

SELLER BRIEF (private):
{task.seller.public_context}
Minimum the seller will accept: {task.seller.reservation_price}

Decide the outcome."""


def _parse(raw: str) -> dict:
    """Tolerant JSON extraction.

    Models wrap JSON in prose or fences often enough that a strict parse would
    measure formatting compliance rather than reasoning. We salvage the first
    balanced object; only if that fails do we treat it as a genuine failure.
    """
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
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


def run_baseline(
    task: Task,
    client: QwenClient,
    run_id: str,
    model: str = MODEL_STANDARD,
) -> SystemResult:
    """One agent, both briefs, one shot."""
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": _render_task(task)},
    ]
    raw = client.chat(
        messages, run_id=run_id, task_id=task.task_id,
        system="baseline", role="solo_analyst", model=model,
    )
    parsed = _parse(raw)

    escalated = bool(parsed.get("escalate", False))
    price = parsed.get("price")
    price = int(price) if isinstance(price, (int, float)) else None

    # A single agent's stated confidence is its ONLY uncertainty signal, and it
    # is self-reported. We invert it to a disagreement-shaped score so both
    # systems feed the same calibration metric on equal terms — this is the
    # fairest available proxy, and the comparison would be unfair without it.
    confidence = parsed.get("confidence")
    disagreement = 1.0 - float(confidence) if isinstance(confidence, (int, float)) else 0.0
    disagreement = max(0.0, min(1.0, disagreement))

    # Exposure is structural here: _render_task() pooled both confidential
    # briefs into one prompt, so every secret crossed the principal boundary
    # before the model was even called.
    exposure = measure_baseline_exposure(task)

    return SystemResult(
        task_id=task.task_id,
        system="baseline",
        escalated=escalated,
        price=price,
        disagreement=disagreement,
        rounds=1,
        escalation_question=parsed.get("question_for_human") if escalated else None,
        transcript=[{"role": "solo_analyst", "content": raw}],
        exposed_secrets=exposure.exposed_secrets,
        total_secrets=exposure.total_secrets,
        exposure_by_construction=True,
    )
