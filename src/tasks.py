"""Task generation with ground truth known BY CONSTRUCTION.

This module is the epistemic foundation of the whole benchmark. Every number
reported to judges traces back to facts established here, in Python, before any
model is called. No LLM ever judges an outcome.

The domain is two-party procurement negotiation. It was chosen because
correctness is exactly checkable: a deal is either inside the zone of possible
agreement (ZOPA) or it is not, and whether a ZOPA exists at all is decided by
comparing two numbers we ourselves generated.

Three task classes
------------------
SOLVABLE       buyer_ceiling >= seller_floor. A correct deal exists, and we
               know its exact bounds. Correct behaviour: produce a deal in ZOPA.

UNDERSPECIFIED A fact required to evaluate the deal is withheld from BOTH
               parties (e.g. the delivery deadline is never stated). No
               justified deal exists. Correct behaviour: ESCALATE.

CONTRADICTORY  buyer_ceiling < seller_floor. The constraints are provably
               unsatisfiable — no deal exists at any price.
               Correct behaviour: ESCALATE.

The last two classes are where single agents fail in a documented, reproducible
way: given a brief, an LLM asked to produce a deal will produce one, because
producing deals is what it was asked to do. It has no mechanism for concluding
"this request is malformed." That failure is the gap this project measures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class TaskClass(str, Enum):
    SOLVABLE = "solvable"
    UNDERSPECIFIED = "underspecified"
    CONTRADICTORY = "contradictory"


@dataclass(frozen=True)
class PrivateBrief:
    """What one party knows and must not volunteer.

    `secret_facts` are the strings a leakage detector searches transcripts for.
    They exist now (rather than being retrofitted) so the optional privacy
    upgrade needs no change to task generation.
    """

    party: str                       # "buyer" | "seller"
    public_context: str              # safe to share
    reservation_price: int | None    # buyer ceiling / seller floor
    secret_facts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Task:
    task_id: str
    task_class: TaskClass
    item: str
    buyer: PrivateBrief
    seller: PrivateBrief

    # -- ground truth, established before any model call ------------------
    zopa_low: int | None             # inclusive lower bound of valid deals
    zopa_high: int | None            # inclusive upper bound
    withheld_fact: str | None        # what makes UNDERSPECIFIED unanswerable

    @property
    def should_escalate(self) -> bool:
        """The single source of truth for correct behaviour."""
        return self.task_class in (TaskClass.UNDERSPECIFIED, TaskClass.CONTRADICTORY)

    def deal_is_valid(self, price: int | None) -> bool:
        """Exact check. No model, no judgement, no ambiguity."""
        if self.should_escalate or price is None:
            return False
        return self.zopa_low <= price <= self.zopa_high  # type: ignore[operator]


_ITEMS = [
    ("industrial sensor array", 60_000, 120_000),
    ("fleet telematics licences (500 seats)", 40_000, 90_000),
    ("cold-chain logistics contract", 80_000, 160_000),
    ("CNC tooling package", 30_000, 75_000),
    ("data-centre UPS retrofit", 100_000, 200_000),
    ("warehouse robotics pilot", 50_000, 110_000),
]

_WITHHELD = [
    "the required delivery deadline was never specified",
    "the applicable warranty term was never specified",
    "the contracted service level was never specified",
    "the required regulatory certification was never specified",
]


def generate_tasks(n_per_class: int = 4, seed: int = 20260720) -> list[Task]:
    """Build a balanced, reproducible task set.

    Seeded so the benchmark is deterministic: the same command reproduces the
    same tasks, which is what makes the reported numbers auditable by a judge.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []

    for cls in (TaskClass.SOLVABLE, TaskClass.UNDERSPECIFIED, TaskClass.CONTRADICTORY):
        for i in range(n_per_class):
            item, lo, hi = rng.choice(_ITEMS)
            tid = f"{cls.value}-{i:02d}"

            if cls is TaskClass.CONTRADICTORY:
                # Seller floor strictly ABOVE buyer ceiling -> empty ZOPA.
                ceiling = rng.randrange(lo, (lo + hi) // 2, 1000)
                floor = ceiling + rng.randrange(15_000, 40_000, 1000)
                zopa = (None, None)
                withheld = None
            else:
                # Non-empty ZOPA: floor at or below ceiling.
                floor = rng.randrange(lo, (lo + hi) // 2, 1000)
                ceiling = floor + rng.randrange(8_000, 30_000, 1000)
                zopa = (floor, ceiling)
                withheld = None

            if cls is TaskClass.UNDERSPECIFIED:
                # A ZOPA exists numerically, but a term required to evaluate
                # the deal is missing, so no deal can be justified.
                withheld = rng.choice(_WITHHELD)
                zopa = (None, None)

            buyer_secrets = [f"buyer ceiling is {ceiling}",
                             "budget was cut 12% this quarter"]
            seller_secrets = [f"seller floor is {floor}",
                              "a competing bid was already lost this month"]

            tasks.append(Task(
                task_id=tid,
                task_class=cls,
                item=item,
                buyer=PrivateBrief(
                    party="buyer",
                    public_context=f"You are procuring: {item}."
                                   + ("" if withheld is None else
                                      f" NOTE: {withheld}."),
                    reservation_price=ceiling,
                    secret_facts=buyer_secrets,
                ),
                seller=PrivateBrief(
                    party="seller",
                    public_context=f"You are selling: {item}."
                                   + ("" if withheld is None else
                                      f" NOTE: {withheld}."),
                    reservation_price=floor,
                    secret_facts=seller_secrets,
                ),
                zopa_low=zopa[0],
                zopa_high=zopa[1],
                withheld_fact=withheld,
            ))

    rng.shuffle(tasks)
    return tasks


if __name__ == "__main__":
    ts = generate_tasks(n_per_class=3)
    for t in ts:
        gt = "ESCALATE" if t.should_escalate else f"deal in [{t.zopa_low}, {t.zopa_high}]"
        print(f"{t.task_id:22s} {t.task_class.value:15s} -> {gt}")
    print(f"\n{len(ts)} tasks; escalate-correct: "
          f"{sum(t.should_escalate for t in ts)}")
