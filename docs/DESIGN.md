# Design note

Technical decisions behind the benchmark, and the reasoning that produced them.
This document exists because the interesting content of the project is the
*measurement design*, not the orchestration code — a multi-agent negotiation
loop is a weekend's work, while a comparison that survives an engineer reading
it adversarially is the hard part.

---

## 1. The metric trap

**Multi-agent systems burn more raw tokens than single agents.** This is not an
implementation flaw to be optimised away; it is structural. N agents means N
system prompts, inter-agent messages, an auditor pass, and coordination
overhead that a single pooled context simply does not incur. Our society issues
several calls per task where the baseline issues one.

So if the headline metric of this benchmark were *tokens consumed*, the
mandatory results table would become evidence **against** our own submission —
and it would be honest evidence. A reader would correctly conclude that we
spent more to do the same job.

### The decision

> **The headline metric is `cost per SUCCESSFUL task`, alongside the success
> rate itself. Raw token totals are reported as a secondary column.**

This is not metric-shopping, and the distinction matters:

- **Raw tokens measure effort.** They answer "what did this run consume?"
- **Cost per successful task measures value delivered.** It answers "what did
  it cost to obtain one outcome I can actually use?"

For a system whose contribution is *knowing when it has failed*, the second is
the only question a production operator asks. A baseline that costs a third as
much per task but silently returns a fabricated price on every impossible task
has a cost per *successful* task that can be arbitrarily worse — or undefined,
if it never succeeds on that class at all.

### Why we still print the raw tokens

Because hiding them would be dishonest, and because an engineer reading the
table will look for them. A results table that reports only the flattering
denominator is a table nobody trusts. The correct move is to report the number
that hurts and explain, in the same view, why it is the wrong number to
optimise.

### Rejected metric: latency via parallelism

A common multi-agent claim is "we're faster, because agents run concurrently."
That claim is only valid when the subtasks are genuinely independent and are
actually executed in parallel. **Ours are not.** Negotiation is inherently
sequential — the seller's counter-offer depends on the buyer's offer. Our
society is therefore *slower* in wall-clock terms than the baseline, and we say
so rather than headlining a number that would be a lie by selective framing.

Wall-clock is reported as a column. It is not a claim.

### Cost figures are estimates, not measurements

The per-token price table in `src/qwen_client.py` is explicitly flagged
`UNVERIFIED` in the source and carries a TODO to confirm against the published
Qwen Cloud pricing. Consequently:

- **Token counts are exact.** They come from the `usage` object on the
  DashScope API response.
- **Latency is exact.** Measured with `time.monotonic()` around the call.
- **Cost is derived** from token counts and an unconfirmed price table, and is
  labelled an estimate everywhere it appears.

The ratio `cost per successful task` between the two systems is robust to the
price table being wrong by a constant factor, since both systems use the same
table — but the absolute dollar figure is not, and is not presented as one.

---

## 2. The controlled-comparison rule

> **The baseline is the same code path with one agent — not a separate
> application.**

Baseline and society share:

- the same **client** (`src/qwen_client.py`), so both are subject to the same
  retry policy, the same degraded-fallback behaviour, the same ledger, and the
  same budget stop;
- the same **task loader** (`src/tasks.py`) and the same seed, so both see
  byte-identical briefs;
- the same **scorer** (`src/scorer.py`), so neither gets a more forgiving
  reading of its output;
- the same **model** by default (`qwen3.7-plus`), so a difference in results is
  not a difference in model capability.

The single independent variable is *how many agents deliberate*. That is the
whole point of the experiment, and any other difference between the two arms
would contaminate it.

This rule has a cost. It forbids the tempting move of giving the society a
better prompt, a stronger model, or a more permissive output parser than the
baseline — each of which would improve the numbers and destroy the claim. Where
the two arms must differ (the baseline has no escalation channel, because it
has no disagreement to threshold), the difference is *the thing being measured*
and is stated explicitly rather than buried.

### On the baseline's fairness

A reasonable objection: could the baseline escalate if we simply told it to?
We give the single agent both briefs and the same latitude to refuse. It can
say "no deal is possible." The claim under test is not that a single agent is
*forbidden* to escalate — it is that a single agent has no **calibrated**
signal for *when* to. Its refusals, if any, are ungrounded assertions from the
same process that produces its confident errors. That is exactly what the
calibration AUC measures, and the baseline is scored on the same axis.

---

## 3. Calibration AUC methodology

The core claim — *does inter-agent disagreement predict failure?* — is a
question about ranking, not about a threshold. AUC is the right instrument
because it is threshold-free: it asks whether the signal orders the cases
correctly, independent of where we choose to cut.

### Construction

1. **Score.** For each task, the society emits a scalar disagreement value.
   Its components: the spread between the parties' standing offers at
   termination, whether the Auditor dissented from the proposed deal, and the
   number of rounds elapsed without convergence. These are combined into one
   number per task.

2. **Label.** The binary label comes from `src/tasks.py` ground truth, not from
   any model: a task is positive if the correct behaviour is **ESCALATE**
   (`UNDERSPECIFIED` or `CONTRADICTORY`) and negative if a valid deal exists
   (`SOLVABLE`). The label is known before the run starts.

3. **AUC.** Compute the area under the ROC curve of disagreement score against
   that label. Equivalently and more intuitively: **the probability that a
   randomly chosen impossible task is assigned a higher disagreement score than
   a randomly chosen solvable one.**

### Reading the result

- **AUC ≈ 0.5** — disagreement carries no information about solvability. The
  core claim is false, and we report that. A society could still win on
  accuracy while failing here; the two are separate questions.
- **AUC > 0.5** — disagreement ranks impossible tasks above solvable ones,
  *provided the score is not saturated and the sample is large enough to trust*.
- **AUC ≈ 1.0** on a small task set is a warning sign, not a triumph: with a
  balanced set of a few tasks per class, the estimate is high-variance. The
  task count is reported alongside the AUC for exactly this reason.

**What the live run actually produced — and why we do NOT claim it as a win.**
The society's AUC came out 0.75 / 0.50 / 0.75 across the three models. On
inspection this is an artifact, not a signal, and we retracted the claim:

1. **Saturation.** The escalation policy floors the disagreement score at 0.85
   whenever it fires, and it fires on almost every task, so five of six scores
   are pinned at 0.85. A metric that is partly *set by* the decision it is meant
   to predict is close to circular.
2. **Sample size.** n = 6, split 4-vs-2. A single task moves the AUC by 0.25.

So the honest reading of the society's calibration is **inconclusive**, and the
uncertainty-signal hypothesis is **not demonstrated** by this run. The one
finding that survives is the baseline's *sub-0.5* AUC (0.06–0.12, consistent
across three models): a single agent's self-confidence, on this set, did not
track its correctness. We report the direction, with the n=6 caveat, and no
more. See README §5.

### How the baseline is scored on the same axis

The baseline has no *inter-agent* dissent to measure — there is only one agent —
so it cannot produce a disagreement score of the kind the society computes. It
can, however, express uncertainty, and excluding it from the calibration
comparison would be convenient and unfair. So we give it **the strongest signal
a single agent can actually have**: its own self-reported confidence, inverted to a
disagreement-shaped score (`1 - confidence`, clamped to `[0, 1]`) in
`src/baseline.py`. Both systems then feed the identical AUC computation.

This is deliberately generous to the control. The claim under test is not that
a single agent is *forbidden* to express uncertainty — it is that self-reported
confidence is produced by the same process that produced the error, and
therefore inherits its blind spot, whereas disagreement between agents holding
different private constraints is an observation about the task. If the
baseline's self-reported confidence turns out to calibrate as well as measured
disagreement, that is a real result and we report it as one.

### Threshold selection

The operating threshold on the disagreement metric is a product decision, not a
statistical one: it trades escalation recall against false-escalation rate.
A system that escalates everything achieves perfect recall and is worthless,
which is precisely why the **false-escalation rate on `SOLVABLE` tasks** is a
mandatory column and not an optional one. Both numbers are reported at the
chosen operating point, and the AUC characterises the curve they sit on.

---

## 4. Why mock mode exists

`QWEN_MODE=mock` returns deterministic canned responses through the identical
wrapper, ledger and scorer. It is not a testing convenience bolted on
afterwards; it was written before live entitlement existed, for three reasons:

1. **The orchestration layer could be built and debugged before the API key
   arrived**, converting the first live call from a debugging session into a
   five-minute verification.
2. **A judge with no Alibaba Cloud account can still run the entire pipeline**
   and see the architecture work end to end. `pip install openai` is not even
   required in this mode.
3. **A green run in mock mode is not a green run against DashScope**, and the
   repository does not claim otherwise. Mock mode proves the orchestration; the
   live smoke test (`python -m src.qwen_client`) proves the endpoint. Both are
   necessary and neither substitutes for the other.

Mock responses are still accounted in the ledger with approximate token counts,
so the budget-stop logic and the results-table generation are exercised by mock
runs rather than being code paths that first execute when real money is at
stake.
