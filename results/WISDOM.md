# Measurable multi-agent gain: self-consistency society vs single agent

Model `qwen3.6-flash` via openrouter, 5 solver agents + majority vote, 12 known-answer reasoning questions. Exact-match scoring, ground truth fixed in code — no LLM-as-judge.

| System | Accuracy | Tokens |
|---|---|---|
| Single agent (baseline) | 12/12 (100%) | 16582 |
| Agent society (5+vote) | 12/12 (100%) | 75186 |

**Measured accuracy gain: +0%** (+0 questions), at the cost of more tokens — the honest trade-off, both shown.
