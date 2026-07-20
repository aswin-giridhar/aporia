"""Cross-model comparison: does the disagreement signal hold across tiers?

Runs the same task set through the same code on several Qwen models and puts
the results side by side. The question this answers is not "which model is
best" — it is whether the *architecture's* behaviour is a property of the
design or an artefact of one model's quirks.

A signal that discriminates on a flagship model but not a cheap one would mean
the technique needs capability above some threshold. A signal that behaves
consistently across tiers is evidence it is architectural. Both are findings;
only a single-model run leaves the question open.

    python -m src.compare_models results results/m_plus results/m_max
"""

from __future__ import annotations

import json
import os
import sys

# Metrics worth comparing across models, in reporting order.
_ROWS = [
    ("solvable_accuracy", "Accuracy on SOLVABLE", "%"),
    ("escalation_recall", "Escalation recall", "%"),
    ("false_escalation_rate", "False-escalation rate", "%"),
    ("hallucinated_deal_rate", "Hallucinated-deal rate", "%"),
    ("overall_correct", "Overall correct", "%"),
    ("cross_principal_exposure", "Cross-principal exposure", "%"),
    ("calibration_auc", "Calibration AUC", "f"),
]


def _load(run_dir: str) -> tuple[str, dict] | None:
    """Read the newest scores-*.json in a run directory."""
    raw = os.path.join(run_dir, "raw")
    if not os.path.isdir(raw):
        return None
    files = sorted(
        (f for f in os.listdir(raw) if f.startswith("scores-")),
        key=lambda f: os.path.getmtime(os.path.join(raw, f)),
    )
    if not files:
        return None
    with open(os.path.join(raw, files[-1])) as fh:
        blob = json.load(fh)
    return blob.get("meta", {}).get("model", "?"), blob


def _cell(metrics: dict, key: str, kind: str) -> str:
    m = metrics.get(key)
    if not isinstance(m, dict):
        return "—"
    v = m.get("value")
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if kind == "%" else f"{v:.3f}"


def main(argv: list[str]) -> int:
    dirs = argv[1:] or ["results"]
    runs = [r for r in (_load(d) for d in dirs) if r]
    if not runs:
        print("No completed runs found. Run src.benchmark first.", file=sys.stderr)
        return 1

    models = [m for m, _ in runs]
    lines = [
        "### Cross-model comparison",
        "",
        "Same task set, same code, same scorer — only the model differs. "
        "Served by OpenRouter using the exact model ids Qwen Cloud publishes.",
        "",
        "| Metric | System | " + " | ".join(models) + " |",
        "|---|---|" + "---|" * len(models),
    ]

    for key, label, kind in _ROWS:
        for system in ("baseline", "society"):
            cells = []
            for _, blob in runs:
                sysblob = blob.get("scores", {}).get("systems", {}).get(system, {})
                cells.append(_cell(sysblob.get("metrics", {}), key, kind))
            lines.append(f"| {label} | {system} | " + " | ".join(cells) + " |")

    out = "\n".join(lines)
    print(out)
    with open("results/CROSS_MODEL.md", "w") as fh:
        fh.write(out + "\n")
    print("\nwrote results/CROSS_MODEL.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
