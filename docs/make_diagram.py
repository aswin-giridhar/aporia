"""Render docs/architecture.png — the submission architecture diagram.

Self-contained: matplotlib only, no network, no external rendering service.

    python3 docs/make_diagram.py

Regenerating the diagram is therefore a repo operation, not a trip to a
drawing tool, which means the diagram cannot drift from the architecture
without someone editing this file.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- palette
BG          = "#FBFBFD"
INK         = "#14161C"
MUTED       = "#5A6273"
ALI_FILL    = "#FFF1E3"
ALI_EDGE    = "#FF6A00"
CHOKE_FILL  = "#EAEFFB"
CHOKE_EDGE  = "#2C4B9B"
BASE_FILL   = "#F1F2F5"
BASE_EDGE   = "#8A93A5"
SOC_FILL    = "#E7F6F1"
SOC_EDGE    = "#128C6A"
HUMAN_FILL  = "#FFF6DC"
HUMAN_EDGE  = "#C98A00"
EVAL_FILL   = "#F3EBFA"
EVAL_EDGE   = "#6D3FA0"
ARROW       = "#39404F"

W, H = 160.0, 107.0

fig, ax = plt.subplots(figsize=(16, 10.7), dpi=170)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(-4.5, H)
ax.axis("off")


# ---------------------------------------------------------------- helpers
def box(x0, y0, x1, y1, fill, edge, lw=1.8, r=1.4, z=2, ls="solid", alpha=1.0):
    p = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z,
        linestyle=ls, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def text(x, y, s, size=11, color=INK, weight="normal", ha="center", va="center",
         style="normal", family="DejaVu Sans", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            fontstyle=style, family=family, zorder=z, linespacing=1.5)


def mono(x, y, s, size=10, color=INK, weight="bold", ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            family="DejaVu Sans Mono", zorder=z, linespacing=1.5)


def arrow(x0, y0, x1, y1, color=ARROW, lw=1.9, style="-|>", rad=0.0, z=4, ls="solid"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=17,
        color=color, linewidth=lw, zorder=z, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)
    return a


def label(x, y, s, size=9.5, color=MUTED, ha="center", rot=0, bg=BG):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", rotation=rot,
            family="DejaVu Sans", zorder=6, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=bg, edgecolor="none"))


# ================================================================ TITLE
text(80, 104.0, "A Society That Knows What It Doesn't Know",
     size=21, weight="bold")
text(80, 100.4,
     "Calibrated escalation from measured inter-agent disagreement   ·   "
     "Track 3: Agent Society   ·   Alibaba Cloud Qwen",
     size=11.5, color=MUTED)


# ================================================================ 1. PROVIDER
box(33, 84.0, 158, 96.5, ALI_FILL, ALI_EDGE, lw=2.4)
text(95.5, 93.6, "ALIBABA CLOUD  ·  MODEL PROVIDER", size=10, color=ALI_EDGE,
     weight="bold")
mono(95.5, 90.0, "dashscope-intl.aliyuncs.com/compatible-mode/v1", size=12.5)
text(95.5, 86.4,
     "qwen3.7-max          qwen3.7-plus  (default)          qwen3.6-flash",
     size=11, color=INK, weight="bold")

label(58.0, 81.0, "OpenAI-compatible surface", size=9.0)


# ================================================================ 2. CHOKE POINT
box(33, 62.0, 158, 78.5, CHOKE_FILL, CHOKE_EDGE, lw=2.6)
mono(95.5, 75.4, "src/qwen_client.py", size=15, color=CHOKE_EDGE)
text(95.5, 72.0, "SINGLE CHOKE POINT — every model call in the project passes through here",
     size=10.5, color=INK, weight="bold")

_chips = [
    (37.0, 64.3, "LEDGER",
     "tokens · latency · cost\nper call, per agent role"),
    (78.0, 64.3, "MOCK MODE",
     "QWEN_MODE=mock\nzero network, zero spend"),
    (119.0, 64.3, "BUDGET STOP",
     "hard token ceiling\nrefuses to overspend"),
]
for cx, cy, head, body in _chips:
    box(cx, cy - 1.2, cx + 36, cy + 6.0, "#FFFFFF", CHOKE_EDGE, lw=1.2, r=1.0)
    text(cx + 18, cy + 4.3, head, size=9.5, color=CHOKE_EDGE, weight="bold")
    text(cx + 18, cy + 1.1, body, size=8.5, color=MUTED)

# provider <-> choke point
arrow(88.0, 78.5, 88.0, 84.0, rad=0)      # request out
arrow(95.5, 84.0, 95.5, 78.5, rad=0)      # completion + usage back
label(122.0, 81.0, "token usage returned → Ledger", size=9.0)


# ================================================================ 3. TASKS (left rail)
box(2, 44.0, 30, 78.5, EVAL_FILL, EVAL_EDGE, lw=2.2)
mono(16, 75.4, "src/tasks.py", size=12.5, color=EVAL_EDGE)
text(16, 72.2, "GROUND TRUTH\nBY CONSTRUCTION", size=9.5, color=INK, weight="bold")
text(16, 66.6,
     "seeded · deterministic\nno model calls, ever",
     size=8.6, color=MUTED)
box(4, 46.0, 28, 63.4, "#FFFFFF", EVAL_EDGE, lw=1.1, r=1.0)
text(16, 61.4, "SOLVABLE", size=9.2, weight="bold", color=INK)
text(16, 59.1, "ceiling ≥ floor → deal in ZOPA", size=7.4, color=MUTED)
text(16, 56.3, "UNDERSPECIFIED", size=9.2, weight="bold", color=INK)
text(16, 54.0, "term withheld → ESCALATE", size=7.4, color=MUTED)
text(16, 51.2, "CONTRADICTORY", size=9.2, weight="bold", color=INK)
text(16, 48.9, "ceiling < floor → empty ZOPA\n→ ESCALATE", size=7.4, color=MUTED)

# tasks -> both systems (private briefs bus, down the right of the rail)
arrow(24, 44.0, 24, 40.0, rad=0, style="-")
arrow(24, 40.0, 33.0, 40.0, rad=0)                      # -> baseline
arrow(24, 40.0, 24, 26.5, rad=0, style="-")
arrow(24, 26.5, 83.0, 26.5, rad=0, style="-")
arrow(83.0, 26.5, 83.0, 34.0, rad=0)                    # -> society
label(28.5, 42.0, "private briefs", size=9.0, ha="left")

# tasks -> scorer (down the left rail)
arrow(8, 44.0, 8, 16.0, rad=0, style="-")
arrow(8, 16.0, 33.0, 16.0, rad=0)
label(8, 31.0, "ground truth", size=9.0, rot=90)


# ================================================================ 4. THE TWO SYSTEMS
# ---- Baseline
box(33, 36.0, 79, 58.0, BASE_FILL, BASE_EDGE, lw=2.2, ls=(0, (6, 3)))
text(56, 55.2, "BASELINE  (control)", size=11.5, weight="bold", color=INK)
box(38, 43.5, 74, 52.5, "#FFFFFF", BASE_EDGE, lw=1.3, r=1.0)
text(56, 49.6, "ONE AGENT", size=10.5, weight="bold", color=INK)
text(56, 46.2, "both briefs pooled\ninto a single context", size=9, color=MUTED)
text(56, 39.8,
     "no second opinion exists, so no\n"
     "disagreement can be measured —\n"
     "only self-reported confidence",
     size=8.4, color="#A03030", style="italic")

# ---- Society
box(83, 30.0, 158, 58.0, SOC_FILL, SOC_EDGE, lw=2.4)
text(120.5, 55.2, "AGENT SOCIETY", size=11.5, weight="bold", color=SOC_EDGE)
for ax0, name, sub in [
    (87.0, "BUYER", "holds ceiling\n(private)"),
    (108.5, "SELLER", "holds floor\n(private)"),
    (130.0, "AUDITOR", "checks feasibility\nof both claims"),
]:
    box(ax0, 43.5, ax0 + 20, 52.5, "#FFFFFF", SOC_EDGE, lw=1.3, r=1.0)
    text(ax0 + 10, 50.2, name, size=10.5, weight="bold", color=INK)
    text(ax0 + 10, 46.5, sub, size=8.4, color=MUTED)

# negotiation loop between buyer and seller
arrow(107.0, 51.0, 89.5, 51.0, rad=0.42, lw=1.5, color=SOC_EDGE, style="<|-|>")
label(98.2, 56.4, "negotiation rounds", size=8.8, color=SOC_EDGE, bg=SOC_FILL)

# each agent's position feeds the disagreement metric
for _x in (97.0, 118.5, 140.0):
    arrow(_x, 43.5, _x, 40.5, lw=1.4, color=SOC_EDGE)

# disagreement metric
box(87, 32.0, 154, 40.5, "#FFFFFF", SOC_EDGE, lw=1.8, r=1.0)
text(120.5, 38.0, "DISAGREEMENT METRIC", size=10.5, weight="bold", color=SOC_EDGE)
text(120.5, 34.4,
     "spread between standing offers  ·  auditor dissent  ·  rounds without convergence\n"
     "→ a scalar confidence signal a single agent structurally cannot produce",
     size=8.6, color=MUTED)

# both systems draw from the choke point
arrow(56, 62.0, 56, 58.0)
arrow(120.5, 62.0, 120.5, 58.0)


# ================================================================ 5. ESCALATION DECISION
box(87, 22.5, 154, 29.0, "#FFFFFF", HUMAN_EDGE, lw=1.8, r=1.0)
text(120.5, 25.7, "ESCALATION DECISION   —   threshold on measured disagreement",
     size=10, weight="bold", color=INK)
arrow(120.5, 32.0, 120.5, 29.0)

# two outcomes
box(87, 11.0, 118, 20.0, "#FFFFFF", SOC_EDGE, lw=1.6, r=1.0)
text(102.5, 17.2, "DEAL", size=10.5, weight="bold", color=SOC_EDGE)
text(102.5, 13.8, "agreed price, checked\nagainst ZOPA exactly", size=8.5, color=MUTED)

box(123, 11.0, 154, 20.0, HUMAN_FILL, HUMAN_EDGE, lw=1.8, r=1.0)
text(138.5, 17.2, "ESCALATE → HUMAN", size=10, weight="bold", color=HUMAN_EDGE)
text(138.5, 13.8, "\"we cannot justify a deal\"\nhuman-in-the-loop", size=8.5, color=MUTED)

arrow(105, 22.5, 102.5, 20.0, rad=0.0)
arrow(136, 22.5, 138.5, 20.0, rad=0.0)
label(94.0, 21.3, "low", size=8.6, color=SOC_EDGE)
label(147.0, 21.3, "high", size=8.6, color=HUMAN_EDGE)

# baseline outcome -> scorer
arrow(56, 36.0, 56, 20.5, rad=0)
label(58.5, 31.0, "deal, or a refusal backed\nonly by self-reported\nconfidence", size=8.8, ha="left")


# ================================================================ 6. SCORER
box(33, 11.0, 79, 20.5, EVAL_FILL, EVAL_EDGE, lw=2.2)
mono(56, 17.6, "src/scorer.py", size=12, color=EVAL_EDGE)
text(56, 14.6, "EXACT SCORING  ·  NO LLM-AS-JUDGE", size=9.6, weight="bold",
     color="#A03030")
text(56, 12.3, "numeric ZOPA check + escalate/no-escalate match", size=8.4, color=MUTED)

# society outcomes -> scorer
arrow(87, 15.5, 79, 15.5, rad=0)
arrow(123, 11.0, 120, 8.5, rad=0)
arrow(120, 8.5, 60, 8.5, rad=0, style="-")
arrow(60, 8.5, 60, 11.0, rad=0)


# ================================================================ 7. RESULTS
box(33, 1.0, 158, 7.0, "#FFFFFF", EVAL_EDGE, lw=1.8, r=1.0)
mono(37, 4.0, "src/benchmark.py", size=10.5, color=EVAL_EDGE, ha="left")
text(107, 4.6,
     "RESULTS TABLE   —   generated from the Ledger, never hand-written",
     size=9.6, weight="bold", color=INK)
text(107, 2.3,
     "success rate  |  escalation recall  |  false-escalation rate  |  hallucinated-deal rate  |  "
     "calibration AUC  |  tokens per SUCCESSFUL task",
     size=8.6, color=MUTED)


# ================================================================ FOOTNOTE
text(80, -2.4,
     "Controlled comparison: the baseline is the SAME code path with one agent — "
     "same client, same task loader, same scorer, same task set.\n"
     "Token counts and latency are exact (taken from the API response); "
     "cost is an estimate, per-token prices pending confirmation.",
     size=9.2, color=MUTED, ha="center", style="italic")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")
fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
print(f"wrote {out}")
