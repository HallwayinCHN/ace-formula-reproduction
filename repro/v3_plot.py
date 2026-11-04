"""Plot completed V3 results only, using the isolated plotting environment."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "repro/artifacts"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".runtime-cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    completion = json.loads((ARTIFACTS / "v3_completion.json").read_text())
    if not completion["status"].startswith("completed_"):
        raise ValueError("Require completed conditional V3 protocol")
    training = json.loads((ARTIFACTS / "v3_training_analysis.json").read_text())
    router = json.loads((ARTIFACTS / "v3_router_lock.json").read_text())
    groups = training["groups"]
    original, nodiag = groups["original"], groups["nodiag"]
    colors = ["#63788d", "#227d69"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for pair in training["pairs"]:
        axes[0].plot([0, 1], [pair[v]["learning_accounted_cny"] for v in ("original", "nodiag")],
                     marker="o", alpha=.8, label=f"Seed {pair['seed']}")
    axes[0].set_xticks([0, 1], ["Original", "No diagnostics"])
    axes[0].set(ylabel="Learning cost estimate (CNY)", title="Paired fresh live training", ylim=(0, None))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar([0, 1], [original["retained_learning_calls"], nodiag["retained_learning_calls"]],
                color=colors, label="Retained learning")
    axes[1].bar([0, 1], [original["diagnostic_calls"], nodiag["diagnostic_calls"]],
                bottom=[original["retained_learning_calls"], nodiag["retained_learning_calls"]],
                color="#d4a45a", label="Measurement only")
    axes[1].set_xticks([0, 1], ["Original", "No diagnostics"])
    axes[1].set(ylabel="Calls across 3 seeds", title="Learning calls by purpose")
    axes[1].legend(frameon=False, fontsize=8)
    if completion["routing_gate_passed"]:
        data = completion["evaluation"]
        for name, group in data["groups"].items():
            x, y = group["accounted_cny_per_100"], group["accuracy"]*100
            axes[2].scatter(x, y, s=55)
            axes[2].annotate(name, (x, y), xytext=(4, 5), textcoords="offset points")
        axes[2].set(xlabel="Inference cost / 100 predictions (CNY)", ylabel="Accuracy (%)",
                    title="160-question project evaluation")
    else:
        candidates = router["candidates"]
        savings = [item["cv"]["estimated_cost_reduction_percent"] for item in candidates]
        bars = axes[2].bar(range(3), savings, color="#63788d", width=.5)
        axes[2].axhline(15, color="#b25140", linestyle="--", label="15% cost gate")
        for bar, item in zip(bars, candidates):
            axes[2].annotate(f"{bar.get_height():.1f}%\n{item['cv']['delta_pp']:+.2f} pp accuracy",
                             (bar.get_x()+bar.get_width()/2, bar.get_height()), ha="center",
                             xytext=(0, 5), textcoords="offset points", fontsize=8)
        axes[2].set_xticks(range(3), [f"Support {item['support']}" for item in candidates])
        axes[2].set(ylabel="Estimated CV cost reduction (%)", title="Routing gate failed; 160 questions sealed",
                    ylim=(min(0, min(savings)-8), max(30, max(savings)+12)))
        axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("ACE V3: conservative output routing and diagnostic-free execution", fontsize=13)
    fig.text(.5, .025, "Fresh retained responses and cache history can differ across paired runs. "
             "Cost estimates are not reconciled bills.\n"
             "Training validation selects books and is not an independent accuracy test.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .94))
    for extension in ("png", "svg"):
        fig.savefig(ARTIFACTS / f"v3_results.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
