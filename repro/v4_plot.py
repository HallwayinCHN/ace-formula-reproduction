"""Standalone research figure from the completed V4 evaluation; isolated plotting environment."""

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
    data = json.loads((ARTIFACTS / "v4_analysis.json").read_text())
    completion = json.loads((ARTIFACTS / "v4_completion.json").read_text())
    if data["predictions"] != 960 or completion["status"] != "complete":
        raise ValueError("Require completed V4 protocol and prompt audit")
    groups = data["groups"]
    original, nodiag = groups["original"], groups["nodiag"]
    training = json.loads((ARTIFACTS / "v3_training_analysis.json").read_text())["groups"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.1))
    colors = ["#63788d", "#227d69"]
    bars = axes[0].bar([0, 1], [original["accuracy"]*100, nodiag["accuracy"]*100],
                       yerr=[original["accuracy_sd_across_three_books"]*100,
                             nodiag["accuracy_sd_across_three_books"]*100], capsize=5, color=colors, width=.55)
    for bar, group in zip(bars, (original, nodiag)):
        axes[0].annotate(f"{group['accuracy']*100:.2f}%", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                         ha="center", xytext=(0, 15), textcoords="offset points")
    axes[0].set_xticks([0, 1], ["Original", "No diagnostics"])
    axes[0].set(ylabel="Accuracy (%)", ylim=(0, 105), title="160-question project holdout")
    delta = data["primary_comparison"]["delta_percentage_points"]
    lower, upper = data["primary_comparison"]["ci95_percentage_points"]
    axes[1].scatter(data["seed_delta_pp"], [0, 1, 2], color="#63788d", s=40)
    axes[1].errorbar([delta], [3], xerr=[[max(0, delta-lower)], [max(0, upper-delta)]],
                     fmt="o", capsize=5, color="#227d69")
    axes[1].axvline(0, color="#777777", linewidth=1)
    axes[1].axvline(-2, color="#b25140", linestyle="--", label="-2 pp lower-bound target")
    axes[1].set_yticks([0, 1, 2, 3], ["Seed 42", "Seed 43", "Seed 44", "Pooled + 95% CI"])
    axes[1].set(xlabel="No diagnostics minus original (pp)", title="Paired accuracy differences", ylim=(-.6, 3.7))
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    ratios = [training["nodiag"]["learning_accounted_cny"]/training["original"]["learning_accounted_cny"]*100,
              nodiag["inference_accounted_cny_per_100"]/original["inference_accounted_cny_per_100"]*100,
              nodiag["assembled_cost_three_books_plus_160_answers_each_cny"]
              /original["assembled_cost_three_books_plus_160_answers_each_cny"]*100]
    bars = axes[2].bar(range(3), ratios, color="#227d69", width=.55)
    axes[2].axhline(100, color="#63788d", linestyle="--", label="Original = 100%")
    for index, value in enumerate(ratios):
        axes[2].text(index, value+3, f"{value:.1f}%", ha="center")
    axes[2].set_xticks(range(3), ["V3 learning", "V4 inference", "Training +\n160 answers/book"])
    axes[2].set(ylabel="Relative cached cost estimate (%)", ylim=(0, max(125, max(ratios)+15)),
                title="Costs: separate stages and assembled total")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("ACE V4: accuracy audit of six preselected training books", fontsize=13)
    verdict = "met" if data["operational_accuracy_target_passed"] else "not met"
    fig.text(.5, .025, f"Registered accuracy target {verdict}. Paired difference {delta:+.2f} pp; "
             f"question-cluster 95% CI [{lower:+.2f}, {upper:+.2f}] pp.\n"
             "Accuracy bars: mean +/- SD over 3 fixed books. No formal noninferiority or causal guarantee.\n"
             "Costs are estimates; the total assembles historical training and fresh inference. "
             "Prior bulk data parsing is disclosed in the report.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, .94))
    for extension in ("png", "svg"):
        path = ARTIFACTS / f"v4_results.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
