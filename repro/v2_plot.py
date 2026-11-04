"""Standalone figure from complete prospective results, using the isolated plot environment."""

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
    data = json.loads((ARTIFACTS / "v2_final_analysis.json").read_text())
    if data["status"] != "prospective_test_complete" or data["prediction_rows"] != 600:
        raise ValueError("Require a completed prospective test")
    a, b = data["groups"]["original"], data["groups"][data["candidate"]]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2))
    bars = axes[0].bar([0, 1], [a["accuracy"]*100, b["accuracy"]*100],
                       yerr=[a["accuracy_sd"]*100, b["accuracy_sd"]*100], capsize=5,
                       color=["#63788d", "#227d69"], width=.5)
    for bar, group in zip(bars, (a, b)):
        axes[0].annotate(f"{group['accuracy']*100:.2f}%", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                         xytext=(0, 12), textcoords="offset points", ha="center")
    axes[0].set_xticks([0, 1], ["Original ACE", "Full book + brief output\n+ parse-failure fallback"])
    axes[0].set(ylim=(0, 105), ylabel="Accuracy (%)", title="Fresh 100-question test")
    metrics = [("Output tokens", "output_tokens"), ("Cached cost estimate", "accounted_cny_per_100"),
               ("Median API time", "median_api_seconds")]
    ratios = [b[key]/a[key]*100 for _, key in metrics]
    bars = axes[1].bar(range(3), ratios, color="#227d69", width=.55)
    axes[1].axhline(100, color="#63788d", linestyle="--", label="Original ACE = 100%")
    for index, value in enumerate(ratios):
        axes[1].text(index, value+3, f"{value:.1f}%", ha="center")
    axes[1].set_xticks(range(3), [name.replace(" ", "\n", 1) for name, _ in metrics])
    axes[1].set(ylim=(0, max(120, max(ratios)+15)), ylabel="Relative resource use (%)",
                title="Inference resources (all fallback calls counted)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    result = data["primary_comparison"]
    lo, hi = result["ci95_percentage_points"]
    fig.suptitle("Accuracy-first ACE: preserve the complete trained playbook", fontsize=13)
    fig.text(.5, .04, f"Paired accuracy difference: {result['delta_percentage_points']:+.2f} pp; "
             f"question-cluster bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}] pp.\n"
             "Accuracy bars: mean +/- SD across 3 books. Costs are estimates, not reconciled bills. "
             "No formal noninferiority claim.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .94))
    for extension in ("png", "svg"):
        fig.savefig(ARTIFACTS / f"v2_accuracy_efficiency.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
