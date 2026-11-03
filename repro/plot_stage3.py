"""Render final, already-unblinded results in an isolated matplotlib environment."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".runtime-cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARTIFACTS = ROOT / "repro/artifacts"
LABELS = {"base": "Empty book", "ace": "ACE", "selective": "Selective reflection",
          "retrieval": "ACE + retrieval", "combined": "Selective + retrieval"}
COLORS = {"base": "#777777", "ace": "#235789", "selective": "#d38b22",
          "retrieval": "#278c75", "combined": "#a3446a"}


def main():
    report = json.loads((ARTIFACTS / "stage3_analysis.json").read_text())
    if report["status"] != "all_frozen_tests_scored" or report["logical_predictions"] != 1500:
        raise ValueError("Plot only the complete frozen test")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), gridspec_kw={"width_ratios": [1.05, 1]})
    for variant, group in report["groups"].items():
        x = group["lifecycle_standardized_cny_per_100"]
        y = group["accuracy_mean"] * 100
        axes[0].errorbar(x, y, yerr=group["accuracy_sd"] * 100, color=COLORS[variant],
                         fmt="o", markersize=7, capsize=4, label=LABELS[variant])
        axes[0].annotate(f"{y:.1f}%", (x, y), xytext=(6, 7), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="Learning + 100 answers: standardized cost (CNY)", ylabel="Accuracy (%)",
                title="Accuracy and complete usage cost")
    axes[0].set_xlim(left=0)
    axes[0].grid(alpha=.17)
    axes[0].legend(loc="best", fontsize=8.5, frameon=False)
    names = ["combined_minus_ace", "selective_minus_ace", "retrieval_minus_ace", "ace_minus_base"]
    labels = ["Combined - ACE (primary)", "Selective - ACE", "Retrieval - ACE", "ACE - empty book"]
    for y, name in enumerate(names):
        result = report["comparisons"][name]
        effect = result["delta_percentage_points"]
        lo, hi = result["ci95_percentage_points"]
        axes[1].errorbar(effect, y, xerr=[[effect - lo], [hi - effect]], fmt="o", capsize=5,
                         color=COLORS["combined"] if y == 0 else "#45515f", linewidth=2 if y == 0 else 1.3)
        axes[1].annotate(f"{effect:+.2f} [{lo:+.2f}, {hi:+.2f}]", (hi, y), xytext=(7, 0),
                         textcoords="offset points", va="center", fontsize=8.5)
    axes[1].axvline(0, color="#777777", linewidth=1)
    axes[1].axvline(-2, color="#bbbbbb", linewidth=1, linestyle="--", label="-2 pp point-estimate goal")
    axes[1].set_yticks(range(len(names)), labels)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Paired accuracy difference (percentage points)",
                title="Question-cluster bootstrap 95% intervals")
    left, right = axes[1].get_xlim()
    axes[1].set_xlim(left - .5, right + max(5, (right - left) * .7))
    axes[1].grid(axis="x", alpha=.17)
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("ACE Formula: 100 held-out questions, three frozen training orders", fontsize=13)
    fig.text(.5, .025, "Left: mean +/- sample SD across 3 seeds. Right: 10,000 paired question-cluster resamples.\n"
             "Cost uses all-cache-miss prices; intervals condition on these trained books. "
             "No formal noninferiority test.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .09, 1, .95))
    for extension in ("png", "svg"):
        fig.savefig(ARTIFACTS / f"stage3_accuracy_cost.{extension}", dpi=180, bbox_inches="tight")
    versions = {"python": platform.python_version(), "purpose": "Plotting only; original uv.lock unchanged",
                "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}}
    (ARTIFACTS / "stage3_plot_environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
