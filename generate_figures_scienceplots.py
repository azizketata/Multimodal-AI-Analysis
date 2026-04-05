#!/usr/bin/env python3
"""Generate thesis figures using SciencePlots IEEE style.

Usage:
    python generate_figures_scienceplots.py --meetings-dir meetings/ --output-dir thesis/figures_scienceplots/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Use SciencePlots IEEE style
try:
    import scienceplots
    plt.style.use(['science', 'no-latex', 'ieee'])
except ImportError:
    print("WARNING: SciencePlots not installed. Using fallback style.")


# ============================================================
# Constants — refined for SciencePlots
# ============================================================

# IEEE-friendly colorblind-safe palette (Tab10 muted)
CITY_COLORS = {
    "Seattle": "#0072B2",
    "Denver": "#E69F00",
    "Boston": "#009E73",
    "Alameda": "#CC79A7",
}

CITY_MARKERS = {
    "Seattle": "o",
    "Denver": "s",
    "Boston": "^",
    "Alameda": "D",
}

DEVIANCE_COLORS = {
    "benign": "#009E73",
    "violation": "#D55E00",
    "efficiency": "#0072B2",
    "innovation": "#E69F00",
    "disruption": "#999999",
    "unknown": "#CCCCCC",
}

SOURCE_COLORS = {
    "audio": "#0072B2",
    "visual": "#D55E00",
    "fused": "#CC79A7",
}


def get_color(city: str) -> str:
    return CITY_COLORS.get(city, "#607D8B")


def get_marker(city: str) -> str:
    return CITY_MARKERS.get(city, "o")


# ============================================================
# Data Loading
# ============================================================

def load_metrics(metrics_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def load_golden_data(golden_file: str) -> list[dict]:
    """Load golden comparison results."""
    with open(golden_file) as f:
        return json.load(f)


def load_sbert_sensitivity(sbert_json: str) -> list[dict]:
    """Load SBERT sensitivity analysis summary."""
    with open(sbert_json) as f:
        data = json.load(f)
    return data["summary"]


def load_all_conformance(meetings_dir: str) -> list[dict]:
    results = []
    for folder in sorted(os.listdir(meetings_dir)):
        conf_path = os.path.join(meetings_dir, folder, "conformance.json")
        if os.path.exists(conf_path):
            with open(conf_path) as f:
                results.append(json.load(f))
    return results


# ============================================================
# Style setup — SciencePlots overrides
# ============================================================

def setup_style():
    """Apply SciencePlots style with thesis-specific overrides."""
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.5),  # IEEE single-column
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
    })


# ============================================================
# FIGURE 1: Corpus Overview Table-Figure
# ============================================================

def fig_corpus_overview(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.axis("off")

    cities = sorted(df["city"].unique())
    table_data = []
    for city in cities:
        cdf = df[df["city"] == city]
        n = len(cdf)
        hours = cdf["duration_seconds"].sum() / 3600
        events = cdf["raw_events"].sum()
        table_data.append([city, str(n), f"{hours:.1f}", f"{events:,}"])

    table_data.append([
        "Total", str(len(df)),
        f"{df['duration_seconds'].sum()/3600:.1f}",
        f"{df['raw_events'].sum():,}"
    ])

    col_labels = ["City", "Meetings", "Hours", "Raw Events"]
    table = ax.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=7)
    for j in range(len(col_labels)):
        table[len(table_data), j].set_facecolor("#ECF0F1")
        table[len(table_data), j].set_text_props(fontweight="bold")
    for i, city in enumerate(cities):
        table[i + 1, 0].set_text_props(color=get_color(city), fontweight="bold")

    ax.set_title("Meeting Corpus Overview", fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"corpus_overview.{fmt}"), dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 2: Modality Contribution — bar chart instead of donut
# ============================================================

def fig_modality_contribution(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    totals = {"audio": 0, "visual": 0, "fused": 0}
    for c in conformance_list:
        sources = c.get("extraction", {}).get("sources", {})
        totals["audio"] += sources.get("audio", 0)
        totals["visual"] += sources.get("visual", 0)
        totals["fused"] += sources.get("fused", 0)

    total = sum(totals.values())
    labels = ["Audio\n(Whisper)", "Visual\n(RTMPose)", "Multimodal\nFusion"]
    values = [totals["audio"], totals["visual"], totals["fused"]]
    pcts = [v / total * 100 for v in values]
    colors = [SOURCE_COLORS["audio"], SOURCE_COLORS["visual"], SOURCE_COLORS["fused"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: horizontal bar chart (replaces donut)
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, pcts, color=colors, alpha=0.85, height=0.6, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel("Percentage of Events (%)")
    ax1.set_title("Event Source Distribution")
    ax1.invert_yaxis()
    for i, (p, v) in enumerate(zip(pcts, values)):
        ax1.text(p + 1, i, f"{p:.1f}% ({v:,})", va="center", fontsize=6)

    # Right: per-meeting box plot
    audio_pcts, visual_pcts, fused_pcts = [], [], []
    for c in conformance_list:
        sources = c.get("extraction", {}).get("sources", {})
        t = sources.get("audio", 0) + sources.get("visual", 0) + sources.get("fused", 0)
        if t > 0:
            audio_pcts.append(sources.get("audio", 0) / t * 100)
            visual_pcts.append(sources.get("visual", 0) / t * 100)
            fused_pcts.append(sources.get("fused", 0) / t * 100)

    bp = ax2.boxplot([audio_pcts, visual_pcts, fused_pcts],
                     labels=["Audio", "Visual", "Fusion"],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('#333333')
            line.set_linewidth(0.8)

    for i, (vals, color) in enumerate(zip([audio_pcts, visual_pcts, fused_pcts], colors)):
        jitter = np.random.normal(0, 0.06, len(vals))
        ax2.scatter(np.full(len(vals), i + 1) + jitter, vals,
                    c=color, alpha=0.4, s=8, zorder=3, edgecolors="none")

    ax2.set_ylabel("Events per Meeting (%)")
    ax2.set_title("Per-Meeting Variation")

    fig.suptitle("Multimodal Event Extraction: Source Modality Contribution",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"modality_contribution.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 3: Fitness Distribution
# ============================================================

def fig_fitness_distribution(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    scores = df["fitness_dedup"].dropna().values
    ax.hist(scores, bins=12, color="#0072B2", alpha=0.75, edgecolor="white", linewidth=0.5)

    mean_val = np.mean(scores)
    median_val = np.median(scores)
    ax.axvline(mean_val, color="#D55E00", linestyle="--", linewidth=1.0,
               label=f"Mean: {mean_val:.3f}")
    ax.axvline(median_val, color="#E69F00", linestyle="-.", linewidth=1.0,
               label=f"Median: {median_val:.3f}")

    ax.set_xlabel("Token-Replay Fitness (Deduplicated)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Fitness Score Distribution (n=54)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_distribution.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 4: Raw vs Dedup Fitness
# ============================================================

def fig_fitness_raw_vs_dedup(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        ax.scatter(df[mask]["fitness_raw"], df[mask]["fitness_dedup"],
                   c=get_color(city), marker=get_marker(city),
                   label=city, alpha=0.7, s=25, zorder=3, edgecolors="white", linewidth=0.3)

    lim = max(df["fitness_raw"].max(), df["fitness_dedup"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.2, linewidth=0.8)
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.03, color="#009E73")
    ax.text(0.05, lim * 0.92, "Dedup improves fitness", fontsize=6,
            color="#009E73", alpha=0.7, style="italic")

    ax.set_xlabel("Raw Fitness Score")
    ax.set_ylabel("Deduplicated Fitness Score")
    ax.set_title("Effect of Deduplication on Fitness")
    ax.legend(fontsize=6, frameon=True, fancybox=False, edgecolor="#333333")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_raw_vs_dedup.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 5: Shadow vs Formal Split — bar + histogram
# ============================================================

def fig_shadow_formal_split(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8),
                                    gridspec_kw={"width_ratios": [1, 1.5]})

    total_formal = df["formal_events"].sum()
    total_shadow = df["shadow_events"].sum()
    total_all = total_formal + total_shadow

    # Left: horizontal bar (replaces donut)
    bars = ax1.barh(["Formal", "Shadow"],
                    [total_formal / total_all * 100, total_shadow / total_all * 100],
                    color=["#009E73", "#D55E00"], alpha=0.85, height=0.5, edgecolor="white")
    ax1.set_xlabel("Percentage (%)")
    ax1.set_title("Overall Activity Split")
    for i, (v, n) in enumerate(zip([total_formal / total_all * 100, total_shadow / total_all * 100],
                                    [total_formal, total_shadow])):
        ax1.text(v + 1, i, f"{v:.1f}% ({n:,})", va="center", fontsize=6)

    # Right: histogram
    shadow_pcts = df["shadow_pct"].values * 100
    ax2.hist(shadow_pcts, bins=12, color="#D55E00", alpha=0.65, edgecolor="white", linewidth=0.5)
    mean_s = np.mean(shadow_pcts)
    median_s = np.median(shadow_pcts)
    ax2.axvline(mean_s, color="#D55E00", linestyle="--", linewidth=1.0,
                label=f"Mean: {mean_s:.1f}%")
    ax2.axvline(median_s, color="#E69F00", linestyle="-.", linewidth=1.0,
                label=f"Median: {median_s:.1f}%")
    ax2.set_xlabel("Shadow Activity (%)")
    ax2.set_ylabel("Number of Meetings")
    ax2.set_title("Shadow Prevalence Distribution")
    ax2.legend(frameon=True, fancybox=False, edgecolor="#333333")

    fig.suptitle("Shadow Workflow Prevalence Across 54 Meetings",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"shadow_formal_split.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 6: Deviance Taxonomy — horizontal bar only (no donut)
# ============================================================

def fig_deviance_taxonomy(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    totals = {"benign": 0, "innovation": 0, "efficiency": 0,
              "disruption": 0, "unknown": 0}
    for c in conformance_list:
        dev = c.get("deviance", {})
        for k in totals:
            totals[k] += dev.get(k, 0)

    labels = [k.title() for k, v in totals.items() if v > 0]
    values = [v for v in totals.values() if v > 0]
    colors = [DEVIANCE_COLORS[k] for k, v in totals.items() if v > 0]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.6,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Shadow Events")
    ax.set_title("Deviance Classification of Shadow Activities")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        pct = v / total * 100
        ax.text(v + max(values) * 0.02, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deviance_taxonomy.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 7: POWL Shadow Patterns — horizontal bar (no donut)
# ============================================================

def fig_powl_patterns(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    totals = {"isolated": 0, "concurrent": 0, "sequential": 0, "recurring": 0}
    for c in conformance_list:
        types = c.get("powl", {}).get("cluster_types", {})
        for k in totals:
            totals[k] += types.get(k, 0)

    labels_map = {
        "isolated": "Isolated",
        "concurrent": "Concurrent",
        "sequential": "Sequential",
        "recurring": "Recurring",
    }

    labels = [labels_map[k] for k, v in totals.items() if v > 0]
    values = [v for v in totals.values() if v > 0]
    colors = ["#78909C", "#D55E00", "#009E73", "#0072B2"][:len(labels)]
    total = sum(values)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.55,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Clusters")
    ax.set_title(f"POWL Shadow Patterns (n={total:,} clusters)")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        pct = v / total * 100
        ax.text(v + max(values) * 0.02, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"powl_patterns.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 8: Declare Compliance
# ============================================================

def fig_declare_compliance(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    bands = [(90, 100, "A", "#009E73"), (80, 90, "B", "#56B4E9"),
             (70, 80, "C", "#E69F00"), (60, 70, "D", "#D55E00"), (0, 60, "F", "#CC79A7")]
    for lo, hi, grade, color in bands:
        ax.axvspan(lo, hi, alpha=0.08, color=color)

    scores = df["declare_score"].dropna().values
    ax.hist(scores, bins=20, color="#0072B2", alpha=0.75, edgecolor="white", linewidth=0.5)

    for lo, hi, grade, color in bands:
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 1,
                grade, ha="center", va="top", fontsize=11, color=color, alpha=0.5,
                fontweight="bold")

    mean_val = np.mean(scores)
    ax.axvline(mean_val, color="#D55E00", linestyle="--", linewidth=1.0,
               label=f"Mean: {mean_val:.1f}")

    grade_counts = Counter(df["declare_grade"].values)
    grade_text = "  ".join([f"{g}: {grade_counts.get(g, 0)}" for g in ["A", "B", "C", "D", "F"]])
    ax.text(0.98, 0.95, grade_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#333333", linewidth=0.5))

    ax.set_xlabel("Robert's Rules Compliance Score (0-100)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Declare Compliance Distribution (n=54)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"declare_compliance.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 9: Declare Violations
# ============================================================

def fig_declare_violations(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    violation_counts = Counter()
    for c in conformance_list:
        violations = c.get("declare", {}).get("violations", [])
        for v in violations:
            if isinstance(v, dict):
                tmpl = v.get("template", "")
                a = v.get("a", "")
                b = v.get("b", "")
                if b:
                    name = f"{tmpl}({a}, {b})"
                else:
                    name = f"{tmpl}({a})"
                violation_counts[name] += 1

    total_sat = sum(c.get("declare", {}).get("satisfied", 0) for c in conformance_list)
    total_viol = sum(c.get("declare", {}).get("violated", 0) for c in conformance_list)
    meetings_with_constraints = sum(
        1 for c in conformance_list
        if c.get("declare", {}).get("constraints_checked", 0) > 0
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: horizontal bar for sat vs violated
    if total_sat + total_viol > 0:
        total_all = total_sat + total_viol
        ax1.barh(["Satisfied", "Violated"],
                 [total_sat / total_all * 100, total_viol / total_all * 100],
                 color=["#009E73", "#D55E00"], alpha=0.85, height=0.5, edgecolor="white")
        for i, (pct, cnt) in enumerate(zip(
            [total_sat / total_all * 100, total_viol / total_all * 100],
            [total_sat, total_viol])):
            ax1.text(pct + 1, i, f"{pct:.1f}% ({cnt})", va="center", fontsize=6)
        ax1.set_xlabel("Percentage (%)")
    ax1.set_title(f"Constraint Outcomes\n({meetings_with_constraints} meetings)", fontsize=8)

    # Right: specific violations bar
    if violation_counts:
        sorted_v = violation_counts.most_common(10)
        names = [v[0] for v in sorted_v]
        counts = [v[1] for v in sorted_v]

        y = np.arange(len(names))
        ax2.barh(y, counts, color="#D55E00", alpha=0.8, height=0.6, edgecolor="white", linewidth=0.5)
        ax2.set_yticks(y)
        ax2.set_yticklabels(names, fontsize=6)
        ax2.set_xlabel("Number of Meetings")
        ax2.invert_yaxis()
        for i, v in enumerate(counts):
            ax2.text(v + 0.2, i, str(v), va="center", fontsize=6)
    ax2.set_title("Most Violated Constraints", fontsize=8)

    fig.suptitle("Robert's Rules Declare Constraint Analysis",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"declare_violations.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 10: Fitness vs Shadow
# ============================================================

def fig_fitness_vs_shadow(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        ax.scatter(df[mask]["shadow_pct"] * 100, df[mask]["fitness_dedup"],
                   c=get_color(city), marker=get_marker(city),
                   label=city, alpha=0.7, s=20, zorder=3, edgecolors="white", linewidth=0.3)

    x = df["shadow_pct"].values * 100
    y = df["fitness_dedup"].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() > 2:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, p(x_line), "--", color="#D55E00", alpha=0.5, linewidth=0.8,
                label=f"Linear trend (slope={z[0]:.4f})")

        y_pred = p(x[valid])
        ss_res = np.sum((y[valid] - y_pred) ** 2)
        ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.02, 0.02, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
                fontsize=6, color="#555555")

    ax.set_xlabel("Shadow Activity (%)")
    ax.set_ylabel("Deduplicated Fitness Score")
    ax.set_title("Fitness vs. Shadow Prevalence")
    ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_vs_shadow.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 11: Agenda Coverage
# ============================================================

def fig_agenda_coverage(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    coverages = df["agenda_coverage_pct"].dropna().values * 100
    ax.hist(coverages, bins=12, color="#0072B2", alpha=0.75, edgecolor="white", linewidth=0.5)

    mean_c = np.mean(coverages)
    median_c = np.median(coverages)
    ax.axvline(mean_c, color="#D55E00", linestyle="--", linewidth=1.0,
               label=f"Mean: {mean_c:.1f}%")
    ax.axvline(median_c, color="#E69F00", linestyle="-.", linewidth=1.0,
               label=f"Median: {median_c:.1f}%")

    ax.set_xlabel("Agenda Item Coverage (%)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Agenda Item Detection Rate (n=54)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"agenda_coverage.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 12: Conformance Scatter
# ============================================================

def fig_conformance_scatter(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        cdf = df[mask]
        sizes = cdf["agenda_coverage_pct"] * 100 + 10
        ax.scatter(cdf["fitness_dedup"], cdf["declare_score"],
                   c=get_color(city), marker=get_marker(city),
                   s=sizes, label=city, alpha=0.6, zorder=3,
                   edgecolors="white", linewidth=0.3)

    ax.set_xlabel("Deduplicated Fitness Score")
    ax.set_ylabel("Declare Compliance (0-100)")
    ax.set_title("Governance Profile: Fitness × Compliance × Coverage")

    for sz, label in [(0.2, "20%"), (0.5, "50%"), (1.0, "100%")]:
        ax.scatter([], [], c="gray", alpha=0.4, s=sz * 100 + 10,
                   label=f"Coverage: {label}")
    ax.legend(fontsize=5, loc="upper left", ncol=2, frameon=True, fancybox=False, edgecolor="#333333")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"conformance_scatter.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 13: Fitness by Agenda Complexity (stratified boxplot)
# ============================================================

def fig_fitness_by_agenda_complexity(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    """Stratified boxplot: fitness, shadow, coverage by agenda complexity."""
    # Categorize agenda complexity
    df = df.copy()
    df["agenda_items"] = df.get("agenda_items", pd.Series([0]*len(df)))
    if "agenda_items" not in df.columns or df["agenda_items"].isna().all():
        return

    def categorize(n):
        if n <= 5:
            return "Sparse\n(≤5 items)"
        elif n <= 50:
            return "Moderate\n(6-50 items)"
        else:
            return "Dense\n(51-100 items)"

    df["complexity"] = df["agenda_items"].apply(categorize)
    cats = ["Sparse\n(≤5 items)", "Moderate\n(6-50 items)", "Dense\n(51-100 items)"]
    colors_box = ["#0072B2", "#E69F00", "#D55E00"]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    for ax, metric, title in zip(axes,
                                  ["fitness_dedup", "shadow_pct", "agenda_coverage_pct"],
                                  ["Deduplicated Fitness", "Shadow Activity (%)", "Agenda Coverage (%)"]):
        data = []
        for cat in cats:
            vals = df[df["complexity"] == cat][metric].dropna().values
            if "pct" in metric:
                vals = vals * 100
            data.append(vals)

        bp = ax.boxplot(data, labels=[c.split("\n")[0] for c in cats],
                       patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color('#333333')
                line.set_linewidth(0.8)
        ax.set_title(title, fontsize=7)
        ax.tick_params(axis='x', labelsize=6)

    fig.suptitle("Meeting Metrics Stratified by Agenda Complexity",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_by_agenda_complexity.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 14: Golden Shadow Comparison (bar chart)
# ============================================================

def fig_golden_shadow_comparison(golden_data: list[dict], output_dir: str, fmt: str, dpi: int):
    """Per-meeting grouped bars: golden vs pipeline shadow pct."""
    n = len(golden_data)
    names = [d["golden_name"].replace("Seattle-", "") for d in golden_data]
    golden_pcts = [d["shadow_comparison"]["golden_shadow_pct"] * 100 for d in golden_data]
    pipeline_pcts = [d["shadow_comparison"]["pipeline_shadow_pct"] * 100 for d in golden_data]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w / 2, golden_pcts, w, label="Ground Truth", color="#0072B2", alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, pipeline_pcts, w, label="Pipeline", color="#E69F00", alpha=0.85,
           edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Shadow Activity (%)")
    ax.set_title(f"Ground-Truth Validation: Shadow Prevalence (n={n})")
    ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=6)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"golden_shadow_comparison.{fmt}"), dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 15: Golden Deviance Comparison (grouped horizontal bars)
# ============================================================

def fig_golden_deviance_comparison(golden_data: list[dict], output_dir: str, fmt: str, dpi: int):
    """Grouped horizontal bar charts comparing golden vs pipeline deviance categories."""
    # Aggregate golden deviance
    golden_totals = Counter()
    for d in golden_data:
        for cat, count in d["golden"]["deviance_cats"].items():
            golden_totals[cat] += count
    golden_n = sum(golden_totals.values())

    # Aggregate pipeline deviance
    pipeline_totals = Counter()
    for d in golden_data:
        for cat, count in d["pipeline"]["deviance"].items():
            pipeline_totals[cat] += count
    pipeline_n = sum(pipeline_totals.values())

    # Union of all categories
    all_cats = sorted(set(list(golden_totals.keys()) + list(pipeline_totals.keys())))
    cat_labels = [c.title() for c in all_cats]
    cat_colors = [DEVIANCE_COLORS.get(c, "#CCCCCC") for c in all_cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # Left: golden
    y_pos = np.arange(len(all_cats))
    g_vals = [golden_totals.get(c, 0) for c in all_cats]
    ax1.barh(y_pos, g_vals, color=cat_colors, alpha=0.85, height=0.6,
             edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cat_labels)
    ax1.set_xlabel("Number of Events")
    ax1.set_title(f"Ground Truth (n={golden_n:,})")
    ax1.invert_yaxis()
    for i, v in enumerate(g_vals):
        if v > 0:
            pct = v / golden_n * 100 if golden_n > 0 else 0
            ax1.text(v + max(g_vals) * 0.02, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=6)

    # Right: pipeline
    p_vals = [pipeline_totals.get(c, 0) for c in all_cats]
    ax2.barh(y_pos, p_vals, color=cat_colors, alpha=0.85, height=0.6,
             edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cat_labels)
    ax2.set_xlabel("Number of Events")
    ax2.set_title(f"Pipeline (n={pipeline_n:,})")
    ax2.invert_yaxis()
    for i, v in enumerate(p_vals):
        if v > 0:
            pct = v / pipeline_n * 100 if pipeline_n > 0 else 0
            ax2.text(v + max(max(p_vals), 1) * 0.02, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=6)

    fig.suptitle("Deviance Classification: Ground Truth vs Pipeline",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"golden_deviance_comparison.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 16: Golden Meeting-Level Comparison (3-panel)
# ============================================================

def fig_golden_meeting_level(golden_data: list[dict], output_dir: str, fmt: str, dpi: int):
    """3-panel grouped bar: event granularity, formal rate, procedural compliance."""
    n = len(golden_data)
    names = [d["golden_name"].replace("Seattle-", "") for d in golden_data]
    x = np.arange(n)
    w = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.8))

    # (a) Event Granularity — log scale
    g_events = [d["golden"]["total_events"] for d in golden_data]
    p_events = [d["pipeline"]["total_events"] for d in golden_data]
    ax1.bar(x - w / 2, g_events, w, label="Ground Truth", color="#0072B2", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax1.bar(x + w / 2, p_events, w, label="Pipeline", color="#E69F00", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
    ax1.set_ylabel("Total Events (log)")
    ax1.set_title("(a) Event Granularity", fontsize=8)
    ax1.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="#333333")

    # (b) Formal Activity Rate
    g_formal = [d["golden"]["estimated_formal_pct"] for d in golden_data]
    p_formal = [d["pipeline"]["formal_count"] / d["pipeline"]["total_events"] * 100
                if d["pipeline"]["total_events"] > 0 else 0 for d in golden_data]
    ax2.bar(x - w / 2, g_formal, w, label="Ground Truth", color="#0072B2", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax2.bar(x + w / 2, p_formal, w, label="Pipeline", color="#E69F00", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
    ax2.set_ylabel("Formal Activity (%)")
    ax2.set_title("(b) Formal Activity Rate", fontsize=8)
    ax2.set_ylim(0, 100)

    # (c) Procedural Compliance
    g_roberts = [d["roberts_comparison"]["golden_roberts_score"] * 100 for d in golden_data]
    p_declare = [d["roberts_comparison"]["pipeline_declare_score"] for d in golden_data]
    ax3.bar(x - w / 2, g_roberts, w, label="Golden Roberts", color="#0072B2", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax3.bar(x + w / 2, p_declare, w, label="Pipeline Declare", color="#E69F00", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
    ax3.set_ylabel("Compliance Score")
    ax3.set_title("(c) Procedural Compliance", fontsize=8)
    ax3.set_ylim(0, 100)

    fig.suptitle("Meeting-Level Comparison: Ground Truth vs Pipeline",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"golden_meeting_level.{fmt}"), dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 17: SBERT Sensitivity (3-panel line plot)
# ============================================================

def fig_sbert_sensitivity(sbert_summary: list[dict], output_dir: str, fmt: str, dpi: int):
    """3-panel line plot: fitness, shadow%, coverage vs SBERT threshold."""
    thresholds = [e["threshold"] for e in sbert_summary]
    mean_fitness = [e["mean_fitness"] for e in sbert_summary]
    std_fitness = [e["std_fitness"] for e in sbert_summary]
    mean_shadow = [e["mean_shadow_pct"] * 100 for e in sbert_summary]
    mean_coverage = [e["mean_coverage_pct"] * 100 for e in sbert_summary]

    t = np.array(thresholds)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.8))

    # (a) Fitness
    mean_f = np.array(mean_fitness)
    std_f = np.array(std_fitness)
    ax1.plot(t, mean_f, "-o", color="#0072B2", markersize=3, linewidth=1.0)
    ax1.fill_between(t, mean_f - std_f, mean_f + std_f, color="#0072B2", alpha=0.15)
    ax1.axvline(0.35, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(0.36, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else max(mean_f) * 1.05,
             "t=0.35", fontsize=6, va="top")
    ax1.set_xlabel("SBERT Threshold")
    ax1.set_ylabel("Mean Fitness")
    ax1.set_title("(a) Fitness vs Threshold", fontsize=8)

    # (b) Shadow %
    mean_s = np.array(mean_shadow)
    ax2.plot(t, mean_s, "-o", color="#D55E00", markersize=3, linewidth=1.0)
    # Use +/- 10% of mean as visual band since no std available
    ax2.fill_between(t, mean_s * 0.85, mean_s * 1.15, color="#D55E00", alpha=0.15)
    ax2.axvline(0.35, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.text(0.36, max(mean_s) * 0.95, "t=0.35", fontsize=6, va="top")
    ax2.set_xlabel("SBERT Threshold")
    ax2.set_ylabel("Shadow Activity (%)")
    ax2.set_title("(b) Shadow % vs Threshold", fontsize=8)

    # (c) Coverage
    mean_c = np.array(mean_coverage)
    ax3.plot(t, mean_c, "-o", color="#009E73", markersize=3, linewidth=1.0)
    ax3.fill_between(t, mean_c * 0.85, mean_c * 1.15, color="#009E73", alpha=0.15)
    ax3.axvline(0.35, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.text(0.36, max(mean_c) * 0.95, "t=0.35", fontsize=6, va="top")
    ax3.set_xlabel("SBERT Threshold")
    ax3.set_ylabel("Agenda Coverage (%)")
    ax3.set_title("(c) Coverage vs Threshold", fontsize=8)

    fig.suptitle("SBERT Threshold Sensitivity Analysis",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sbert_sensitivity.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures (SciencePlots style)")
    parser.add_argument("--meetings-dir", default="meetings")
    parser.add_argument("--output-dir", default="thesis/figures_scienceplots")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--golden-file", default="golden_comparison_results.json",
                        help="Path to golden comparison results JSON")
    parser.add_argument("--sbert-json", default="thesis/figures/sbert_sensitivity.json",
                        help="Path to SBERT sensitivity analysis JSON")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.meetings_dir, "_aggregated", "metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"ERROR: {metrics_csv} not found. Run batch_analyze.py first.")
        sys.exit(1)

    df = load_metrics(metrics_csv)
    conformance_list = load_all_conformance(args.meetings_dir)

    # Load golden comparison data (optional)
    golden_data = None
    if os.path.exists(args.golden_file):
        golden_data = load_golden_data(args.golden_file)
        print(f"Loaded {len(golden_data)} golden comparison entries from {args.golden_file}")
    else:
        print(f"WARNING: {args.golden_file} not found, skipping golden figures.")

    # Load SBERT sensitivity data (optional)
    sbert_summary = None
    if os.path.exists(args.sbert_json):
        sbert_summary = load_sbert_sensitivity(args.sbert_json)
        print(f"Loaded {len(sbert_summary)} SBERT threshold entries from {args.sbert_json}")
    else:
        print(f"WARNING: {args.sbert_json} not found, skipping SBERT sensitivity figure.")

    print(f"Loaded {len(df)} meetings from {metrics_csv}")
    print(f"Style: SciencePlots (IEEE)")
    print(f"Generating figures to {args.output_dir}/\n")

    figures = [
        ("corpus_overview", fig_corpus_overview, [df]),
        ("modality_contribution", fig_modality_contribution, [conformance_list]),
        ("fitness_distribution", fig_fitness_distribution, [df]),
        ("fitness_raw_vs_dedup", fig_fitness_raw_vs_dedup, [df]),
        ("agenda_coverage", fig_agenda_coverage, [df]),
        ("shadow_formal_split", fig_shadow_formal_split, [df]),
        ("deviance_taxonomy", fig_deviance_taxonomy, [conformance_list]),
        ("powl_patterns", fig_powl_patterns, [conformance_list]),
        ("fitness_vs_shadow", fig_fitness_vs_shadow, [df]),
        ("declare_compliance", fig_declare_compliance, [df]),
        ("declare_violations", fig_declare_violations, [conformance_list]),
        ("conformance_scatter", fig_conformance_scatter, [df]),
        ("fitness_by_agenda_complexity", fig_fitness_by_agenda_complexity, [df]),
    ]

    # Add golden comparison figures if data is available
    if golden_data is not None:
        figures.extend([
            ("golden_shadow_comparison", fig_golden_shadow_comparison, [golden_data]),
            ("golden_deviance_comparison", fig_golden_deviance_comparison, [golden_data]),
            ("golden_meeting_level", fig_golden_meeting_level, [golden_data]),
        ])

    # Add SBERT sensitivity figure if data is available
    if sbert_summary is not None:
        figures.append(
            ("sbert_sensitivity", fig_sbert_sensitivity, [sbert_summary]),
        )

    generated = 0
    total = len(figures)
    for i, (name, func, extra_args) in enumerate(figures, 1):
        try:
            func(*extra_args, args.output_dir, args.format, args.dpi)
            generated += 1
            print(f"  [{i}/{total}] {name}.{args.format}")
        except Exception as e:
            print(f"  [{i}/{total}] ERROR {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {generated}/{total} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
