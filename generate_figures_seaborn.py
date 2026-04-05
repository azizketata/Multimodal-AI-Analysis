#!/usr/bin/env python3
"""Generate thesis figures using Seaborn whitegrid style with colorblind palette.

Usage:
    python generate_figures_seaborn.py --meetings-dir meetings/ --output-dir thesis/figures_seaborn/
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
import seaborn as sns

# Apply seaborn theme
sns.set_theme(
    style="whitegrid",
    font_scale=1.1,
    palette="colorblind",
    rc={
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }
)

# Seaborn colorblind palette
CB_PALETTE = sns.color_palette("colorblind", 10)

# ============================================================
# Constants
# ============================================================

CITY_COLORS = {
    "Seattle": CB_PALETTE[0],   # blue
    "Denver": CB_PALETTE[1],    # orange
    "Boston": CB_PALETTE[2],    # green
    "Alameda": CB_PALETTE[4],   # purple
}

CITY_MARKERS = {
    "Seattle": "o",
    "Denver": "s",
    "Boston": "^",
    "Alameda": "D",
}

DEVIANCE_COLORS = {
    "benign": CB_PALETTE[2],     # green
    "violation": CB_PALETTE[3],  # red
    "efficiency": CB_PALETTE[0], # blue
    "innovation": CB_PALETTE[1], # orange
    "disruption": CB_PALETTE[7], # gray
    "unknown": "#CCCCCC",
}

SOURCE_COLORS = {
    "audio": CB_PALETTE[0],   # blue
    "visual": CB_PALETTE[3],  # red
    "fused": CB_PALETTE[4],   # purple
}


def get_color(city: str):
    return CITY_COLORS.get(city, CB_PALETTE[7])


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


def load_all_conformance(meetings_dir: str) -> list[dict]:
    results = []
    for folder in sorted(os.listdir(meetings_dir)):
        conf_path = os.path.join(meetings_dir, folder, "conformance.json")
        if os.path.exists(conf_path):
            with open(conf_path) as f:
                results.append(json.load(f))
    return results


def load_golden_data(golden_file: str) -> list[dict] | None:
    """Load golden comparison results JSON."""
    if os.path.exists(golden_file):
        with open(golden_file) as f:
            return json.load(f)
    return None


def load_sbert_sensitivity(sbert_json: str) -> list[dict] | None:
    """Load SBERT sensitivity analysis summary."""
    if os.path.exists(sbert_json):
        with open(sbert_json) as f:
            data = json.load(f)
        return data.get("summary", [])
    return None


def _short_golden_name(name: str) -> str:
    """Strip 'Seattle-' prefix and abbreviate for axis labels."""
    s = name.replace("Seattle-", "").replace("Seattle_", "")
    if len(s) > 14:
        s = s[:12] + ".."
    return s


# ============================================================
# FIGURE 1: Corpus Overview Table-Figure
# ============================================================

def fig_corpus_overview(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 3.2))
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
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#34495E")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(col_labels)):
        table[len(table_data), j].set_facecolor("#F0F3F4")
        table[len(table_data), j].set_text_props(fontweight="bold")
    for i, city in enumerate(cities):
        c = get_color(city)
        table[i + 1, 0].set_text_props(
            color=c if isinstance(c, str) else "#{:02x}{:02x}{:02x}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255)),
            fontweight="bold")

    ax.set_title("Meeting Corpus Overview", fontsize=13, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"corpus_overview.{fmt}"), dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 2: Modality Contribution
# ============================================================

def fig_modality_contribution(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    totals = {"audio": 0, "visual": 0, "fused": 0}
    for c in conformance_list:
        sources = c.get("extraction", {}).get("sources", {})
        totals["audio"] += sources.get("audio", 0)
        totals["visual"] += sources.get("visual", 0)
        totals["fused"] += sources.get("fused", 0)

    total = sum(totals.values())
    labels = ["Audio (Whisper)", "Visual (RTMPose)", "Multimodal Fusion"]
    values = [totals["audio"], totals["visual"], totals["fused"]]
    pcts = [v / total * 100 for v in values]
    colors = [SOURCE_COLORS["audio"], SOURCE_COLORS["visual"], SOURCE_COLORS["fused"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: horizontal bar
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, pcts, color=colors, alpha=0.8, height=0.55, edgecolor="white", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Percentage of Events (%)")
    ax1.set_title("Event Source Distribution", fontsize=11)
    ax1.invert_yaxis()
    for i, (p, v) in enumerate(zip(pcts, values)):
        ax1.text(p + 0.8, i, f"{p:.1f}% ({v:,})", va="center", fontsize=8)

    # Right: per-meeting strip/box plot using seaborn
    audio_pcts, visual_pcts, fused_pcts = [], [], []
    for c in conformance_list:
        sources = c.get("extraction", {}).get("sources", {})
        t = sources.get("audio", 0) + sources.get("visual", 0) + sources.get("fused", 0)
        if t > 0:
            audio_pcts.append(sources.get("audio", 0) / t * 100)
            visual_pcts.append(sources.get("visual", 0) / t * 100)
            fused_pcts.append(sources.get("fused", 0) / t * 100)

    strip_data = pd.DataFrame({
        "Modality": (["Audio"] * len(audio_pcts) +
                     ["Visual"] * len(visual_pcts) +
                     ["Fusion"] * len(fused_pcts)),
        "Percentage": audio_pcts + visual_pcts + fused_pcts,
    })

    sns.boxplot(data=strip_data, x="Modality", y="Percentage", ax=ax2,
                palette=[SOURCE_COLORS["audio"], SOURCE_COLORS["visual"], SOURCE_COLORS["fused"]],
                width=0.5, linewidth=0.8, fliersize=3, saturation=0.4)
    sns.stripplot(data=strip_data, x="Modality", y="Percentage", ax=ax2,
                  palette=[SOURCE_COLORS["audio"], SOURCE_COLORS["visual"], SOURCE_COLORS["fused"]],
                  alpha=0.4, size=3, jitter=0.15)

    ax2.set_ylabel("Events per Meeting (%)")
    ax2.set_title("Per-Meeting Modality Variation", fontsize=11)

    fig.suptitle("Multimodal Event Extraction: Source Modality Contribution",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"modality_contribution.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 3: Fitness Distribution
# ============================================================

def fig_fitness_distribution(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4))

    scores = df["fitness_dedup"].dropna().values
    sns.histplot(scores, bins=12, color=CB_PALETTE[0], alpha=0.7, edgecolor="white",
                 linewidth=0.5, ax=ax, kde=True, line_kws={"linewidth": 1.2, "alpha": 0.5})

    mean_val = np.mean(scores)
    median_val = np.median(scores)
    ax.axvline(mean_val, color=CB_PALETTE[3], linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_val:.3f}")
    ax.axvline(median_val, color=CB_PALETTE[1], linestyle="-.", linewidth=1.2,
               label=f"Median: {median_val:.3f}")

    ax.set_xlabel("Token-Replay Fitness (Deduplicated)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Process Conformance: Fitness Score Distribution (n=54)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_distribution.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 4: Raw vs Dedup Fitness
# ============================================================

def fig_fitness_raw_vs_dedup(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        ax.scatter(df[mask]["fitness_raw"], df[mask]["fitness_dedup"],
                   c=[get_color(city)], marker=get_marker(city),
                   label=city, alpha=0.7, s=35, zorder=3, edgecolors="white", linewidth=0.5)

    lim = max(df["fitness_raw"].max(), df["fitness_dedup"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.15, linewidth=0.8)
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.03, color=CB_PALETTE[2])
    ax.text(0.05, lim * 0.92, "Dedup improves fitness", fontsize=8,
            color=CB_PALETTE[2], alpha=0.7, style="italic")

    ax.set_xlabel("Raw Fitness Score")
    ax.set_ylabel("Deduplicated Fitness Score")
    ax.set_title("Effect of Deduplication on Fitness Measurement")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_raw_vs_dedup.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 5: Shadow vs Formal Split
# ============================================================

def fig_shadow_formal_split(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [1, 1.5]})

    total_formal = df["formal_events"].sum()
    total_shadow = df["shadow_events"].sum()
    total_all = total_formal + total_shadow

    # Left: horizontal bar
    bars = ax1.barh(["Formal", "Shadow"],
                    [total_formal / total_all * 100, total_shadow / total_all * 100],
                    color=[CB_PALETTE[2], CB_PALETTE[3]], alpha=0.8, height=0.45, edgecolor="white")
    ax1.set_xlabel("Percentage (%)")
    ax1.set_title("Overall Activity Split", fontsize=11)
    for i, (v, n) in enumerate(zip([total_formal / total_all * 100, total_shadow / total_all * 100],
                                    [total_formal, total_shadow])):
        ax1.text(v + 0.5, i, f"{v:.1f}% ({n:,})", va="center", fontsize=8)

    # Right: histogram with KDE
    shadow_pcts = df["shadow_pct"].values * 100
    sns.histplot(shadow_pcts, bins=12, color=CB_PALETTE[3], alpha=0.6, edgecolor="white",
                 linewidth=0.5, ax=ax2, kde=True, line_kws={"linewidth": 1.0, "alpha": 0.5})
    mean_s = np.mean(shadow_pcts)
    median_s = np.median(shadow_pcts)
    ax2.axvline(mean_s, color=CB_PALETTE[3], linestyle="--", linewidth=1.2,
                label=f"Mean: {mean_s:.1f}%")
    ax2.axvline(median_s, color=CB_PALETTE[1], linestyle="-.", linewidth=1.2,
                label=f"Median: {median_s:.1f}%")
    ax2.set_xlabel("Shadow Activity Percentage (%)")
    ax2.set_ylabel("Number of Meetings")
    ax2.set_title("Shadow Prevalence Distribution", fontsize=11)
    ax2.legend()

    fig.suptitle("Shadow Workflow Prevalence Across 54 Meetings",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"shadow_formal_split.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 6: Deviance Taxonomy — horizontal bar
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

    fig, ax = plt.subplots(figsize=(7, 3.5))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.55,
                   edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Shadow Events")
    ax.set_title("Process Deviance Classification of Shadow Activities",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        pct = v / total * 100
        ax.text(v + max(values) * 0.015, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deviance_taxonomy.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 7: POWL Shadow Patterns — horizontal bar
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
    colors = [CB_PALETTE[7], CB_PALETTE[3], CB_PALETTE[2], CB_PALETTE[0]][:len(labels)]
    total = sum(values)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(7, 3))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.55,
            edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Clusters")
    ax.set_title(f"POWL Shadow Workflow Structural Patterns (n={total:,} clusters)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        pct = v / total * 100
        ax.text(v + max(values) * 0.015, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"powl_patterns.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 8: Declare Compliance
# ============================================================

def fig_declare_compliance(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4))

    bands = [(90, 100, "A", CB_PALETTE[2]), (80, 90, "B", CB_PALETTE[0]),
             (70, 80, "C", CB_PALETTE[1]), (60, 70, "D", CB_PALETTE[3]), (0, 60, "F", CB_PALETTE[4])]
    for lo, hi, grade, color in bands:
        ax.axvspan(lo, hi, alpha=0.07, color=color)

    scores = df["declare_score"].dropna().values
    sns.histplot(scores, bins=20, color=CB_PALETTE[0], alpha=0.7, edgecolor="white",
                 linewidth=0.5, ax=ax)

    for lo, hi, grade, color in bands:
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 1,
                grade, ha="center", va="top", fontsize=14, color=color, alpha=0.45,
                fontweight="bold")

    mean_val = np.mean(scores)
    ax.axvline(mean_val, color=CB_PALETTE[3], linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_val:.1f}")

    grade_counts = Counter(df["declare_grade"].values)
    grade_text = "  ".join([f"{g}: {grade_counts.get(g, 0)}" for g in ["A", "B", "C", "D", "F"]])
    ax.text(0.98, 0.95, grade_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#666666", linewidth=0.5))

    ax.set_xlabel("Robert's Rules Compliance Score (0-100)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Declare Constraint Compliance Distribution (n=54)")
    ax.legend()
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: horizontal bar
    if total_sat + total_viol > 0:
        total_all = total_sat + total_viol
        bars = ax1.barh(["Satisfied", "Violated"],
                       [total_sat / total_all * 100, total_viol / total_all * 100],
                       color=[CB_PALETTE[2], CB_PALETTE[3]], alpha=0.8, height=0.45, edgecolor="white")
        for i, (pct, cnt) in enumerate(zip(
            [total_sat / total_all * 100, total_viol / total_all * 100],
            [total_sat, total_viol])):
            ax1.text(pct + 0.5, i, f"{pct:.1f}% ({cnt})", va="center", fontsize=8)
        ax1.set_xlabel("Percentage (%)")
    ax1.set_title(f"Constraint Outcomes\n({meetings_with_constraints} meetings)", fontsize=10)

    # Right: specific violations
    if violation_counts:
        sorted_v = violation_counts.most_common(10)
        names = [v[0] for v in sorted_v]
        counts = [v[1] for v in sorted_v]

        y = np.arange(len(names))
        ax2.barh(y, counts, color=CB_PALETTE[3], alpha=0.8, height=0.6, edgecolor="white", linewidth=0.8)
        ax2.set_yticks(y)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("Number of Meetings")
        ax2.invert_yaxis()
        for i, v in enumerate(counts):
            ax2.text(v + 0.2, i, str(v), va="center", fontsize=8)
    ax2.set_title("Most Violated Constraints", fontsize=10)

    fig.suptitle("Robert's Rules Declare Constraint Analysis",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"declare_violations.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 10: Fitness vs Shadow
# ============================================================

def fig_fitness_vs_shadow(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 5))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        ax.scatter(df[mask]["shadow_pct"] * 100, df[mask]["fitness_dedup"],
                   c=[get_color(city)], marker=get_marker(city),
                   label=city, alpha=0.7, s=35, zorder=3, edgecolors="white", linewidth=0.5)

    x = df["shadow_pct"].values * 100
    y = df["fitness_dedup"].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() > 2:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, p(x_line), "--", color=CB_PALETTE[3], alpha=0.5, linewidth=1.0,
                label=f"Linear trend (slope={z[0]:.4f})")

        y_pred = p(x[valid])
        ss_res = np.sum((y[valid] - y_pred) ** 2)
        ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.02, 0.02, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
                fontsize=9, color="#555555")

    ax.set_xlabel("Shadow Activity Percentage (%)")
    ax.set_ylabel("Deduplicated Fitness Score")
    ax.set_title("Conformance Fitness vs. Shadow Workflow Prevalence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_vs_shadow.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 11: Agenda Coverage
# ============================================================

def fig_agenda_coverage(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4))

    coverages = df["agenda_coverage_pct"].dropna().values * 100
    sns.histplot(coverages, bins=12, color=CB_PALETTE[0], alpha=0.7, edgecolor="white",
                 linewidth=0.5, ax=ax, kde=True, line_kws={"linewidth": 1.0, "alpha": 0.5})

    mean_c = np.mean(coverages)
    median_c = np.median(coverages)
    ax.axvline(mean_c, color=CB_PALETTE[3], linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_c:.1f}%")
    ax.axvline(median_c, color=CB_PALETTE[1], linestyle="-.", linewidth=1.2,
               label=f"Median: {median_c:.1f}%")

    ax.set_xlabel("Agenda Item Coverage (%)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Agenda Item Detection Rate Distribution (n=54)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"agenda_coverage.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 12: Conformance Scatter
# ============================================================

def fig_conformance_scatter(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(8, 5))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        cdf = df[mask]
        sizes = cdf["agenda_coverage_pct"] * 150 + 15
        ax.scatter(cdf["fitness_dedup"], cdf["declare_score"],
                   c=[get_color(city)], marker=get_marker(city),
                   s=sizes, label=city, alpha=0.6, zorder=3,
                   edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Deduplicated Fitness Score")
    ax.set_ylabel("Declare Compliance Score (0-100)")
    ax.set_title("Meeting Governance Profile: Fitness × Compliance × Coverage")

    for sz, label in [(0.2, "20%"), (0.5, "50%"), (1.0, "100%")]:
        ax.scatter([], [], c="gray", alpha=0.4, s=sz * 150 + 15,
                   label=f"Coverage: {label}")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"conformance_scatter.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 13: Fitness by Agenda Complexity (seaborn boxplot)
# ============================================================

def fig_fitness_by_agenda_complexity(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    df = df.copy()
    if "agenda_items" not in df.columns or df["agenda_items"].isna().all():
        return

    def categorize(n):
        if n <= 5:
            return "Sparse (≤5)"
        elif n <= 50:
            return "Moderate (6-50)"
        else:
            return "Dense (51-100)"

    df["complexity"] = df["agenda_items"].apply(categorize)
    order = ["Sparse (≤5)", "Moderate (6-50)", "Dense (51-100)"]
    colors_box = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[3]]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    for ax, metric, title in zip(axes,
                                  ["fitness_dedup", "shadow_pct", "agenda_coverage_pct"],
                                  ["Deduplicated Fitness", "Shadow Activity (%)", "Agenda Coverage (%)"]):
        plot_df = df[["complexity", metric]].copy()
        if "pct" in metric:
            plot_df[metric] = plot_df[metric] * 100

        sns.boxplot(data=plot_df, x="complexity", y=metric, order=order,
                   palette=colors_box, ax=ax, width=0.6, linewidth=0.8,
                   fliersize=3, saturation=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelsize=8)

    fig.suptitle("Meeting Metrics Stratified by Agenda Complexity",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_by_agenda_complexity.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 14: Golden Shadow Comparison (bar chart)
# ============================================================

def fig_golden_shadow_comparison(golden_data: list[dict], output_dir: str, fmt: str, dpi: int):
    names = [_short_golden_name(g["golden_name"]) for g in golden_data]
    golden_pcts = [g["shadow_comparison"]["golden_shadow_pct"] * 100 for g in golden_data]
    pipeline_pcts = [g["shadow_comparison"]["pipeline_shadow_pct"] * 100 for g in golden_data]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, golden_pcts, width, label="Ground Truth",
                   color=CB_PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, pipeline_pcts, width, label="Pipeline",
                   color=CB_PALETTE[1], alpha=0.8, edgecolor="white", linewidth=0.8)

    # Mean difference line
    mean_diff = np.mean([g - p for g, p in zip(golden_pcts, pipeline_pcts)])
    ax.axhline(np.mean(golden_pcts), color=CB_PALETTE[0], linestyle="--",
               linewidth=0.8, alpha=0.5)
    ax.axhline(np.mean(pipeline_pcts), color=CB_PALETTE[1], linestyle="--",
               linewidth=0.8, alpha=0.5)

    # Annotate mean difference
    ax.text(0.98, 0.95, f"Mean gap: {mean_diff:+.1f} pp",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="#666666", linewidth=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Shadow Activity (%)")
    ax.set_title("Ground-Truth Validation: Shadow Prevalence (n=8 meetings)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"golden_shadow_comparison.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 15: Golden Deviance Comparison (grouped horizontal bars)
# ============================================================

def fig_golden_deviance_comparison(golden_data: list[dict], output_dir: str, fmt: str, dpi: int):
    # Aggregate golden deviance
    golden_deviance = Counter()
    for g in golden_data:
        cats = g.get("golden", {}).get("deviance_cats", {})
        for k, v in cats.items():
            golden_deviance[k] += v
    golden_total = sum(golden_deviance.values())

    # Aggregate pipeline deviance
    pipeline_deviance = Counter()
    for g in golden_data:
        dev = g.get("pipeline", {}).get("deviance", {})
        for k, v in dev.items():
            pipeline_deviance[k] += v
    pipeline_total = sum(pipeline_deviance.values())

    # Union of all categories
    all_cats = sorted(set(list(golden_deviance.keys()) + list(pipeline_deviance.keys())))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # Left: golden deviance
    g_vals = [golden_deviance.get(c, 0) for c in all_cats]
    g_colors = [DEVIANCE_COLORS.get(c, "#CCCCCC") for c in all_cats]
    y_pos = np.arange(len(all_cats))

    ax1.barh(y_pos, g_vals, color=g_colors, alpha=0.8, height=0.55,
             edgecolor="white", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([c.title() for c in all_cats], fontsize=10)
    ax1.set_xlabel("Number of Events")
    ax1.set_title(f"Ground Truth (n={golden_total:,})", fontsize=11)
    ax1.invert_yaxis()
    for i, v in enumerate(g_vals):
        if v > 0:
            pct = v / golden_total * 100 if golden_total > 0 else 0
            ax1.text(v + max(g_vals) * 0.02, i, f"{v:,} ({pct:.1f}%)",
                     va="center", fontsize=8)

    # Right: pipeline deviance
    p_vals = [pipeline_deviance.get(c, 0) for c in all_cats]
    p_colors = [DEVIANCE_COLORS.get(c, "#CCCCCC") for c in all_cats]

    ax2.barh(y_pos, p_vals, color=p_colors, alpha=0.8, height=0.55,
             edgecolor="white", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([c.title() for c in all_cats], fontsize=10)
    ax2.set_xlabel("Number of Events")
    ax2.set_title(f"Pipeline (n={pipeline_total:,})", fontsize=11)
    ax2.invert_yaxis()
    for i, v in enumerate(p_vals):
        if v > 0:
            pct = v / pipeline_total * 100 if pipeline_total > 0 else 0
            ax2.text(v + max(max(p_vals), 1) * 0.02, i, f"{v:,} ({pct:.1f}%)",
                     va="center", fontsize=8)

    fig.suptitle("Deviance Classification: Ground Truth vs. Pipeline",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"golden_deviance_comparison.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 16: Golden Meeting-Level Comparison (3-panel)
# ============================================================

def fig_golden_meeting_level(golden_data: list[dict], output_dir: str, fmt: str, dpi: int):
    names = [_short_golden_name(g["golden_name"]) for g in golden_data]
    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    # (a) Event Granularity (log scale)
    golden_events = [g["golden"]["total_events"] for g in golden_data]
    pipeline_events = [g["pipeline"]["total_events"] for g in golden_data]

    ax1.bar(x - width / 2, golden_events, width, label="Ground Truth",
            color=CB_PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.8)
    ax1.bar(x + width / 2, pipeline_events, width, label="Pipeline",
            color=CB_PALETTE[1], alpha=0.8, edgecolor="white", linewidth=0.8)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Total Events (log scale)")
    ax1.set_title("(a) Event Granularity", fontsize=10)
    ax1.legend(fontsize=7)

    # (b) Formal Activity Rate
    golden_formal = [g["golden"]["estimated_formal_pct"] for g in golden_data]
    pipeline_formal = [
        (g["pipeline"]["formal_count"] / g["pipeline"]["total_events"] * 100)
        if g["pipeline"]["total_events"] > 0 else 0
        for g in golden_data
    ]

    ax2.bar(x - width / 2, golden_formal, width, label="Ground Truth",
            color=CB_PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.8)
    ax2.bar(x + width / 2, pipeline_formal, width, label="Pipeline",
            color=CB_PALETTE[1], alpha=0.8, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Formal Activity Rate (%)")
    ax2.set_title("(b) Formal Activity Rate", fontsize=10)
    ax2.legend(fontsize=7)

    # (c) Procedural Compliance
    golden_roberts = [g["roberts_comparison"]["golden_roberts_score"] * 100
                      for g in golden_data]
    pipeline_declare = [g["roberts_comparison"]["pipeline_declare_score"]
                        for g in golden_data]

    ax3.bar(x - width / 2, golden_roberts, width, label="Ground Truth (Roberts)",
            color=CB_PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.8)
    ax3.bar(x + width / 2, pipeline_declare, width, label="Pipeline (Declare)",
            color=CB_PALETTE[1], alpha=0.8, edgecolor="white", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Compliance Score (%)")
    ax3.set_title("(c) Procedural Compliance", fontsize=10)
    ax3.legend(fontsize=7)

    fig.suptitle("Meeting-Level Ground-Truth Comparison (n=8 Seattle meetings)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"golden_meeting_level.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 17: SBERT Sensitivity Analysis (3-panel line plot)
# ============================================================

def fig_sbert_sensitivity(sbert_summary: list[dict], output_dir: str, fmt: str, dpi: int):
    thresholds = [s["threshold"] for s in sbert_summary]
    mean_fitness = [s["mean_fitness"] for s in sbert_summary]
    std_fitness = [s["std_fitness"] for s in sbert_summary]
    mean_shadow = [s["mean_shadow_pct"] * 100 for s in sbert_summary]
    mean_coverage = [s["mean_coverage_pct"] * 100 for s in sbert_summary]

    # Approximate std bands for shadow/coverage using median spread
    shadow_band = [abs(s["mean_shadow_pct"] - s["median_shadow_pct"]) * 100
                   for s in sbert_summary]
    coverage_band = [abs(s["mean_coverage_pct"] - s["median_coverage_pct"]) * 100
                     for s in sbert_summary]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Conformance Fitness
    ax1.plot(thresholds, mean_fitness, "-o", color=CB_PALETTE[0], linewidth=1.5,
             markersize=4, label="Mean fitness")
    ax1.fill_between(thresholds,
                     [m - s for m, s in zip(mean_fitness, std_fitness)],
                     [m + s for m, s in zip(mean_fitness, std_fitness)],
                     color=CB_PALETTE[0], alpha=0.15, label="$\\pm$1 std")
    ax1.axvline(0.35, color=CB_PALETTE[3], linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(0.355, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.5,
             "t=0.35", fontsize=8, color=CB_PALETTE[3], va="top")
    ax1.set_xlabel("SBERT Threshold")
    ax1.set_ylabel("Fitness Score")
    ax1.set_title("(a) Conformance Fitness", fontsize=10)
    ax1.legend(fontsize=7)

    # (b) Shadow Prevalence
    ax2.plot(thresholds, mean_shadow, "-o", color=CB_PALETTE[3], linewidth=1.5,
             markersize=4, label="Mean shadow %")
    ax2.fill_between(thresholds,
                     [max(0, m - b) for m, b in zip(mean_shadow, shadow_band)],
                     [m + b for m, b in zip(mean_shadow, shadow_band)],
                     color=CB_PALETTE[3], alpha=0.15, label="Mean-median spread")
    ax2.axvline(0.35, color=CB_PALETTE[3], linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.text(0.355, max(mean_shadow) * 0.95, "t=0.35", fontsize=8,
             color=CB_PALETTE[3], va="top")
    ax2.set_xlabel("SBERT Threshold")
    ax2.set_ylabel("Shadow Prevalence (%)")
    ax2.set_title("(b) Shadow Prevalence", fontsize=10)
    ax2.legend(fontsize=7)

    # (c) Agenda Coverage
    ax3.plot(thresholds, mean_coverage, "-o", color=CB_PALETTE[2], linewidth=1.5,
             markersize=4, label="Mean coverage %")
    ax3.fill_between(thresholds,
                     [max(0, m - b) for m, b in zip(mean_coverage, coverage_band)],
                     [m + b for m, b in zip(mean_coverage, coverage_band)],
                     color=CB_PALETTE[2], alpha=0.15, label="Mean-median spread")
    ax3.axvline(0.35, color=CB_PALETTE[3], linestyle="--", linewidth=1.0, alpha=0.7)
    ax3.text(0.355, max(mean_coverage) * 0.95, "t=0.35", fontsize=8,
             color=CB_PALETTE[3], va="top")
    ax3.set_xlabel("SBERT Threshold")
    ax3.set_ylabel("Agenda Coverage (%)")
    ax3.set_title("(c) Agenda Coverage", fontsize=10)
    ax3.legend(fontsize=7)

    fig.suptitle("SBERT Threshold Sensitivity Analysis (n=54 meetings)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sbert_sensitivity.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures (Seaborn style)")
    parser.add_argument("--meetings-dir", default="meetings")
    parser.add_argument("--output-dir", default="thesis/figures_seaborn")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--golden-file", default="golden_comparison_results.json",
                        help="Path to golden comparison results JSON")
    parser.add_argument("--sbert-json", default="thesis/figures/sbert_sensitivity.json",
                        help="Path to SBERT sensitivity analysis JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.meetings_dir, "_aggregated", "metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"ERROR: {metrics_csv} not found. Run batch_analyze.py first.")
        sys.exit(1)

    df = load_metrics(metrics_csv)
    conformance_list = load_all_conformance(args.meetings_dir)

    # Load golden comparison data
    golden_data = load_golden_data(args.golden_file)
    if golden_data:
        print(f"Loaded {len(golden_data)} golden comparison entries from {args.golden_file}")
    else:
        print(f"WARNING: Golden data not found at {args.golden_file} (skipping golden figures)")

    # Load SBERT sensitivity data
    sbert_summary = load_sbert_sensitivity(args.sbert_json)
    if sbert_summary:
        print(f"Loaded {len(sbert_summary)} SBERT thresholds from {args.sbert_json}")
    else:
        print(f"WARNING: SBERT sensitivity data not found at {args.sbert_json} (skipping)")

    print(f"Loaded {len(df)} meetings from {metrics_csv}")
    print(f"Style: Seaborn (whitegrid + colorblind)")
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
    if golden_data:
        figures.extend([
            ("golden_shadow_comparison", fig_golden_shadow_comparison, [golden_data]),
            ("golden_deviance_comparison", fig_golden_deviance_comparison, [golden_data]),
            ("golden_meeting_level", fig_golden_meeting_level, [golden_data]),
        ])

    # Add SBERT sensitivity figure if data is available
    if sbert_summary:
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
