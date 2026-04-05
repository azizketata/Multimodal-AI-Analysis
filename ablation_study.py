#!/usr/bin/env python3
"""
Multimodal ablation study for Meeting Process Twin.

For each meeting, filters raw events by source modality (Audio, Video, Full)
and re-runs SBERT mapping + fitness computation to quantify each modality's
contribution to pipeline metrics.

No API key needed — uses regex agenda parsing + SBERT only.

Usage:
    python ablation_study.py
    python ablation_study.py --meetings-dir meetings --output ablation_results.csv
"""

import sys
import os
import argparse
from unittest.mock import MagicMock
from datetime import datetime, timedelta

# Mock streamlit before project imports
_mock_st = MagicMock()
_mock_st.error = lambda m: print(f"[ST ERROR] {m}", file=sys.stderr)
_mock_st.warning = lambda m: print(f"[ST WARN] {m}")
sys.modules["streamlit"] = _mock_st

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pm4py.objects.bpmn.obj import BPMN
import pm4py

# Reuse project functions
from run_no_api import (
    parse_agenda_activities,
    build_sequential_bpmn,
    map_events_sbert,
    dedup_for_fitness,
    parse_time,
)


def compute_fitness(bpmn_graph, df_mapped, activities):
    """Compute raw and dedup token-replay fitness."""
    result = {"raw": 0.0, "dedup": 0.0, "agenda_coverage": 0, "agenda_coverage_pct": 0.0}

    # Filter to formal events only
    formal = df_mapped[~df_mapped["mapped_activity"].str.startswith("Deviation:", na=False)].copy()
    if formal.empty:
        return result

    try:
        net, im, fm = pm4py.convert_to_petri_net(bpmn_graph)
    except Exception:
        return result

    def to_log(df):
        log = df.copy()
        log["case:concept:name"] = "Meeting_1"
        log["concept:name"] = log["mapped_activity"]
        log["time:timestamp"] = log["timestamp"].apply(parse_time)
        return log

    # Raw fitness (all formal events)
    try:
        log_raw = to_log(formal)
        fit = pm4py.fitness_token_based_replay(log_raw, net, im, fm)
        result["raw"] = fit.get("log_fitness", 0.0)
    except Exception:
        pass

    # Dedup fitness (first occurrence per activity)
    try:
        dedup_df = dedup_for_fitness(df_mapped, activities)
        if not dedup_df.empty:
            log_dedup = to_log(dedup_df)
            fit = pm4py.fitness_token_based_replay(log_dedup, net, im, fm)
            result["dedup"] = fit.get("log_fitness", 0.0)
    except Exception:
        pass

    # Agenda coverage
    matched_activities = set(formal["mapped_activity"].unique())
    covered = matched_activities.intersection(set(activities))
    result["agenda_coverage"] = len(covered)
    result["agenda_coverage_pct"] = len(covered) / len(activities) if activities else 0.0

    return result


def main():
    parser = argparse.ArgumentParser(description="Multimodal ablation study")
    parser.add_argument("--meetings-dir", default="meetings", help="Path to meetings directory")
    parser.add_argument("--output", default="ablation_results.csv", help="Output CSV path")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2", help="SBERT model name")
    parser.add_argument("--threshold", type=float, default=0.35, help="SBERT threshold")
    args = parser.parse_args()

    print(f"Loading SBERT model: {args.sbert_model}")
    model = SentenceTransformer(args.sbert_model)

    conditions = {
        "Audio-only": lambda df: df[df["source"] == "Audio"],
        "Visual-only": lambda df: df[df["source"] == "Video"],
        "Full": lambda df: df.copy(),
    }

    results = []
    meeting_dirs = sorted([
        d for d in os.listdir(args.meetings_dir)
        if os.path.isdir(os.path.join(args.meetings_dir, d)) and not d.startswith("_")
    ])

    print(f"Processing {len(meeting_dirs)} meetings...\n")

    for i, meeting_id in enumerate(meeting_dirs):
        meeting_path = os.path.join(args.meetings_dir, meeting_id)
        raw_csv = os.path.join(meeting_path, "raw_events.csv")
        agenda_path = os.path.join(meeting_path, "agenda.txt")

        if not os.path.exists(raw_csv) or not os.path.exists(agenda_path):
            print(f"  [{i+1}/{len(meeting_dirs)}] {meeting_id}: SKIP (missing files)")
            continue

        df_raw = pd.read_csv(raw_csv)
        if df_raw.empty:
            print(f"  [{i+1}/{len(meeting_dirs)}] {meeting_id}: SKIP (empty CSV)")
            continue

        activities = parse_agenda_activities(agenda_path)
        if not activities:
            print(f"  [{i+1}/{len(meeting_dirs)}] {meeting_id}: SKIP (no agenda items)")
            continue

        bpmn = build_sequential_bpmn(activities)
        city = meeting_id.rsplit("_", 1)[0] if "_" in meeting_id else meeting_id

        for cond_name, filter_fn in conditions.items():
            filtered = filter_fn(df_raw)
            n_events = len(filtered)

            if n_events == 0:
                results.append({
                    "meeting_id": meeting_id, "city": city, "condition": cond_name,
                    "n_events": 0, "shadow_pct": 0.0, "fitness_raw": 0.0,
                    "fitness_dedup": 0.0, "agenda_coverage_pct": 0.0, "match_rate": 0.0,
                })
                continue

            # SBERT mapping
            mapped = map_events_sbert(filtered, activities, model, threshold=args.threshold)

            # Shadow %
            shadow_mask = mapped["mapped_activity"].str.startswith("Deviation:", na=False)
            shadow_pct = shadow_mask.sum() / len(mapped) if len(mapped) > 0 else 0.0
            match_rate = 1.0 - shadow_pct

            # Fitness
            fit = compute_fitness(bpmn, mapped, activities)

            results.append({
                "meeting_id": meeting_id,
                "city": city,
                "condition": cond_name,
                "n_events": n_events,
                "shadow_pct": round(shadow_pct, 4),
                "match_rate": round(match_rate, 4),
                "fitness_raw": round(fit["raw"], 4),
                "fitness_dedup": round(fit["dedup"], 4),
                "agenda_coverage_pct": round(fit["agenda_coverage_pct"], 4),
            })

        print(f"  [{i+1}/{len(meeting_dirs)}] {meeting_id}: {len(activities)} agenda items, "
              f"{len(df_raw)} events")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (mean across all meetings)")
    print("=" * 80)
    summary = df_results.groupby("condition").agg({
        "n_events": "mean",
        "shadow_pct": "mean",
        "match_rate": "mean",
        "fitness_raw": "mean",
        "fitness_dedup": "mean",
        "agenda_coverage_pct": "mean",
    }).round(4)

    # Reorder rows
    order = ["Audio-only", "Visual-only", "Full"]
    summary = summary.reindex([o for o in order if o in summary.index])

    print(f"\n{'Condition':<15} {'Events':>8} {'Shadow%':>9} {'Match%':>9} "
          f"{'Fit(raw)':>10} {'Fit(ded)':>10} {'Coverage':>10}")
    print("-" * 75)
    for cond, row in summary.iterrows():
        print(f"{cond:<15} {row['n_events']:>8.0f} {row['shadow_pct']*100:>8.1f}% "
              f"{row['match_rate']*100:>8.1f}% {row['fitness_raw']:>10.4f} "
              f"{row['fitness_dedup']:>10.4f} {row['agenda_coverage_pct']*100:>9.1f}%")

    # LaTeX table
    print("\n\n% LaTeX table for thesis")
    print("\\begin{table}[htbp]")
    print("    \\centering")
    print("    \\caption{Multimodal ablation study: pipeline metrics when restricted to single-modality event subsets versus the full multimodal pipeline. Mean values across 54~meetings.}")
    print("    \\label{tab:ablation}")
    print("    \\small")
    print("    \\begin{tabular}{lccccc}")
    print("        \\toprule")
    print("        Condition & Events & Shadow~\\% & Fitness (dedup) & Coverage~\\% & Match~\\% \\\\")
    print("        \\midrule")
    for cond, row in summary.iterrows():
        tex_cond = cond.replace("-", " ")
        print(f"        {tex_cond} & {row['n_events']:.0f} & "
              f"{row['shadow_pct']*100:.1f} & {row['fitness_dedup']:.3f} & "
              f"{row['agenda_coverage_pct']*100:.1f} & {row['match_rate']*100:.1f} \\\\")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("\\end{table}")

    # Also print median table for robustness
    print("\n\nMEDIAN values:")
    med = df_results.groupby("condition").agg({
        "fitness_dedup": "median",
        "shadow_pct": "median",
    }).round(4)
    med = med.reindex([o for o in order if o in med.index])
    for cond, row in med.iterrows():
        print(f"  {cond}: fitness_dedup={row['fitness_dedup']:.4f}, shadow={row['shadow_pct']*100:.1f}%")


if __name__ == "__main__":
    main()
