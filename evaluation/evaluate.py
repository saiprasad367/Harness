"""
Comparative evaluation suite: Adaptive Harness vs. Fixed Threshold Baselines.
Evaluates FIR, FWR, Precision, Recall, F1, and Latency percentiles across
partitions, sensitivity presets, and dysarthric/control cohorts.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tabulate import tabulate

from config.settings import OUTPUT_DIR
from config.sensitivity import SensitivityPreset
from vad.extract_pauses import extract_candidate_pauses
from classifier.pause_classifier import AdaptivePauseClassifier
from evaluation.baseline import FixedThresholdBaseline
from evaluation.metrics import compute_turn_metrics, TurnMetrics


def generate_benchmark_labels(pauses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Establish conversational ground truth for candidate pauses.
    Per project specification:
    - Intra-utterance mid-speech pauses (not trailing, not pre-speech) are CONTINUATION opportunities.
    - Trailing pauses (after speech completes) represent turn completions (END).
    """
    df = pauses_df.copy()

    def assign_label(row):
        if row.get("is_trailing", False):
            return "END"
        return "CONTINUE"

    df["ground_truth"] = df.apply(assign_label, axis=1)
    return df


def run_comparative_benchmark(
    pauses_csv: Optional[Path] = None,
    split_to_eval: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs benchmark comparing:
    1. Fixed Thresholds (500ms, 800ms, 1200ms, 1500ms)
    2. Adaptive Classifier (Patient, Normal, Quick presets)
    Exports detailed metrics and confusion matrices.
    """
    if pauses_csv is None:
        pauses_csv = OUTPUT_DIR / "candidate_pauses.csv"
        # If default candidate pauses doesn't exist or is small, fallback to pilot candidates
        if not pauses_csv.exists() or pauses_csv.stat().st_size < 100:
            pilot_csv = OUTPUT_DIR / "annotation_candidates_pilot.csv"
            if pilot_csv.exists():
                pauses_csv = pilot_csv

    if not pauses_csv.exists():
        print(f"{pauses_csv} not found. Running pause extraction first...")
        df_pauses = extract_candidate_pauses(limit=60)
    else:
        df_pauses = pd.read_csv(pauses_csv)

    labeled_df = generate_benchmark_labels(df_pauses)

    if split_to_eval and "split" in labeled_df.columns:
        subset_df = labeled_df[labeled_df["split"] == split_to_eval]
        if len(subset_df) > 0:
            labeled_df = subset_df

    durations = labeled_df["duration_sec"].tolist()
    ground_truth = labeled_df["ground_truth"].tolist()
    speakers = labeled_df["speaker_id"].tolist() if "speaker_id" in labeled_df.columns else ["spk"] * len(durations)

    print(f"\nEvaluating on {len(labeled_df)} pauses...")
    print(f"Ground-truth distribution: CONTINUE={ground_truth.count('CONTINUE')}, END={ground_truth.count('END')}")

    results = []
    confusion_matrices = {}

    # 1. Fixed Threshold Baselines
    thresholds = [0.5, 0.8, 1.2, 1.5]
    for th in thresholds:
        name = f"Fixed ({int(th * 1000)}ms)"
        baseline = FixedThresholdBaseline(threshold_sec=th)
        preds = baseline.evaluate_batch(durations)
        m = compute_turn_metrics(preds, ground_truth, durations)
        confusion_matrices[name] = m.confusion_matrix
        results.append({
            "Policy / Model": name,
            "FIR (False Interruption)": f"{m.false_interruption_rate * 100:.1f}%",
            "FWR (False Wait)": f"{m.false_wait_rate * 100:.1f}%",
            "Accuracy": f"{m.accuracy * 100:.1f}%",
            "Precision": f"{m.precision * 100:.1f}%",
            "Recall": f"{m.recall * 100:.1f}%",
            "F1-Score": f"{m.f1_score * 100:.1f}%",
            "Avg Latency": f"{m.avg_latency_sec:.2f}s",
            "P90 Latency": f"{m.p90_latency_sec:.2f}s",
        })

    # 2. Adaptive Pause Classifier across Sensitivity Modes
    classifier = AdaptivePauseClassifier()
    for preset in [SensitivityPreset.QUICK, SensitivityPreset.NORMAL, SensitivityPreset.PATIENT]:
        name = f"Adaptive ({preset.value.capitalize()})"
        preds = []
        for d, spk in zip(durations, speakers):
            p, _ = classifier.predict(current_pause_sec=d, speaker_id=spk, preset=preset)
            preds.append(p)

        m = compute_turn_metrics(preds, ground_truth, durations)
        confusion_matrices[name] = m.confusion_matrix
        results.append({
            "Policy / Model": name,
            "FIR (False Interruption)": f"{m.false_interruption_rate * 100:.1f}%",
            "FWR (False Wait)": f"{m.false_wait_rate * 100:.1f}%",
            "Accuracy": f"{m.accuracy * 100:.1f}%",
            "Precision": f"{m.precision * 100:.1f}%",
            "Recall": f"{m.recall * 100:.1f}%",
            "F1-Score": f"{m.f1_score * 100:.1f}%",
            "Avg Latency": f"{m.avg_latency_sec:.2f}s",
            "P90 Latency": f"{m.p90_latency_sec:.2f}s",
        })

    summary_df = pd.DataFrame(results)
    print("\n" + tabulate(summary_df, headers="keys", tablefmt="github", showindex=False))

    # Save benchmark table and confusion matrices
    out_file = OUTPUT_DIR / "benchmark_comparison.csv"
    summary_df.to_csv(out_file, index=False)

    cm_file = OUTPUT_DIR / "confusion_matrices.json"
    with open(cm_file, "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # 3. Cohort Comparison (Dysarthric vs. Control)
    if "cohort" in labeled_df.columns:
        cohort_results = []
        for ch in ["dysarthric", "control"]:
            ch_df = labeled_df[labeled_df["cohort"] == ch]
            if len(ch_df) > 0:
                ch_dur = ch_df["duration_sec"].tolist()
                ch_gt = ch_df["ground_truth"].tolist()
                ch_spk = ch_df["speaker_id"].tolist()

                # Test standard Fixed 800ms vs Adaptive Patient
                base_800 = FixedThresholdBaseline(0.8).evaluate_batch(ch_dur)
                m_base = compute_turn_metrics(base_800, ch_gt, ch_dur)

                adap_preds = [classifier.predict(d, s, SensitivityPreset.PATIENT)[0] for d, s in zip(ch_dur, ch_spk)]
                m_adap = compute_turn_metrics(adap_preds, ch_gt, ch_dur)

                cohort_results.append({
                    "Cohort": ch.capitalize(),
                    "Count": len(ch_df),
                    "Fixed 800ms FIR": f"{m_base.false_interruption_rate * 100:.1f}%",
                    "Adaptive Patient FIR": f"{m_adap.false_interruption_rate * 100:.1f}%",
                    "FIR Reduction": f"{(m_base.false_interruption_rate - m_adap.false_interruption_rate) * 100:+.1f}%",
                    "Adaptive Accuracy": f"{m_adap.accuracy * 100:.1f}%",
                })
        cohort_df = pd.DataFrame(cohort_results)
        cohort_file = OUTPUT_DIR / "cohort_comparison.csv"
        cohort_df.to_csv(cohort_file, index=False)

    print(f"\nSaved benchmark results to {out_file} and {cm_file}")
    return summary_df


if __name__ == "__main__":
    run_comparative_benchmark()
