"""
Evaluation metrics for intelligent turn-taking:
False Interruption Rate (FIR), False Wait Rate (FWR), Latency percentiles,
Precision, Recall, F1 score, and Confusion Matrix.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np


@dataclass
class TurnMetrics:
    total_evaluations: int
    continuation_opportunities: int
    completed_turns: int
    false_interruptions: int     # FP: Predicted END during CONTINUE opportunity
    false_waits: int             # FN: Predicted CONTINUE when user has reached END
    true_completions: int        # TP: Correctly identified END
    true_continuations: int      # TN: Correctly waited during CONTINUE
    false_interruption_rate: float # FIR = FP / (FP + TN) (Lower is better)
    false_wait_rate: float         # FWR = FN / (TP + FN) (Lower is better)
    accuracy: float
    precision: float             # TP / (TP + FP)
    recall: float                # TP / (TP + FN)
    f1_score: float              # 2 * (P * R) / (P + R)
    avg_latency_sec: float
    p50_latency_sec: float
    p90_latency_sec: float
    confusion_matrix: Dict[str, int]


def compute_turn_metrics(
    predictions: Sequence[str],
    ground_truth: Sequence[str],
    pause_durations: Sequence[float],
) -> TurnMetrics:
    """
    Computes rigorous empirical turn-taking metrics:
      - ground_truth: Sequence of 'CONTINUE' or 'END'
      - predictions: Sequence of 'CONTINUE' or 'END'
      - pause_durations: Observed silence durations in seconds
    """
    n = len(predictions)
    if n == 0 or len(ground_truth) != n:
        raise ValueError("Mismatched prediction and ground truth lengths.")

    tp = 0  # GT=END, Pred=END (Correct completion detection)
    fp = 0  # GT=CONTINUE, Pred=END (False Interruption / premature barge-in)
    tn = 0  # GT=CONTINUE, Pred=CONTINUE (Correct hesitation wait)
    fn = 0  # GT=END, Pred=CONTINUE (False Wait / unnecessary latency)
    latencies = []

    for pred, gt, duration in zip(predictions, ground_truth, pause_durations):
        p_clean = str(pred).upper().strip()
        g_clean = str(gt).upper().strip()

        if g_clean == "CONTINUE":
            if p_clean == "END":
                fp += 1
            else:
                tn += 1
        elif g_clean == "END":
            if p_clean == "CONTINUE":
                fn += 1
            else:
                tp += 1
                latencies.append(duration)

    continuation_opps = fp + tn
    completed_turns = tp + fn

    fir = (fp / continuation_opps) if continuation_opps > 0 else 0.0
    fwr = (fn / completed_turns) if completed_turns > 0 else 0.0
    
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p90_lat = float(np.percentile(latencies, 90)) if latencies else 0.0

    cm = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    return TurnMetrics(
        total_evaluations=n,
        continuation_opportunities=continuation_opps,
        completed_turns=completed_turns,
        false_interruptions=fp,
        false_waits=fn,
        true_completions=tp,
        true_continuations=tn,
        false_interruption_rate=round(fir, 4),
        false_wait_rate=round(fwr, 4),
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        avg_latency_sec=round(avg_lat, 4),
        p50_latency_sec=round(p50_lat, 4),
        p90_latency_sec=round(p90_lat, 4),
        confusion_matrix=cm,
    )
