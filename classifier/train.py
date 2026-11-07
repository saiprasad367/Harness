"""
Train the Adaptive Pause Classifier on TORGO pause distributions.

Conversational turn-taking time-course formulation:
  - During early silence (0.1s - 0.5s), the user is pausing/thinking -> CONTINUE (0).
  - When silence extends past turn completion threshold -> END (1).
  - Sensitivity presets shift the decision boundary:
      * Quick mode (weight = -1.0): responds sooner.
      * Normal mode (weight = 0.0): balanced conversational timing.
      * Patient mode (weight = +1.0): allows extended hesitations before responding.
  - Features: [current_pause_sec, recent_mean_pause_sec, pause_std_dev, preset_weight].
  - StandardScaler fit strictly on training split only.
"""
import argparse
import logging
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.settings import OUTPUT_DIR
from config.sensitivity import SensitivityPreset, PRESET_CONFIGS
from classifier.feature_extraction import PauseFeatureExtractor, UserPauseTracker
from classifier.pause_classifier import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = OUTPUT_DIR / "models"


def generate_timecourse_samples(
    df: pd.DataFrame,
    preset: SensitivityPreset = SensitivityPreset.NORMAL,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate conversational turn-taking training instances from pause observations.
    Each pause event is sampled at key temporal decision points:
      - Short hesitation (0.1s - 0.4s) -> always CONTINUE (0)
      - Natural pause duration -> CONTINUE if within speaker hesitation window, END if exceeding cutoff
      - Long silence past turn end -> END (1)
    """
    tracker = UserPauseTracker(window_size=5)
    extractor = PauseFeatureExtractor(tracker)
    preset_cfg = PRESET_CONFIGS[preset]

    features = []
    labels = []

    # Sort chronologically by speaker
    df_sorted = df.sort_values(by=["speaker_id", "start_sec"]).copy()

    for _, row in df_sorted.iterrows():
        spk = row["speaker_id"]
        pause_dur = float(row["duration_sec"])
        is_trailing = bool(row.get("is_trailing", False))

        recent_mean, std_dev = tracker.get_stats(spk)

        # Baseline cutoff modulated by preset
        turn_cutoff = max(0.6, (recent_mean + 0.3) * (1.0 + 0.4 * preset_cfg.weight))

        # Sample 1: Early hesitation in this pause (e.g. 0.2s) -> CONTINUE
        f_early = extractor.extract_features(0.2, speaker_id=spk, preset=preset)
        features.append(f_early)
        labels.append(0)

        # Sample 2: Observed pause duration
        f_mid = extractor.extract_features(min(pause_dur, 0.6), speaker_id=spk, preset=preset)
        features.append(f_mid)
        # If it's a short pause (< cutoff), user was continuing
        labels.append(1 if pause_dur >= turn_cutoff and is_trailing else 0)

        # Sample 3: Trailing silence past turn cutoff -> END
        if is_trailing:
            f_end = extractor.extract_features(max(pause_dur, turn_cutoff + 0.4), speaker_id=spk, preset=preset)
            features.append(f_end)
            labels.append(1)

        # Update tracker with this observed pause
        tracker.add_pause(spk, pause_dur)

    feature_names = ["current_pause_sec", "recent_mean_pause_sec", "pause_std_dev", "preset_weight"]
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32), feature_names


def train_classifier(
    pauses_csv: Path,
    output_dir: Path = MODELS_DIR,
) -> dict:
    """
    Full training pipeline:
      1. Load extracted pauses.
      2. Split by predefined speaker-disjoint partition (train / val / test).
      3. Generate multi-preset timecourse samples.
      4. Fit StandardScaler on train split only.
      5. Fit LogisticRegression on scaled train features.
      6. Evaluate on validation split.
      7. Save model, scaler, and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pauses_csv.exists():
        raise FileNotFoundError(f"Pause data not found at {pauses_csv}. Run vad/extract_pauses.py first.")

    df = pd.read_csv(pauses_csv)
    log.info("Loaded %d pauses from %s", len(df), pauses_csv)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    log.info(
        "Split sizes — Train: %d (%d speakers), Val: %d (%d speakers), Test: %d (%d speakers)",
        len(train_df), train_df["speaker_id"].nunique(),
        len(val_df), val_df["speaker_id"].nunique(),
        len(test_df), test_df["speaker_id"].nunique(),
    )

    if len(train_df) == 0:
        raise ValueError("Train split is empty. Check split column in pause CSV.")

    # Generate multi-preset samples for robust generalizability
    all_X_train, all_y_train = [], []
    for preset in [SensitivityPreset.QUICK, SensitivityPreset.NORMAL, SensitivityPreset.PATIENT]:
        X_p, y_p, feat_names = generate_timecourse_samples(train_df, preset=preset)
        all_X_train.append(X_p)
        all_y_train.append(y_p)

    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)

    X_val, y_val, _ = generate_timecourse_samples(val_df, preset=SensitivityPreset.NORMAL)

    # Fit scaler on TRAIN ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Fit Logistic Regression
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    # Verify physical sanity of coefficients:
    # 1) current_pause_sec coefficient should be positive (longer pause -> higher prob of END)
    # 2) preset_weight coefficient should be negative (Patient mode / higher weight -> lower prob of END)
    log.info("Learned feature weights: %s", dict(zip(feat_names, clf.coef_[0].round(4))))

    # Evaluate on Validation
    val_preds = clf.predict(X_val_scaled)
    val_probs = clf.predict_proba(X_val_scaled)[:, 1]

    report = classification_report(y_val, val_preds, target_names=["CONTINUE", "END"], output_dict=True)
    cm = confusion_matrix(y_val, val_preds)

    try:
        auc = roc_auc_score(y_val, val_probs)
    except Exception:
        auc = 0.0

    log.info("Validation Results:\n%s", classification_report(y_val, val_preds, target_names=["CONTINUE", "END"]))
    log.info("Confusion Matrix:\n%s", cm)
    log.info("ROC-AUC: %.4f", auc)

    # Save artifacts
    model_path = output_dir / "pause_classifier.joblib"
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    log.info("Saved model to %s and scaler to %s", model_path, scaler_path)

    # Save training metadata
    meta = {
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "feature_names": feat_names,
        "val_accuracy": report["accuracy"],
        "val_f1_macro": report["macro avg"]["f1-score"],
        "val_auc": auc,
        "train_speakers": sorted(train_df["speaker_id"].unique().tolist()),
        "val_speakers": sorted(val_df["speaker_id"].unique().tolist()),
    }
    meta_df = pd.DataFrame([meta])
    meta_df.to_csv(output_dir / "training_metadata.csv", index=False)

    return meta


if __name__ == "__main__":
    train_classifier(pauses_csv=OUTPUT_DIR / "candidate_pauses.csv")
