"""
Adaptive Pause Classifier using Scikit-Learn Logistic Regression + StandardScaler.

Design notes:
- Features are scaled before classification (StandardScaler fit on training data only).
- If no trained model exists, a synthetic fallback model is used (clearly documented).
- The synthetic model is NOT trained on real data and should NOT be used for research claims.
- Saves both the scaler and model to outputs/models/.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.settings import OUTPUT_DIR
from config.sensitivity import SensitivityPreset
from .feature_extraction import PauseFeatureExtractor, UserPauseTracker

log = logging.getLogger(__name__)

MODELS_DIR = OUTPUT_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "pause_classifier.joblib"
DEFAULT_SCALER_PATH = MODELS_DIR / "scaler.joblib"


class AdaptivePauseClassifier:
    """
    Classifies detected pauses as CONTINUE (0) or END (1).

    Two modes:
      - Trained mode: Uses a model fit on real TORGO-extracted features.
      - Synthetic fallback: Uses a 7-sample synthetic initialization for
        software testing only. NOT suitable for research claims.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        history_window: int = 5,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH

        self.tracker = UserPauseTracker(window_size=history_window)
        self.extractor = PauseFeatureExtractor(self.tracker)
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained_on_real_data: bool = False

        if self.model_path.exists() and self.scaler_path.exists():
            self._load_trained()
        else:
            log.warning(
                "No trained model found at %s. Using synthetic fallback. "
                "Do NOT use this for research evaluation.",
                self.model_path,
            )
            self._init_synthetic_fallback()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_synthetic_fallback(self):
        """
        Initialize with 7 hand-crafted examples for basic software testing.
        Features: [current_pause_sec, recent_mean_sec, std_dev, preset_weight]
        Labels:   0=CONTINUE, 1=END
        NOT for research use.
        """
        X = np.array([
            [0.2, 0.5, 0.1,  0.0],
            [0.4, 0.8, 0.2,  0.0],
            [0.8, 0.8, 0.2,  0.0],
            [1.5, 0.8, 0.2,  0.0],
            [2.5, 1.0, 0.3,  0.0],
            [1.2, 1.5, 0.4,  1.0],
            [0.5, 0.4, 0.1, -1.0],
        ], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1, 0, 1], dtype=np.int32)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained_on_real_data = False

    def _load_trained(self):
        """Load a previously saved trained model and scaler."""
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.is_trained_on_real_data = True
        log.info("Loaded trained classifier from %s", self.model_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        current_pause_sec: float,
        speaker_id: str = "default_user",
        preset: "str | SensitivityPreset" = SensitivityPreset.NORMAL,
    ) -> Tuple[str, float]:
        """
        Classify the current pause duration.

        Returns:
            decision: 'CONTINUE' or 'END'
            end_probability: probability ∈ [0.0, 1.0] that the user has finished their turn.
        """
        raw_features = self.extractor.extract_features(
            current_pause_sec, speaker_id, preset
        )
        X = raw_features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        prob_end = float(self.model.predict_proba(X_scaled)[0, 1])
        decision = "END" if prob_end >= 0.5 else "CONTINUE"
        return decision, prob_end

    def register_completed_pause(self, speaker_id: str, pause_duration_sec: float):
        """Update speaker's rolling pause history after a confirmed pause."""
        self.tracker.add_pause(speaker_id, pause_duration_sec)

    # ------------------------------------------------------------------
    # Training (called from classifier/train.py)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        max_iter: int = 500,
    ):
        """
        Fit the classifier on a labeled feature matrix.
        Fits the scaler on X (must be training data only — never fit on test/val).
        """
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained_on_real_data = True
        log.info("Classifier fitted on %d samples (%d END, %d CONTINUE)", len(y), (y==1).sum(), (y==0).sum())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """Save both model and scaler."""
        m_path = Path(model_path or self.model_path)
        s_path = Path(scaler_path or self.scaler_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, m_path)
        joblib.dump(self.scaler, s_path)
        log.info("Saved classifier to %s and scaler to %s", m_path, s_path)

    def load(self, model_path: Path, scaler_path: Path):
        """Load model and scaler from disk."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained_on_real_data = True
        log.info("Loaded model from %s", model_path)
