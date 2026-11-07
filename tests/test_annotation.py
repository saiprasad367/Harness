"""
Unit tests for the Human Annotation Pilot subsystem.
Tests:
  - Annotation saving and loading
  - Duplicate prevention and overwriting
  - Invalid timestamps rejection
  - Missing audio file detection
  - Label validation (strictly CONTINUE, END, UNCERTAIN)
  - Resume functionality
  - Speaker and recording metadata integrity
  - Quality report generation
"""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from dataset.annotation_pilot import AnnotationManager, VALID_LABELS, slice_audio_context


class TestAnnotationPilot(unittest.TestCase):

    def setUp(self):
        # Create isolated temporary directory for testing annotations
        self.test_dir = tempfile.TemporaryDirectory()
        self.ann_file = Path(self.test_dir.name) / "test_annotations.csv"
        self.manager = AnnotationManager(annotation_file=self.ann_file)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_save_and_load_valid_annotation(self):
        ok, msg = self.manager.save_annotation(
            candidate_id="cand_001",
            file_name="sample_01.wav",
            speaker_id="FC01",
            cohort="control",
            start_sec=1.5,
            end_sec=2.5,
            duration_sec=1.0,
            label="CONTINUE",
            annotator="annotator_A",
            confidence="high",
            notes="Audible breath before continuing",
        )
        self.assertTrue(ok)
        self.assertIn("saved successfully", msg)

        # Verify loaded data
        df = self.manager.load_annotations()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["label"], "CONTINUE")
        self.assertEqual(df.iloc[0]["speaker_id"], "FC01")
        self.assertEqual(df.iloc[0]["start_sec"], 1.5)

    def test_duplicate_prevention(self):
        # Save first annotation
        self.manager.save_annotation(
            candidate_id="cand_001",
            file_name="sample_01.wav",
            speaker_id="FC01",
            cohort="control",
            start_sec=1.5,
            end_sec=2.5,
            duration_sec=1.0,
            label="CONTINUE",
            annotator="annotator_A",
        )

        # Attempt to save duplicate (same file and start_sec) with updated label
        ok, msg = self.manager.save_annotation(
            candidate_id="cand_001",
            file_name="sample_01.wav",
            speaker_id="FC01",
            cohort="control",
            start_sec=1.5,
            end_sec=2.5,
            duration_sec=1.0,
            label="END",
            annotator="annotator_A",
            notes="Revised label",
        )
        self.assertTrue(ok)
        self.assertIn("Updated existing", msg)

        # Total rows should still be 1 (no duplicate row added)
        df = self.manager.load_annotations()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["label"], "END")
        self.assertEqual(df.iloc[0]["notes"], "Revised label")

    def test_label_validation(self):
        # Invalid label should be rejected
        ok, msg = self.manager.save_annotation(
            candidate_id="cand_002",
            file_name="sample_02.wav",
            speaker_id="F01",
            cohort="dysarthric",
            start_sec=0.5,
            end_sec=1.5,
            duration_sec=1.0,
            label="INVALID_LABEL",
            annotator="annotator_A",
        )
        self.assertFalse(ok)
        self.assertIn("Invalid label", msg)

        # Valid labels should all be accepted
        for lbl in ["CONTINUE", "END", "UNCERTAIN"]:
            ok, _ = self.manager.save_annotation(
                candidate_id=f"cand_{lbl}",
                file_name=f"sample_{lbl}.wav",
                speaker_id="M01",
                cohort="dysarthric",
                start_sec=1.0,
                end_sec=2.0,
                duration_sec=1.0,
                label=lbl,
                annotator="annotator_A",
            )
            self.assertTrue(ok)

    def test_invalid_timestamps_rejection(self):
        # start >= end should fail
        ok, msg = self.manager.save_annotation(
            candidate_id="cand_bad_ts",
            file_name="sample_bad.wav",
            speaker_id="FC01",
            cohort="control",
            start_sec=3.0,
            end_sec=2.0,  # Invalid
            duration_sec=-1.0,
            label="CONTINUE",
            annotator="annotator_A",
        )
        self.assertFalse(ok)
        self.assertIn("Invalid timestamps", msg)

        # negative start_sec should fail
        ok2, msg2 = self.manager.save_annotation(
            candidate_id="cand_neg_ts",
            file_name="sample_neg.wav",
            speaker_id="FC01",
            cohort="control",
            start_sec=-0.5,
            end_sec=1.0,
            duration_sec=1.5,
            label="CONTINUE",
            annotator="annotator_A",
        )
        self.assertFalse(ok2)

    def test_missing_audio_file_detection(self):
        # If an explicit file_path is passed that doesn't exist, fail
        ok, msg = self.manager.save_annotation(
            candidate_id="cand_missing",
            file_name="missing.wav",
            speaker_id="FC01",
            cohort="control",
            start_sec=1.0,
            end_sec=2.0,
            duration_sec=1.0,
            label="CONTINUE",
            annotator="annotator_A",
            file_path="C:/non_existent_path/fake_audio.wav",
        )
        self.assertFalse(ok)
        self.assertIn("does not exist", msg)

    def test_resume_functionality(self):
        # Create mock candidate list with 3 candidates
        candidates = pd.DataFrame([
            {"file_name": "f1.wav", "start_sec": 1.0, "end_sec": 2.0},
            {"file_name": "f2.wav", "start_sec": 3.0, "end_sec": 4.0},
            {"file_name": "f3.wav", "start_sec": 5.0, "end_sec": 6.0},
        ])

        # Initially, resume index is 0
        idx0 = self.manager.get_resume_index(candidates)
        self.assertEqual(idx0, 0)

        # Annotate f1
        self.manager.save_annotation(
            candidate_id="c1", file_name="f1.wav", speaker_id="FC01",
            cohort="control", start_sec=1.0, end_sec=2.0, duration_sec=1.0,
            label="CONTINUE", annotator="ann_1"
        )

        # Next unannotated should be index 1 (f2.wav)
        idx1 = self.manager.get_resume_index(candidates)
        self.assertEqual(idx1, 1)

    def test_quality_report_metrics(self):
        # Add 3 annotations: 2 CONTINUE, 1 END
        self.manager.save_annotation("c1", "f1.wav", "FC01", "control", 1.0, 2.0, 1.0, "CONTINUE", "ann_1")
        self.manager.save_annotation("c2", "f2.wav", "M01", "dysarthric", 2.0, 3.5, 1.5, "CONTINUE", "ann_1")
        self.manager.save_annotation("c3", "f3.wav", "FC01", "control", 0.5, 1.5, 1.0, "END", "ann_1")

        report = self.manager.generate_quality_report()
        self.assertEqual(report["total_annotated"], 3)
        self.assertEqual(report["counts_by_label"]["CONTINUE"], 2)
        self.assertEqual(report["counts_by_label"]["END"], 1)
        self.assertEqual(report["num_speakers"], 2)
        self.assertEqual(report["invalid_records_count"], 0)


if __name__ == "__main__":
    unittest.main()

