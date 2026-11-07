"""Comprehensive unit and integration test suite for the Adaptive Voice Harness."""
import unittest
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from config.settings import AudioConfig, VADConfig, OUTPUT_DIR
from config.sensitivity import SensitivityPreset, get_preset_config
from audio.preprocessing import frame_generator, normalize_audio, resample_audio
from vad.silero_detector import SileroVADDetector, VADState, StreamingVADUpdate, SILERO_FRAME_SIZE
from classifier.feature_extraction import PauseFeatureExtractor, UserPauseTracker
from classifier.pause_classifier import AdaptivePauseClassifier
from controller.turn_controller import TurnController, HarnessState, ControllerAction
from evaluation.baseline import FixedThresholdBaseline
from evaluation.metrics import compute_turn_metrics, TurnMetrics
from dataset.dataset_manifest import get_speaker_split, SPEAKER_SPLITS, TORGOMeta
from model.moshi_adapter import MoshiModelAdapter


class TestAudioAndPreprocessing(unittest.TestCase):

    def test_audio_framing_exact_and_padding(self):
        # 16000 samples = 1 sec
        audio = torch.zeros(16000)
        frames = list(frame_generator(audio, frame_size_samples=1600))
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[0].shape[0], 1600)

        # Partial frame padding test
        audio_odd = torch.zeros(1500)
        frames_odd = list(frame_generator(audio_odd, frame_size_samples=1000))
        self.assertEqual(len(frames_odd), 2)
        self.assertEqual(frames_odd[0].shape[0], 1000)
        self.assertEqual(frames_odd[1].shape[0], 1000)

    def test_normalize_audio(self):
        arr = np.array([-2.0, 0.0, 1.0, 4.0], dtype=np.float32)
        norm = normalize_audio(arr)
        self.assertAlmostEqual(float(np.max(np.abs(norm))), 1.0)
        self.assertEqual(norm[3], 1.0)
        self.assertEqual(norm[0], -0.5)


class TestSileroVADRobustness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = SileroVADDetector()

    def test_vad_arbitrary_chunk_sizes_no_crash(self):
        """Silero requires 512 samples. Verify detector buffers and handles any chunk size."""
        test_sizes = [100, 512, 1024, 2560, 3200]
        for size in test_sizes:
            chunk = torch.randn(size)
            update = self.detector.process_chunk(chunk)
            self.assertIsInstance(update, StreamingVADUpdate)
            self.assertIn(update.state, (VADState.SPEECH, VADState.SILENCE))
            self.assertGreaterEqual(update.speech_prob, 0.0)
            self.assertLessEqual(update.speech_prob, 1.0)
        self.detector.reset_stream()

    def test_vad_offline_extraction_synthetic_audio(self):
        # 2 seconds of silence, 1 second of pseudo-speech (high amplitude noise), 2 seconds of silence
        sr = 16000
        silence_1 = torch.zeros(sr * 2)
        speech = torch.sin(torch.linspace(0, 2 * np.pi * 300, sr)) * 0.8
        silence_2 = torch.zeros(sr * 2)
        audio = torch.cat([silence_1, speech, silence_2])

        speech_segs, pauses = self.detector.extract_speech_and_pauses(audio, sample_rate=sr)
        self.assertIsInstance(speech_segs, list)
        self.assertIsInstance(pauses, list)
        self.assertGreater(len(pauses), 0)


class TestSpeakerDisjointSplits(unittest.TestCase):

    def test_no_speaker_leakage(self):
        train = set(SPEAKER_SPLITS["train"])
        val = set(SPEAKER_SPLITS["val"])
        test = set(SPEAKER_SPLITS["test"])

        self.assertEqual(len(train.intersection(val)), 0, "Speaker leakage: Train & Val overlap!")
        self.assertEqual(len(train.intersection(test)), 0, "Speaker leakage: Train & Test overlap!")
        self.assertEqual(len(val.intersection(test)), 0, "Speaker leakage: Val & Test overlap!")

    def test_path_metadata_parsing(self):
        p1 = Path("dataset/M_Con/wav_headMic_MC01S01/file.wav")
        meta1 = TORGOMeta.parse_path(p1)
        self.assertIsNotNone(meta1)
        self.assertEqual(meta1["speaker_id"], "MC01")
        self.assertEqual(meta1["cohort"], "control")
        self.assertEqual(meta1["split"], "train")

        p2 = Path("dataset/M_Dys/wav_headMic_M05S02/file.wav")
        meta2 = TORGOMeta.parse_path(p2)
        self.assertIsNotNone(meta2)
        self.assertEqual(meta2["speaker_id"], "M05")
        self.assertEqual(meta2["cohort"], "dysarthric")
        self.assertEqual(meta2["split"], "test")


class TestClassifierAndScaler(unittest.TestCase):

    def test_feature_extractor_and_tracker(self):
        tracker = UserPauseTracker(window_size=3)
        tracker.add_pause("spk1", 1.0)
        tracker.add_pause("spk1", 2.0)
        tracker.add_pause("spk1", 3.0)
        mean_p, std_p = tracker.get_stats("spk1")
        self.assertAlmostEqual(mean_p, 2.0)

        extractor = PauseFeatureExtractor(tracker)
        vec = extractor.extract_features(1.5, speaker_id="spk1", preset=SensitivityPreset.PATIENT)
        self.assertEqual(len(vec), 4)
        self.assertEqual(vec[0], 1.5)
        self.assertAlmostEqual(vec[1], 2.0)
        self.assertEqual(vec[3], 1.0)  # Patient weight = 1.0

    def test_classifier_fit_save_and_predict(self):
        clf = AdaptivePauseClassifier()
        # Train synthetic toy matrix
        X = np.array([
            [0.2, 0.5, 0.1, 0.0],
            [0.4, 0.8, 0.2, 0.0],
            [1.8, 0.8, 0.2, 0.0],
            [2.5, 0.8, 0.2, 0.0],
        ], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        clf.fit(X, y)

        dec_short, prob_short = clf.predict(0.2, preset=SensitivityPreset.NORMAL)
        self.assertEqual(dec_short, "CONTINUE")
        self.assertLess(prob_short, 0.5)

        dec_long, prob_long = clf.predict(2.2, preset=SensitivityPreset.NORMAL)
        self.assertEqual(dec_long, "END")
        self.assertGreater(prob_long, 0.5)


class TestTurnControllerAndBargeIn(unittest.TestCase):

    def test_state_transitions(self):
        controller = TurnController(preset=SensitivityPreset.NORMAL)

        # Speech input -> LISTENING
        v_speech = StreamingVADUpdate(speech_prob=0.85, state=VADState.SPEECH, current_pause_sec=0.0, is_new_pause_event=False)
        state, action, _ = controller.step(v_speech)
        self.assertEqual(state, HarnessState.LISTENING)
        self.assertEqual(action, ControllerAction.WAIT_AND_LISTEN)

        # Short pause -> PAUSE_EVAL & CONTINUE
        v_short_pause = StreamingVADUpdate(speech_prob=0.02, state=VADState.SILENCE, current_pause_sec=0.2, is_new_pause_event=True)
        state, action, meta = controller.step(v_short_pause)
        self.assertEqual(state, HarnessState.PAUSE_EVAL)
        self.assertEqual(action, ControllerAction.WAIT_AND_LISTEN)

        # Long pause -> SPEAKING (response gate opened)
        v_long_pause = StreamingVADUpdate(speech_prob=0.01, state=VADState.SILENCE, current_pause_sec=3.0, is_new_pause_event=False)
        state, action, meta = controller.step(v_long_pause)
        self.assertEqual(state, HarnessState.SPEAKING)
        self.assertEqual(action, ControllerAction.RELEASE_RESPONSE)

        # Barge-in: user speaks while assistant is speaking -> INTERRUPTED
        state, action, _ = controller.step(v_speech)
        self.assertEqual(state, HarnessState.INTERRUPTED)
        self.assertEqual(action, ControllerAction.HALT_BARGE_IN)


class TestEvaluationMetrics(unittest.TestCase):

    def test_fir_and_fwr_calculation(self):
        # 2 CONTINUE opportunities: 1 correctly predicted CONTINUE, 1 falsely predicted END (premature interruption)
        # 2 END opportunities: 1 correctly predicted END, 1 falsely predicted CONTINUE (false wait)
        preds = ["CONTINUE", "END", "CONTINUE", "END"]
        gt = ["CONTINUE", "CONTINUE", "END", "END"]
        durations = [0.4, 0.9, 1.2, 1.6]

        m = compute_turn_metrics(preds, gt, durations)
        self.assertEqual(m.continuation_opportunities, 2)
        self.assertEqual(m.false_interruptions, 1)
        self.assertEqual(m.false_interruption_rate, 0.5)

        self.assertEqual(m.completed_turns, 2)
        self.assertEqual(m.false_waits, 1)
        self.assertEqual(m.false_wait_rate, 0.5)
        self.assertEqual(m.accuracy, 0.5)


class TestMoshiAdapterMock(unittest.TestCase):

    async def _async_test_moshi(self):
        adapter = MoshiModelAdapter(mock_mode=True)
        await adapter.connect()
        self.assertTrue(adapter.is_connected)

        chunk = torch.zeros(1600)
        await adapter.send_audio_chunk(chunk)

        responses = []
        async for r in adapter.receive_audio_stream():
            responses.append(r)
        self.assertGreater(len(responses), 0)
        await adapter.disconnect()
        self.assertFalse(adapter.is_connected)

    def test_moshi_mock_execution(self):
        import asyncio
        asyncio.run(self._async_test_moshi())


if __name__ == "__main__":
    unittest.main()
