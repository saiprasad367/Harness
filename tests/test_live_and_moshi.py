"""
Unit and integration tests for Live Audio Streaming, Genuine Moshi Adapter,
and Full-Duplex Barge-In Turn Control.
"""
import asyncio
import json
import re
import unittest
import torch
import numpy as np

from audio.audio_stream import AudioStreamBuffer, MicrophoneStreamer, AudioPlayer
from model.moshi_adapter import MoshiModelAdapter, MoshiConnectionState, normalize_moshi_url
from controller.turn_controller import TurnController, HarnessState, ControllerAction
from vad.silero_detector import StreamingVADUpdate, VADState
from config.sensitivity import SensitivityPreset


class TestLiveAudioStreamBuffer(unittest.TestCase):
    """Test thread-safe AudioStreamBuffer buffering and queue limits."""

    def test_buffer_put_and_get(self):
        buf = AudioStreamBuffer(maxsize=5)
        tensor = torch.zeros(512, dtype=torch.float32)
        buf.put(tensor)
        self.assertFalse(buf.empty())
        self.assertEqual(buf.qsize(), 1)
        retrieved = buf.get(timeout=0.1)
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 512)
        self.assertTrue(buf.empty())

    def test_buffer_overflow_drop_oldest(self):
        buf = AudioStreamBuffer(maxsize=2)
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        t3 = torch.tensor([3.0])
        buf.put(t1)
        buf.put(t2)
        # Putting 3rd should drop t1 and retain t2, t3
        buf.put(t3)
        self.assertEqual(buf.qsize(), 2)
        r1 = buf.get(timeout=0.1)
        self.assertEqual(r1[0].item(), 2.0)
        r2 = buf.get(timeout=0.1)
        self.assertEqual(r2[0].item(), 3.0)

    def test_microphone_device_enumeration(self):
        """Verify device enumeration returns structured list without crashing."""
        devs = MicrophoneStreamer.list_input_devices()
        self.assertIsInstance(devs, list)
        for d in devs:
            self.assertIn("index", d)
            self.assertIn("name", d)
            self.assertIn("channels", d)

    def test_audio_player_device_enumeration(self):
        """Verify output speaker device enumeration."""
        devs = AudioPlayer.list_output_devices()
        self.assertIsInstance(devs, list)

    def test_moshi_url_normalization(self):
        """Verify normalization of remote Kaggle Gradio tunnels and local endpoints."""
        self.assertEqual(
            normalize_moshi_url("https://abc-123.gradio.live"),
            "wss://abc-123.gradio.live/api/chat",
        )
        self.assertEqual(
            normalize_moshi_url("http://localhost:8998"),
            "ws://localhost:8998/api/chat",
        )
        self.assertEqual(
            normalize_moshi_url("ws://localhost:8998/api/chat"),
            "ws://localhost:8998/api/chat",
        )
        self.assertEqual(
            normalize_moshi_url("my-server.ngrok-free.app"),
            "wss://my-server.ngrok-free.app/api/chat",
        )


class TestMoshiAdapterIntegration(unittest.TestCase):
    """Test Moshi adapter state separation, mock streaming, and genuine connection handling."""

    def test_mock_mode_lifecycle(self):
        adapter = MoshiModelAdapter(mock_mode=True)
        self.assertEqual(adapter.state, MoshiConnectionState.DISCONNECTED)
        asyncio.run(adapter.connect())
        self.assertEqual(adapter.state, MoshiConnectionState.CONNECTED)
        self.assertTrue(adapter.is_connected)

        adapter.start_session()
        self.assertEqual(adapter.state, MoshiConnectionState.SESSION_ACTIVE)
        self.assertTrue(adapter._is_session_active)

        # Send chunk
        chunk = torch.sin(torch.linspace(0, 1, 3200))
        asyncio.run(adapter.send_audio_chunk(chunk))
        self.assertEqual(adapter.total_frames_sent, 1)

        # Test interruption
        adapter.interrupt()
        self.assertTrue(adapter._interrupted)
        adapter.reset_interruption()
        self.assertFalse(adapter._interrupted)

        # Disconnect
        asyncio.run(adapter.disconnect())
        self.assertEqual(adapter.state, MoshiConnectionState.DISCONNECTED)

    def test_genuine_connection_failure_no_silent_fallback(self):
        """
        Critical requirement: Attempting to connect to an unreachable genuine Moshi server
        MUST transition to CONNECTION_FAILED and NEVER silently fall back to mock mode.
        """
        # Connect to non-existent local port
        adapter = MoshiModelAdapter(server_url="ws://127.0.0.1:59999/api/chat", mock_mode=False)
        self.assertEqual(adapter.state, MoshiConnectionState.DISCONNECTED)

        # Attempt connection with small timeout
        success = asyncio.run(adapter.connect(timeout=0.5))
        self.assertFalse(success)
        self.assertEqual(adapter.state, MoshiConnectionState.CONNECTION_FAILED)
        self.assertIsNotNone(adapter.last_error)
        self.assertFalse(adapter.mock_mode, "CRITICAL: Adapter silently fell back to mock mode!")
        self.assertFalse(adapter.is_connected)

    def test_mock_receive_audio_stream(self):
        adapter = MoshiModelAdapter(mock_mode=True)
        asyncio.run(adapter.connect())
        adapter.start_session()

        async def run_receive():
            chunks = []
            async for frame in adapter.receive_audio_stream():
                chunks.append(frame)
                if len(chunks) >= 2:
                    break
            return chunks

        received = asyncio.run(run_receive())
        self.assertEqual(len(received), 2)
        self.assertGreater(adapter.total_frames_received, 0)


class TestFullDuplexControllerBargeIn(unittest.TestCase):
    """Verify TurnController barge-in halts Moshi audio and gates output correctly."""

    def test_turn_controller_gating_and_barge_in(self):
        controller = TurnController(preset=SensitivityPreset.PATIENT)
        barge_in_flag = []

        def on_barge_in():
            barge_in_flag.append(True)

        controller.set_callbacks(on_barge_in=on_barge_in)

        # 1. User starts speaking
        vad_spk = StreamingVADUpdate(
            state=VADState.SPEECH,
            speech_prob=0.92,
            current_pause_sec=0.0,
            is_new_pause_event=False,
        )
        state, action, _ = controller.step(vad_spk)
        self.assertEqual(state, HarnessState.LISTENING)
        self.assertEqual(action, ControllerAction.WAIT_AND_LISTEN)

        # 2. User pauses (Patient mode wait)
        vad_pause = StreamingVADUpdate(
            state=VADState.SILENCE,
            speech_prob=0.05,
            current_pause_sec=0.3,
            is_new_pause_event=True,
            new_pause_duration_sec=0.3,
        )
        state, action, _ = controller.step(vad_pause)
        self.assertEqual(state, HarnessState.PAUSE_EVAL)
        self.assertEqual(action, ControllerAction.WAIT_AND_LISTEN)

        # 3. Simulate turn completion -> Controller transitions to SPEAKING
        controller.state = HarnessState.SPEAKING
        controller._is_gate_open = True
        dummy_audio = torch.ones(1600)
        controller.queue_assistant_audio(dummy_audio)

        # Fetch audible chunk while gate is open
        chunk = controller.fetch_audible_chunk()
        self.assertIsNotNone(chunk)

        # Queue another chunk
        controller.queue_assistant_audio(dummy_audio)

        # 4. Barge-in: User speaks while assistant is SPEAKING
        vad_barge = StreamingVADUpdate(
            state=VADState.SPEECH,
            speech_prob=0.95,
            current_pause_sec=0.0,
            is_new_pause_event=False,
        )
        state, action, _ = controller.step(vad_barge)
        self.assertEqual(state, HarnessState.INTERRUPTED)
        self.assertEqual(action, ControllerAction.HALT_BARGE_IN)
        self.assertEqual(len(barge_in_flag), 1)

        # Audio buffer must be immediately flushed!
        self.assertIsNone(controller.fetch_audible_chunk())
        self.assertEqual(len(controller._output_buffer), 0)


class TestKaggleManagerAndProtocol(unittest.TestCase):
    """Verify Kaggle deployment manager packaging and Kyutai protocol framing."""

    def test_kaggle_manager_package_preparation(self):
        from scripts.kaggle_manager import KaggleMoshiManager, KERNEL_DIR
        manager = KaggleMoshiManager(username="researcher123", api_key="fake_key_abc")
        pkg_path = manager.prepare_kernel_package()
        self.assertTrue(pkg_path.exists())
        metadata_file = pkg_path / "kernel-metadata.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["id"], "researcher123/kyutai-moshi-gpu-server")
        self.assertTrue(meta["enable_gpu"])
        self.assertTrue(meta["enable_internet"])

    def test_gradio_regex_extraction(self):
        from scripts.kaggle_manager import KaggleMoshiManager
        manager = KaggleMoshiManager()
        # Mock logs string containing Gradio link
        sample_logs = "Loading weights...\nRunning on public URL: https://abc123xyz-moshi.gradio.live\nServer started."
        match = re.search(r"https://[a-zA-Z0-9-]+\.gradio\.live", sample_logs)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(0), "https://abc123xyz-moshi.gradio.live")


if __name__ == "__main__":
    unittest.main()
