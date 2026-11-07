"""
Interactive Terminal Simulation for the Adaptive Voice Harness.
Simulates real-time full-duplex turn-taking and barge-in in the terminal.
Supports both offline TORGO file streaming and live microphone streaming.

Usage:
  python cli_simulate.py [--speaker FC01] [--preset patient] [--barge-in]
  python cli_simulate.py --mic [--device-index 1] [--preset patient]
"""
import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import torch

from config.settings import VADConfig
from config.sensitivity import SensitivityPreset, PRESET_CONFIGS
from audio.preprocessing import load_audio
from audio.audio_stream import MicrophoneStreamer
from vad.silero_detector import SileroVADDetector, VADState
from classifier.pause_classifier import AdaptivePauseClassifier
from controller.turn_controller import TurnController, HarnessState, ControllerAction
from evaluation.baseline import FixedThresholdBaseline
from dataset.dataset_manifest import load_manifest
from model.moshi_adapter import MoshiModelAdapter


def run_live_mic_cli(
    preset: str = "patient",
    device_index: Optional[int] = None,
    max_duration_sec: float = 30.0,
):
    print("=" * 75)
    print("🎙️ ADAPTIVE VOICE HARNESS — LIVE MICROPHONE TERMINAL STREAMING")
    print("=" * 75)

    preset_enum = SensitivityPreset(preset.lower())
    preset_cfg = PRESET_CONFIGS[preset_enum]

    print(f"Preset Mode:      {preset.upper()} (wait boost: +{preset_cfg.min_wait_boost_sec}s, cutoff: {preset_cfg.baseline_threshold_sec}s)")
    print(f"Max Session Time: {max_duration_sec}s (Press Ctrl+C to terminate early)")

    mic = MicrophoneStreamer(sample_rate=16000, chunk_size_ms=200, device_index=device_index)
    mic.start()
    print("Microphone active! Speak into your microphone...")
    print("-" * 75)

    detector = SileroVADDetector()
    classifier = AdaptivePauseClassifier()
    controller = TurnController(classifier=classifier, preset=preset_enum)
    controller.speaker_id = "live_user"
    baseline = FixedThresholdBaseline(threshold_sec=preset_cfg.baseline_threshold_sec)
    moshi = MoshiModelAdapter(mock_mode=True)

    print(f"{'TIME':<8} | {'VAD':<10} | {'PROB':<6} | {'SILENCE':<8} | {'ADAPTIVE':<12} | {'BASELINE':<10} | {'ACTION':<18}")
    print("-" * 75)

    start_time = time.time()
    chunk_idx = 0
    try:
        while time.time() - start_time < max_duration_sec:
            chunk = mic.get_chunk(timeout=0.4)
            if chunk is None:
                continue

            chunk_idx += 1
            t_sec = round(chunk_idx * 0.20, 2)

            vad_update = detector.process_chunk(chunk)
            new_state, action, meta = controller.step(vad_update)

            curr_pause = vad_update.current_pause_sec
            base_dec = baseline.predict(curr_pause) if vad_update.state == VADState.SILENCE else "SPEAKING"
            adap_dec = meta.get("decision") or ("SPEAKING" if vad_update.state == VADState.SPEECH else "CONTINUE")

            vad_str = "SPEECH" if vad_update.state == VADState.SPEECH else "SILENCE"
            action_str = action.value
            if action == ControllerAction.RELEASE_RESPONSE:
                action_str = ">>> RELEASE (END)"
            elif action == ControllerAction.HALT_BARGE_IN:
                action_str = "!!! BARGE-IN CUT"

            print(f"{t_sec:5.2f}s   | {vad_str:<10} | {vad_update.speech_prob:<6.2f} | {curr_pause:5.2f}s  | {adap_dec:<12} | {base_dec:<10} | {action_str:<18}")

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        mic.stop()
        print("-" * 75)
        print("Live microphone stream closed cleanly.")


def run_cli_simulation(
    file_path: Optional[str] = None,
    speaker_id: Optional[str] = None,
    preset: str = "patient",
    simulate_barge_in: bool = False,
):
    print("=" * 75)
    print("🎙️ ADAPTIVE VOICE HARNESS — REAL-TIME FULL-DUPLEX SIMULATION")
    print("=" * 75)

    # 1. Pick audio file
    if file_path is None:
        df = load_manifest()
        if speaker_id:
            df = df[df["speaker_id"] == speaker_id]
        if len(df) == 0:
            print("No matching recordings found.")
            return
        sample_row = df.iloc[0]
        file_path = sample_row["file_path"]
        speaker_id = sample_row["speaker_id"]
        cohort = sample_row["cohort"]
    else:
        speaker_id = speaker_id or "user"
        cohort = "unknown"

    preset_enum = SensitivityPreset(preset.lower())
    preset_cfg = PRESET_CONFIGS[preset_enum]

    print(f"Audio Sample: {Path(file_path).name}")
    print(f"Speaker ID:   {speaker_id} ({cohort})")
    print(f"Preset Mode:  {preset.upper()} (wait boost: +{preset_cfg.min_wait_boost_sec}s, cutoff: {preset_cfg.baseline_threshold_sec}s)")
    print(f"Simulate Barge-In: {simulate_barge_in}")
    print("-" * 75)

    # 2. Initialize harness pipeline
    detector = SileroVADDetector()
    classifier = AdaptivePauseClassifier()
    controller = TurnController(classifier=classifier, preset=preset_enum)
    controller.speaker_id = speaker_id
    baseline = FixedThresholdBaseline(threshold_sec=preset_cfg.baseline_threshold_sec)
    moshi = MoshiModelAdapter(mock_mode=True)

    audio_tensor, sr = load_audio(file_path, target_sr=16000)
    chunk_samples = 3200  # 200ms
    total_chunks = len(audio_tensor) // chunk_samples

    print(f"Audio loaded: {len(audio_tensor)/16000:.2f}s ({total_chunks} streaming chunks)")
    print(f"{'TIME':<8} | {'VAD':<10} | {'PROB':<6} | {'SILENCE':<8} | {'ADAPTIVE':<12} | {'BASELINE':<10} | {'ACTION':<18}")
    print("-" * 75)

    for i in range(total_chunks):
        t_sec = i * 0.20
        chunk = audio_tensor[i * chunk_samples : (i + 1) * chunk_samples]

        # VAD update
        vad_update = detector.process_chunk(chunk)

        # Optional simulated barge-in: inject speech when assistant is speaking
        if simulate_barge_in and controller.state == HarnessState.SPEAKING and i % 3 == 0:
            vad_update.state = VADState.SPEECH
            vad_update.speech_prob = 0.95

        new_state, action, meta = controller.step(vad_update)

        curr_pause = vad_update.current_pause_sec
        base_dec = baseline.predict(curr_pause) if vad_update.state == VADState.SILENCE else "SPEAKING"
        adap_dec = meta.get("decision") or ("SPEAKING" if vad_update.state == VADState.SPEECH else "CONTINUE")

        vad_str = "SPEECH" if vad_update.state == VADState.SPEECH else "SILENCE"
        action_str = action.value

        if action == ControllerAction.RELEASE_RESPONSE:
            action_str = ">>> RELEASE (END)"
        elif action == ControllerAction.HALT_BARGE_IN:
            action_str = "!!! BARGE-IN CUT"

        print(f"{t_sec:5.2f}s   | {vad_str:<10} | {vad_update.speech_prob:<6.2f} | {curr_pause:5.2f}s  | {adap_dec:<12} | {base_dec:<10} | {action_str:<18}")

        if controller._is_gate_open:
            dummy_resp = torch.zeros(1600)
            controller.queue_assistant_audio(dummy_resp)

        time.sleep(0.04)

    print("-" * 75)
    print("Simulation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLI turn-taking simulation.")
    parser.add_argument("--mic", action="store_true", help="Capture live microphone input")
    parser.add_argument("--device-index", type=int, default=None, help="Input audio device index for microphone")
    parser.add_argument("--file", type=str, default=None, help="Path to custom WAV file")
    parser.add_argument("--speaker", type=str, default="FC01", help="Speaker ID from TORGO")
    parser.add_argument("--preset", type=str, default="patient", choices=["patient", "normal", "quick"])
    parser.add_argument("--barge-in", action="store_true", help="Simulate user barge-in while assistant speaks")
    parser.add_argument("--duration", type=float, default=20.0, help="Max recording duration for live mic in seconds")
    args = parser.parse_args()

    if args.mic:
        run_live_mic_cli(
            preset=args.preset,
            device_index=args.device_index,
            max_duration_sec=args.duration,
        )
    else:
        run_cli_simulation(
            file_path=args.file,
            speaker_id=args.speaker,
            preset=args.preset,
            simulate_barge_in=args.barge_in,
        )
