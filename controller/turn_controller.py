"""Turn-taking controller, state machine, and response output gate."""
import time
from enum import Enum
from typing import Callable, List, Optional
import torch
from config.settings import ControllerConfig
from config.sensitivity import SensitivityPreset
from vad.silero_detector import StreamingVADUpdate, VADState
from classifier.pause_classifier import AdaptivePauseClassifier


class HarnessState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PAUSE_EVAL = "PAUSE_EVAL"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class ControllerAction(str, Enum):
    WAIT_AND_LISTEN = "WAIT_AND_LISTEN"
    BACKCHANNEL = "BACKCHANNEL"
    RELEASE_RESPONSE = "RELEASE_RESPONSE"
    HALT_BARGE_IN = "HALT_BARGE_IN"
    NO_OP = "NO_OP"


class TurnController:
    """Coordinates turn-taking, pause evaluation, gating, and barge-in interruptions."""

    def __init__(
        self,
        classifier: Optional[AdaptivePauseClassifier] = None,
        config: Optional[ControllerConfig] = None,
        preset: str | SensitivityPreset = SensitivityPreset.NORMAL,
    ):
        self.config = config or ControllerConfig()
        self.classifier = classifier or AdaptivePauseClassifier()
        self.preset = preset
        self.state = HarnessState.IDLE
        self.speaker_id = "default_user"
        
        # Audio response buffers
        self._output_buffer: List[torch.Tensor] = []
        self._is_gate_open: bool = False
        self._on_response_release_callback: Optional[Callable[[], None]] = None
        self._on_barge_in_callback: Optional[Callable[[], None]] = None

    def set_preset(self, preset: str | SensitivityPreset):
        self.preset = preset

    def set_callbacks(
        self,
        on_release: Optional[Callable[[], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
    ):
        self._on_response_release_callback = on_release
        self._on_barge_in_callback = on_barge_in

    def step(self, vad_update: StreamingVADUpdate) -> tuple[HarnessState, ControllerAction, dict]:
        """
        Processes a single VAD update and transitions state accordingly.
        Returns:
            (new_state, action_to_take, metadata_dict)
        """
        t0 = time.perf_counter()
        action = ControllerAction.NO_OP
        meta = {
            "speech_prob": vad_update.speech_prob,
            "pause_sec": vad_update.current_pause_sec,
            "decision": None,
            "end_prob": None,
            "decision_latency_ms": 0.0,
        }

        def finalize_step(state, act):
            meta["decision_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            return state, act, meta

        # 1. User is speaking
        if vad_update.state == VADState.SPEECH:
            if self.state == HarnessState.SPEAKING:
                # Barge-in event: User started speaking while assistant was talking
                if self.config.barge_in_enabled and vad_update.speech_prob >= self.config.barge_in_sensitivity_threshold:
                    self.state = HarnessState.INTERRUPTED
                    action = ControllerAction.HALT_BARGE_IN
                    self._clear_output_gate()
                    if self._on_barge_in_callback:
                        self._on_barge_in_callback()
                    return finalize_step(self.state, action)

            self.state = HarnessState.LISTENING
            action = ControllerAction.WAIT_AND_LISTEN
            return finalize_step(self.state, action)

        # 2. Silence / Pause observed
        if self.state in (HarnessState.LISTENING, HarnessState.PAUSE_EVAL):
            self.state = HarnessState.PAUSE_EVAL
            
            # Consult Adaptive Pause Classifier
            decision, end_prob = self.classifier.predict(
                current_pause_sec=vad_update.current_pause_sec,
                speaker_id=self.speaker_id,
                preset=self.preset,
            )
            meta["decision"] = decision
            meta["end_prob"] = end_prob

            if decision == "CONTINUE":
                action = ControllerAction.WAIT_AND_LISTEN
            else:  # decision == "END"
                # Turn finished! Open response gate
                self.state = HarnessState.SPEAKING
                action = ControllerAction.RELEASE_RESPONSE
                self._is_gate_open = True
                
                # Update user pause history with finished pause
                self.classifier.register_completed_pause(
                    self.speaker_id,
                    vad_update.current_pause_sec,
                )
                
                if self._on_response_release_callback:
                    self._on_response_release_callback()

            return finalize_step(self.state, action)

        # 3. Currently SPEAKING
        if self.state == HarnessState.SPEAKING:
            # Check if all assistant output has finished playing
            if len(self._output_buffer) == 0:
                self.state = HarnessState.IDLE
                self._is_gate_open = False

        return finalize_step(self.state, action)

    def queue_assistant_audio(self, audio_chunk: torch.Tensor):
        """Buffer assistant audio from full-duplex model."""
        self._output_buffer.append(audio_chunk)

    def fetch_audible_chunk(self) -> Optional[torch.Tensor]:
        """Release audio chunk to speaker only if output gate is open."""
        if self._is_gate_open and len(self._output_buffer) > 0:
            return self._output_buffer.pop(0)
        return None

    def _clear_output_gate(self):
        """Immediately drop buffered audio upon barge-in."""
        self._output_buffer.clear()
        self._is_gate_open = False

