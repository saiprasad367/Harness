"""Extract candidate pauses across TORGO recordings using Silero VAD."""
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm
from config.settings import OUTPUT_DIR, VADConfig
from audio.preprocessing import load_audio
from vad.silero_detector import SileroVADDetector
from dataset.dataset_manifest import load_manifest


def extract_candidate_pauses(
    manifest_df: Optional[pd.DataFrame] = None,
    output_csv: Optional[Path] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run Silero VAD across recordings and extract structured candidate pause events.
    """
    if manifest_df is None:
        manifest_df = load_manifest()
        
    if limit is not None:
        manifest_df = manifest_df.head(limit)
        
    if output_csv is None:
        output_csv = OUTPUT_DIR / "candidate_pauses.csv"

    detector = SileroVADDetector(config=VADConfig())
    records = []

    print(f"Extracting candidate pauses for {len(manifest_df)} recordings...")
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        file_path = row["file_path"]
        try:
            audio_tensor, sr = load_audio(file_path, target_sr=16000)
            speech_segs, pauses = detector.extract_speech_and_pauses(audio_tensor, sample_rate=sr)
            
            for p_idx, pause in enumerate(pauses):
                records.append({
                    "file_path": file_path,
                    "file_name": row["file_name"],
                    "speaker_id": row["speaker_id"],
                    "cohort": row["cohort"],
                    "gender": row["gender"],
                    "split": row["split"],
                    "pause_index": p_idx,
                    "start_sec": pause.start_sec,
                    "end_sec": pause.end_sec,
                    "duration_sec": pause.duration_sec,
                    "is_pre_speech": pause.is_pre_speech,
                    "is_trailing": pause.is_trailing,
                    "num_speech_segments": len(speech_segs),
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    pauses_df = pd.DataFrame(records)
    pauses_df.to_csv(output_csv, index=False)
    print(f"Extracted {len(pauses_df)} pauses -> Saved to {output_csv}")
    return pauses_df


if __name__ == "__main__":
    # Run pilot extraction on 50 recordings first (as in report pilot) or full set
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    extract_candidate_pauses(limit=limit)

