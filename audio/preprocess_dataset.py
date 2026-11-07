"""
Batch Audio Preprocessing Pipeline for the TORGO Speech Dataset.
Performs:
  - Format validation & corrupt file detection
  - Multi-channel to mono conversion
  - Resampling to target 16 kHz using scipy.signal
  - Peak amplitude normalization
  - Writing standardized WAV files to dataset/processed/
  - Generating outputs/preprocessing_report.csv
"""
import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config.settings import DATASET_DIR, OUTPUT_DIR
from audio.preprocessing import load_audio
from dataset.dataset_manifest import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = DATASET_DIR / "processed"


def preprocess_dataset(
    manifest_df: Optional[pd.DataFrame] = None,
    output_dir: Path = PROCESSED_DIR,
    report_path: Optional[Path] = None,
    target_sr: int = 16000,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Standardizes audio files into target_sr mono WAVs and generates an audit report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_path = OUTPUT_DIR / "preprocessing_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_df is None:
        manifest_df = load_manifest()

    if limit is not None and limit > 0:
        manifest_df = manifest_df.head(limit)

    log.info("Starting audio preprocessing for %d files -> %s", len(manifest_df), output_dir)

    report_records = []

    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Preprocessing audio"):
        raw_path = Path(row["file_path"])
        file_name = row["file_name"]
        speaker_id = row["speaker_id"]

        record = {
            "file_name": file_name,
            "speaker_id": speaker_id,
            "cohort": row["cohort"],
            "split": row["split"],
            "raw_path": str(raw_path),
            "original_sample_rate": None,
            "original_channels": None,
            "original_duration_sec": None,
            "processed_sample_rate": target_sr,
            "processed_duration_sec": None,
            "processed_path": None,
            "has_clipping": False,
            "status": "PENDING",
            "error_msg": None,
        }

        try:
            # Inspect original file metadata before processing
            raw_info = sf.info(str(raw_path))
            record["original_sample_rate"] = raw_info.samplerate
            record["original_channels"] = raw_info.channels
            record["original_duration_sec"] = round(raw_info.duration, 3)

            # Load, convert to mono, resample to 16kHz, normalize
            audio_tensor, sr = load_audio(raw_path, target_sr=target_sr)
            audio_np = audio_tensor.cpu().numpy()

            # Quality checks
            record["has_clipping"] = bool(np.any(np.abs(audio_np) >= 0.999))
            record["processed_duration_sec"] = round(len(audio_np) / target_sr, 3)

            # Output path partitioned by speaker
            speaker_out_dir = output_dir / speaker_id
            speaker_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = speaker_out_dir / file_name

            # Write standard 16-bit PCM 16kHz WAV
            sf.write(str(out_file), audio_np, target_sr, subtype="PCM_16")

            record["processed_path"] = str(out_file.resolve())
            record["status"] = "SUCCESS"

        except Exception as e:
            record["status"] = "FAILED"
            record["error_msg"] = str(e)
            log.warning("Failed processing %s: %e", raw_path, e)

        report_records.append(record)

    report_df = pd.DataFrame(report_records)
    report_df.to_csv(report_path, index=False)
    log.info("Preprocessing complete: %d files processed. Report written to %s", len(report_df), report_path)
    return report_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TORGO audio files.")
    parser.add_argument("--limit", type=int, default=100, help="Number of files to preprocess (default 100 for pilot)")
    args = parser.parse_args()

    preprocess_dataset(limit=args.limit)
