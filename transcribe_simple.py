import argparse
import sys
from pathlib import Path

import whisper
import torch
from tqdm import tqdm


class ProgressCallback:
    """Callback to track Whisper transcription progress."""
    
    def __init__(self, total_duration: float):
        self.pbar = tqdm(total=100, desc="Transcribing", unit="%", ncols=80, bar_format='{l_bar}{bar}| {n:.1f}%')
        self.total_duration = total_duration
        self.last_progress = 0
    
    def __call__(self, segment):
        if self.total_duration > 0:
            current_time = segment.get('end', segment.get('start', 0))
            progress = min(100, (current_time / self.total_duration) * 100)
            delta = progress - self.last_progress
            if delta > 0:
                self.pbar.update(delta)
                self.last_progress = progress
    
    def close(self):
        # Ensure we reach 100%
        if self.last_progress < 100:
            self.pbar.update(100 - self.last_progress)
        self.pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper medium on GPU.")
    parser.add_argument("audio", type=Path, help="Path to the audio file (WAV, MP3, etc.)")
    parser.add_argument("--output", type=Path, default=None, help="Output text file for transcript (default: audio_name.txt)")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g., en, hi for Hindi, or leave blank for auto-detect)")
    
    args = parser.parse_args()
    
    if not args.audio.exists():
        sys.exit(f"Audio file not found: {args.audio}")
    
    output_file = args.output or args.audio.with_suffix(".txt")
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    
    print("=" * 60)
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {gpu_name}")
    print("=" * 60)
    
    # Load Whisper medium model on GPU
    print("[1/3] Loading Whisper medium model...")
    model = whisper.load_model("medium", device=device)
    
    # Get audio duration for progress tracking
    print(f"[2/3] Analyzing audio: {args.audio.name}")
    audio = whisper.load_audio(str(args.audio))
    duration = len(audio) / whisper.audio.SAMPLE_RATE
    print(f"      Duration: {duration:.1f} seconds")
    
    # Transcribe with progress callback
    print(f"[3/3] Transcribing (language: {args.language or 'auto-detect'})...")
    
    progress_callback = ProgressCallback(duration)
    result = model.transcribe(
        str(args.audio),
        language=args.language,
        fp16=(device == "cuda"),
        verbose=False,  # Disable Whisper's own progress output
        task="transcribe"
    )
    progress_callback.close()
    
    transcript = result.get("text", "").strip()
    detected_language = result.get("language", "unknown")
    
    # Save to file
    output_file.write_text(transcript, encoding="utf-8")
    print(f"\n✓ Transcript saved to: {output_file}")
    print(f"✓ Detected language: {detected_language}")
    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(transcript)
    print("=" * 60)


if __name__ == "__main__":
    main()
