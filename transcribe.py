import argparse
import subprocess
import sys
from pathlib import Path

import whisper


def extract_audio(input_video: Path, output_audio: Path) -> None:
    """Extract lossless PCM audio from a video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output if it exists
        "-i",
        str(input_video),
        "-vn",  # drop video
        "-acodec",
        "pcm_s16le",  # lossless PCM
        "-ar",
        "16000",  # resample to 16 kHz for Whisper
        "-ac",
        "1",  # mono
        str(output_audio),
    ]

    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {completed.returncode}:\n{completed.stdout}")


def transcribe_audio(audio_path: Path, language: str | None = None) -> str:
    """Run Whisper (medium) transcription on the given audio file."""
    model = whisper.load_model("medium")
    result = model.transcribe(str(audio_path), language=language)
    return result.get("text", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio from a video and transcribe it with Whisper medium.")
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument("--audio", type=Path, default=None, help="Optional path for the extracted WAV file (default: alongside video)")
    parser.add_argument("--language", type=str, default=None, help="Force a language code (e.g., en, es). Default: auto-detect")

    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Video file not found: {args.video}")

    audio_path = args.audio or args.video.with_suffix(".wav")

    extract_audio(args.video, audio_path)
    print(f"[ok] Extracted audio -> {audio_path}")

    text = transcribe_audio(audio_path, language=args.language)
    print("[ok] Transcription:")
    print(text)


if __name__ == "__main__":
    main()
