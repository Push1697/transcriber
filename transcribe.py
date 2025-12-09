import argparse
import sys
import subprocess
import torch
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel

def extract_audio(input_video: Path, output_audio: Path) -> None:
    """Extract lossless PCM audio from a video using ffmpeg."""
    print(f"[Audio] Extracting audio from {input_video.name}...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_audio),
    ]
    # Suppress verbose ffmpeg output, show only errors
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    print(f"[Audio] Saved to {output_audio.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe video/audio using faster-whisper.")
    parser.add_argument("input", type=Path, help="Path to the input video or audio file")
    parser.add_argument("--output", type=Path, default=None, help="Output text file (default: input_name.txt)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for inference")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size (tiny, small, medium, large-v2)")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g. en, hi). Default: auto-detect")

    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: File not found: {args.input}")

    # Determine input type
    is_audio = args.input.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
    audio_path = args.input

    # Extract audio if it's a video
    if not is_audio:
        audio_path = args.input.with_suffix(".wav")
        try:
            extract_audio(args.input, audio_path)
        except RuntimeError as e:
            sys.exit(f"Error extracting audio: {e}")

    # Determine Device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    compute_type = "float16" if device == "cuda" else "int8"
    
    print("=" * 60)
    print(f"Device: {device.upper()} | Compute: {compute_type} | Model: {args.model}")
    print("=" * 60)

    # Load Model
    print("[1/2] Loading model...")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:
        sys.exit(f"Failed to load model: {e}")

    # Transcribe
    print("[2/2] Transcribing...")
    segments, info = model.transcribe(str(audio_path), language=args.language, beam_size=5)

    print(f"      Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Collect segments with a progress bar (since faster-whisper returns a generator)
    # Note: faster-whisper segments generator doesn't know total duration upfront in strict sense easily without overhead, 
    # but we can just print segments as they come or show a spinner. 
    # For a total progress bar, we need duration. 
    
    # Let's try to get duration for better UX if possible, else just simple progress
    total_duration = info.duration
    
    transcript_parts = []
    with tqdm(total=total_duration, unit="s", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s") as pbar:
        for segment in segments:
            transcript_parts.append(segment.text)
            pbar.update(segment.end - pbar.n)
            
    transcript = "".join(transcript_parts).strip()

    # Save output
    output_file = args.output or args.input.with_suffix(".txt")
    output_file.write_text(transcript, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"✓ Saved transcript to: {output_file}")
    print("=" * 60)

    # Cleanup temp audio if we extracted it
    if not is_audio and audio_path != args.input:
        # Optional: Uncomment to keep the extracted WAV
        # audio_path.unlink(missing_ok=True)
        pass

if __name__ == "__main__":
    main()
