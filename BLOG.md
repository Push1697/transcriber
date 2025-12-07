# Whisper Transcriber: Local, Fast, Private

This project turns a laptop (CPU or NVIDIA GPU) into a local transcription station using OpenAI Whisper. It ships with a FastAPI web dashboard (WAV/MP3 ≤10 MB), live upload/transcribe progress, language picker (auto/en/hi), transcript download, and CI/CD assets (GitHub Actions, Jenkins, Argo CD, Docker).

## Features
- Web UI: upload WAV/MP3 (10 MB), live upload + transcription progress bars, auto/en/hi selection, download transcript as .txt
- API: FastAPI `/upload` endpoint with size guard and GPU/CPU auto-selection
- CLI: `transcribe.py` (video→audio→text) and `transcribe_simple.py` (audio→text with progress)
- GPU aware: uses CUDA when available; falls back to CPU
- Formats: WAV/MP3 input; manual FFmpeg extraction supported for other containers

## Quickstart (local)
```powershell
# 1) Create/activate venv
python -m venv transcriber-venv
.\transcriber-venv\Scripts\activate

# 2) Install deps (CPU). For GPU, swap torch URL to matching CUDA.
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3) Run the dashboard
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## Web UI behavior
- Accepts WAV/MP3 up to 10 MB (client + server enforced)
- Shows upload percentage and indeterminate bar during transcription
- Language dropdown: Auto (default), en, hi
- GPU/CPU auto-detect; displays device and detected language
- Download transcript as `.txt` after completion

## CLI usage
- Extract + transcribe video (auto audio extraction):
  ```powershell
  python transcribe.py video.mp4 --language hi  # or omit for auto
  ```
- Transcribe existing audio with progress bar:
  ```powershell
  python transcribe_simple.py audio.wav --output out.txt --language auto
  ```

## Docker
Build a CPU image (includes ffmpeg) and run:
```bash
docker build -t video-transcriber:local .
docker run --rm -p 8000:8000 video-transcriber:local
# open http://localhost:8000
```

## Jenkins pipeline (Jenkinsfile)
Stages: checkout → venv + pip → install deps (CPU torch + requirements) → syntax check → docker build → docker push (uses `docker-registry-creds` for auth, pushes to `$REGISTRY_USR/$IMAGE_NAME:$BUILD_NUMBER`). Adjust credentials ID and registry as needed.

## Argo CD

- `k8s/deployment.yaml`: Deployment + Service (ClusterIP 80→8000), image placeholder `your-registry/video-transcriber:latest`, CPU-friendly resources.
- `k8s/argo-application.yaml`: Argo CD Application pointing to repo `path: k8s` and namespace `video-transcriber` (CreateNamespace=true). Update `repoURL` and image tag before syncing.

## GitHub Actions CI

- `.github/workflows/ci.yml`: Python 3.12, cached pip, installs CPU torch + whisper + tqdm, import check, `python -m compileall` on app and CLIs.

## Notes on models and performance

- Default model: `medium`. Change in `app.py`/CLI via `whisper.load_model("medium")` to `small`/`base`/`tiny` for lower VRAM.
- First run downloads ~1.4 GB; cache persists between runs/requests.

## Troubleshooting

- "FP16 not supported on CPU": install CUDA-enabled torch or run on GPU.
- File rejected: ensure WAV/MP3 and ≤10 MB.
- FFmpeg missing: install and add to PATH (Windows) or ensure present in Docker/K8s image (already handled in Dockerfile).

## Roadmap ideas

- Add timestamped SRT/VTT export
- Add auth/rate limits for shared deployments
- Add small/heavy model toggle in UI
- GPU-enabled Dockerfile variant (nvidia/cuda base) for on-cluster acceleration

**Last updated:** December 7, 2025
