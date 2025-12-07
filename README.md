---
title: Video Transcriber
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# Video & Audio Transcriber 🎙️

A beautiful, modern web application for transcribing video and audio files using **OpenAI Whisper**. Built with **FastAPI**.

![Screenshot](https://dummyimage.com/1200x630/0f172a/06b6d4&text=Video+Transcriber+UI)

## Features

- 🚀 **Fast Transcription**: Uses OpenAI's generic Whisper model.
- 🎨 **Modern UI**: Dark mode, responsive design, and real-time progress updates.
- 📁 **File Support**: Supports `.wav`, `.mp3` (with auto-conversion).
- 🐍 **Python Powered**: Built on the robust FastAPI framework.

## Quick Start

### Local Development

1. **Clone the repo**

    ```bash
    git clone https://github.com/your-username/video-transcriber.git
    cd video-transcriber
    ```

2. **Install Dependencies**

    ```bash
    # Create virtual env (optional but recommended)
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate

    # Install requirements
    pip install -r requirements.txt
    ```

3. **Run the App**

    ```bash
    uvicorn app:app --reload
    ```

    Open [http://localhost:8000](http://localhost:8000) in your browser.

### Docker

```bash
docker build -t video-transcriber .
docker run -p 8000:8000 video-transcriber
```

## Deployment

### Hugging Face Spaces (Recommended Free Tier)

This project is configured for deployment on [Hugging Face Spaces](https://huggingface.co/spaces).

1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Push this repository to your Space's remote.

*Note: The `Dockerfile` includes permissions settings (creating a non-root user) specifically for HF Spaces compatibility.*

## License

MIT
