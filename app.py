"""Minimal FastAPI app for uploading a WAV file (<=10 MB) and transcribing with Whisper.
Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import html
import os
import tempfile
from pathlib import Path

import torch
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 8192

app = FastAPI(title="Whisper Transcriber", version="1.0")

_model = None
_device = None


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Whisper Transcriber</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600&family=Space+Grotesk:wght@400;600&display=swap');
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --accent: #06b6d4;
      --accent-2: #8b5cf6;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --border: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      background: radial-gradient(circle at 20% 20%, #1e293b, #0f172a 45%),
                  radial-gradient(circle at 80% 0%, #111827, #0f172a 55%),
                  radial-gradient(circle at 50% 80%, #0b1224, #0f172a 70%);
      font-family: 'Sora', 'Space Grotesk', sans-serif;
      color: var(--text);
      padding: 32px 16px;
    }
    .card {
      width: min(960px, 100%);
      background: linear-gradient(135deg, #111827 0%, #0b1224 100%);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    h1 { margin: 0 0 10px; font-size: 1.8rem; letter-spacing: -0.02em; }
    p { margin: 4px 0 18px; color: var(--muted); }
    form { border: 1px dashed var(--border); padding: 18px; border-radius: 12px; background: rgba(255,255,255,0.02); }
    label { display: block; margin-bottom: 10px; font-weight: 600; }
    input[type=file] {
      width: 100%;
      padding: 12px;
      background: #0b1224;
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
    }
    button {
      margin-top: 14px;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #0b1224;
      border: none;
      padding: 12px 18px;
      border-radius: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 10px 30px rgba(6,182,212,0.35); }
    .meta { margin-top: 14px; color: var(--muted); font-size: 0.9rem; }
    .result { margin-top: 20px; background: #0b1224; border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(6,182,212,0.15); color: var(--text); font-size: 0.85rem; margin-right: 8px; }
    pre { white-space: pre-wrap; word-break: break-word; font-family: 'Space Grotesk', 'Sora', sans-serif; }
    .progress-wrap { margin-top: 14px; }
    .progress-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }
    .progress {
      width: 100%; height: 10px; background: #0b1224; border: 1px solid var(--border);
      border-radius: 999px; overflow: hidden; position: relative;
    }
    .progress-bar {
      height: 100%; width: 0%; background: linear-gradient(120deg, var(--accent), var(--accent-2));
      transition: width 120ms ease;
    }
    .indeterminate {
      position: absolute; left: 0; top: 0; bottom: 0;
      width: 30%; min-width: 80px;
      animation: slide 1.2s infinite ease-in-out;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
    }
    @keyframes slide { 0% { transform: translateX(-40%); } 50% { transform: translateX(80%); } 100% { transform: translateX(200%); } }
  </style>
</head>
<body>
  <div class="card">
    <h1>Whisper Transcriber</h1>
    <p>Upload a WAV or MP3 file (max 10 MB) to transcribe locally with Whisper.</p>
    <form id="upload-form">
      <label for="file">Select WAV or MP3 file</label>
      <input id="file" name="file" type="file" accept="audio/wav,audio/mpeg" required />
      <label for="language" style="margin-top:10px; display:block;">Language</label>
      <select id="language" name="language" style="width:100%; padding:10px; background:#0b1224; border:1px solid var(--border); border-radius:10px; color:var(--text);">
        <option value="auto" selected>Auto (detect)</option>
        <option value="en">English (en)</option>
        <option value="hi">Hindi (hi)</option>
      </select>
      <button type="submit">Transcribe</button>
      <div class="meta">GPU used when available. Supported languages: 90+ (auto-detect).</div>
    </form>
    <div id="status" class="meta" style="margin-top:12px;"></div>
    <div class="progress-wrap" id="upload-wrap" style="display:none;">
      <div class="progress-label">Upload progress</div>
      <div class="progress"><div id="upload-bar" class="progress-bar"></div></div>
    </div>
    <div class="progress-wrap" id="transcribe-wrap" style="display:none;">
      <div class="progress-label">Transcribing</div>
      <div class="progress"><div id="transcribe-bar" class="indeterminate"></div></div>
    </div>
    <div id="result" class="result" style="display:none;"></div>
    <button id="download" style="display:none; margin-top:12px; background:#0b1224; color: var(--text); border:1px solid var(--border);">
      Download transcript (.txt)
    </button>
  </div>
  <script>
    const form = document.getElementById('upload-form');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    const downloadBtn = document.getElementById('download');
    const uploadWrap = document.getElementById('upload-wrap');
    const uploadBar = document.getElementById('upload-bar');
    const transcribeWrap = document.getElementById('transcribe-wrap');
    const transcribeBar = document.getElementById('transcribe-bar');
    let lastBlobUrl = null;

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('file');
      const languageInput = document.getElementById('language');
      if (!fileInput.files.length) return;
      const file = fileInput.files[0];
      if (file.size > 10 * 1024 * 1024) {
        statusEl.textContent = 'File too large. Limit is 10 MB.';
        resultEl.style.display = 'none';
        downloadBtn.style.display = 'none';
        return;
      }
      statusEl.textContent = 'Uploading and transcribing...';
      resultEl.style.display = 'none';
      downloadBtn.style.display = 'none';
      uploadWrap.style.display = 'block';
      transcribeWrap.style.display = 'block';
      uploadBar.style.width = '0%';
      transcribeBar.classList.add('indeterminate');
      transcribeBar.style.width = '0%';
      if (lastBlobUrl) {
        URL.revokeObjectURL(lastBlobUrl);
        lastBlobUrl = null;
      }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', languageInput.value);

      const res = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');
        xhr.upload.onprogress = (evt) => {
          if (evt.lengthComputable) {
            const pct = (evt.loaded / evt.total) * 100;
            uploadBar.style.width = `${pct}%`;
          }
        };
        xhr.onloadstart = () => {
          uploadBar.style.width = '0%';
          transcribeBar.classList.add('indeterminate');
        };
        xhr.onloadend = () => {
          uploadBar.style.width = '100%';
        };
        xhr.onreadystatechange = () => {
          if (xhr.readyState === XMLHttpRequest.DONE) {
            resolve(xhr);
          }
        };
        xhr.onerror = (err) => reject(err);
        xhr.send(formData);
      });

      if (res.status < 200 || res.status >= 300) {
        const text = res.responseText;
        statusEl.textContent = `Error: ${text || res.statusText}`;
        downloadBtn.style.display = 'none';
        transcribeBar.classList.remove('indeterminate');
        transcribeBar.style.width = '0%';
        return;
      }

      transcribeBar.classList.remove('indeterminate');
      transcribeBar.style.width = '100%';

      const data = JSON.parse(res.responseText);
      statusEl.textContent = `Done. Detected: ${data.language} | Device: ${data.device}`;
      resultEl.style.display = 'block';
      resultEl.innerHTML = `<div class="badge">${data.device.toUpperCase()}</div><div class="badge">${(data.size_mb).toFixed(2)} MB</div><pre>${data.transcript}</pre>`;

      const blob = new Blob([data.transcript], { type: 'text/plain' });
      lastBlobUrl = URL.createObjectURL(blob);
      const baseName = file.name.replace(/\\.[^.]+$/, '') || 'transcript';
      downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = lastBlobUrl;
        a.download = `${baseName}_transcript.txt`;
        a.click();
      };
      downloadBtn.style.display = 'inline-block';
    });
  </script>
</body>
</html>
"""


def get_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)
    _model, _device = model, device
    return model, device


async def save_upload_file_tmp(upload: UploadFile) -> tuple[Path, int]:
    # Enforce WAV or MP3 mime and extension
    allowed_mimes = {"audio/wav", "audio/x-wav", "audio/mpeg"}
    allowed_exts = {".wav", ".mp3"}
    if upload.content_type not in allowed_mimes:
        raise HTTPException(status_code=400, detail="Only WAV or MP3 files are allowed.")
    if upload.filename and Path(upload.filename).suffix.lower() not in allowed_exts:
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are allowed.")

    total = 0
    suffix = Path(upload.filename).suffix if upload.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large; limit is 10 MB.")
            tmp.write(chunk)
    return Path(tmp.name), total


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post("/upload")
async def upload(file: UploadFile = File(...), language: str = Form("auto")):
    tmp_path = None
    try:
        tmp_path, total = await save_upload_file_tmp(file)
        model, device = get_model()
        lang_arg = None if language == "auto" else language
        result = await asyncio.get_event_loop().run_in_executor(
            None,
          lambda: model.transcribe(
            str(tmp_path),
            fp16=(device == "cuda"),
            verbose=False,
            language=lang_arg,
          ),
        )
        transcript = result.get("text", "").strip()
        language = result.get("language", "unknown")
        return {
            "transcript": transcript,
            "language": language,
            "device": device,
            "size_mb": round(total / (1024 * 1024), 2),
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # For local debugging without uvicorn CLI
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
