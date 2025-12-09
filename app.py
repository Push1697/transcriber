"""Minimal FastAPI app for uploading a WAV file (<=10 MB) and transcribing with Whisper (Faster).
Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import uuid
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from faster_whisper import WhisperModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 8192

app = FastAPI(title="Whisper Transcriber", version="2.2")

# Global state to track tasks (in-memory)
# Format: {task_id: {"status": "processing", "progress": 0, "transcript": "", "error": None}}
tasks: Dict[str, Dict[str, Any]] = {}

_model = None
_model_device = None

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Whisper Transcriber (Faster)</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600&family=Space+Grotesk:wght@400;600&display=swap');
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --accent: #06b6d4;
      --accent-2: #8b5cf6;
      --danger: #ef4444;
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
    button.stop-btn {
      background: var(--danger);
      box-shadow: none;
      margin-left: 10px;
    }
    button.stop-btn:hover {
       transform: translateY(-1px); 
       box-shadow: 0 10px 30px rgba(239, 68, 68, 0.35); 
    }
    .meta { margin-top: 14px; color: var(--muted); font-size: 0.9rem; }
    .result { margin-top: 20px; background: #0b1224; border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(6,182,212,0.15); color: var(--text); font-size: 0.85rem; margin-right: 8px; }
    pre { white-space: pre-wrap; word-break: break-word; font-family: 'Space Grotesk', 'Sora', sans-serif; }
    .progress-wrap { margin-top: 14px; }
    .progress-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }
    .progress {
      width: 100%; height: 20px; background: #0b1224; border: 1px solid var(--border);
      border-radius: 999px; overflow: hidden; position: relative;
    }
    .progress-bar {
      height: 100%; width: 0%; background: linear-gradient(120deg, var(--accent), var(--accent-2));
      transition: width 120ms ease;
      display: flex; align-items: center; justify-content: center;
      font-size: 0.75rem; font-weight: bold; color: white;
      min-width: 20px; /* ensure text fits if possible */
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
    <h1>Whisper Transcriber (Faster)</h1>
    <p>Upload a WAV or MP3 file (max 10 MB) to transcribe locally with faster-whisper.</p>
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
      <div class="progress-label" style="display:flex; justify-content:space-between; align-items:center;">
        <span>Transcribing (<span id="transcribe-pct">0</span>%)</span>
        <button id="stop-btn" class="stop-btn" style="margin:0; padding: 4px 10px; font-size:0.8rem;">Stop</button>
      </div>
      <div class="progress" style="margin-top:6px;"><div id="transcribe-bar" class="progress-bar">0%</div></div>
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
    const transcribePct = document.getElementById('transcribe-pct');
    const stopBtn = document.getElementById('stop-btn');
    
    let lastBlobUrl = null;
    let eventSource = null;
    let currentTaskId = null;

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      // Reset UI
      statusEl.textContent = 'Preparing upload...';
      resultEl.style.display = 'none';
      downloadBtn.style.display = 'none';
      uploadWrap.style.display = 'block';
      transcribeWrap.style.display = 'none';
      uploadBar.style.width = '0%';
      transcribeBar.style.width = '0%';
      transcribeBar.textContent = '0%';
      transcribePct.textContent = '0';
      if (lastBlobUrl) { URL.revokeObjectURL(lastBlobUrl); lastBlobUrl = null; }
      if (eventSource) { eventSource.close(); eventSource = null; }

      const fileInput = document.getElementById('file');
      const languageInput = document.getElementById('language');
      if (!fileInput.files.length) return;
      const file = fileInput.files[0];
      if (file.size > 10 * 1024 * 1024) {
        statusEl.textContent = 'File too large. Limit is 10 MB.';
        return;
      }

      // 1. Upload
      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', languageInput.value);

      try {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');
        xhr.upload.onprogress = (evt) => {
          if (evt.lengthComputable) {
             const pct = Math.round((evt.loaded / evt.total) * 100);
             uploadBar.style.width = `${pct}%`;
             uploadBar.textContent = `${pct}%`;
          }
        };
        const res = await new Promise((resolve, reject) => {
            xhr.onload = () => resolve(xhr);
            xhr.onerror = () => reject(new Error("Upload failed"));
            xhr.send(formData);
        });
        
        if (res.status !== 202) {
             throw new Error(JSON.parse(res.responseText).detail || res.statusText);
        }
        currentTaskId = JSON.parse(res.responseText).task_id;
      } catch (err) {
          statusEl.textContent = `Error: ${err.message}`;
          return;
      }

      // 2. Listen for Progress (SSE)
      transcribeWrap.style.display = 'block';
      statusEl.textContent = 'Transcribing...';
      
      eventSource = new EventSource(`/progress/${currentTaskId}`);
      eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.status === 'processing') {
              const p = Math.round(data.progress);
              transcribeBar.style.width = `${p}%`;
              transcribeBar.textContent = `${p}%`;
              transcribePct.textContent = p;
          } else if (data.status === 'completed') {
              closeEventSource();
              transcribeBar.style.width = '100%';
              transcribeBar.textContent = '100%';
              transcribePct.textContent = '100';
              statusEl.textContent = 'Done!';
              showResult(data.result);
          } else if (data.status === 'error') {
              closeEventSource();
              statusEl.textContent = `Error: ${data.error}`;
          } else if (data.status === 'cancelled') {
              closeEventSource();
              statusEl.textContent = 'Transcription stopped by user.';
              transcribeBar.style.backgroundColor = '#ef4444'; // Red
          }
      };
      eventSource.onerror = () => {
          console.error("SSE Error");
          closeEventSource();
      };
    });

    stopBtn.addEventListener('click', async () => {
       if (!currentTaskId) return;
       try {
           statusEl.textContent = "Stopping...";
           await fetch(`/stop/${currentTaskId}`, { method: 'POST' });
       } catch (e) {
           console.error("Failed to stop", e);
       }
    });

    function closeEventSource() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    function showResult(data) {
        resultEl.style.display = 'block';
        // Cleanup formatting
        const sizeMb = data.size_mb || 0;
        resultEl.innerHTML = `<div class="badge">${data.device.toUpperCase()}</div><div class="badge">${sizeMb} MB</div><pre>${data.transcript}</pre>`;
        
        const blob = new Blob([data.transcript], { type: 'text/plain' });
        lastBlobUrl = URL.createObjectURL(blob);
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = lastBlobUrl;
            a.download = 'transcript.txt';
            a.click();
        };
        downloadBtn.style.display = 'inline-block';
    }
  </script>
</body>
</html>
"""


def get_model():
    """Load faster-whisper model."""
    global _model, _model_device
    if _model is not None:
        return _model, _model_device

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # int8 is safe for CPU, float16 better for CUDA
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Loading model on {device} ({compute_type})...")
        model = WhisperModel("medium", device=device, compute_type=compute_type)
        _model, _model_device = model, device
        return model, device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def process_transcription(task_id: str, file_path: Path, language: str):
    """Background task to run transcription."""
    try:
        tasks[task_id]["status"] = "processing"
        
        model, device = get_model()
        lang_arg = None if language == "auto" else language
        
        segments, info = model.transcribe(str(file_path), language=lang_arg, beam_size=5)
        total_duration = info.duration
        
        transcript_parts = []
        for segment in segments:
            # Check for cancellation
            if tasks.get(task_id, {}).get("status") == "cancelled":
                logger.info(f"Task {task_id} cancelled by user.")
                return 

            transcript_parts.append(segment.text)
            # Update progress
            if total_duration > 0:
                current_progress = (segment.end / total_duration) * 100
                tasks[task_id]["progress"] = min(99, current_progress)
            
        full_text = "".join(transcript_parts).strip()
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["result"] = {
            "transcript": full_text,
            "language": info.language,
            "device": device,
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
        }

    except Exception as e:
        logger.exception("Transcription failed")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
    finally:
        # Cleanup
        file_path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post("/upload", status_code=202)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    language: str = Form("auto")
):
    # Validation
    allowed_exts = {".wav", ".mp3", ".mp4", ".m4a"}
    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if suffix not in allowed_exts:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save to temp
    task_id = str(uuid.uuid4())
    tmp_path = Path(tempfile.gettempdir()) / f"{task_id}{suffix}"
    
    total_size = 0
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk: break
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (10MB limit)")
            f.write(chunk)
            
    # Init Task
    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "transcript": "",
        "created_at": time.time()
    }
    
    # Start Background Job
    background_tasks.add_task(process_transcription, task_id, tmp_path, language)
    
    return {"task_id": task_id, "message": "Queued"}


@app.post("/stop/{task_id}")
async def stop_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Only cancel if pending or processing
    if tasks[task_id]["status"] in ["queued", "processing"]:
        tasks[task_id]["status"] = "cancelled"
        
    return {"message": "Task cancelled"}


@app.get("/progress/{task_id}")
async def stream_progress(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        while True:
            task = tasks.get(task_id)
            if not task: break
            
            data = json.dumps({
                "status": task["status"],
                "progress": task.get("progress", 0),
                "error": task.get("error"),
                "result": task.get("result")
            })
            yield f"data: {data}\n\n"
            
            if task["status"] in ["completed", "error", "cancelled"]:
                break
                
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
