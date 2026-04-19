"""
SimpleAutoSubs Headless API Server
REST API for the Hub to control video processing without the GUI.
Port: 9020
"""
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PORT = 9020
SETTINGS_FILE = os.path.join(_HERE, "..", "youtube_hub", "hub_settings.json")

# ─── Shared state ─────────────────────────────────────────────────────────────

_files: List[Dict[str, Any]] = []
_settings: Dict[str, Any] = {
    "animation_type": "Auto",
    "sync_offset": -0.15,
    "output_dir": os.path.join(os.path.expanduser("~"), "Desktop"),
    "enable_trimming": True,
}
_logs: deque = deque(maxlen=500)
_processing = False
_stop_requested = False
_current_index = -1
_lock = threading.Lock()


def _load_settings() -> None:
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        import json
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k in _settings:
            if k in saved:
                _settings[k] = saved[k]
        print(f"[settings] Loaded from {SETTINGS_FILE}", flush=True)
    except Exception as e:
        print(f"[settings] Could not load {SETTINGS_FILE}: {e}", flush=True)


def _save_settings() -> None:
    try:
        import json
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_settings, f, indent=2)
    except Exception as e:
        print(f"[settings] Could not save {SETTINGS_FILE}: {e}", flush=True)


_load_settings()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _logs.append(line)
    print(line, flush=True)


def _unique_output_path(input_path: str, output_dir: str) -> str:
    name = os.path.splitext(os.path.basename(input_path))[0]
    base = os.path.join(output_dir, f"{name}-as.mp4")
    if not os.path.exists(base):
        return base
    i = 1
    while os.path.exists(os.path.join(output_dir, f"{name}-as-{i}.mp4")):
        i += 1
    return os.path.join(output_dir, f"{name}-as-{i}.mp4")


def _processing_worker() -> None:
    global _processing, _stop_requested, _current_index

    try:
        from core.video_processor import VideoProcessor
    except Exception as e:
        _log(f"❌ Failed to import VideoProcessor: {e}")
        import traceback
        _log(traceback.format_exc())
        _processing = False
        return

    queued_indices = [
        i for i, f in enumerate(_files) if f["status"] == "queued"
    ]
    _log(f"=== Batch started: {len(queued_indices)} queued file(s) ===")

    for idx in queued_indices:
        if _stop_requested:
            _log("⏹  Stop requested — halting batch.")
            break

        with _lock:
            _current_index = idx
            _files[idx]["status"] = "processing"

        entry = _files[idx]
        _log(f"\n{'='*40}")
        _log(
            f"Processing {idx + 1}/{len(_files)}: "
            f"{os.path.basename(entry['input_path'])}"
        )
        _log(f"{'='*40}")

        try:
            final_path, _title, _meta = VideoProcessor.process_single_video(
                input_file=entry["input_path"],
                output_file=entry["output_path"],
                animation_type=_settings["animation_type"],
                sync_offset=_settings["sync_offset"],
                detailed_logs=True,
                log_func=_log,
                enable_trimming=_settings["enable_trimming"],
            )
            with _lock:
                _files[idx]["output_path"] = (
                    final_path or entry["output_path"])
                _files[idx]["status"] = "done"
            _log(
                f"✅ Done: "
                f"{os.path.basename(_files[idx]['output_path'])}"
            )

        except Exception as e:
            import traceback
            err = str(e)
            with _lock:
                _files[idx]["status"] = "error"
                _files[idx]["error"] = err
            _log(
                f"❌ Error on "
                f"{os.path.basename(entry['input_path'])}: {err}"
            )
            _log(traceback.format_exc())

    with _lock:
        _processing = False
        _current_index = -1
    _log("=== Batch complete ===")


# ─── Pydantic models ──────────────────────────────────────────────────────────

class SubtitlerSettings(BaseModel):
    animation_type: str
    sync_offset: float
    output_dir: str
    enable_trimming: bool


class FilesPayload(BaseModel):
    paths: List[str]


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log(f"✅ SimpleAutoSubs API ready on :{PORT}")
    yield
    _log("SimpleAutoSubs API stopping.")


app = FastAPI(title="SimpleAutoSubs API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "port": PORT}


# ─── Files ────────────────────────────────────────────────────────────────────

@app.get("/files")
def get_files():
    return list(_files)


@app.post("/files")
def add_files(payload: FilesPayload):
    output_dir = _settings["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    existing = {f["input_path"] for f in _files}
    added = 0
    for raw in payload.paths:
        path = raw.strip()
        if not path:
            continue
        if not os.path.exists(path):
            _log(f"⚠️  Skipping missing file: {path}")
            continue
        if path in existing:
            continue
        with _lock:
            _files.append({
                "input_path": path,
                "output_path": _unique_output_path(path, output_dir),
                "status": "queued",
                "error": "",
            })
        existing.add(path)
        added += 1
    return {"ok": True, "added": added, "total": len(_files)}


@app.delete("/files/{index}")
def remove_file(index: int):
    if _processing:
        raise HTTPException(400, "Cannot remove files while processing")
    if index < 0 or index >= len(_files):
        raise HTTPException(404, "Index out of range")
    with _lock:
        _files.pop(index)
    return {"ok": True, "total": len(_files)}


@app.delete("/files")
def clear_files():
    global _files
    if _processing:
        raise HTTPException(400, "Cannot clear while processing")
    with _lock:
        _files.clear()
    return {"ok": True}


@app.post("/files/reset")
def reset_files():
    """Set all done/error files back to queued."""
    if _processing:
        raise HTTPException(400, "Cannot reset while processing")
    with _lock:
        for f in _files:
            if f["status"] in ("done", "error"):
                f["status"] = "queued"
                f["error"] = ""
    return {"ok": True}


@app.get("/files/browse")
async def browse_files():
    import asyncio

    def _dialog():
        if sys.platform == "darwin":
            import subprocess
            script = (
                "set theFiles to choose file "
                'of type {"mp4", "mkv", "avi"} '
                "with multiple selections allowed "
                'with prompt "Select Video Files"\n'
                "set out to \"\"\n"
                "repeat with f in theFiles\n"
                "    set out to out & POSIX path of f & (ASCII character 10)\n"
                "end repeat\n"
                "return out"
            )
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return []
            return [p for p in r.stdout.strip().splitlines() if p.strip()]
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            paths = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=[("Video files", "*.mp4 *.mkv *.avi")],
            )
            root.destroy()
            return list(paths)

    try:
        paths = await asyncio.get_event_loop().run_in_executor(None, _dialog)
        return {"paths": paths}
    except Exception as e:
        raise HTTPException(500, f"File dialog failed: {e}")


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.get("/settings")
def get_settings():
    return dict(_settings)


@app.post("/settings")
def post_settings(s: SubtitlerSettings):
    dir_changed = s.output_dir != _settings["output_dir"]
    _settings.update({
        "animation_type": s.animation_type,
        "sync_offset": round(s.sync_offset, 3),
        "output_dir": s.output_dir,
        "enable_trimming": s.enable_trimming,
    })
    if dir_changed:
        os.makedirs(s.output_dir, exist_ok=True)
        with _lock:
            for f in _files:
                if f["status"] == "queued":
                    f["output_path"] = _unique_output_path(
                        f["input_path"], s.output_dir)
    _save_settings()
    return {"ok": True}


@app.get("/settings/browse-dir")
async def browse_output_dir():
    import asyncio

    def _dialog():
        if sys.platform == "darwin":
            import subprocess
            script = (
                'POSIX path of (choose folder with prompt '
                '"Select Output Directory")'
            )
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return ""
            return r.stdout.strip().rstrip("/")
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            path = filedialog.askdirectory(title="Select Output Directory")
            root.destroy()
            return path or ""

    try:
        path = await asyncio.get_event_loop().run_in_executor(None, _dialog)
        return {"path": path}
    except Exception as e:
        raise HTTPException(500, f"Directory dialog failed: {e}")


# ─── Session log files ────────────────────────────────────────────────────────

@app.get("/session-logs/list")
def list_session_logs(dir: str = ""):
    target = dir.strip() or _settings.get("output_dir", "")
    if not target or not os.path.isdir(target):
        return {"files": [], "dir": target}
    try:
        files = []
        for name in os.listdir(target):
            if not name.endswith(".txt"):
                continue
            full = os.path.join(target, name)
            if not os.path.isfile(full):
                continue
            stat = os.stat(full)
            files.append({
                "name": name,
                "path": full,
                "modified": stat.st_mtime,
                "size": stat.st_size,
            })
        files.sort(key=lambda x: x["modified"], reverse=True)
        return {"files": files, "dir": target}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/session-logs/read")
def read_session_log(path: str = ""):
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"name": os.path.basename(path), "content": content}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Processing ───────────────────────────────────────────────────────────────

@app.post("/process/start")
def start_processing():
    global _processing, _stop_requested
    if _processing:
        return {"ok": False, "reason": "already_running"}
    queued = [f for f in _files if f["status"] == "queued"]
    if not queued:
        return {"ok": False, "reason": "no_queued_files"}
    _stop_requested = False
    _processing = True
    threading.Thread(target=_processing_worker, daemon=True).start()
    return {"ok": True}


@app.post("/process/stop")
def stop_processing():
    global _stop_requested
    _stop_requested = True
    return {
        "ok": True,
        "message": "Stop requested — will halt after current file.",
    }


@app.get("/process/status")
def process_status():
    return {
        "processing": _processing,
        "stop_requested": _stop_requested,
        "current_index": _current_index,
        "total": len(_files),
        "queued": sum(1 for f in _files if f["status"] == "queued"),
        "done": sum(1 for f in _files if f["status"] == "done"),
        "errors": sum(1 for f in _files if f["status"] == "error"),
    }


# ─── Logs ─────────────────────────────────────────────────────────────────────

@app.get("/logs")
def get_logs(last: int = 200):
    return {"lines": list(_logs)[-last:]}


@app.delete("/logs")
def clear_logs():
    _logs.clear()
    return {"ok": True}


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")