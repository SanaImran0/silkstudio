"""
API for the Next.js app. Photos are uploaded here, then create_dress() runs.

Run from the repository root:
    python backend/serve.py
"""

import json
import mimetypes
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
os.chdir(ROOT)

from generate_dress import (  # noqa: E402
    IMAGE_SUFFIXES,
    MAX_REFERENCE_IMAGES,
    CollectedImage,
    create_dress,
)
from prompt import GARMENT_SLOTS  # noqa: E402

UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
MAX_FILE_BYTES = 50 * 1024 * 1024
SLOT_KEYS = {slot.key: slot for slot in GARMENT_SLOTS}

_sessions: dict[str, dict[str, list[Path]]] = {}
_sessions_lock = threading.Lock()
_generate_lock = threading.Lock()


def _safe_session(value: str) -> str:
    if len(value) == 36 and all(char in "0123456789abcdef-" for char in value.lower()):
        return value.lower()
    raise ValueError("Invalid upload session.")


def _safe_filename(value: str) -> str:
    name = Path(unquote(value)).name.strip()
    cleaned = "".join(char if char.isalnum() or char in ".-_ " else "_" for char in name)
    cleaned = cleaned.strip(" .") or "image"
    return cleaned[:120]


class DressHandler(BaseHTTPRequestHandler):
    server_version = "SilkStudio"

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} {format % args}")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_json(200, {
                "service": "silk-studio",
                "ui": "http://127.0.0.1:3000",
            })
            return
        if path == "/api/slots":
            payload = [
                {
                    "key": slot.key,
                    "ask": slot.ask,
                    "many": slot.many,
                    "required": slot.required,
                }
                for slot in GARMENT_SLOTS
            ]
            self._send_json(200, payload)
            return
        if path.startswith("/outputs/"):
            self._send_output(path[len("/outputs/"):])
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload":
            self._upload()
            return
        if path == "/api/generate":
            self._generate()
            return
        self._send_json(404, {"error": "Not found."})

    def _upload(self) -> None:
        try:
            session = _safe_session(self.headers.get("X-Session", ""))
            slot_key = self.headers.get("X-Slot", "")
            if slot_key not in SLOT_KEYS:
                raise ValueError("Unknown garment piece.")
            filename = _safe_filename(self.headers.get("X-Filename", "image"))
            length = int(self.headers.get("Content-Length", "0"))
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"error": str(exc) or "Invalid upload."})
            return

        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            self._send_json(400, {"error": "Use a jpeg, png, webp, or gif file."})
            return
        if length <= 0 or length > MAX_FILE_BYTES:
            self._send_json(400, {"error": "Each photo must be under 50MB."})
            return

        with _sessions_lock:
            slots = _sessions.setdefault(session, {key: [] for key in SLOT_KEYS})
            count = sum(len(paths) for paths in slots.values())
            if count >= MAX_REFERENCE_IMAGES:
                self._discard(length)
                self._send_json(400, {"error": f"The limit is {MAX_REFERENCE_IMAGES} photos."})
                return
            folder = UPLOADS / session / slot_key
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / f"{count + 1:02d}_{filename}"
            slots[slot_key].append(destination)

        try:
            self._write_body(destination, length)
        except Exception:
            destination.unlink(missing_ok=True)
            with _sessions_lock:
                stored = _sessions.get(session, {}).get(slot_key, [])
                if destination in stored:
                    stored.remove(destination)
            self._send_json(400, {"error": "The upload was interrupted."})
            return
        self._send_json(200, {"name": destination.name})

    def _generate(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            session = _safe_session(str(body.get("session", "")))
        except (ValueError, TypeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid generate request."})
            return

        with _sessions_lock:
            stored = _sessions.get(session)
            if not stored:
                self._send_json(400, {"error": "Upload the photos first."})
                return
            images = [
                CollectedImage(slot=SLOT_KEYS[slot.key], path=path)
                for slot in GARMENT_SLOTS
                for path in stored.get(slot.key, [])
            ]
        if not images:
            self._send_json(400, {"error": "Add at least one photo."})
            return
        if len(images) > MAX_REFERENCE_IMAGES:
            self._send_json(400, {"error": f"The limit is {MAX_REFERENCE_IMAGES} photos."})
            return

        if not _generate_lock.acquire(blocking=False):
            self._send_json(409, {"error": "A portrait is already being generated."})
            return
        try:
            image_path, notes_path, summary = create_dress(images)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        finally:
            _generate_lock.release()
            self._clear_session(session)

        self._send_json(200, {
            "image": "/outputs/" + image_path.resolve().relative_to(OUTPUTS.resolve()).as_posix(),
            "notes": "/outputs/" + notes_path.resolve().relative_to(OUTPUTS.resolve()).as_posix(),
            "summary": summary,
        })

    def _clear_session(self, session: str) -> None:
        with _sessions_lock:
            _sessions.pop(session, None)
        folder = UPLOADS / session
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    def _write_body(self, destination: Path, length: int) -> None:
        remaining = length
        with destination.open("wb") as handle:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Upload ended early.")
                handle.write(chunk)
                remaining -= len(chunk)

    def _discard(self, length: int) -> None:
        remaining = length
        while remaining:
            chunk = self.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _send_output(self, relative: str) -> None:
        target = (OUTPUTS / unquote(relative)).resolve()
        if not target.is_relative_to(OUTPUTS.resolve()) or not target.is_file():
            self._send_json(404, {"error": "Image not found."})
            return
        mime_type, _ = mimetypes.guess_type(target.name)
        self._send_bytes(200, target.read_bytes(), mime_type or "application/octet-stream")

    def _send_json(self, status: int, payload: object) -> None:
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session, X-Slot, X-Filename")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = None
    for port in range(8765, 8775):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), DressHandler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port between 8765 and 8774.")

    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"API at {url}", flush=True)
    print("UI at http://127.0.0.1:3000  (npm run dev --prefix frontend)", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
