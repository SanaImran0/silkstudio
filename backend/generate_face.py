"""
Generate one new adult face for the Silk Studio model library.

The photos are references for photographic realism and South Asian appearance.
The result is a new person, not a portrait of the person in the photos.

Run from the repository root. This opens a page where you upload the photos
and fill in the source note, age range, and hairstyle:

    python backend/generate_face.py

You can still run one face from the terminal:

    python backend/generate_face.py a.jpg b.jpg c.jpg d.jpg \\
        --note "public-figure set A" --age 25-34 --hair "black, center part"

Uses google/gemini-3.1-flash-image-preview (Nano Banana 2) through OpenRouter.
Nano Banana Pro is google/gemini-3-pro-image-preview, which Google AI Studio
blocks from this location.
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import shutil
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv

from compress_references import compress_images, report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.1-flash-image-preview"
ASPECT_RATIO = "3:4"
IMAGE_SIZE = "2K"
ADULT_AGE_RANGES = ("18-24", "25-34")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MIN_REFERENCES = 2
MAX_REFERENCES = 8
MAX_FILE_BYTES = 50 * 1024 * 1024

_sessions: dict[str, list[Path]] = {}
_sessions_lock = threading.Lock()
_generate_lock = threading.Lock()


def face_prompt(age_range: str, hair: str) -> str:
    return f"""
Create one photograph of a new adult woman for a commercial fashion model library. Exactly one person, in one frame. Not a grid, collage, or set of alternate looks.

The attached photos are reference material only. They show a real person. Do not depict that person. Do not make a lookalike, a relative, an aged or younger version, or a face that someone who knows her would recognize. Change the face shape, the distance between the eyes, the nose, the lips, the jaw, the chin, the brow, and the ears so the result is a different individual. Do not copy her moles, marks, or distinctive features.

She should look like a working fashion model during a catalog shoot, not a person sitting for a passport or identity photo. Turn her body slightly, with one shoulder closer to the camera, and let her head sit at a small natural angle. Her eyes meet the lens and are alive: a little warmth in them, as if she has just been spoken to. A small real smile, the kind that changes the eyes, not a blank closed mouth and not a wide grin. No sorrow, no heavy lids, no vacant stare, and no frown.

She is an original adult, apparent age {age_range}, with South Asian features suitable for Pakistani and Indian fashion photography. Hair: {hair}. Use that description for this one portrait only. A few loose strands and flyaways, not a solid sheet of hair. She wears a simple black fine-knit top with a real neckline, seams, and fabric texture. Shoulders are covered. Light daytime makeup only: groomed brows, a thin coat of mascara, a soft muted lip color, and skin that still shows pores, faint color variation, and fine facial hair underneath. No glam, no contour stripes, no glassy highlight. Eyes have depth, a moist surface, and uneven lashes. The face is slightly asymmetrical, the way a real face is. Light comes from one side, so the cheek and neck have a gentle shadow. Plain light-grey studio background. This must look like a photograph from a camera, not a rendered or retouched face. No beauty filter, no plastic skin, no waxy highlights, no extra fingers, no distorted ears. No text, logo, watermark, or other people.
""".strip()


def image_part(path: Path) -> dict:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def collect_paths(given: list[str]) -> list[Path]:
    paths = [Path(item).expanduser() for item in given]
    if len(paths) < MIN_REFERENCES:
        raise SystemExit(f"Provide at least {MIN_REFERENCES} photos of the reference person.")
    if len(paths) > MAX_REFERENCES:
        raise SystemExit(f"The limit is {MAX_REFERENCES} photos.")
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"No file at {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Use a jpeg, png, webp, or gif file: {path}")
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def generate(prompt: str, images: list[Path], destination: Path) -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or the environment.")
    content: list[dict] = [{
        "type": "text",
        "text": (
            "The following photos are of one real adult. Use them only as "
            "photographic reference. The face you create must be a different person."
        ),
    }]
    for index, path in enumerate(images, start=1):
        content.append({
            "type": "text",
            "text": f"Reference {index}, file {path.name}. Do not reproduce this person.",
        })
        content.append(image_part(path))
    content.append({"type": "text", "text": prompt})
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"],
            "image_config": {"aspect_ratio": ASPECT_RATIO, "image_size": IMAGE_SIZE},
        },
        timeout=600,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter returned {response.status_code}:\n{response.text}")
    result = response.json()
    try:
        data_url = result["choices"][0]["message"]["images"][0]["image_url"]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        preview = json.dumps(result, indent=2)[:4000]
        raise RuntimeError(f"No image in the response.\n{preview}") from exc
    if "," not in data_url:
        raise RuntimeError("The returned image was not a base64 data URL.")
    _, encoded = data_url.split(",", 1)
    destination.write_bytes(base64.b64decode(encoded))


def write_provenance(
    path: Path,
    face_id: str,
    sources: list[Path],
    sent: list[Path],
    prompt: str,
    note: str,
    age_range: str,
    hair: str,
) -> None:
    lines = [
        f"face_id: {face_id}",
        f"created_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"model: {MODEL}",
        f"aspect_ratio: {ASPECT_RATIO}",
        f"image_size: {IMAGE_SIZE}",
        f"age_range: {age_range}",
        f"hair: {hair}",
        f"source_note: {note}",
        "approval: pending",
        "",
        "process:",
        "Reference photos were sent only as realism and appearance context.",
        "The prompt requires a new adult identity and forbids a recognizable copy of the reference person.",
        "",
        "source_files:",
    ]
    for index, source in enumerate(sources, start=1):
        lines.append(f"{index}. {source.resolve()}")
        lines.append(f"   sha256: {file_sha256(source)}")
    lines.extend(["", "files_sent_to_model:"])
    for index, item in enumerate(sent, start=1):
        lines.append(f"{index}. {item.resolve()}")
    lines.extend(["", "prompt:", prompt, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_name(source: Path, index: int) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in source.stem)
    stem = stem.strip("_") or "image"
    suffix = source.suffix.lower() if source.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    return f"{index:02d}_{stem[:80]}{suffix}"


def create_face(sources: list[Path], note: str, age_range: str, hair: str) -> tuple[Path, Path, str]:
    if age_range not in ADULT_AGE_RANGES:
        raise ValueError(f"Age must be one of {', '.join(ADULT_AGE_RANGES)}.")
    note = note.strip()
    hair = hair.strip()
    if not note:
        raise ValueError("Enter a source note for these reference photos.")
    if not hair:
        raise ValueError("Enter a hairstyle for the new face.")
    if len(sources) < MIN_REFERENCES or len(sources) > MAX_REFERENCES:
        raise ValueError(f"Use {MIN_REFERENCES} to {MAX_REFERENCES} photos of one adult.")

    prompt = face_prompt(age_range, hair)
    face_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = PROJECT_ROOT / "outputs" / "faces" / face_id
    source_dir = folder / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Path] = []
    for index, source in enumerate(sources, start=1):
        destination = source_dir / _safe_name(source, index)
        shutil.copy2(source, destination)
        kept.append(destination)

    prepared = compress_images(kept, folder / "references")
    summary = report(prepared)
    print(summary)
    image_path = folder / "face.png"
    print("Generating a new face.")
    generate(prompt, [item.output for item in prepared], image_path)
    notes_path = folder / "provenance.txt"
    write_provenance(
        notes_path,
        face_id,
        kept,
        [item.output for item in prepared],
        prompt,
        note,
        age_range,
        hair,
    )
    print(f"Saved -> {image_path}")
    print(f"Provenance -> {notes_path}")
    return image_path, notes_path, summary


def _page() -> str:
    options = "\n".join(
        f'<option value="{age}">{age}</option>' for age in ADULT_AGE_RANGES
    )
    return PAGE_TEMPLATE.replace("<!-- ages -->", options)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Silk Studio — New face</title>
  <style>
    body { margin: 0; background: #f7f6f4; color: #1a1a1a; font-family: system-ui, sans-serif; }
    main { max-width: 720px; margin: 0 auto; padding: 40px 20px 80px; }
    h1 { font-size: 1.7rem; font-weight: 560; margin: 0 0 8px; }
    .lead { color: #4d4a46; line-height: 1.5; margin: 0 0 28px; }
    label { display: block; font-size: 0.92rem; margin: 18px 0 6px; }
    input[type="text"], select, textarea {
      width: 100%; box-sizing: border-box; padding: 10px 12px;
      border: 1px solid #d6d3ce; border-radius: 8px; background: #fff; font: inherit;
    }
    textarea { min-height: 88px; resize: vertical; }
    .drop {
      border: 1px dashed #b9b4ad; border-radius: 12px; padding: 22px; background: #fff;
      cursor: pointer;
    }
    .drop strong { display: block; margin-bottom: 4px; }
    .drop span { color: #6b6762; font-size: 0.9rem; }
    .previews { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
    .shot { position: relative; }
    .shot img { width: 92px; height: 92px; object-fit: cover; border-radius: 8px; display: block; }
    .shot button {
      position: absolute; top: 4px; right: 4px; width: 22px; height: 22px; padding: 0;
      border: 0; border-radius: 999px; background: #1a1a1a; color: #fff; cursor: pointer;
    }
    button.primary {
      margin-top: 22px; background: #1a1a1a; color: #fff; border: 0; border-radius: 999px;
      padding: 12px 22px; font: inherit; cursor: pointer;
    }
    button.primary:disabled { opacity: 0.45; cursor: default; }
    .status { min-height: 1.4em; margin-top: 14px; color: #6b3a2a; }
    .result { margin-top: 28px; }
    .result img { width: min(100%, 420px); border-radius: 12px; display: block; }
    .result a { color: #1a1a1a; }
    input[type="file"] { display: none; }
  </style>
</head>
<body>
  <main>
    <h1>New face</h1>
    <p class="lead">Upload several photos of one adult. The result is a new person for the model library, not a copy of the person in the photos.</p>
    <form id="form">
      <div class="drop" id="drop">
        <strong>Reference photos</strong>
        <span>Choose 2 to 8 photos of the same person. Jpeg, png, webp, or gif.</span>
      </div>
      <input id="files" type="file" accept="image/jpeg,image/png,image/webp,image/gif" multiple>
      <div class="previews" id="previews"></div>
      <label for="note">Source note</label>
      <textarea id="note" name="note" required placeholder="Where these photos came from, for the provenance record"></textarea>
      <label for="age">Age range</label>
      <select id="age" name="age" required>
        <!-- ages -->
      </select>
      <label for="hair">Hairstyle</label>
      <input id="hair" name="hair" type="text" required placeholder="Black hair, center part, just past the shoulders">
      <button class="primary" id="go" type="submit">Generate face</button>
    </form>
    <p class="status" id="status"></p>
    <section class="result" id="result" hidden>
      <img id="portrait" alt="Generated face">
      <p><a id="notes" href="">Provenance record</a></p>
    </section>
  </main>
  <script>
    const filesInput = document.getElementById("files");
    const previews = document.getElementById("previews");
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    const go = document.getElementById("go");
    let chosen = [];

    document.getElementById("drop").addEventListener("click", () => filesInput.click());
    filesInput.addEventListener("change", () => addFiles(filesInput.files));

    function addFiles(list) {
      const room = 8 - chosen.length;
      const next = Array.from(list || []).slice(0, room);
      if ((list || []).length > room) {
        status.textContent = "The limit is 8 photos.";
      }
      for (const file of next) {
        chosen.push({ file, url: URL.createObjectURL(file) });
      }
      filesInput.value = "";
      draw();
    }

    function draw() {
      previews.innerHTML = "";
      chosen.forEach((item, index) => {
        const shot = document.createElement("div");
        shot.className = "shot";
        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.file.name;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.setAttribute("aria-label", "Remove " + item.file.name);
        remove.addEventListener("click", (event) => {
          event.stopPropagation();
          URL.revokeObjectURL(item.url);
          chosen.splice(index, 1);
          draw();
        });
        shot.append(img, remove);
        previews.append(shot);
      });
    }

    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (chosen.length < 2) {
        status.textContent = "Add at least two photos of the same person.";
        return;
      }
      const note = document.getElementById("note").value.trim();
      const hair = document.getElementById("hair").value.trim();
      const age = document.getElementById("age").value;
      if (!note || !hair) {
        status.textContent = "Enter the source note and the hairstyle.";
        return;
      }
      go.disabled = true;
      result.hidden = true;
      status.textContent = "Uploading photos…";
      const session = crypto.randomUUID();
      try {
        for (const item of chosen) {
          const response = await fetch("/api/upload", {
            method: "POST",
            headers: {
              "X-Session": session,
              "X-Filename": encodeURIComponent(item.file.name),
            },
            body: item.file,
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.error || "Upload failed.");
        }
        status.textContent = "Generating a new face. This can take a few minutes.";
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session, note, age, hair }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Generation failed.");
        document.getElementById("portrait").src = payload.image;
        const notes = document.getElementById("notes");
        notes.href = payload.notes;
        result.hidden = false;
        status.textContent = "Saved. Compare this face with the references before approving it.";
      } catch (error) {
        status.textContent = error.message || "Something went wrong.";
      } finally {
        go.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def _safe_session(value: str) -> str:
    if len(value) == 36 and all(char in "0123456789abcdef-" for char in value.lower()):
        return value.lower()
    raise ValueError("Invalid upload session.")


def _safe_filename(value: str) -> str:
    name = Path(unquote(value)).name.strip()
    cleaned = "".join(char if char.isalnum() or char in ".-_ " else "_" for char in name)
    cleaned = cleaned.strip(" .") or "image"
    return cleaned[:120]


class FaceHandler(BaseHTTPRequestHandler):
    server_version = "SilkStudioFace"

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} {format % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = _page().encode("utf-8")
            self._send_bytes(200, body, "text/html; charset=utf-8")
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
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid upload."})
            return
        try:
            session = _safe_session(self.headers.get("X-Session", ""))
            filename = _safe_filename(self.headers.get("X-Filename", "image"))
        except ValueError as exc:
            self._discard(length)
            self._send_json(400, {"error": str(exc)})
            return
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            self._discard(length)
            self._send_json(400, {"error": "Use a jpeg, png, webp, or gif file."})
            return
        if length <= 0 or length > MAX_FILE_BYTES:
            self._discard(length)
            self._send_json(400, {"error": "Each photo must be under 50MB."})
            return

        with _sessions_lock:
            stored = _sessions.setdefault(session, [])
            if len(stored) >= MAX_REFERENCES:
                self._discard(length)
                self._send_json(400, {"error": f"The limit is {MAX_REFERENCES} photos."})
                return
            folder = PROJECT_ROOT / "uploads" / "face-sessions" / session
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / f"{len(stored) + 1:02d}_{filename}"
            stored.append(destination)

        try:
            self._write_body(destination, length)
        except Exception:
            destination.unlink(missing_ok=True)
            with _sessions_lock:
                stored = _sessions.get(session, [])
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
            note = str(body.get("note", ""))
            age = str(body.get("age", ""))
            hair = str(body.get("hair", ""))
        except (ValueError, TypeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid generate request."})
            return

        with _sessions_lock:
            sources = list(_sessions.get(session, []))
        if len(sources) < MIN_REFERENCES:
            self._send_json(400, {"error": f"Upload at least {MIN_REFERENCES} photos first."})
            return

        if not _generate_lock.acquire(blocking=False):
            self._send_json(409, {"error": "A face is already being generated."})
            return
        try:
            image_path, notes_path, summary = create_face(sources, note, age, hair)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        finally:
            _generate_lock.release()

        self._clear_session(session)
        outputs = PROJECT_ROOT / "outputs"
        self._send_json(200, {
            "image": "/outputs/" + image_path.resolve().relative_to(outputs.resolve()).as_posix(),
            "notes": "/outputs/" + notes_path.resolve().relative_to(outputs.resolve()).as_posix(),
            "summary": summary,
        })

    def _clear_session(self, session: str) -> None:
        with _sessions_lock:
            _sessions.pop(session, None)
        folder = PROJECT_ROOT / "uploads" / "face-sessions" / session
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
        root = (PROJECT_ROOT / "outputs").resolve()
        target = (root / unquote(relative)).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            self._send_json(404, {"error": "File not found."})
            return
        mime_type, _ = mimetypes.guess_type(target.name)
        self._send_bytes(200, target.read_bytes(), mime_type or "application/octet-stream")

    def _send_json(self, status: int, payload: object) -> None:
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(open_browser: bool) -> None:
    server = None
    for port in range(8780, 8790):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), FaceHandler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port between 8780 and 8789.")

    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Face page at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open a page to generate a new adult face from reference photos."
    )
    parser.add_argument("images", nargs="*", help="Optional. With photos, run once in the terminal instead of opening the page.")
    parser.add_argument("--note", default="", help="Provenance note, such as which public-figure set this is")
    parser.add_argument("--age", default="25-34", choices=ADULT_AGE_RANGES)
    parser.add_argument("--hair", default="", help="Hairstyle for this new identity")
    parser.add_argument("--no-browser", action="store_true", help="Start the page without opening a browser window")
    args = parser.parse_args()

    if not args.images:
        serve(open_browser=not args.no_browser)
        return

    sources = collect_paths(args.images)
    try:
        image_path, notes_path, _summary = create_face(
            sources,
            args.note.strip(),
            args.age,
            args.hair.strip(),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print("Approval is pending. Compare the result with the references before it enters the library.")
    print(image_path)
    print(notes_path)


if __name__ == "__main__":
    main()
