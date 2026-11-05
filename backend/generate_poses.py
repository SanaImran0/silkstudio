"""
Save eight separate photographs of one model, in the views of a turnaround sheet.

Four are full length: front, left profile, right profile, back.
Four are head and shoulders: front, left profile, right three-quarter, back of the head.

The model photograph is the only person sent to the generator. Each view is its
own image, and later views are locked to the full-length front so the face,
hair, body, and clothes stay the same.

Run from the repository root. This opens a page:

    python backend/generate_poses.py

Or generate straight from a model photograph:

    python backend/generate_poses.py outputs/faces/20261027_021301/face.png
"""

import argparse
import base64
import json
import mimetypes
import shutil
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from generate_face import (
    API_URL,
    IMAGE_SUFFIXES,
    MAX_FILE_BYTES,
    MODEL,
    OPENROUTER_API_KEY,
    PROJECT_ROOT,
    image_part,
)

IMAGE_SIZE = "2K"

POSES: tuple[dict[str, str], ...] = (
    {
        "id": "01_full_front",
        "label": "Full body, front",
        "aspect": "3:4",
        "instruction": (
            "Full-length photograph. She stands facing the camera, weight even, "
            "arms relaxed at her sides, feet slightly apart. Include her shoes, "
            "with a little space above her head and below her feet."
        ),
    },
    {
        "id": "02_full_left",
        "label": "Full body, left profile",
        "aspect": "3:4",
        "instruction": (
            "Full-length photograph. Strict left profile: she stands side-on, "
            "facing the left edge of the frame, arms relaxed. Include her shoes."
        ),
    },
    {
        "id": "03_full_right",
        "label": "Full body, right profile",
        "aspect": "3:4",
        "instruction": (
            "Full-length photograph. Strict right profile: she stands side-on, "
            "facing the right edge of the frame, arms relaxed. Include her shoes."
        ),
    },
    {
        "id": "04_full_back",
        "label": "Full body, back",
        "aspect": "3:4",
        "instruction": (
            "Full-length photograph from directly behind. Her back faces the camera, "
            "arms relaxed. Include her shoes. Her face is not visible."
        ),
    },
    {
        "id": "05_head_front",
        "label": "Head and shoulders, front",
        "aspect": "3:4",
        "instruction": (
            "Head-and-shoulders photograph from the chest up. She faces the camera. "
            "The top of her head is in frame."
        ),
    },
    {
        "id": "06_head_left",
        "label": "Head and shoulders, left profile",
        "aspect": "3:4",
        "instruction": (
            "Head-and-shoulders photograph from the chest up, in strict left profile. "
            "She faces the left edge of the frame. The top of her head is in frame."
        ),
    },
    {
        "id": "07_head_three_quarter",
        "label": "Head and shoulders, right three-quarter",
        "aspect": "3:4",
        "instruction": (
            "Head-and-shoulders photograph from the chest up. Her head is turned "
            "about three-quarters toward the right edge of the frame, so one cheek "
            "and the side of her face are visible. The top of her head is in frame."
        ),
    },
    {
        "id": "08_head_back",
        "label": "Head and shoulders, back",
        "aspect": "3:4",
        "instruction": (
            "Head-and-shoulders photograph from the chest up, taken from directly "
            "behind. Show the back of her head and hair. Her face is not visible."
        ),
    },
)
POSE_BY_ID = {pose["id"]: pose for pose in POSES}

_sessions: dict[str, "PoseJob"] = {}
_sessions_lock = threading.Lock()
_generate_lock = threading.Lock()


@dataclass
class PoseJob:
    model: Path
    folder: Path
    anchor: Path | None = None


def pose_prompt(pose: dict[str, str], has_anchor: bool) -> str:
    outfit = (
        "Match the clothes, shoes, hair, body, and face in the full-length photograph exactly."
        if has_anchor
        else (
            "This is the first full-length photograph. Dress her in a plain black long-sleeve "
            "top with no print, plain black straight trousers, and plain black shoes. No logos."
        )
    )
    return f"""
Photograph the same adult woman shown in the attached model image. Keep her face, age, skin, and hair. Do not invent a different person, and do not copy anyone who is not in the attached photographs.

{pose["instruction"]}

{outfit}

When her face is visible, keep the expression from the model photograph: warm eyes and a small living smile, with the same light makeup. Not a blank passport face, and not a sad face. Light comes from one side, on a seamless light-grey studio background. Real fabric texture, real skin, a few loose hairs. One person, one photograph, not a grid or contact sheet. No text, logo, watermark, or extra people.
""".strip()


def generate_pose(pose: dict[str, str], model: Path, anchor: Path | None, destination: Path) -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or the environment.")
    content: list[dict] = [{
        "type": "text",
        "text": "Image 1 is the model. Depict this exact person, not a new face.",
    }]
    content.append(image_part(model))
    if anchor is not None:
        content.append({
            "type": "text",
            "text": "Image 2 is the same model full length. Match her outfit, body, hair, and face.",
        })
        content.append(image_part(anchor))
    content.append({"type": "text", "text": pose_prompt(pose, anchor is not None)})
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
            "image_config": {"aspect_ratio": pose["aspect"], "image_size": IMAGE_SIZE},
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))


def write_notes(folder: Path, model: Path) -> None:
    lines = [
        f"created_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"model: {MODEL}",
        f"image_size: {IMAGE_SIZE}",
        f"identity: {model.resolve()}",
        "",
        "Each file is one camera angle of this same model.",
        "The full-length front locks the clothes and body for the other seven.",
        "",
        "poses:",
    ]
    for pose in POSES:
        lines.append(f"- {pose['id']}.png: {pose['label']}")
    (folder / "poses.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_pose(job: PoseJob, pose: dict[str, str]) -> Path:
    if pose["id"] != POSES[0]["id"] and job.anchor is None:
        raise ValueError("The full-body front photograph has to be made first.")
    destination = job.folder / f"{pose['id']}.png"
    anchor = None if pose["id"] == POSES[0]["id"] else job.anchor
    print(f"Generating {pose['label']}.")
    generate_pose(pose, job.model, anchor, destination)
    if pose["id"] == POSES[0]["id"]:
        job.anchor = destination
    write_notes(job.folder, job.model)
    print(f"Saved -> {destination}")
    return destination


def _page() -> str:
    cards = []
    for pose in POSES:
        cards.append(
            f'<figure data-pose="{pose["id"]}">'
            f"<figcaption>{pose['label']}</figcaption>"
            f'<img alt="" hidden>'
            f'<p class="wait">Waiting</p>'
            f"</figure>"
        )
    return PAGE_TEMPLATE.replace("<!-- cards -->", "\n".join(cards))


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Silk Studio — Model poses</title>
  <style>
    body { margin: 0; background: #f7f6f4; color: #1a1a1a; font-family: system-ui, sans-serif; }
    main { max-width: 980px; margin: 0 auto; padding: 40px 20px 80px; }
    h1 { font-size: 1.7rem; font-weight: 560; margin: 0 0 8px; }
    .lead { color: #4d4a46; line-height: 1.5; margin: 0 0 28px; max-width: 640px; }
    .drop {
      border: 1px dashed #b9b4ad; border-radius: 12px; padding: 22px; background: #fff; cursor: pointer;
      max-width: 640px;
    }
    .drop strong { display: block; margin-bottom: 4px; }
    .drop span { color: #6b6762; font-size: 0.9rem; }
    .chosen { display: flex; align-items: center; gap: 12px; margin-top: 14px; }
    .chosen[hidden], figure img[hidden], .wait[hidden] { display: none; }
    .chosen img { width: 72px; height: 96px; object-fit: cover; border-radius: 8px; }
    button.primary {
      margin-top: 22px; background: #1a1a1a; color: #fff; border: 0; border-radius: 999px;
      padding: 12px 22px; font: inherit; cursor: pointer;
    }
    button.primary:disabled { opacity: 0.45; cursor: default; }
    .status { min-height: 1.4em; margin-top: 14px; color: #6b3a2a; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 28px; }
    figure { margin: 0; background: #fff; border-radius: 12px; padding: 10px; }
    figcaption { font-size: 0.82rem; margin-bottom: 8px; }
    figure img { width: 100%; border-radius: 8px; display: block; }
    .wait { margin: 8px 0 4px; color: #6b6762; font-size: 0.85rem; }
    input[type="file"] { display: none; }
    @media (max-width: 800px) { .grid { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <main>
    <h1>Model poses</h1>
    <p class="lead">Upload one photograph of your model. This saves eight separate photographs of that same person: four full-length views and four head-and-shoulders views.</p>
    <form id="form">
      <div class="drop" id="drop">
        <strong>Model photograph</strong>
        <span>One image of the model these poses should follow.</span>
      </div>
      <input id="file" type="file" accept="image/jpeg,image/png,image/webp,image/gif">
      <div class="chosen" id="chosen" hidden>
        <img id="preview" alt="Selected model">
        <span id="filename"></span>
      </div>
      <button class="primary" id="go" type="submit">Generate 8 poses</button>
    </form>
    <p class="status" id="status"></p>
    <div class="grid">
      <!-- cards -->
    </div>
  </main>
  <script>
    const fileInput = document.getElementById("file");
    const status = document.getElementById("status");
    const go = document.getElementById("go");
    let chosen = null;

    document.getElementById("drop").addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      if (chosen) URL.revokeObjectURL(chosen.url);
      chosen = { file, url: URL.createObjectURL(file) };
      document.getElementById("preview").src = chosen.url;
      document.getElementById("filename").textContent = file.name;
      document.getElementById("chosen").hidden = false;
      fileInput.value = "";
    });

    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!chosen) {
        status.textContent = "Add a photograph of the model first.";
        return;
      }
      go.disabled = true;
      status.textContent = "Uploading the model…";
      const session = crypto.randomUUID();
      try {
        const uploaded = await fetch("/api/upload", {
          method: "POST",
          headers: {
            "X-Session": session,
            "X-Filename": encodeURIComponent(chosen.file.name),
          },
          body: chosen.file,
        });
        const uploadPayload = await uploaded.json();
        if (!uploaded.ok) throw new Error(uploadPayload.error || "Upload failed.");

        const cards = Array.from(document.querySelectorAll("figure"));
        for (const card of cards) {
          const pose = card.dataset.pose;
          const caption = card.querySelector("figcaption").textContent;
          status.textContent = "Generating " + caption + ".";
          card.querySelector(".wait").textContent = "Generating…";
          const response = await fetch("/api/pose", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session, pose }),
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.error || "That pose failed.");
          const img = card.querySelector("img");
          img.src = payload.image;
          img.hidden = false;
          card.querySelector(".wait").hidden = true;
        }
        status.textContent = "Saved eight photographs of this model.";
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


class PoseHandler(BaseHTTPRequestHandler):
    server_version = "SilkStudioPoses"

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} {format % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(200, _page().encode("utf-8"), "text/html; charset=utf-8")
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
        if path == "/api/pose":
            self._pose()
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
            self._send_json(400, {"error": "The photograph must be under 50MB."})
            return

        folder = PROJECT_ROOT / "uploads" / "pose-sessions" / session
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / filename
        try:
            self._write_body(destination, length)
        except Exception:
            destination.unlink(missing_ok=True)
            self._send_json(400, {"error": "The upload was interrupted."})
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = PROJECT_ROOT / "outputs" / "poses" / stamp
        output.mkdir(parents=True, exist_ok=True)
        kept = output / f"model{suffix}"
        shutil.copy2(destination, kept)
        with _sessions_lock:
            _sessions[session] = PoseJob(model=kept, folder=output)
        self._send_json(200, {"name": kept.name})

    def _pose(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            session = _safe_session(str(body.get("session", "")))
            pose_id = str(body.get("pose", ""))
        except (ValueError, TypeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid pose request."})
            return
        pose = POSE_BY_ID.get(pose_id)
        if pose is None:
            self._send_json(400, {"error": "Unknown pose."})
            return
        with _sessions_lock:
            job = _sessions.get(session)
        if job is None:
            self._send_json(400, {"error": "Upload the model photograph first."})
            return
        if not _generate_lock.acquire(blocking=False):
            self._send_json(409, {"error": "A pose is already being generated."})
            return
        try:
            image_path = render_pose(job, pose)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        finally:
            _generate_lock.release()
        relative = image_path.resolve().relative_to((PROJECT_ROOT / "outputs").resolve()).as_posix()
        self._send_json(200, {"image": "/outputs/" + relative, "label": pose["label"]})

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
    for port in range(8790, 8800):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), PoseHandler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port between 8790 and 8799.")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Pose page at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def run_all(model: Path) -> None:
    if not model.is_file():
        raise SystemExit(f"No file at {model}")
    if model.suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit("Use a jpeg, png, webp, or gif file.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = PROJECT_ROOT / "outputs" / "poses" / stamp
    folder.mkdir(parents=True, exist_ok=True)
    kept = folder / f"model{model.suffix.lower()}"
    shutil.copy2(model, kept)
    job = PoseJob(model=kept, folder=folder)
    for pose in POSES:
        render_pose(job, pose)
    print(f"Saved eight photographs -> {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create eight separate pose photographs of one model.")
    parser.add_argument("model", nargs="?", help="Optional model photograph. Without it, a page opens.")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.model:
        run_all(Path(args.model).expanduser())
        return
    serve(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
