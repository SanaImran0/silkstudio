"""
Ask for each piece of a dress, then generate one 4K photograph of it being worn.

Uses the same OpenRouter Gemini image call as gemini_multi_image.py.
The wording of the request lives in prompt.py.

Setup:
    pip install requests python-dotenv pillow
    Put OPENROUTER_API_KEY in .env

Run from the repository root:
    python backend/serve.py
    npm run dev --prefix frontend

The terminal interview is still available:
    python backend/generate_dress.py
"""

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from compress_references import compress_images, report
from prompt import GARMENT_SLOTS, GarmentSlot, build_prompt, image_caption

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Same model that produced the portraits in
# outputs/google/gemini-3.1-flash-image-preview. The failed run used
# gemini-3-pro-image-preview, which Google AI Studio blocks from this location.
MODEL = "google/gemini-3.1-flash-image-preview"
PROVIDER = None
ASPECT_RATIO = "9:16"
IMAGE_SIZE = "4K"
SEND_IMAGE_SIZE = True
MAX_REFERENCE_IMAGES = 14
VALID_ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1",
}
VALID_SIZES = {"0.5K", "1K", "2K", "4K"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@dataclass
class CollectedImage:
    slot: GarmentSlot
    path: Path


def clean_path(raw: str) -> Path:
    """Accept a pasted path or a macOS Terminal drag-and-drop path."""
    text = raw.strip()
    if text.startswith("& "):
        text = text[2:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    else:
        text = (
            text.replace("\\ ", " ")
            .replace("\\(", "(")
            .replace("\\)", ")")
            .replace("\\'", "'")
        )
    return Path(text).expanduser()


def validate_image(path: Path) -> str | None:
    """Return an error message, or None when the file can be uploaded."""
    if not path.exists():
        return f"No file at {path}"
    if not path.is_file():
        return f"Not a file: {path}"
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return f"Use a jpeg, png, webp, or gif file. Got {path.suffix or 'no extension'}."
    return None


def _to_content_part(image_path: Path) -> dict:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def collect_images() -> list[CollectedImage]:
    """Walk the garment slots and collect local image paths."""
    print(
        "Upload each piece of this one outfit.\n"
        "Drag a file into the terminal, or paste its full path, then press Enter.\n"
        "\n"
        "  none   skip this piece\n"
        "  undo   remove the last image added for this piece\n"
        "  done   finish a detail piece (or press Enter on an empty line)\n"
        "  quit   stop before any image is generated\n"
        f"\nThis model accepts {MAX_REFERENCE_IMAGES} reference images. "
        "Add the full pieces before spending slots on close-ups.\n"
    )
    collected: list[CollectedImage] = []

    for slot in GARMENT_SLOTS:
        slot_images: list[CollectedImage] = []
        while True:
            remaining = MAX_REFERENCE_IMAGES - len(collected) - len(slot_images)
            if remaining <= 0:
                if slot_images:
                    print(f"Stopped {slot.label} at the {MAX_REFERENCE_IMAGES}-image limit.")
                else:
                    print(
                        f"\n{slot.ask}\n"
                        f"The {MAX_REFERENCE_IMAGES}-image limit is full, so this piece is skipped."
                    )
                break

            later_required = [
                later.ask
                for later in GARMENT_SLOTS
                if later.required and GARMENT_SLOTS.index(later) > GARMENT_SLOTS.index(slot)
            ]
            print(f"\n{slot.ask}")
            if slot.many:
                print("None or many. One path per line. Type done when finished.")
            elif slot.required:
                print("One image. Type none only if this outfit does not have this piece.")
            print(f"Images used so far: {len(collected) + len(slot_images)} of {MAX_REFERENCE_IMAGES}.")
            if later_required:
                print("Still to come: " + "; ".join(later_required) + ".")

            raw = input("> ").strip()
            command = raw.lower()
            if command == "quit":
                raise SystemExit("Stopped before generating.")
            if command == "undo":
                if slot_images:
                    removed = slot_images.pop()
                    print(f"Removed {removed.path.name}")
                else:
                    print("Nothing to undo for this piece.")
                continue
            if command == "done":
                if not slot.many:
                    print("Paste a path, or type none to skip.")
                    continue
                break
            if command == "none":
                if slot_images:
                    print("Images are already added. Type done to keep them, or undo to remove them.")
                    continue
                if slot.required:
                    print(f"Skipped {slot.label}. The dress will not invent this piece.")
                break
            if command == "":
                if slot.many:
                    break
                print("Paste a path, or type none to skip.")
                continue

            path = clean_path(raw)
            error = validate_image(path)
            if error:
                print(error)
                continue
            if any(item.path.resolve() == path.resolve() for item in collected + slot_images):
                print("That file is already added.")
                continue
            slot_images.append(CollectedImage(slot=slot, path=path))
            print(f"Added {path.name}")
            if not slot.many:
                break

        collected.extend(slot_images)

    if not collected:
        raise SystemExit("No images added. Nothing was generated.")
    return collected


def confirm(images: list[CollectedImage]) -> None:
    print("\nReferences to send:")
    for number, image in enumerate(images, start=1):
        print(f"  {number}. {image.slot.label}: {image.path}")
    answer = input("\nGenerate the 4K image now? [y/n] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Cancelled. Nothing was generated.")


def output_paths() -> tuple[Path, Path]:
    """Folder is model / aspect ratio / size, and the file name is the time."""
    model_name = MODEL.split("/")[-1]
    aspect = ASPECT_RATIO.replace(":", "x")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = PROJECT_ROOT / "outputs" / model_name / aspect / IMAGE_SIZE
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stamp}.png", directory / f"{stamp}.txt"


def generate_image(prompt: str, images: list[CollectedImage], image_path: Path) -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or the environment.")
    if ASPECT_RATIO not in VALID_ASPECT_RATIOS:
        raise ValueError(f"aspect_ratio must be one of {sorted(VALID_ASPECT_RATIOS)}")
    if IMAGE_SIZE not in VALID_SIZES:
        raise ValueError(f"image_size must be one of {sorted(VALID_SIZES)}")

    content: list[dict] = [{
        "type": "text",
        "text": (
            "The following images are labeled parts of one Pakistani outfit. "
            "Use them only as specified. The full instructions are after the images."
        ),
    }]
    for number, image in enumerate(images, start=1):
        content.append({
            "type": "text",
            "text": image_caption(number, image.slot.label, image.slot.fidelity),
        })
        content.append(_to_content_part(image.path))
    content.append({"type": "text", "text": prompt})

    image_config = {"aspect_ratio": ASPECT_RATIO}
    if SEND_IMAGE_SIZE:
        image_config["image_size"] = IMAGE_SIZE
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "image_config": image_config,
    }
    if PROVIDER:
        payload["provider"] = PROVIDER
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=600,
    )
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter returned {response.status_code}:\n{response.text}"
        )

    result = response.json()
    try:
        message = result["choices"][0]["message"]
        data_url = message["images"][0]["image_url"]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        preview = json.dumps(result, indent=2)[:4000]
        raise RuntimeError(f"No image in the response.\n{preview}") from exc
    if "," not in data_url:
        raise RuntimeError("The returned image was not a base64 data URL.")

    _, encoded = data_url.split(",", 1)
    image_path.write_bytes(base64.b64decode(encoded))
    if "usage" in result:
        print(f"Usage: {result['usage']}")


def write_notes(
    notes_path: Path,
    images: list[CollectedImage],
    sent_paths: list[Path],
    prompt: str,
) -> None:
    lines = [
        f"model: {MODEL}",
        f"aspect_ratio: {ASPECT_RATIO}",
        f"image_size: {IMAGE_SIZE}",
        "",
        "references:",
    ]
    for number, (image, sent) in enumerate(zip(images, sent_paths), start=1):
        lines.append(f"{number}. {image.slot.label}")
        lines.append(f"   original: {image.path}")
        lines.append(f"   sent: {sent}")
    lines.extend(["", prompt, ""])
    notes_path.write_text("\n".join(lines), encoding="utf-8")


def create_dress(images: list[CollectedImage]) -> tuple[Path, Path, str]:
    """Compress the references, generate the portrait, and return the saved paths."""
    prompt = build_prompt([
        {
            "index": number,
            "label": image.slot.label,
            "fidelity": image.slot.fidelity,
            "filename": image.path.name,
        }
        for number, image in enumerate(images, start=1)
    ])
    image_path, notes_path = output_paths()
    print("\nFitting references under Google's 20MB request limit.")
    prepared = compress_images(
        [image.path for image in images],
        PROJECT_ROOT / "uploads" / image_path.stem,
    )
    summary = report(prepared)
    print(summary)
    sent = [
        CollectedImage(slot=image.slot, path=item.output)
        for image, item in zip(images, prepared)
    ]
    print("\nGenerating. A 4K image with many references can take several minutes.")
    generate_image(prompt, sent, image_path)
    write_notes(notes_path, images, [item.output for item in prepared], prompt)
    print(f"Saved -> {image_path}")
    print(f"Notes -> {notes_path}")
    return image_path, notes_path, summary


def main() -> None:
    images = collect_images()
    confirm(images)
    create_dress(images)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nStopped.")
