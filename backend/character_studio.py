"""
Silk Studio character studio: build a fashion model as a complete character.

One page, three jobs:

  1. New character, face only.
  2. New character, face plus the full character sheet (8 views).
  3. Existing character: pick one from the library, edit its details (metadata),
     edit its face with a prompt, generate its missing sheet views, or make a new
     image of that character from your own prompt.

Everything for one character lives in one folder:

    outputs/faces/<id>/
        face.png              the casting face (identity)
        metadata.json         name, age, size, height, silhouette, hair, traits ...
        provenance.txt        record for legal and compliance review
        sheet/<view>.png      the 8 sheet views (front, sides, back, head shots)
        extras/<stamp>_*.png  prompt-made images, with prompts.jsonl
        face_edits/           candidate face edits waiting for your decision
        face_history/         every face that was replaced by an accepted edit

Run from the repository root:

    python backend/character_studio.py
"""

import argparse
import base64
import json
import os
import random
import re
import shutil
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

from generate_face import (
    ADULT_AGE_RANGES,
    API_URL,
    ASPECT_RATIO,
    EYE_COLORS,
    HAIR_COLORS,
    HEIGHTS,
    IMAGE_SIZE,
    MODEL,
    OPENROUTER_API_KEY,
    PROJECT_ROOT,
    SHAPES,
    SIZES,
    SKIN_TONES,
    DISTINCTIVE,
    TRAIT_KEYS,
    VARIATIONS,
    VIBES,
    FaceHandler,
    _safe_name,
    _safe_session,
    blend_traits,
    body_description,
    create_face,
    describe_traits,
    face_prompt,
    file_sha256,
    image_part,
)
from generate_poses import POSES, POSE_BY_ID

FACES_DIR = PROJECT_ROOT / "outputs" / "faces"
SHEET_SIZE = "2K"
CUSTOM_ASPECTS = ("3:4", "4:5", "9:16", "1:1", "16:9")
DEFAULT_OUTFIT = "a plain white skin-tight athletic vest top, plain white full-length skin-tight athletic tights, and plain white low sneakers"
MAX_RETRIES = 4

_ID = re.compile(r"^\d{8}_\d{6}$")
DETAIL_KEYS = ("name", "age_range", "size", "height", "shape", "skin", "eye_color", "hair_color", "hairstyle", "vibe")
_work_lock = threading.Lock()


# --------------------------------------------------------------------------
# Library helpers
# --------------------------------------------------------------------------


def _outputs_url(path: Path) -> str:
    return "/outputs/" + path.resolve().relative_to((PROJECT_ROOT / "outputs").resolve()).as_posix()


def _meta(folder: Path) -> dict:
    try:
        return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _folder(char_id: str) -> Path:
    if not _ID.match(char_id or ""):
        raise ValueError("Invalid character id.")
    folder = FACES_DIR / char_id
    if not (folder / "face.png").is_file():
        raise ValueError("That character has no face image.")
    return folder


def list_characters() -> list[dict]:
    items = []
    if not FACES_DIR.is_dir():
        return items
    for folder in sorted(FACES_DIR.iterdir(), reverse=True):
        if not (folder.is_dir() and _ID.match(folder.name) and (folder / "face.png").is_file()):
            continue
        meta = _meta(folder)
        sheet_dir, extras_dir = folder / "sheet", folder / "extras"
        sheet = {p.stem: _outputs_url(p) for p in sorted(sheet_dir.glob("*.png"))} if sheet_dir.is_dir() else {}
        extras = [_outputs_url(p) for p in sorted(extras_dir.glob("*.png"), reverse=True)] if extras_dir.is_dir() else []
        edits_dir = folder / "face_edits"
        candidates = (
            [{"file": p.name, "url": _outputs_url(p)} for p in sorted(edits_dir.glob("*.png"), reverse=True)]
            if edits_dir.is_dir() else []
        )
        name = (meta.get("name") or "").strip()
        detail = ", ".join(str(meta[k]) for k in ("age_range", "size") if meta.get(k))
        items.append({
            "id": folder.name,
            "label": f"{name or folder.name}" + (f" ({detail})" if detail else ""),
            "face": _outputs_url(folder / "face.png") + f"?v={int((folder / 'face.png').stat().st_mtime)}",
            "sheet": sheet,
            "extras": extras,
            "candidates": candidates,
            "meta": {k: meta.get(k, "") for k in DETAIL_KEYS},
        })
    return items


def character_text(meta: dict) -> str:
    """The latest saved details, written out. Used so every new image follows them."""
    parts = []
    if meta.get("age_range"):
        parts.append(f"Apparent age: {meta['age_range']}")
    if meta.get("body_description"):
        parts.append(f"Body and build: {meta['body_description']}")
    if meta.get("hair_color") or meta.get("hairstyle"):
        parts.append(f"Hair: {meta.get('hair_color', '')} {meta.get('hairstyle', '')}".strip())
    if meta.get("skin") in SKIN_TONES:
        parts.append(f"Skin: {SKIN_TONES[meta['skin']]}")
    if meta.get("eye_color"):
        parts.append(f"Eye colour: {meta['eye_color']}")
    return ". ".join(parts)


def _reference_jpeg(src: Path, dest: Path, max_side: int = 1536) -> Path:
    """Small JPEG copy: keeps each request far below the provider's size limit."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    image.save(dest, "JPEG", quality=92)
    return dest


def _post_image(content: list[dict], aspect: str, size: str) -> bytes:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or the environment.")
    last = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "modalities": ["image", "text"],
                    "image_config": {"aspect_ratio": aspect, "image_size": size},
                },
                timeout=600,
            )
            if response.ok:
                data = response.json()
                try:
                    url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
                    return base64.b64decode(url.split(",", 1)[1])
                except (KeyError, IndexError, TypeError, ValueError):
                    last = "No image in the response: " + json.dumps(data)[:3000]
            else:
                last = f"OpenRouter returned {response.status_code}: {response.text[:3000]}"
                if response.status_code in (401, 402, 403):
                    break
        except requests.RequestException as exc:
            last = f"Network error: {exc}"
        time.sleep(4 * attempt)
    raise RuntimeError(last)


# --------------------------------------------------------------------------
# Sheet views and custom images
# --------------------------------------------------------------------------


def view_prompt(pose: dict, meta: dict, outfit: str, has_anchor: bool) -> str:
    outfit_line = (
        "Match the clothes, shoes, hair, body and face in image 2 exactly."
        if has_anchor else f"She wears {outfit}. No logos, no prints."
    )
    body = character_text(meta)
    body_line = (
        f"Her latest saved details: {body}. These written details are the most recent. Her face comes from image 1, "
        "but if the images show a different hair colour, hairstyle, skin tone, size or build than these details, "
        "follow the written details.\n\n"
        if body else "Match her body to what her face and build suggest.\n\n"
    )
    size = meta.get("size", "")
    fit_line = (
        "Her clothing is opaque, modest, skin-tight athletic wear, so her real body shape, proportions and height are "
        "clearly visible and match the build described. "
        + (f"She should look like a person who wears size {size}. " if size else "")
        + "Do not slim, enlarge or idealise her body."
    )
    return f"""
Photograph the same adult woman shown in image 1, her face reference. Keep her exact face, age, skin tone, hair and expression. Do not invent a different person.

{body_line}{pose["instruction"]}

{outfit_line}
{fit_line}

When her face is visible, keep the look of the face photograph: confident and poised, bright engaged eyes, very light makeup, professionally styled fashion-model hair. Not a dull or blank face. Real skin texture, real fabric texture, a few fine flyaways. Soft studio light from one side on a plain mid-grey seamless backdrop. One person, one photograph, not a grid or contact sheet. No text, logo, watermark, or extra people.
""".strip()


def render_view(char_id: str, view_id: str, outfit: str = "") -> Path:
    pose = POSE_BY_ID.get(view_id)
    if pose is None:
        raise ValueError("Unknown view.")
    folder = _folder(char_id)
    meta = _meta(folder)
    anchor = folder / "sheet" / f"{POSES[0]['id']}.png"
    is_front = view_id == POSES[0]["id"]
    if not is_front and not anchor.is_file():
        raise ValueError("Generate the full-body front view first. The other views are matched to it.")
    cache = folder / ".cache"
    content: list[dict] = [
        {"type": "text", "text": "Image 1 is the character's face. Depict this exact woman, not a new face."},
        image_part(_reference_jpeg(folder / "face.png", cache / "face_ref.jpg")),
    ]
    if not is_front:
        content.append({"type": "text", "text": "Image 2 is the same woman full length. Match her outfit, body, hair and face."})
        content.append(image_part(_reference_jpeg(anchor, cache / "anchor_ref.jpg")))
    content.append({"type": "text", "text": view_prompt(pose, meta, outfit.strip() or DEFAULT_OUTFIT, not is_front)})
    data = _post_image(content, pose["aspect"], SHEET_SIZE)
    destination = folder / "sheet" / f"{view_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def render_custom(char_id: str, prompt: str, aspect: str) -> Path:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Describe the image you want.")
    if len(prompt) > 1500:
        raise ValueError("Keep the prompt under 1500 characters.")
    if aspect not in CUSTOM_ASPECTS:
        raise ValueError(f"Aspect ratio must be one of {', '.join(CUSTOM_ASPECTS)}.")
    folder = _folder(char_id)
    meta = _meta(folder)
    anchor = folder / "sheet" / f"{POSES[0]['id']}.png"
    cache = folder / ".cache"
    content: list[dict] = [
        {"type": "text", "text": "Image 1 is the character's face. Depict this exact woman, not a new face."},
        image_part(_reference_jpeg(folder / "face.png", cache / "face_ref.jpg")),
    ]
    if anchor.is_file():
        content.append({"type": "text", "text": "Image 2 is the same woman full length. Keep her body, build and proportions."})
        content.append(image_part(_reference_jpeg(anchor, cache / "anchor_ref.jpg")))
    body = character_text(meta)
    text = f"""
Photograph the same adult woman shown in the attached images. Keep her exact face, skin tone, hair and body proportions. Do not invent a different person.

{body + "." if body else ""}

Requested image: {prompt}

A real photograph: real skin texture, real fabric, natural light falloff, the confident presence of a professional fashion model unless the request says otherwise. One person unless the request says otherwise. No text, logo, or watermark.
""".strip()
    content.append({"type": "text", "text": text})
    data = _post_image(content, aspect, SHEET_SIZE)
    extras = folder / "extras"
    extras.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "image"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = extras / f"{stamp}_{slug}.png"
    destination.write_bytes(data)
    with (extras / "prompts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "file": destination.name, "prompt": prompt, "aspect": aspect,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }) + "\n")
    return destination


# --------------------------------------------------------------------------
# Edit an existing character: details and face
# --------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(folder: Path, line: str) -> None:
    """Append to the edit log in provenance.txt, for later legal and compliance review."""
    path = folder / "provenance.txt"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    header = "" if "edit_log:" in existing else ("\n" if existing else "") + "edit_log:\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{header}- {_now()}: {line}\n")


def _write_meta(folder: Path, meta: dict) -> None:
    temp = folder / "metadata.json.tmp"
    temp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(temp, folder / "metadata.json")


def update_meta(char_id: str, fields: dict) -> None:
    folder = _folder(char_id)
    meta = _meta(folder)
    old = {k: meta.get(k, "") for k in DETAIL_KEYS}
    new = dict(old)
    for key, allowed in (("size", SIZES), ("height", HEIGHTS), ("shape", SHAPES), ("skin", SKIN_TONES),
                         ("eye_color", EYE_COLORS), ("vibe", VIBES)):
        value = str(fields.get(key, "")).strip()
        if value not in allowed:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be one of {', '.join(allowed)}.")
        new[key] = value
    age = str(fields.get("age_range", "")).strip()
    if age not in ADULT_AGE_RANGES and age != old["age_range"]:
        raise ValueError(f"Age range must be one of {', '.join(ADULT_AGE_RANGES)}.")
    new["age_range"] = age
    hair_color = str(fields.get("hair_color", "")).strip()
    hairstyle = str(fields.get("hairstyle", "")).strip()
    if not hair_color or len(hair_color) > 40:
        raise ValueError("Enter a hair color (up to 40 characters).")
    if not hairstyle or len(hairstyle) > 200:
        raise ValueError("Enter a hairstyle (up to 200 characters).")
    new["hair_color"], new["hairstyle"] = hair_color, hairstyle
    new["name"] = str(fields.get("name", "")).strip()[:60]
    meta.update(new)
    meta.setdefault("face_id", char_id)
    meta.setdefault("approval", "pending")
    meta["body_description"] = body_description(new["size"], new["height"], new["shape"], meta.get("reference_body", ""))
    meta["updated_at"] = _now()
    _write_meta(folder, meta)
    changed = [k for k in DETAIL_KEYS if old[k] != new[k]]
    if changed:
        _log(folder, "details updated: " + "; ".join(f"{k}: {old[k] or '(empty)'} -> {new[k]}" for k in changed))


_FEATURE_RE = r"(face shape|forehead|eyebrows?|brows?|eyes?|nose|lips?|mouth|jawline|jaw|chin|cheekbones?|cheeks?)"
_ASSIGN_A = re.compile(_FEATURE_RE + r"(?:\s+(?:features?|shape|structure))?\s*(?:of|from|in|like|:|=)?\s*(?:the\s+)?reference\s*#?\s*(\d+)", re.I)
_ASSIGN_B = re.compile(r"reference\s*#?\s*(\d+)(?:'s|s')?\s+" + _FEATURE_RE, re.I)
_ALIAS = {
    "face shape": "face_shape", "forehead": "forehead", "eyebrow": "brows", "eyebrows": "brows", "brow": "brows",
    "brows": "brows", "eye": "eyes", "eyes": "eyes", "nose": "nose", "lip": "lips", "lips": "lips", "mouth": "lips",
    "jaw": "jaw_and_chin", "jawline": "jaw_and_chin", "chin": "jaw_and_chin", "cheekbone": "cheekbones",
    "cheekbones": "cheekbones", "cheek": "cheekbones", "cheeks": "cheekbones",
}


def parse_assignments(prompt: str) -> dict[str, int]:
    """Read 'eyes from reference 1, nose from reference 2' style instructions. {trait: reference number}."""
    found: dict[str, int] = {}
    for match in _ASSIGN_A.finditer(prompt):
        found[_ALIAS[match.group(1).lower()]] = int(match.group(2))
    for match in _ASSIGN_B.finditer(prompt):
        found.setdefault(_ALIAS[match.group(2).lower()], int(match.group(1)))
    return found


def _resolved_details(meta: dict) -> dict:
    """The latest saved details with safe fallbacks for older characters."""
    size = meta.get("size") if meta.get("size") in SIZES else "S-M"
    height = meta.get("height") if meta.get("height") in HEIGHTS else "average"
    shape = meta.get("shape") if meta.get("shape") in SHAPES else "straight"
    return {
        "age_range": meta.get("age_range") or "25-40",
        "hair_color": meta.get("hair_color") or "black",
        "hairstyle": meta.get("hairstyle") or "long, loose, professionally styled",
        "size": size,
        "skin": meta.get("skin") if meta.get("skin") in SKIN_TONES else "fair",
        "eye_color": meta.get("eye_color") if meta.get("eye_color") in EYE_COLORS else "dark brown",
        "vibe": meta.get("vibe") if meta.get("vibe") in VIBES else "confident",
        "body": meta.get("body_description") or body_description(size, height, shape, ""),
    }


def edit_face(char_id: str, prompt: str, sources: list[Path] | None = None, mode: str = "rebuild") -> Path:
    """Make a candidate edit of the face. The face itself changes only on accept.

    mode "rebuild": a new photograph written from the latest saved details, her current
        features, the reference feature notes and your direction. Changes always show.
    mode "tweak": the current photo is edited. She stays recognisable, changes are smaller.

    Optional reference photos (3 to 6) are read by the trait model into written feature
    notes. No reference photo is sent to the image model.
    """
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Describe what to change about her face.")
    if len(prompt) > 1000:
        raise ValueError("Keep the prompt under 1000 characters.")
    if mode not in ("rebuild", "tweak"):
        raise ValueError("Edit mode must be rebuild or tweak.")
    sources = sources or []
    if sources and not 3 <= len(sources) <= 6:
        raise ValueError("Use 3 to 6 reference photos, or none.")
    folder = _folder(char_id)
    meta = _meta(folder)
    details = _resolved_details(meta)
    cache = folder / ".cache"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = random.SystemRandom()

    readable: list[tuple[int, dict[str, str]]] = []
    refs_log = ""
    if sources:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("Set OPENROUTER_API_KEY in .env or the environment.")
        refs_dir = folder / "face_edit_refs" / stamp
        refs_dir.mkdir(parents=True, exist_ok=True)
        kept = []
        for index, source in enumerate(sources, start=1):
            destination = refs_dir / _safe_name(source, index)
            shutil.copy2(source, destination)
            kept.append(destination)
        described = [(index, describe_traits(path)) for index, path in enumerate(kept, start=1)]
        readable = [(index, traits) for index, traits in described if traits]
        if len(readable) < 2:
            raise RuntimeError("Could not read facial traits from at least two reference photos. Try clearer photos.")
        hashes = "; ".join(f"{path.name} sha256 {file_sha256(path)}" for path in kept)
        refs_log = f"face_edit_refs/{stamp}/ ({hashes})"

    # exact feature assignments from the prompt, e.g. "eyes from reference 1"
    by_ref = dict(readable)
    assignments = {k: n for k, n in parse_assignments(prompt).items() if n in by_ref and k in by_ref[n]}
    assigned = {k: by_ref[n][k] for k, n in assignments.items()}
    assigned_text = "; ".join(f"{k.replace('_', ' ')}: {v} (from reference {assignments[k]})" for k, v in assigned.items())

    current = {k: v for k, v in (meta.get("traits") or {}).items() if k in TRAIT_KEYS}
    notes = ""
    if readable:
        notes = "Reference feature notes (written descriptions only, no photo is attached):\n" + "\n".join(
            f"Reference {i}: " + "; ".join(f"{k.replace('_', ' ')}: {t[k]}" for k in TRAIT_KEYS if k in t)
            for i, t in readable
        )

    realism = (
        "Real skin texture with visible pores, no airbrushing, no beauty filter. Very light professional makeup. "
        "Professionally styled fashion-model hair with individual strands. A confident, lively, engaged expression. "
        "She must clearly be an adult woman."
    )
    target = (
        f"- apparent age: {details['age_range']}\n- skin: {SKIN_TONES[details['skin']]}\n"
        f"- eye colour: {details['eye_color']}\n- hair: {details['hair_color']} hair, {details['hairstyle']}\n"
        f"- body: {details['body']}"
    )

    if mode == "rebuild":
        if readable and not assigned:
            traits, _ = blend_traits(readable, rng)        # no explicit mapping: blend the references
        else:
            traits = dict(current)
            for key in TRAIT_KEYS:
                traits.setdefault(key, rng.choice(VARIATIONS[key]))
        traits.update(assigned)                            # explicit mapping always wins
        direction = prompt + (f" Use exactly these feature descriptions: {assigned_text}." if assigned_text else "")
        text = face_prompt(
            details["age_range"], details["hair_color"], details["hairstyle"], details["size"], details["skin"],
            traits, rng.sample(DISTINCTIVE, 2), eye_color=details["eye_color"], vibe=details["vibe"],
            body=details["body"], direction=direction,
        )
        content = [{"type": "text", "text": text}]
    else:
        text = f"""
Image 1 is the current casting photograph. Edit it so the woman matches the target details and the director's request below. The changes must be clearly visible: do not return an unchanged copy. Keep the framing, white athletic vest top, plain mid-grey backdrop and lighting. Keep her recognisable as the same model except where the request or the target details change her.

Target details (the latest saved details; they win over what image 1 shows):
{target}

Director's request: {prompt}
{("Use exactly these feature descriptions: " + assigned_text) if assigned_text else ""}
{notes}

{realism} One person, one frame. No text, logo, or watermark.
""".strip()
        content = [
            {"type": "text", "text": "Image 1 is the current face photograph of the character."},
            image_part(_reference_jpeg(folder / "face.png", cache / "face_ref.jpg")),
            {"type": "text", "text": text},
        ]
    data = _post_image(content, ASPECT_RATIO, IMAGE_SIZE)
    edits = folder / "face_edits"
    edits.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "edit"
    destination = edits / f"{stamp}_{slug}.png"
    destination.write_bytes(data)
    with (edits / "prompts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"file": destination.name, "prompt": prompt, "mode": mode,
                             "assigned": assigned_text, "references": refs_log, "created_at": _now()}) + "\n")
    _log(folder, (
        f"face edit candidate {destination.name} made in {mode} mode"
        + (f" with {len(sources)} reference photos kept in {refs_log}. Only written feature notes went to the image model" if refs_log else "")
        + (f". Feature assignments: {assigned_text}" if assigned_text else "")
        + f". Prompt: {prompt}"
    ))
    return destination


def _candidate(folder: Path, file: str) -> Path:
    candidate = folder / "face_edits" / Path(file).name
    if not candidate.is_file() or candidate.suffix.lower() != ".png":
        raise ValueError("That edited face no longer exists.")
    return candidate


def accept_face(char_id: str, file: str) -> None:
    folder = _folder(char_id)
    candidate = _candidate(folder, file)
    history = folder / "face_history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(folder / "face.png", history / f"{stamp}_previous.png")
    shutil.copy2(candidate, folder / "face.png")
    (folder / ".cache" / "face_ref.jpg").unlink(missing_ok=True)
    prompt = ""
    log_file = folder / "face_edits" / "prompts.jsonl"
    if log_file.is_file():
        for line in log_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("file") == candidate.name:
                prompt = row.get("prompt", "")
    candidate.unlink()
    meta = _meta(folder)
    meta.setdefault("face_id", char_id)
    meta.setdefault("approval", "pending")
    meta.setdefault("face_edits", []).append({"at": _now(), "prompt": prompt})
    meta["approval"] = "pending"
    meta["updated_at"] = _now()
    _write_meta(folder, meta)
    _log(folder, f"face replaced by an AI edit (previous face kept in face_history/). Edit prompt: {prompt}. Approval reset to pending.")


def discard_face(char_id: str, file: str) -> None:
    _candidate(_folder(char_id), file).unlink()


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def _options(values, selected: str = "") -> str:
    return "\n".join(
        f'<option value="{v}"{" selected" if v == selected else ""}>{v}</option>' for v in values
    )


def _page() -> str:
    views = json.dumps([{"id": p["id"], "label": p["label"]} for p in POSES])
    return (
        PAGE_TEMPLATE
        .replace("<!-- ages -->", _options(ADULT_AGE_RANGES, "25-40"))
        .replace("<!-- sizes -->", _options(SIZES, "S-M"))
        .replace("<!-- heights -->", _options(HEIGHTS, "average"))
        .replace("<!-- shapes -->", _options(SHAPES, "straight"))
        .replace("<!-- skins -->", _options(SKIN_TONES, "fair"))
        .replace("<!-- eyes -->", _options(EYE_COLORS, "dark brown"))
        .replace("<!-- hair_colors -->", _options(HAIR_COLORS, "black"))
        .replace("<!-- vibes -->", _options(VIBES, "confident"))
        .replace("<!-- aspects -->", _options(CUSTOM_ASPECTS, "3:4"))
        .replace("__VIEWS__", views)
        .replace("__OUTFIT__", DEFAULT_OUTFIT)
    )


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Silk Studio: Character studio</title>
  <style>
    body { margin: 0; background: #f7f6f4; color: #1a1a1a; font-family: system-ui, sans-serif; }
    main { max-width: 860px; margin: 0 auto; padding: 40px 20px 80px; }
    h1 { font-size: 1.7rem; font-weight: 560; margin: 0 0 8px; }
    h2 { font-size: 1.1rem; font-weight: 560; margin: 30px 0 6px; }
    .lead { color: #4d4a46; line-height: 1.5; margin: 0 0 22px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 18px; }
    .tabs button { border: 1px solid #d6d3ce; background: #fff; border-radius: 999px; padding: 8px 16px; font: inherit; cursor: pointer; }
    .tabs button.on { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
    section[hidden], [hidden] { display: none !important; }
    label { display: block; font-size: 0.92rem; margin: 16px 0 6px; }
    input[type="text"], select, textarea {
      width: 100%; box-sizing: border-box; padding: 10px 12px;
      border: 1px solid #d6d3ce; border-radius: 8px; background: #fff; font: inherit;
    }
    textarea { min-height: 80px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .drop { border: 1px dashed #b9b4ad; border-radius: 12px; padding: 20px; background: #fff; cursor: pointer; }
    .drop strong { display: block; margin-bottom: 4px; }
    .drop span { color: #6b6762; font-size: 0.9rem; }
    .previews { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .shot { position: relative; }
    .shot img { width: 84px; height: 84px; object-fit: cover; border-radius: 8px; display: block; }
    .shot span { position: absolute; left: 4px; bottom: 4px; background: #1a1a1a; color: #fff; border-radius: 999px; font-size: 0.68rem; padding: 1px 7px; }
    .shot button { position: absolute; top: 4px; right: 4px; width: 22px; height: 22px; padding: 0; border: 0; border-radius: 999px; background: #1a1a1a; color: #fff; cursor: pointer; }
    .choice { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 0.95rem; }
    .choice input { margin: 0; }
    .views { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 14px; margin-top: 8px; }
    button.primary { margin-top: 20px; background: #1a1a1a; color: #fff; border: 0; border-radius: 999px; padding: 12px 22px; font: inherit; cursor: pointer; }
    button.primary:disabled { opacity: 0.45; cursor: default; }
    .status { min-height: 1.4em; margin-top: 14px; color: #6b3a2a; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }
    figure { margin: 0; background: #fff; border-radius: 12px; padding: 8px; }
    figure img { width: 100%; border-radius: 8px; display: block; }
    figcaption { font-size: 0.78rem; margin-top: 6px; color: #4d4a46; }
    .acts { display: flex; gap: 6px; margin-top: 6px; }
    .acts button { flex: 1; border: 1px solid #d6d3ce; background: #fff; border-radius: 8px; padding: 6px; font: inherit; font-size: 0.78rem; cursor: pointer; }
    input[type="file"] { display: none; }
    @media (max-width: 800px) { .grid { grid-template-columns: repeat(2, 1fr); } .row, .views { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>Character studio</h1>
  <p class="lead">Build a fashion model as a complete character: a casting face, then the full character sheet, then any new image you want of her.</p>
  <div class="tabs">
    <button type="button" id="tabNew" class="on">New character</button>
    <button type="button" id="tabExisting">Existing character</button>
  </div>

  <section id="paneNew">
    <form id="newForm">
      <div class="drop" id="drop">
        <strong>Reference photos</strong>
        <span>Choose 3 to 8 photos, ideally of different women. Their general features and body type are blended into a new person, not a copy of anyone.</span>
      </div>
      <input id="files" type="file" accept="image/jpeg,image/png,image/webp,image/gif" multiple>
      <div class="previews" id="previews"></div>
      <label for="note">Source note</label>
      <textarea id="note" required placeholder="Where these photos came from, for the provenance record"></textarea>
      <label for="name">Character name (optional)</label>
      <input id="name" type="text" placeholder="e.g. Zara">
      <div class="row">
        <div><label for="age">Age range</label><select id="age"><!-- ages --></select></div>
        <div><label for="size">Size</label><select id="size"><!-- sizes --></select></div>
        <div><label for="height">Height</label><select id="height"><!-- heights --></select></div>
        <div><label for="shape">Silhouette</label><select id="shape"><!-- shapes --></select></div>
        <div><label for="skin">Skin tone</label><select id="skin"><!-- skins --></select></div>
        <div><label for="eyes">Eye color</label><select id="eyes"><!-- eyes --></select></div>
        <div><label for="haircolor">Hair color</label><select id="haircolor"><!-- hair_colors --></select></div>
        <div><label for="vibe">Presence</label><select id="vibe"><!-- vibes --></select></div>
      </div>
      <label for="hair">Hairstyle</label>
      <input id="hair" type="text" required placeholder="Long, sleek and blow-dried, side part, past the shoulders">
      <label>What to generate</label>
      <div class="choice"><input type="radio" name="output" id="outFace" value="face" checked><label for="outFace" style="margin:0">Face only</label></div>
      <div class="choice"><input type="radio" name="output" id="outSheet" value="sheet"><label for="outSheet" style="margin:0">Face and full character sheet (8 views)</label></div>
      <label for="outfit">Base outfit for the sheet (fitted wear shows body type and size; to change an existing sheet, regenerate the front view first, then the others)</label>
      <input id="outfit" type="text" value="__OUTFIT__">
      <button class="primary" id="goNew" type="submit">Create character</button>
    </form>
  </section>

  <section id="paneExisting" hidden>
    <label for="charSelect">Character</label>
    <select id="charSelect"></select>

    <h2>Details</h2>
    <p class="lead" style="margin-bottom:0">Saved to this character's metadata. Face edits and new sheet views follow these details. Existing images do not change until you edit or regenerate them.</p>
    <label for="e_name">Character name</label>
    <input id="e_name" type="text">
    <div class="row">
      <div><label for="e_age">Age range</label><select id="e_age"><!-- ages --></select></div>
      <div><label for="e_size">Size</label><select id="e_size"><!-- sizes --></select></div>
      <div><label for="e_height">Height</label><select id="e_height"><!-- heights --></select></div>
      <div><label for="e_shape">Silhouette</label><select id="e_shape"><!-- shapes --></select></div>
      <div><label for="e_skin">Skin tone</label><select id="e_skin"><!-- skins --></select></div>
      <div><label for="e_eyes">Eye color</label><select id="e_eyes"><!-- eyes --></select></div>
      <div><label for="e_haircolor">Hair color</label><select id="e_haircolor"><!-- hair_colors --></select></div>
      <div><label for="e_vibe">Presence</label><select id="e_vibe"><!-- vibes --></select></div>
    </div>
    <label for="e_hair">Hairstyle</label>
    <input id="e_hair" type="text">
    <button class="primary" id="goMeta" type="button">Save details</button>

    <h2>Edit her face with a prompt</h2>
    <p class="lead" style="margin-bottom:0">Describe what to change. The edit uses her latest saved details above (skin, eye color, hair, age, size). You get a candidate to review. Her face changes only if you choose "Use as face", and the old one is kept.</p>
    <label for="editMode">Edit mode</label>
    <select id="editMode">
      <option value="rebuild">Rebuild from the saved details (strong change, a new look)</option>
      <option value="tweak">Tweak the current photo (keeps her recognisable, smaller changes)</option>
    </select>
    <textarea id="faceEdit" placeholder="e.g. Make the hair auburn brown with a soft side part, add a few light freckles across the nose"></textarea>
    <div class="drop" id="editDrop" style="margin-top:12px">
      <strong>Reference photos (optional)</strong>
      <span>3 to 6 photos of different women. Only their general facial structure is read. Say in the prompt what to borrow, for example "take the eye shape from reference 2".</span>
    </div>
    <input id="editFiles" type="file" accept="image/jpeg,image/png,image/webp,image/gif" multiple>
    <div class="previews" id="editPreviews"></div>
    <button class="primary" id="goFaceEdit" type="button">Generate edited face</button>

    <h2>Character sheet</h2>
    <div class="views" id="viewChecks"></div>
    <button type="button" id="tickAll" style="margin-top:8px;border:1px solid #d6d3ce;background:#fff;border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer">Tick all views (regenerate with the latest details)</button>
    <button class="primary" id="goViews" type="button">Generate selected views</button>

    <h2>New image from a prompt</h2>
    <textarea id="customPrompt" placeholder="e.g. Full-length photo walking toward the camera in an emerald silk kurta, soft window light"></textarea>
    <div class="row">
      <div><label for="aspect">Aspect ratio</label><select id="aspect"><!-- aspects --></select></div>
    </div>
    <button class="primary" id="goCustom" type="button">Generate image</button>
  </section>

  <p class="status" id="status"></p>
  <div class="grid" id="gallery"></div>
</main>
<script>
  const VIEWS = __VIEWS__;
  const $ = (id) => document.getElementById(id);
  let characters = [];
  let chosen = [];

  function setTab(which) {
    $("paneNew").hidden = which !== "new";
    $("paneExisting").hidden = which !== "existing";
    $("tabNew").classList.toggle("on", which === "new");
    $("tabExisting").classList.toggle("on", which === "existing");
    renderGallery();
  }
  $("tabNew").addEventListener("click", () => setTab("new"));
  $("tabExisting").addEventListener("click", () => setTab("existing"));

  async function post(path, body) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Request failed.");
    return payload;
  }

  function current() { return characters.find((c) => c.id === $("charSelect").value) || null; }

  let filledFor = null;
  function setField(id, value) {
    const el = $(id);
    if (!value) return;
    if (el.tagName === "SELECT" && !Array.from(el.options).some((o) => o.value === value)) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      el.append(option);
    }
    el.value = value;
  }
  function fillDetails() {
    const item = current();
    const key = item ? item.id : null;
    if (key === filledFor) return;
    filledFor = key;
    const m = item ? item.meta : {};
    setField("e_name", m.name || "");
    $("e_name").value = m.name || "";
    setField("e_age", m.age_range);
    setField("e_size", m.size);
    setField("e_height", m.height);
    setField("e_shape", m.shape);
    setField("e_skin", m.skin);
    setField("e_eyes", m.eye_color);
    setField("e_haircolor", m.hair_color);
    setField("e_vibe", m.vibe);
    $("e_hair").value = m.hairstyle || "";
  }

  async function loadCharacters(selectId) {
    const response = await fetch("/api/characters");
    characters = (await response.json()).characters;
    const select = $("charSelect");
    const keep = selectId || select.value;
    select.innerHTML = "";
    for (const item of characters) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.label;
      select.append(option);
    }
    if (characters.some((c) => c.id === keep)) select.value = keep;
    fillDetails();
    renderViewChecks();
    renderGallery();
  }
  $("charSelect").addEventListener("change", () => { fillDetails(); renderViewChecks(); renderGallery(); });

  function renderViewChecks() {
    const box = $("viewChecks");
    box.innerHTML = "";
    const item = current();
    for (const view of VIEWS) {
      const done = !!(item && item.sheet[view.id]);
      const row = document.createElement("div");
      row.className = "choice";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = view.id;
      input.id = "v_" + view.id;
      input.checked = !done;
      const label = document.createElement("label");
      label.htmlFor = input.id;
      label.style.margin = "0";
      label.textContent = view.label + (done ? " (done)" : "");
      row.append(input, label);
      box.append(row);
    }
  }

  function renderGallery() {
    const grid = $("gallery");
    grid.innerHTML = "";
    const item = current();
    if (!item) return;
    const add = (src, caption, actions) => {
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = src;
      img.alt = caption;
      img.loading = "lazy";
      const cap = document.createElement("figcaption");
      cap.textContent = caption;
      figure.append(img, cap);
      if (actions) {
        const row = document.createElement("div");
        row.className = "acts";
        for (const [text, handler] of actions) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = text;
          button.addEventListener("click", handler);
          row.append(button);
        }
        figure.append(row);
      }
      grid.append(figure);
    };
    add(item.face, "Face");
    for (const candidate of item.candidates) {
      add(candidate.url, "Edited face (not saved yet)", [
        ["Use as face", () => faceChoice("/api/character/face-accept", item.id, candidate.file,
          "Face replaced. Views made earlier still show the old face: tick them and regenerate to match.")],
        ["Discard", () => faceChoice("/api/character/face-discard", item.id, candidate.file, "Edited face discarded.")],
      ]);
    }
    for (const view of VIEWS) if (item.sheet[view.id]) add(item.sheet[view.id], view.label);
    item.extras.forEach((src, index) => add(src, "Custom image " + (item.extras.length - index)));
  }

  async function faceChoice(path, id, file, message) {
    busy(true);
    try {
      await post(path, { id, file });
      await loadCharacters(id);
      $("status").textContent = message;
    } catch (error) {
      $("status").textContent = error.message || "Something went wrong.";
    } finally {
      busy(false);
    }
  }

  function busy(on) { for (const id of ["goNew", "goViews", "goCustom", "goMeta", "goFaceEdit"]) $(id).disabled = on; }

  async function runViews(id, ids, outfit) {
    for (const viewId of ids) {
      const label = VIEWS.find((v) => v.id === viewId).label;
      $("status").textContent = "Generating: " + label + ". This can take a minute.";
      await post("/api/character/view", { id, view: viewId, outfit });
      await loadCharacters(id);
    }
  }

  // ---- new character
  $("drop").addEventListener("click", () => $("files").click());
  $("files").addEventListener("change", () => {
    const room = 8 - chosen.length;
    for (const file of Array.from($("files").files).slice(0, room)) chosen.push({ file, url: URL.createObjectURL(file) });
    $("files").value = "";
    drawPreviews();
  });
  function drawPreviews() {
    const box = $("previews");
    box.innerHTML = "";
    chosen.forEach((item, index) => {
      const shot = document.createElement("div");
      shot.className = "shot";
      const img = document.createElement("img");
      img.src = item.url;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "x";
      remove.addEventListener("click", (event) => { event.stopPropagation(); URL.revokeObjectURL(item.url); chosen.splice(index, 1); drawPreviews(); });
      shot.append(img, remove);
      box.append(shot);
    });
  }

  $("newForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const status = $("status");
    if (chosen.length < 3) { status.textContent = "Add at least three photos, ideally of different women."; return; }
    busy(true);
    try {
      const session = crypto.randomUUID();
      status.textContent = "Uploading photos...";
      for (const item of chosen) {
        const response = await fetch("/api/upload", { method: "POST", headers: { "X-Session": session, "X-Filename": encodeURIComponent(item.file.name) }, body: item.file });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Upload failed.");
      }
      status.textContent = "Creating the face. This can take a few minutes.";
      const made = await post("/api/character/face", {
        session, note: $("note").value.trim(), name: $("name").value.trim(), age: $("age").value, size: $("size").value,
        height: $("height").value, shape: $("shape").value, skin: $("skin").value, eyecolor: $("eyes").value,
        haircolor: $("haircolor").value, vibe: $("vibe").value, hair: $("hair").value.trim(),
      });
      await loadCharacters(made.id);
      setTab("existing");
      if (document.querySelector('input[name="output"]:checked').value === "sheet") {
        await runViews(made.id, VIEWS.map((v) => v.id), $("outfit").value);
        status.textContent = "Character saved with the full sheet. Compare her with the references before approving.";
      } else {
        status.textContent = "Face saved. Compare it with the references before approving. Open it under Existing character to make the sheet.";
      }
    } catch (error) {
      status.textContent = error.message || "Something went wrong.";
    } finally {
      busy(false);
    }
  });

  // ---- existing character
  $("tickAll").addEventListener("click", () => {
    document.querySelectorAll("#viewChecks input").forEach((box) => { box.checked = true; });
  });

  $("goViews").addEventListener("click", async () => {
    const item = current();
    if (!item) { $("status").textContent = "Choose a character first."; return; }
    const picked = new Set(Array.from(document.querySelectorAll("#viewChecks input:checked")).map((i) => i.value));
    if (!picked.size) { $("status").textContent = "Tick at least one view."; return; }
    if (!item.sheet[VIEWS[0].id]) picked.add(VIEWS[0].id);
    busy(true);
    try {
      await runViews(item.id, VIEWS.map((v) => v.id).filter((id) => picked.has(id)), $("outfit").value);
      $("status").textContent = "Done.";
    } catch (error) {
      $("status").textContent = error.message || "Something went wrong.";
    } finally {
      busy(false);
    }
  });

  $("goCustom").addEventListener("click", async () => {
    const item = current();
    if (!item) { $("status").textContent = "Choose a character first."; return; }
    const prompt = $("customPrompt").value.trim();
    if (!prompt) { $("status").textContent = "Describe the image you want."; return; }
    busy(true);
    $("status").textContent = "Generating the image. This can take a minute.";
    try {
      await post("/api/character/custom", { id: item.id, prompt, aspect: $("aspect").value });
      await loadCharacters(item.id);
      $("status").textContent = "Saved to this character.";
    } catch (error) {
      $("status").textContent = error.message || "Something went wrong.";
    } finally {
      busy(false);
    }
  });

  $("goMeta").addEventListener("click", async () => {
    const item = current();
    if (!item) { $("status").textContent = "Choose a character first."; return; }
    busy(true);
    try {
      await post("/api/character/meta", {
        id: item.id, name: $("e_name").value.trim(), age_range: $("e_age").value, size: $("e_size").value,
        height: $("e_height").value, shape: $("e_shape").value, skin: $("e_skin").value, eye_color: $("e_eyes").value,
        hair_color: $("e_haircolor").value, hairstyle: $("e_hair").value.trim(), vibe: $("e_vibe").value,
      });
      filledFor = null;
      await loadCharacters(item.id);
      $("status").textContent = "Details saved. Images are unchanged.";
    } catch (error) {
      $("status").textContent = error.message || "Something went wrong.";
    } finally {
      busy(false);
    }
  });

  // ---- reference photos for face edits
  let editChosen = [];
  $("editDrop").addEventListener("click", () => $("editFiles").click());
  $("editFiles").addEventListener("change", () => {
    const room = 6 - editChosen.length;
    for (const file of Array.from($("editFiles").files).slice(0, room)) editChosen.push({ file, url: URL.createObjectURL(file) });
    $("editFiles").value = "";
    drawEditPreviews();
  });
  function drawEditPreviews() {
    const box = $("editPreviews");
    box.innerHTML = "";
    editChosen.forEach((entry, index) => {
      const shot = document.createElement("div");
      shot.className = "shot";
      const img = document.createElement("img");
      img.src = entry.url;
      const number = document.createElement("span");
      number.textContent = "Reference " + (index + 1);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "x";
      remove.addEventListener("click", () => { URL.revokeObjectURL(entry.url); editChosen.splice(index, 1); drawEditPreviews(); });
      shot.append(img, number, remove);
      box.append(shot);
    });
  }

  $("goFaceEdit").addEventListener("click", async () => {
    const item = current();
    if (!item) { $("status").textContent = "Choose a character first."; return; }
    const prompt = $("faceEdit").value.trim();
    if (!prompt) { $("status").textContent = "Describe what to change about her face."; return; }
    if (editChosen.length && editChosen.length < 3) {
      $("status").textContent = "Add 3 to 6 reference photos, or remove them all.";
      return;
    }
    busy(true);
    try {
      let session = "";
      if (editChosen.length) {
        session = crypto.randomUUID();
        $("status").textContent = "Uploading reference photos...";
        for (const entry of editChosen) {
          const response = await fetch("/api/upload", { method: "POST", headers: { "X-Session": session, "X-Filename": encodeURIComponent(entry.file.name) }, body: entry.file });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.error || "Upload failed.");
        }
      }
      $("status").textContent = "Generating the edited face. This can take a few minutes.";
      await post("/api/character/face-edit", { id: item.id, prompt, session, mode: $("editMode").value });
      editChosen.forEach((entry) => URL.revokeObjectURL(entry.url));
      editChosen = [];
      drawEditPreviews();
      await loadCharacters(item.id);
      $("status").textContent = "Edited face ready below. Use it as her face, or discard it.";
    } catch (error) {
      $("status").textContent = error.message || "Something went wrong.";
    } finally {
      busy(false);
    }
  });

  loadCharacters();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


class StudioHandler(FaceHandler):
    server_version = "SilkStudioCharacters"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(200, _page().encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/characters":
            self._send_json(200, {"characters": list_characters()})
        elif path.startswith("/outputs/"):
            self._send_output(path[len("/outputs/"):])
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        routes = {
            "/api/upload": self._upload,
            "/api/character/face": self._character_face,
            "/api/character/view": self._character_view,
            "/api/character/custom": self._character_custom,
            "/api/character/meta": self._character_meta,
            "/api/character/face-edit": self._character_face_edit,
            "/api/character/face-accept": self._character_face_accept,
            "/api/character/face-discard": self._character_face_discard,
        }
        handler = routes.get(path)
        if handler is None:
            self._send_json(404, {"error": "Not found."})
            return
        handler()

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            return data if isinstance(data, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def _guarded(self, work) -> None:
        if not _work_lock.acquire(blocking=False):
            self._send_json(409, {"error": "Another image is being generated. Wait for it to finish."})
            return
        try:
            payload = work()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        finally:
            _work_lock.release()
        self._send_json(200, payload)

    def _character_face(self) -> None:
        body = self._body()

        def work() -> dict:
            session = _safe_session(str(body.get("session", "")))
            with _sessions_lock_ref:
                sources = list(_sessions_ref.get(session, []))
            image_path, notes_path, summary = create_face(
                sources,
                str(body.get("note", "")),
                str(body.get("age", "")),
                str(body.get("haircolor", "")),
                str(body.get("hair", "")),
                str(body.get("size", "")),
                str(body.get("skin", "fair")),
                name=str(body.get("name", "")),
                height=str(body.get("height", "average")),
                shape=str(body.get("shape", "straight")),
                eye_color=str(body.get("eyecolor", "dark brown")),
                vibe=str(body.get("vibe", "confident")),
            )
            self._clear_session(session)
            return {"id": image_path.parent.name, "image": _outputs_url(image_path),
                    "notes": _outputs_url(notes_path), "summary": summary}

        self._guarded(work)

    def _character_view(self) -> None:
        body = self._body()
        self._guarded(lambda: {
            "image": _outputs_url(render_view(str(body.get("id", "")), str(body.get("view", "")), str(body.get("outfit", ""))))
        })

    def _simple(self, work) -> None:
        """Quick actions that never call the image model, so they skip the generation lock."""
        try:
            work()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True})

    def _character_meta(self) -> None:
        body = self._body()
        self._simple(lambda: update_meta(str(body.get("id", "")), body))

    def _character_face_edit(self) -> None:
        body = self._body()

        def work() -> dict:
            session = str(body.get("session", "")).strip()
            sources: list[Path] = []
            if session:
                session = _safe_session(session)
                with _sessions_lock_ref:
                    sources = list(_sessions_ref.get(session, []))
            image = edit_face(str(body.get("id", "")), str(body.get("prompt", "")), sources,
                              str(body.get("mode", "rebuild")))
            if session:
                self._clear_session(session)
            return {"image": _outputs_url(image)}

        self._guarded(work)

    def _character_face_accept(self) -> None:
        body = self._body()
        self._simple(lambda: accept_face(str(body.get("id", "")), str(body.get("file", ""))))

    def _character_face_discard(self) -> None:
        body = self._body()
        self._simple(lambda: discard_face(str(body.get("id", "")), str(body.get("file", ""))))

    def _character_custom(self) -> None:
        body = self._body()
        self._guarded(lambda: {
            "image": _outputs_url(render_custom(str(body.get("id", "")), str(body.get("prompt", "")), str(body.get("aspect", ""))))
        })


# the upload endpoint inherited from FaceHandler stores sessions in generate_face
import generate_face as _gf  # noqa: E402

_sessions_ref = _gf._sessions
_sessions_lock_ref = _gf._sessions_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the Silk Studio character studio.")
    parser.add_argument("--no-browser", action="store_true", help="Start the page without opening a browser window")
    args = parser.parse_args()
    server = None
    for port in range(8790, 8800):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port between 8790 and 8799.")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Character studio at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()