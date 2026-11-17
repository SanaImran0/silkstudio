"""
Generate one new adult face for the Silk Studio model library.

Reference photos are used to learn general facial STRUCTURE (face shape, eyes,
nose, lips, jaw, ...). A vision model writes those traits as short text phrases for
each photo. The traits are then blended across the photos, with deliberate
variation, into one new feature list. The image model receives only that text. No
reference photo is ever sent to the image model.

Use 3 or more DIFFERENT people as references. One person's photos only teach the
model that one person's face.

Run from the repository root. This opens a page where you upload the photos and
choose age range, size, skin tone, hair color, and hairstyle:

    python backend/generate_face.py

You can also run one face from the terminal:

    python backend/generate_face.py a.jpg b.jpg c.jpg \\
        --note "public-figure set A" --age 25-40 --size S-M --skin fair \\
        --hair-color "dark brown" --hairstyle "long, soft waves, side part"

Image model: google/gemini-3.1-flash-image-preview through OpenRouter.
Trait reader: FACE_ANALYSIS_MODEL (default google/gemini-2.5-flash).
"""

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import shutil
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.1-flash-image-preview"
ASPECT_RATIO = "5:4"
IMAGE_SIZE = "4K"
ADULT_AGE_RANGES = ("18-24", "25-40")
ANALYSIS_MODEL = os.environ.get("FACE_ANALYSIS_MODEL", "google/gemini-2.5-flash")
SKIN_TONES = {
    "fair": "fair skin with a warm neutral to faintly golden undertone, the way a fair Pakistani complexion looks in real life",
    "light wheatish": "light wheatish skin with a warm golden undertone, a light Pakistani complexion",
    "wheatish": "wheatish skin with a warm golden-olive undertone, a medium Pakistani complexion",
}
EYE_COLORS = ("dark brown", "brown", "light brown", "hazel")
HEIGHTS = {
    "petite": "petite, about 5'2\" to 5'4\"",
    "average": "average height, about 5'5\" to 5'7\"",
    "tall": "tall, about 5'8\" or taller",
}
SHAPES = {
    "straight": "a straight silhouette: shoulders and hips of similar width, a softly defined waist",
    "athletic": "an athletic silhouette: defined shoulders, a lean waist, toned arms",
    "pear": "a pear silhouette: narrower shoulders, fuller hips and thighs",
    "hourglass": "an hourglass silhouette: balanced shoulders and hips, a defined waist",
}
VIBES = {
    "confident": "Confident and poised. Chin level to slightly lifted, a direct, bright, self-assured gaze into the lens, lips relaxed with at most a faint hint of a smile at one corner. Eyes alive and engaged.",
    "warm": "Confident and warm. A soft, genuine half-smile with lips closed or barely parted, the eyes lightly crinkled, a direct friendly gaze.",
    "editorial": "Cool editorial intensity. Chin slightly lowered, a strong, steady, slightly intense gaze, lips relaxed and closed, no smile.",
    "fresh": "Fresh and bright. A light natural smile with lips together, bright open eyes, the head tilted a touch.",
}
BODY_KEYS = ("build", "shoulders", "leg_to_torso")
HAIR_COLORS = ("black", "dark brown", "brown", "chestnut brown", "auburn brown", "honey brown")
SIZES = {
    "XS-S": "slim build: narrow shoulders, a slim neck, visible collarbones, a slim face with light cheeks",
    "S-M": "average build: medium shoulder width, a natural neck, softly visible collarbones, a face of average fullness",
    "L-XL": "fuller build: broad shoulders, a softer neck and fuller upper arms, a fuller face with soft cheeks and a softer jawline",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MIN_REFERENCES = 3
MAX_REFERENCES = 8
MAX_FILE_BYTES = 50 * 1024 * 1024

_sessions: dict[str, list[Path]] = {}
_sessions_lock = threading.Lock()
_generate_lock = threading.Lock()


TRAIT_KEYS = (
    "face_shape", "forehead", "brows", "eyes", "nose", "lips", "jaw_and_chin", "cheekbones",
)

# Natural, non-idealised alternatives. A few traits of every new face are swapped for
# these, so the result drifts away from every reference instead of copying one.
VARIATIONS = {
    "face_shape": ("a soft oval face", "a slightly longer oval face", "a heart-shaped face with a narrower chin",
                   "a softly rounded oval face", "a gently squared oval face"),
    "forehead": ("a medium-height smooth forehead", "a slightly high forehead", "a lower, softly curved forehead"),
    "brows": ("naturally full, softly arched brows", "straight, slightly thick brows",
              "thinner, gently curved brows", "full brows with a flatter shape"),
    "eyes": ("almond eyes with a slight upward tilt", "large, rounder dark eyes", "softly hooded almond eyes",
             "deep-set eyes with a defined crease", "eyes set slightly wide apart"),
    "nose": ("a straight nose with a soft rounded tip", "a slim nose with a subtle bridge",
             "a medium nose with a gently defined tip", "a nose with a softly curved profile"),
    "lips": ("full lips with a soft cupid's bow", "a fuller lower lip", "medium lips with a clear outline",
             "a slightly wide mouth with soft corners"),
    "jaw_and_chin": ("a soft jaw with a rounded chin", "a slim jaw with a gently pointed chin",
                     "a defined jaw with a soft chin", "a clean oval jawline"),
    "cheekbones": ("softly defined cheekbones", "high, prominent cheekbones",
                   "gently rounded cheeks with soft cheekbones"),
}

# Small real-life irregularities. Two are picked per face so no two faces share a "finish".
DISTINCTIVE = (
    "one eyebrow very slightly higher than the other",
    "a faint natural beauty mark near the jaw or cheek",
    "lips that are slightly uneven, one corner a touch higher",
    "eyes that are very slightly different in size",
    "subtle pigment variation across the bridge of the nose",
    "ears set at very slightly different heights",
    "a hairline that is a little uneven at one temple",
    "a faint natural crease beside one eye",
)


def _jpeg_part(path: Path, max_side: int = 1024) -> dict:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def describe_traits(path: Path) -> dict[str, str]:
    """Ask a vision model for general structure phrases. Returns {} on failure.

    Facial keys are required. Body keys are included only when the photo shows the body.
    """
    instruction = (
        "A casting team is building a catalogue of generic facial-structure vocabulary. "
        "Look at the face in this photo and describe ONLY these visible structural traits, each as a "
        "short neutral phrase of 3 to 8 words: " + ", ".join(TRAIT_KEYS) + ". "
        "If the photo also shows her body from the shoulders down, add the keys "
        + ", ".join(BODY_KEYS) + " (leg_to_torso is the proportion of legs to torso), each a short neutral "
        "phrase. If the body is not visible, set those keys to \"not visible\". "
        "Describe shapes and proportions only. Do not identify the person or guess who they are. "
        "Do not mention fame, ethnicity, age, makeup, hair, skin colour, expression, or clothing. "
        "Reply with one JSON object using exactly those keys and nothing else."
    )
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": ANALYSIS_MODEL,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": [{"type": "text", "text": instruction}, _jpeg_part(path)]}],
            },
            timeout=120,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception as exc:  # a bad photo should not stop the run
        print(f"Could not read traits from {path.name}: {exc}")
        return {}
    traits = {}
    for key in TRAIT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and 3 <= len(value.strip()) <= 90:
            traits[key] = value.strip().rstrip(".")
    if len(traits) < 5:
        return {}
    for key in BODY_KEYS:
        value = data.get(key)
        if isinstance(value, str) and 3 <= len(value.strip()) <= 90 and "not visible" not in value.lower():
            traits[key] = value.strip().rstrip(".")
    return traits


def blend_body(described: list[tuple[int, dict[str, str]]], rng: random.Random) -> str:
    """Pick body traits from whichever photos show a body. Returns '' if none do."""
    parts = []
    for key in BODY_KEYS:
        options = [traits[key] for _, traits in described if key in traits]
        if options:
            parts.append(f"{key.replace('_', ' ')}: {rng.choice(options)}")
    return "; ".join(parts)


def body_description(size: str, height: str, shape: str, reference_body: str) -> str:
    text = f"{SIZES[size]}; height: {HEIGHTS[height]}; {SHAPES[shape]}"
    if reference_body:
        text += f". Loose influence from the references, never overriding the settings above: {reference_body}"
    return text


def blend_traits(described: list[tuple[int, dict[str, str]]], rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    """Mix traits from several photos, then swap 2-3 of them for natural variations.

    Returns (trait -> phrase, trait -> origin). Origins go in the provenance record.
    """
    valid = [(index, traits) for index, traits in described if traits]
    if not valid:
        raise RuntimeError(
            "Could not read facial traits from any photo. The trait model may have declined or failed. "
            "Try different photos, or set FACE_ANALYSIS_MODEL."
        )
    order: list[int] = []
    while len(order) < len(TRAIT_KEYS):
        order += rng.sample(range(len(valid)), len(valid))  # cycle so every photo contributes
    blended: dict[str, str] = {}
    origin: dict[str, str] = {}
    for key, pick in zip(TRAIT_KEYS, order):
        index, traits = valid[pick]
        blended[key] = traits[key] if key in traits else rng.choice(VARIATIONS[key])
        origin[key] = f"photo {index}" if key in traits else "variation"
    for key in rng.sample(TRAIT_KEYS, rng.choice((2, 3))):
        blended[key] = rng.choice(VARIATIONS[key])
        origin[key] = "variation"
    return blended, origin


def face_prompt(age_range: str, hair_color: str, hairstyle: str, size: str, skin: str,
                traits: dict[str, str], distinctive: list[str], *, eye_color: str = "dark brown",
                vibe: str = "confident", body: str = "", direction: str = "") -> str:
    feature_lines = "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in traits.items())
    direction_block = (
        f"\n\nDirection from the casting director. Follow it, and where it conflicts with the lines above it wins: {direction}"
        if direction else ""
    )
    return f"""
A real digital casting photograph of one adult Pakistani woman, taken by an agency photographer for a fashion model portfolio. Exactly one person, one frame. Not a grid, collage, or set of alternate looks. It must look like a photograph taken with a camera, not like AI art or a render.

No photograph of a person is attached. Do not draw any real or public figure. Build only the woman described here.

Her face, built from this feature list:
{feature_lines}
Eye colour: {eye_color}. Real irregularities, kept subtle: {"; ".join(distinctive)}. Her face is not perfectly symmetrical and not idealised. The features are attractive but real, the way a working fashion model looks. Apparent age {age_range}.

Expression and presence: {VIBES[vibe]} She looks like a professional fashion model who knows the camera. Never dull, sleepy, blank, or shy. No teeth, no frown.{direction_block}

Skin: {SKIN_TONES[skin]}. Healthy and even, with real life in it: a little more colour on the cheeks and nose, a touch cooler under the eyes. Visible pores on the cheeks, nose and forehead, fine texture, tiny vellus hairs catching the light along the jaw. No airbrushing, no skin smoothing, no beauty filter, no glassy or waxy highlight, no plastic look.

Makeup: very light, professional fashion-casting makeup that you notice but would not call heavy. A skin-like base that leaves pores visible, soft natural blush, groomed and defined brows, brown mascara and a soft smudge of liner at the lash line, a subtle highlight on the cheekbones, a tinted rose lip with a soft finish. Not bare and dull, not glam.

Hair: {hair_color} hair, {hairstyle}. Styled by a professional hairstylist for a fashion shoot: a clean shape, healthy volume at the roots, smooth movement, neat ends, and a natural healthy shine. Still real hair, with individual strands and only a few fine flyaways. Not a wig, not helmet-like, no AI-smooth clumps.

Body: {body or SIZES[size]}. She wears a plain white skin-tight athletic vest top, opaque and modest, with a clean round neckline and visible fabric texture. It fits snugly so her real build is visible, and she looks like a person who wears size {size}, neither slimmed nor idealised. No jewellery, no text.

Framing: head and shoulders in a 5:4 frame. Her face fills about half of the frame height, both shoulders are in frame, and there is soft background on both sides. Eye level, head turned very slightly from her shoulders, shoulders relaxed and down. Not a stiff passport pose.

Camera and light: 85mm lens at f/2.8, ISO 200, a large softbox slightly above and to one side with gentle fill, natural light falloff, a clear catchlight in each eye, sharp focus on the eyes. Plain mid-grey seamless backdrop. Natural colour, mild photographic grain, true-to-life contrast. No watermark, no logo, no extra people, no extra fingers, no distorted ears.
""".strip()


def image_part(path: Path) -> dict:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def collect_paths(given: list[str]) -> list[Path]:
    paths = [Path(item).expanduser() for item in given]
    if len(paths) < MIN_REFERENCES:
        raise SystemExit(f"Provide at least {MIN_REFERENCES} reference photos, ideally of different women.")
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


def _mean_color(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    pixels = [pixel for pixel in image.crop(box).getdata() if 25 < sum(pixel) / 3 < 235]
    if len(pixels) < 20:
        pixels = list(image.crop(box).getdata())
    count = len(pixels)
    return tuple(sum(pixel[channel] for pixel in pixels) // count for channel in range(3))


def color_swatches(sources: list[Path], folder: Path) -> list[Path]:
    """Return flat skin and hair swatches. The reference faces are not included."""
    skin_samples = []
    hair_samples = []
    for path in sources:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        width, height = image.size
        skin_samples.append(_mean_color(image, (
            int(width * 0.35), int(height * 0.38), int(width * 0.65), int(height * 0.58),
        )))
        hair_samples.append(_mean_color(image, (
            int(width * 0.30), int(height * 0.02), int(width * 0.70), int(height * 0.16),
        )))
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, samples in (("skin_swatch.png", skin_samples), ("hair_swatch.png", hair_samples)):
        count = len(samples)
        color = tuple(sum(sample[channel] for sample in samples) // count for channel in range(3))
        destination = folder / name
        Image.new("RGB", (64, 64), color).save(destination)
        paths.append(destination)
    return paths


def generate(prompt: str, destination: Path) -> None:
    """Text-only request. No reference photo goes to the image model."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or the environment.")
    last = ""
    for attempt in range(1, 4):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    "modalities": ["image", "text"],
                    "image_config": {"aspect_ratio": ASPECT_RATIO, "image_size": IMAGE_SIZE},
                },
                timeout=600,
            )
            if response.ok:
                result = response.json()
                try:
                    data_url = result["choices"][0]["message"]["images"][0]["image_url"]["url"]
                    destination.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
                    return
                except (KeyError, IndexError, TypeError, ValueError):
                    last = "No image in the response.\n" + json.dumps(result, indent=2)[:4000]
            else:
                last = f"OpenRouter returned {response.status_code}:\n{response.text[:4000]}"
                if response.status_code in (401, 402, 403):
                    break
        except requests.RequestException as exc:
            last = f"Network error: {exc}"
        time.sleep(4 * attempt)
    raise RuntimeError(last)


def write_provenance(path: Path, face_id: str, sources: list[Path], prompt: str, note: str,
                     meta: dict, traits: dict[str, str], origin: dict[str, str]) -> None:
    lines = [
        f"face_id: {face_id}",
        f"created_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"image_model: {MODEL}",
        f"trait_model: {ANALYSIS_MODEL}",
        f"aspect_ratio: {ASPECT_RATIO}",
        f"image_size: {IMAGE_SIZE}",
        f"age_range: {meta['age_range']}",
        f"size: {meta['size']}",
        f"skin: {meta['skin']}",
        f"hair_color: {meta['hair_color']}",
        f"hairstyle: {meta['hairstyle']}",
        f"name: {meta.get('name', '')}",
        f"height: {meta.get('height', '')}",
        f"silhouette: {meta.get('shape', '')}",
        f"eye_color: {meta.get('eye_color', '')}",
        f"vibe: {meta.get('vibe', '')}",
        f"source_note: {note}",
        "approval: pending",
        "",
        "process:",
        "Each reference photo was sent to the trait model only to write general facial-structure phrases.",
        "No reference photo was sent to the image model. It received the written feature list below.",
        "Traits were mixed across the photos, and two or three were replaced by natural variations.",
        "",
        "blended_traits:",
    ]
    for key, value in traits.items():
        lines.append(f"- {key}: {value}  [{origin.get(key, '?')}]")
    lines.extend(["", "source_files:"])
    for index, source in enumerate(sources, start=1):
        lines.append(f"{index}. {source.resolve()}")
        lines.append(f"   sha256: {file_sha256(source)}")
    lines.extend(["", "prompt:", prompt, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_name(source: Path, index: int) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in source.stem)
    stem = stem.strip("_") or "image"
    suffix = source.suffix.lower() if source.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    return f"{index:02d}_{stem[:80]}{suffix}"


def create_face(sources: list[Path], note: str, age_range: str, hair_color: str,
                hairstyle: str, size: str, skin: str = "fair", *, name: str = "",
                height: str = "average", shape: str = "straight", eye_color: str = "dark brown",
                vibe: str = "confident") -> tuple[Path, Path, str]:
    if age_range not in ADULT_AGE_RANGES:
        raise ValueError(f"Age must be one of {', '.join(ADULT_AGE_RANGES)}.")
    if size not in SIZES:
        raise ValueError(f"Size must be one of {', '.join(SIZES)}.")
    if skin not in SKIN_TONES:
        raise ValueError(f"Skin tone must be one of {', '.join(SKIN_TONES)}.")
    for label, value, allowed in (("Height", height, HEIGHTS), ("Silhouette", shape, SHAPES),
                                  ("Eye color", eye_color, EYE_COLORS), ("Vibe", vibe, VIBES)):
        if value not in allowed:
            raise ValueError(f"{label} must be one of {', '.join(allowed)}.")
    name = name.strip()[:60]
    note, hair_color, hairstyle = note.strip(), hair_color.strip(), hairstyle.strip()
    if not note:
        raise ValueError("Enter a source note for these reference photos.")
    if not hair_color or not hairstyle:
        raise ValueError("Enter a hair color and a hairstyle for the new face.")
    if len(sources) < MIN_REFERENCES or len(sources) > MAX_REFERENCES:
        raise ValueError(f"Use {MIN_REFERENCES} to {MAX_REFERENCES} photos, ideally of different women.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or the environment.")

    face_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = PROJECT_ROOT / "outputs" / "faces" / face_id
    source_dir = folder / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Path] = []
    for index, source in enumerate(sources, start=1):
        destination = source_dir / _safe_name(source, index)
        shutil.copy2(source, destination)
        kept.append(destination)

    print("Reading general facial structure from the photos.", flush=True)
    described = [(index, describe_traits(path)) for index, path in enumerate(kept, start=1)]
    readable = sum(1 for _, traits in described if traits)
    if readable < 2:
        raise RuntimeError("Could not read facial traits from at least two photos. Try clearer, front-facing photos.")
    rng = random.SystemRandom()
    traits, origin = blend_traits(described, rng)
    distinctive = rng.sample(DISTINCTIVE, 2)

    reference_body = blend_body(described, rng)
    body = body_description(size, height, shape, reference_body)
    prompt = face_prompt(age_range, hair_color, hairstyle, size, skin, traits, distinctive,
                         eye_color=eye_color, vibe=vibe, body=body)
    image_path = folder / "face.png"
    print("Generating a new face.", flush=True)
    generate(prompt, image_path)

    meta = {
        "face_id": face_id, "age_range": age_range, "size": size, "skin": skin,
        "hair_color": hair_color, "hairstyle": hairstyle, "traits": traits,
        "trait_origin": origin, "name": name, "height": height, "shape": shape,
        "eye_color": eye_color, "vibe": vibe, "reference_body": reference_body,
        "body_description": body, "image": "face.png", "approval": "pending",
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    notes_path = folder / "provenance.txt"
    write_provenance(notes_path, face_id, kept, prompt, note, meta, traits, origin)
    summary = (
        f"Blended traits from {readable} of {len(kept)} photos. "
        "No reference photo was sent to the image model."
    )
    print(f"Saved -> {image_path}")
    print(f"Provenance -> {notes_path}")
    return image_path, notes_path, summary


def _options(values, selected: str = "") -> str:
    return "\n".join(
        f'<option value="{v}"{" selected" if v == selected else ""}>{v}</option>' for v in values
    )


def _page() -> str:
    return (
        PAGE_TEMPLATE
        .replace("<!-- ages -->", _options(ADULT_AGE_RANGES, "25-40"))
        .replace("<!-- sizes -->", _options(SIZES, "S-M"))
        .replace("<!-- skins -->", _options(SKIN_TONES, "fair"))
        .replace("<!-- hair_colors -->", _options(HAIR_COLORS, "black"))
    )


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
    <p class="lead">Upload photos of 3 or more different women. Their general features are blended into a new person for the model library, not a copy of anyone.</p>
    <form id="form">
      <div class="drop" id="drop">
        <strong>Reference photos</strong>
        <span>Choose 3 to 8 photos, ideally of different women. Jpeg, png, webp, or gif.</span>
      </div>
      <input id="files" type="file" accept="image/jpeg,image/png,image/webp,image/gif" multiple>
      <div class="previews" id="previews"></div>
      <label for="note">Source note</label>
      <textarea id="note" name="note" required placeholder="Where these photos came from, for the provenance record"></textarea>
      <label for="age">Age range</label>
      <select id="age" name="age" required>
        <!-- ages -->
      </select>
      <label for="size">Size</label>
      <select id="size" name="size" required>
        <!-- sizes -->
      </select>
      <label for="skin">Skin tone</label>
      <select id="skin" name="skin" required>
        <!-- skins -->
      </select>
      <label for="haircolor">Hair color</label>
      <select id="haircolor" name="haircolor" required>
        <!-- hair_colors -->
      </select>
      <label for="hair">Hairstyle</label>
      <input id="hair" name="hair" type="text" required placeholder="Long, soft waves, side part, just past the shoulders">
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
      if (chosen.length < 3) {
        status.textContent = "Add at least three photos, ideally of different women.";
        return;
      }
      const note = document.getElementById("note").value.trim();
      const hair = document.getElementById("hair").value.trim();
      const age = document.getElementById("age").value;
      const size = document.getElementById("size").value;
      const skin = document.getElementById("skin").value;
      const haircolor = document.getElementById("haircolor").value;
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
          body: JSON.stringify({ session, note, age, hair, size, skin, haircolor }),
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
            size = str(body.get("size", ""))
            skin = str(body.get("skin", "fair"))
            hair_color = str(body.get("haircolor", ""))
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
            image_path, notes_path, summary = create_face(sources, note, age, hair_color, hair, size, skin)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
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
    parser.add_argument("--age", default="25-40", choices=ADULT_AGE_RANGES)
    parser.add_argument("--size", default="S-M", choices=tuple(SIZES))
    parser.add_argument("--skin", default="fair", choices=tuple(SKIN_TONES))
    parser.add_argument("--hair-color", default="black", help="e.g. black, dark brown, chestnut brown")
    parser.add_argument("--hairstyle", "--hair", dest="hairstyle", default="", help="Hairstyle for this new identity")
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
            args.hair_color.strip(),
            args.hairstyle.strip(),
            args.size,
            args.skin,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print("Approval is pending. Compare the result with the references before it enters the library.")
    print(image_path)
    print(notes_path)


if __name__ == "__main__":
    main()