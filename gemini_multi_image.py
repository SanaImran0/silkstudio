"""
Generate ONE high-resolution image from MULTIPLE reference images using
Google's Gemini ("Nano Banana 2") image models via OpenRouter.

Setup:
    pip install requests
    export OPENROUTER_API_KEY="sk-or-v1-..."

Usage:
    python gemini_multi_image.py
"""

import base64
import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pick the Gemini / "Nano Banana" variant you want:
#   "google/gemini-2.5-flash-image"          -> Nano Banana (original)
#   "google/gemini-3.1-flash-image-preview"  -> Nano Banana 2 (extended aspect ratios, up to 4K)
#   "google/gemini-3-pro-image-preview"      -> Nano Banana Pro (best quality, up to 14 reference images)
MODEL = "google/gemini-3.1-flash-image-preview"
OUTPUT_DIR_PATH = f"outputs/{MODEL}"
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

VALID_ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1",
}
VALID_SIZES = {"0.5K", "1K", "2K", "4K"}



def _to_content_part(image_ref: str) -> dict:
    """
    Turn one reference image (local file path OR http(s) URL) into the
    image_url content part the chat completions API expects.
    """
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        url = image_ref
    else:
        path = Path(image_ref)
        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        url = f"data:{mime_type};base64,{b64}"
    return {"type": "image_url", "image_url": {"url": url}}


def generate_image(
    prompt: str,
    reference_images: list[str],
    aspect_ratio: str = "9:16",
    image_size: str = "4K",
    model: str = MODEL,
    output_dir_path: str = OUTPUT_DIR_PATH
) -> str:
    """
    Send a text prompt + multiple reference images to Gemini via OpenRouter
    and save the single returned image to disk.

    reference_images: list of local file paths and/or http(s) URLs.
        Nano Banana / Nano Banana 2 handle several reference images fine;
        Nano Banana Pro (gemini-3-pro-image-preview) accepts up to 14.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable first.")
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        raise ValueError(f"aspect_ratio must be one of {sorted(VALID_ASPECT_RATIOS)}")
    if image_size not in VALID_SIZES:
        raise ValueError(f"image_size must be one of {sorted(VALID_SIZES)}")
    if not reference_images:
        raise ValueError("Provide at least one reference image.")

    content = [{"type": "text", "text": prompt}]
    content.extend(_to_content_part(img) for img in reference_images)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
        },
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()

    message = result["choices"][0]["message"]
    images = message.get("images")
    if not images:
        raise RuntimeError(
            "No image returned. Full response:\n" + json.dumps(result, indent=2)
        )

    # These models return one image per request by default -> take the first
    data_url = images[0]["image_url"]["url"]
    _, b64_data = data_url.split(",", 1)
    try:
        img_path = f"{OUTPUT_DIR_PATH}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        Path(img_path).write_bytes(base64.b64decode(b64_data))
        print(f"Saved -> {img_path}")
        if "usage" in result:
            print(f"Usage: {result['usage']}")
        return img_path
    except Exception as e:
        return f"Error saving image: {e}"

if __name__ == "__main__":
    generate_image(
        prompt=(
            "Combine the parts of the garment from the reference images into a single cohesive garment, "
            "and create aphotorealistic image of a fashion model wearing this garment/dress. "
            "Ensure the image is in a realistic and natural pose, "
            "the background should be light weight and minimalistic, "
            "the fashion model should look like a very real human being, "
            "with realistic expressions on the face and realistic face features and skin texture."
            "Produce a high-resolution image with a resolution of 4K or higher such that the garment is visible in detail when zoomed in."
        ),
        reference_images=[
            "/Users/macbookpro/Desktop/Dresses Data/4/1.jpeg",
            "/Users/macbookpro/Desktop/Dresses Data/4/2.jpeg",
            "/Users/macbookpro/Desktop/Dresses Data/4/3.jpeg",
            "/Users/macbookpro/Desktop/Dresses Data/4/4.jpeg",

        ],
        aspect_ratio="9:16",
        image_size="4K",
        output_dir_path = OUTPUT_DIR_PATH
    )
