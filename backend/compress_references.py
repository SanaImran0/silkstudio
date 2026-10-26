"""
Shrink a batch of reference photos so Google AI Studio will accept the request.

Google rejects the whole request body above 20MB. The photos travel inside
that body as base64, which is about a third larger than the JPEG files, and
the prompt is in the body too. This module keeps each photo at the largest
size whose encoded request still fits:

- originals are sent unchanged when the batch is already under the cap
- otherwise each photo is resized with Lanczos, from the full image, never
  from a previous smaller copy
- a light unsharp mask is applied only after a downscale, to put edge
  contrast back into stitches, stones, and lace
- JPEG is saved at high quality with 4:4:4 color (no chroma subsampling)
  until a smaller file is required

The first setting that fits is kept, so detail is given up only as far as
the 20MB request cap requires.

Run on its own:
    python compress_references.py photo1.jpg photo2.jpg -o compressed_references
"""

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

# Google AI Studio rejects a request body over 20,000,000 bytes. Stay 1MB
# under that. Base64 expands each photo by 4/3, and the prompt shares the body.
GOOGLE_BODY_LIMIT = 20_000_000
BODY_MARGIN = 1_000_000
TEXT_ALLOWANCE = 256 * 1024
PER_IMAGE_OVERHEAD = 1024

# Tried in order. long_edge, JPEG quality, chroma subsampling.
# subsampling 0 is 4:4:4 (full color). 2 is 4:2:0 (smaller, softer color).
SETTINGS: tuple[tuple[int, int, int], ...] = (
    (3072, 92, 0),
    (3072, 88, 0),
    (3072, 84, 0),
    (2560, 92, 0),
    (2560, 88, 0),
    (2560, 84, 0),
    (2304, 90, 0),
    (2304, 86, 0),
    (2048, 90, 0),
    (2048, 85, 0),
    (2048, 85, 2),
    (1600, 85, 2),
    (1280, 80, 2),
    (1024, 75, 2),
)


@dataclass(frozen=True)
class PreparedImage:
    source: Path
    output: Path
    original_bytes: int
    sent_bytes: int
    long_edge: int | None
    quality: int | None


def format_megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f}MB"


def jpeg_budget(image_count: int) -> int:
    """Largest photo-byte total whose base64 request stays under Google's cap."""
    room = GOOGLE_BODY_LIMIT - BODY_MARGIN - TEXT_ALLOWANCE - image_count * PER_IMAGE_OVERHEAD
    if room <= 0:
        raise RuntimeError("No room left in the request for reference photos.")
    return (room // 4) * 3


def estimate_request_bytes(jpeg_bytes: int, image_count: int) -> int:
    """Approximate the JSON body Google measures, including base64 and the prompt."""
    base64_bytes = 4 * ((jpeg_bytes + 2) // 3)
    return base64_bytes + TEXT_ALLOWANCE + image_count * PER_IMAGE_OVERHEAD


def report(prepared: list[PreparedImage]) -> str:
    """One block of text describing what will actually be uploaded."""
    lines = []
    for item in prepared:
        if item.long_edge is None:
            lines.append(f"  {item.source.name}: {format_megabytes(item.original_bytes)} unchanged")
        else:
            lines.append(
                f"  {item.source.name}: {format_megabytes(item.original_bytes)} -> "
                f"{format_megabytes(item.sent_bytes)} "
                f"({item.long_edge}px, quality {item.quality})"
            )
    sent = sum(item.sent_bytes for item in prepared)
    request_bytes = estimate_request_bytes(sent, len(prepared))
    lines.append(
        f"Photos: {format_megabytes(sent)}. "
        f"Estimated request: {request_bytes / 1_000_000:.1f}MB of Google's 20MB limit."
    )
    return "\n".join(lines)


def _load(path: Path) -> Image.Image:
    """Open one photo upright, in RGB, at its full resolution."""
    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image)
        return upright.convert("RGB")


def _fit(image: Image.Image, long_edge: int) -> Image.Image:
    """Shrink so the long side is at most long_edge. Never enlarge."""
    width, height = image.size
    longest = max(width, height)
    if longest <= long_edge:
        return image
    scale = long_edge / longest
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized.filter(ImageFilter.UnsharpMask(radius=0.8, percent=60, threshold=2))


def _encode(image: Image.Image, quality: int, subsampling: int) -> bytes:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=True,
    )
    return buffer.getvalue()


def _output_name(index: int, source: Path) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in source.stem)
    stem = stem.strip("_") or "image"
    return f"{index:02d}_{stem}.jpg"


def compress_images(sources: list[Path], destination: Path) -> list[PreparedImage]:
    """
    Return one prepared file per source.

    Files already small enough to send together are returned as the originals.
    Otherwise JPEG copies are written into destination.
    """
    if not sources:
        raise ValueError("No images to compress.")

    original_sizes = [path.stat().st_size for path in sources]
    budget = jpeg_budget(len(sources))
    if sum(original_sizes) <= budget:
        destination.mkdir(parents=True, exist_ok=True)
        prepared: list[PreparedImage] = []
        for index, (path, size) in enumerate(zip(sources, original_sizes), start=1):
            output = destination / _output_name(index, path)
            if path.resolve() != output.resolve():
                output.write_bytes(path.read_bytes())
            prepared.append(
                PreparedImage(
                    source=path,
                    output=output,
                    original_bytes=size,
                    sent_bytes=size,
                    long_edge=None,
                    quality=None,
                )
            )
        return prepared

    masters = [_load(path) for path in sources]
    chosen: tuple[int, int, int, list[bytes]] | None = None
    for long_edge, quality, subsampling in SETTINGS:
        encoded = [
            _encode(_fit(master, long_edge), quality, subsampling)
            for master in masters
        ]
        if sum(len(blob) for blob in encoded) <= budget:
            chosen = (long_edge, quality, subsampling, encoded)
            break

    if chosen is None:
        raise RuntimeError(
            "These photos stay above Google's 20MB request limit even after the "
            "strongest compression. Remove a close-up and try again."
        )

    long_edge, quality, _, encoded = chosen
    destination.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedImage] = []
    for index, (source, original_size, blob) in enumerate(
        zip(sources, original_sizes, encoded),
        start=1,
    ):
        output = destination / _output_name(index, source)
        output.write_bytes(blob)
        prepared.append(
            PreparedImage(
                source=source,
                output=output,
                original_bytes=original_size,
                sent_bytes=len(blob),
                long_edge=long_edge,
                quality=quality,
            )
        )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shrink reference photos so the Gemini request stays under Google's 20MB limit."
    )
    parser.add_argument("images", nargs="+", type=Path, help="Image files to prepare")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("compressed_references"),
        help="Folder for the smaller copies (default: compressed_references)",
    )
    args = parser.parse_args()
    missing = [path for path in args.images if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise SystemExit(f"No such file: {names}")
    prepared = compress_images(args.images, args.output)
    print(report(prepared))


if __name__ == "__main__":
    main()
