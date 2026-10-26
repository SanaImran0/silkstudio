# Dress photograph workflow

The Python API lives in `backend/`. The page is a Next.js app in `frontend/`, styled with Tailwind. You choose photos in the browser, and the page calls `backend/generate_dress.py`. The instructions the model receives are written in `backend/prompt.py`. The finished portrait is shown on the page and saved in `outputs/`.

## Before you run

From the project folder, with the virtual environment active:

```bash
pip install -r backend/requirements.txt
npm install --prefix frontend
```

`.env` in the project root must contain:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Start the API and the page in two terminals:

```bash
python backend/serve.py
npm run dev --prefix frontend
```

Open `http://127.0.0.1:3000`. The API listens on `http://127.0.0.1:8765`. Stop either process with Ctrl+C. The same commands are used on macOS, Windows, and Linux.

## The page

Each piece is a row. Choose a photo, or add several close-ups where the row says “none or many.” Tick “Skip this piece” when the outfit has no separate picture of a required part. The counter stops at 14 photos. Generate portrait runs the compression step, then the model, and shows the image underneath. Leave the page open while it runs.

| Question | How many | What the photo must show |
| --- | --- | --- |
| Shirt/top's Front | one image | The whole front: neck to hem, both sides, true color |
| Shirt/top's Back | one image | The whole back, including the back neck |
| Shirt/top's Neck | one image | Neckline shape and all neck work |
| Neck detailing | none or many | Close-ups of neck stitches, stones, glass, beads |
| Right sleeve | one image | The whole right sleeve and cuff |
| Left sleeve | one image | The whole left sleeve and cuff |
| Border | one image | The border's full width and where it sits |
| Border detailing | none or many | Lace, tassels, crystals, hangings |
| Shirt detailing patch | none or many | Each separate patch of handwork |
| Trouser/bottom | one image | The whole trouser: cut, length, hem |
| Trouser detailing | none or many | Close-ups of hem or ankle work |
| Dupatta | one image | The whole dupatta, borders, and pallu |
| Dupatta detailing | none or many | Close-ups of dupatta border or pallu work |

`compress_references.py` checks the batch against Google's 20MB request limit before it is sent. The photos are base64 inside that request, so the JPEG files together are kept near 13MB, which leaves room for the expansion and the prompt. Originals that already fit are sent unchanged. A larger batch is resized with Lanczos, then saved as a high-quality JPEG, only as small as the cap requires. A light sharpen is applied after that resize so stitches and stones stay crisp. The smaller copies are written to `uploads/<timestamp>/`. The originals on your computer are left as they are. `outputs/` receives only the finished portrait and its notes file. The page shows the before-and-after sizes under the portrait.

To prepare photos without generating:

```bash
python backend/compress_references.py photo1.jpg photo2.jpg -o compressed_references
```

## The 14-image limit

The page uses `google/gemini-3.1-flash-image-preview`, the same model that saved the portraits already in `outputs/google/gemini-3.1-flash-image-preview`. A later run switched to the Pro image model, and Google AI Studio rejected that model with “User location is not supported for the API use.”

It accepts at most 14 reference images. Eight of those are the full pieces (front, back, neck, two sleeves, border, trouser, dupatta), which leaves six for close-ups. The page counter shows how many are used. Extra files past 14 are not added.

Spend close-ups on the work that must survive a zoom: neck, border, patches, cuffs, trouser hem, dupatta edge. Skip a detail shot when the full-piece photo is already sharp.

## How to photograph the pieces

- Shoot the full piece first, then the close-up of the same area. The prompt treats close-ups as magnification, not as new designs.
- Use a plain background and the same daylight for every shot. Filters and strong color casts make the model recolor the fabric.
- Keep the entire hem, cuff, or pallu inside the full-piece frame. A cropped motif gets completed by guesswork.
- If the left and right sleeves differ, do not upload one sleeve twice.
- Photograph borders at their real width. A crop of only the embroidery leaves the width unknown.

## What the model is allowed to create

The dress itself is closed: colors, motifs, embroidery, glasswork, stones, crystals, hangings, tassels, lace, and cut come only from the photos. The prompt forbids adding or removing garment detail.

The woman, pose, hair, jewellery, footwear, background, and light are not in those photos, so the prompt does create them. It asks for an original adult Pakistani woman with realistic skin, eyes, hands, and feet, jewellery and shoes that suit the outfit, and a quiet background whose light does not recolor the cloth. It does not ask for the face of a real influencer or public figure.

## Where the file goes

```text
outputs/<model>/<aspect>/<size>/<timestamp>.png
outputs/<model>/<aspect>/<size>/<timestamp>.txt
uploads/<timestamp>/
```

A 9:16, 4K run looks like:

```text
outputs/gemini-3.1-flash-image-preview/9x16/4K/20261015_191000.png
uploads/20261015_191000/
```

The `.txt` file stores the model settings, the paths you uploaded, and the exact prompt. Aspect ratio and size are changed with `ASPECT_RATIO` and `IMAGE_SIZE` at the top of `generate_dress.py`.

Generation can take several minutes. If OpenRouter rejects the request, the script prints the status code and the response body.

## Editing the prompt

Change wording in `prompt.py`. Each slot has a `fidelity` sentence that is sent immediately before that image, and `build_prompt()` assembles the full brief: catalog, missing pieces, garment rules, the woman, pose, light, and sharpness. After a run, read the saved `.txt` to see the brief that was actually sent.
