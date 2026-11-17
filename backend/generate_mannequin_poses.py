"""
Build the mannequin POSE LIBRARY for Silk Studio.

Step 1: generate ONE master mannequin (or reuse one with --master).
Step 2: render every pose in POSES from that master, so the same mannequin
        (body, finish, proportions) appears in every pose.

Output (9:16, 4K, grey studio background):

    outputs/mannequin_poses/<run>/
        master.png
        poses/<pose_id>.png
        manifest.json      <- id, label, category, file, status (for the frontend / DB)

Run from the repository root:

    python backend/generate_mannequin_poses.py                    # master + all poses
    python backend/generate_mannequin_poses.py --list             # show pose ids
    python backend/generate_mannequin_poses.py --master my.png    # skip master step
    python backend/generate_mannequin_poses.py --only standing_front walk_front_a
    python backend/generate_mannequin_poses.py --category walking
    python backend/generate_mannequin_poses.py --resume outputs/mannequin_poses/20261116_150000
    python backend/generate_mannequin_poses.py --workers 3        # parallel poses

--resume skips poses already saved, so a crash or rate limit never costs you finished images.
"""

import argparse
import base64
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from generate_face import (
    API_URL,
    IMAGE_SUFFIXES,
    MODEL,
    OPENROUTER_API_KEY,
    PROJECT_ROOT,
    image_part,
)

ASPECT_RATIO = "9:16"
IMAGE_SIZE = "4K"
MAX_RETRIES = 4
OUT_ROOT = PROJECT_ROOT / "outputs" / "mannequin_poses"

# --------------------------------------------------------------------------
# Master mannequin
# --------------------------------------------------------------------------

MASTER_PROMPT = """
Studio photograph of a single full-length female fashion mannequin (dress form), standing upright, facing the camera, arms relaxed slightly away from the body, feet slightly apart.

Look: matte warm-beige fabric-wrapped finish, smooth and seamless, elegant slender South Asian fashion proportions (about 8 heads tall), softly articulated shoulders, elbows, wrists, hips, knees and ankles, simplified hands with gently separated fingers, simplified feet without shoes, an egg-shaped featureless head with no eyes, nose, mouth or hair. Slim neck, defined bust, waist and hips, natural shoulder width. The mannequin is unclothed and has no jewellery, no stand, no pole, no base plate, and no visible joints hardware.

Seamless mid-grey studio background, soft even light from the front-left, gentle floor shadow. Head to toe fully in frame with small margins above and below. One mannequin, one photograph. No text, logo, watermark, or props.
""".strip()

# --------------------------------------------------------------------------
# Pose library. Edit freely: add a dict and it is generated next run.
# "instruction" describes ONLY the pose and camera angle. The mannequin
# identity comes from the master image.
# --------------------------------------------------------------------------


def P(id_: str, category: str, label: str, instruction: str) -> dict:
    return {"id": id_, "category": category, "label": label, "instruction": instruction}


POSES: tuple[dict, ...] = (
    # ---- standing --------------------------------------------------------
    P("standing_front", "standing", "Standing, front",
      "Standing straight facing the camera, weight even, arms relaxed beside the body with a small gap from the torso, feet hip-width apart."),
    P("standing_front_narrow", "standing", "Standing, front, feet together",
      "Standing straight facing the camera, feet together, arms hanging straight and close to the body so the full side seams of a dress would be visible."),
    P("standing_back", "standing", "Standing, back",
      "Seen from directly behind, standing straight, arms relaxed beside the body, feet hip-width apart. Back of the head toward the camera."),
    P("standing_left", "standing", "Standing, left profile",
      "Strict left profile, body side-on facing the left edge of the frame, arms relaxed, feet slightly apart."),
    P("standing_right", "standing", "Standing, right profile",
      "Strict right profile, body side-on facing the right edge of the frame, arms relaxed, feet slightly apart."),
    P("standing_three_quarter_left", "standing", "Standing, three-quarter left",
      "Body turned about 45 degrees toward the left edge of the frame, head turned slightly back toward the camera, arms relaxed."),
    P("standing_three_quarter_right", "standing", "Standing, three-quarter right",
      "Body turned about 45 degrees toward the right edge of the frame, head turned slightly back toward the camera, arms relaxed."),
    P("standing_back_three_quarter", "standing", "Standing, back three-quarter",
      "Seen from behind and slightly to the side, body turned about 45 degrees away from the camera, arms relaxed, head turned slightly toward the side."),
    # ---- hand on hip -----------------------------------------------------
    P("hip_right_front", "hand_on_hip", "Right hand on hip, front",
      "Facing the camera, right hand resting on the hip with the elbow out, left arm relaxed, weight on the left leg, right knee slightly bent."),
    P("hip_left_front", "hand_on_hip", "Left hand on hip, front",
      "Facing the camera, left hand resting on the hip with the elbow out, right arm relaxed, weight on the right leg, left knee slightly bent."),
    P("hip_both_front", "hand_on_hip", "Both hands on hips, front",
      "Facing the camera, both hands on the hips with the elbows out, feet shoulder-width apart, upright confident stance."),
    P("hip_right_three_quarter", "hand_on_hip", "Hand on hip, three-quarter",
      "Body turned about 45 degrees to the right, right hand on the hip, left arm relaxed, one foot slightly forward."),
    # ---- walking ---------------------------------------------------------
    P("walk_front_a", "walking", "Walking toward camera, right foot forward",
      "Mid-stride walking toward the camera, right foot forward, arms swinging naturally, steps of moderate length, upright posture."),
    P("walk_front_b", "walking", "Walking toward camera, left foot forward",
      "Mid-stride walking toward the camera, left foot forward, arms swinging naturally, steps of moderate length, upright posture."),
    P("walk_side_left", "walking", "Walking, left profile",
      "Mid-stride walking in strict left profile toward the left edge of the frame, one foot ahead of the other, arms swinging gently."),
    P("walk_side_right", "walking", "Walking, right profile",
      "Mid-stride walking in strict right profile toward the right edge of the frame, one foot ahead of the other, arms swinging gently."),
    P("walk_away", "walking", "Walking away",
      "Seen from behind, mid-stride walking away from the camera, one foot ahead, arms swinging gently."),
    P("step_pause_front", "walking", "Runway pause, front",
      "Runway pause facing the camera: one foot placed slightly in front of the other, hips slightly angled, arms relaxed at the sides."),
    # ---- dupatta / arm-hold poses (mannequin has NO dupatta: arms only) ---
    P("dupatta_hold_one_arm_front", "dupatta_hold", "Right forearm raised, front",
      "Facing the camera, right elbow bent with the forearm held up across the front of the torso at waist height, left arm relaxed. Leave the pose open so a dupatta could drape over the forearm."),
    P("dupatta_hold_both_arms_front", "dupatta_hold", "Both forearms forward, front",
      "Facing the camera, both elbows bent and both forearms held forward at waist height, hands slightly apart, as if holding the ends of a scarf. Hands empty."),
    P("dupatta_shoulder_left", "dupatta_hold", "Left hand at shoulder, front",
      "Facing the camera, left hand raised to touch the left shoulder as if adjusting a dupatta, right arm relaxed. Hand empty."),
    P("dupatta_shoulder_right", "dupatta_hold", "Right hand at shoulder, front",
      "Facing the camera, right hand raised to touch the right shoulder as if adjusting a dupatta, left arm relaxed. Hand empty."),
    P("dupatta_hold_back", "dupatta_hold", "Both hands low behind, back",
      "Seen from behind, both arms slightly bent and held a little away from the body so a dupatta's back drape would stay visible."),
    P("dupatta_hold_side", "dupatta_hold", "Forearm raised, left profile",
      "Strict left profile, near forearm raised in front of the torso at waist height, elbow bent, other arm relaxed."),
    # ---- gesture / fashion -----------------------------------------------
    P("arms_crossed_front", "gesture", "Arms crossed, front",
      "Facing the camera, arms crossed loosely at the waist, upright posture, feet hip-width apart."),
    P("hands_clasped_front", "gesture", "Hands clasped, front",
      "Facing the camera, both hands lightly clasped in front of the waist, shoulders relaxed, feet together."),
    P("hands_behind_front", "gesture", "Hands behind back, front",
      "Facing the camera, both hands held behind the back, chest slightly open, feet hip-width apart."),
    P("one_arm_raised_front", "gesture", "One hand to hair, front",
      "Facing the camera, right hand raised toward the side of the head, left arm relaxed, head slightly tilted, weight on the left leg."),
    P("lift_skirt_front", "gesture", "Hands at skirt level, front",
      "Facing the camera, both hands slightly lifted at hip level and held a little away from the body, as if gathering a long skirt. Hands empty."),
    P("twirl_back_look", "gesture", "Looking back over shoulder",
      "Body turned away from the camera at three-quarter back, head turned back over the left shoulder toward the camera, arms relaxed."),
    P("seated_front", "seated", "Seated on a stool, front",
      "Seated on a simple plain grey cube stool facing the camera, back straight, knees together, hands resting on the thighs. The stool is part of the pose."),
    P("seated_side", "seated", "Seated on a stool, left profile",
      "Seated on a simple plain grey cube stool in strict left profile, back straight, knees together, hands resting on the thighs."),
)
POSE_BY_ID = {p["id"]: p for p in POSES}
assert len(POSE_BY_ID) == len(POSES), "Duplicate pose id"

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def pose_prompt(pose: dict) -> str:
    return f"""
Image 1 is the master mannequin. Photograph this exact same mannequin: same beige matte finish, same proportions, same featureless egg-shaped head, same hands and feet. Do not add a face, hair, clothes, jewellery, shoes, or props. Do not turn her into a human.

New pose and camera angle: {pose["instruction"]}

Keep the body limbs anatomically plausible for a posable mannequin, with all fingers and both feet clearly visible and no merged limbs. Seamless mid-grey studio background, soft even light from the front-left, gentle floor shadow. Full body from head to feet with small margins above and below, centered. One mannequin, one photograph. No text, logo, watermark, grid, or extra figures.
""".strip()


def _call(content: list[dict]) -> bytes:
    if not OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY in .env or the environment.")
    last = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
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
            if r.ok:
                data = r.json()
                try:
                    url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
                    return base64.b64decode(url.split(",", 1)[1])
                except (KeyError, IndexError, TypeError, ValueError):
                    last = "No image in response: " + json.dumps(data)[:4000]
            else:
                last = f"HTTP {r.status_code}: {r.text[:4000]}"
                if r.status_code in (401, 402, 403):  # auth/credits: retrying won't help
                    break
        except requests.RequestException as exc:
            last = f"Network error: {exc}"
        time.sleep(4 * attempt)
    raise RuntimeError(last)


def make_master(dest: Path) -> None:
    print("Generating master mannequin...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_call([{"type": "text", "text": MASTER_PROMPT}]))
    print(f"Saved master -> {dest}", flush=True)


def make_reference(master: Path, dest: Path, max_side: int = 1536) -> Path:
    """Small JPEG copy of the master. Keeps requests far below the 20MB limit."""
    from PIL import Image

    with Image.open(master) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(dest, "JPEG", quality=92)
    return dest


def make_pose(master: Path, pose: dict, dest: Path) -> None:
    content = [
        {"type": "text", "text": "Image 1 is the master mannequin. Keep this exact mannequin."},
        image_part(master),
        {"type": "text", "text": pose_prompt(pose)},
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_call(content))


# --------------------------------------------------------------------------
# Run management
# --------------------------------------------------------------------------


def write_manifest(run: Path, status: dict[str, str]) -> None:
    items = [
        {
            "id": p["id"],
            "category": p["category"],
            "label": p["label"],
            "instruction": p["instruction"],
            "file": f"poses/{p['id']}.png",
            "status": status.get(p["id"], "pending"),
        }
        for p in POSES
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL,
        "aspect_ratio": ASPECT_RATIO,
        "image_size": IMAGE_SIZE,
        "master": "master.png",
        "poses": items,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the mannequin pose library.")
    ap.add_argument("--list", action="store_true", help="List pose ids and exit.")
    ap.add_argument("--master", help="Existing master mannequin image to reuse.")
    ap.add_argument("--resume", help="Existing run folder; skips poses already saved.")
    ap.add_argument("--only", nargs="+", metavar="ID", help="Only these pose ids.")
    ap.add_argument("--category", nargs="+", help="Only these categories.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel pose requests (default 1).")
    args = ap.parse_args()

    if args.list:
        for p in POSES:
            print(f"{p['id']:34s} [{p['category']}] {p['label']}")
        print(f"\n{len(POSES)} poses")
        return

    selected = list(POSES)
    if args.only:
        unknown = [i for i in args.only if i not in POSE_BY_ID]
        if unknown:
            sys.exit(f"Unknown pose id(s): {', '.join(unknown)}. Use --list.")
        selected = [POSE_BY_ID[i] for i in args.only]
    if args.category:
        selected = [p for p in selected if p["category"] in args.category]
    if not selected:
        sys.exit("No poses selected.")

    # run folder + master
    if args.resume:
        run = Path(args.resume).expanduser()
        if not run.is_dir():
            sys.exit(f"No folder at {run}")
        master = run / "master.png"
        if not master.is_file():
            sys.exit("This run has no master.png.")
    else:
        run = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        run.mkdir(parents=True, exist_ok=True)
        master = run / "master.png"
        if args.master:
            src = Path(args.master).expanduser()
            if not src.is_file() or src.suffix.lower() not in IMAGE_SUFFIXES:
                sys.exit("--master must be an existing jpeg/png/webp/gif file.")
            shutil.copy2(src, master)
        else:
            make_master(master)
            input("Check master.png now. Press Enter to continue (Ctrl+C to stop and tweak MASTER_PROMPT)... ")

    reference = make_reference(master, run / "master_ref.jpg")
    errors_log = run / "errors.log"

    status: dict[str, str] = {}
    for p in POSES:
        status[p["id"]] = "done" if (run / "poses" / f"{p['id']}.png").is_file() else "pending"
    write_manifest(run, status)

    todo = [p for p in selected if status[p["id"]] != "done"]
    print(f"{len(todo)} to generate, {len(selected) - len(todo)} already done.", flush=True)

    def work(p: dict) -> tuple[str, str]:
        try:
            make_pose(reference, p, run / "poses" / f"{p['id']}.png")
            return p["id"], "done"
        except Exception as exc:  # keep going; report at the end
            return p["id"], f"failed: {exc}"

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, p) for p in todo]
        for fut in as_completed(futures):
            pid, state = fut.result()
            status[pid] = state
            if state.startswith("failed"):
                with errors_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"=== {pid}\n{state}\n\n")
            done += 1
            print(f"[{done}/{len(todo)}] {pid}: {state[:120]}", flush=True)
            write_manifest(run, status)

    failed = [k for k, v in status.items() if v.startswith("failed")]
    print(f"\nFinished -> {run}")
    if failed:
        print(f"{len(failed)} failed. Full errors in {errors_log}")
        print(f"Re-run with: --resume {run} (only missing poses are retried)")


if __name__ == "__main__":
    main()