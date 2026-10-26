"""
Prompt for a faithful Pakistani dress reconstruction, worn by a model.

Edit the text in this file when the photograph should change. generate_dress.py
loads these slots and this prompt; it does not keep a second copy of the wording.

What this prompt is for
-----------------------
The reference photos are pieces of one outfit: shirt/kameez, sleeves, neck,
borders, patches, trouser, and dupatta, plus close-ups of handwork. The model
must assemble those pieces into the same dress, then photograph an adult
Pakistani woman wearing it.

Two jobs are kept separate on purpose:

1. The dress. Every panel, color, motif, border, embroidery stitch, glass
   piece, stone, crystal, tassel, and lace edge comes only from the uploaded
   photos. Nothing on the garment is invented, duplicated into an empty area,
   or left out.
2. The person and the setting. The woman, her pose, hair, jewellery, footwear,
   background, and light are not in the garment photos. They are created for
   the photograph and must not cover or replace the dress.

The woman is an original person. The prompt does not ask for the face of a
real influencer, actress, or other public figure.

Image order
-----------
The page and generate_dress.py ask for the slots below in this order. Required slots expect
one photo. Slots marked many=True accept none or several close-ups. There is a
hard cap of 14 reference images on the Pro image model, so extra close-ups
should be saved for the work that must stay sharp when the picture is zoomed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GarmentSlot:
    """One question in the upload interview, and the rule for using those photos."""

    key: str
    ask: str
    label: str
    fidelity: str
    many: bool = False
    required: bool = True


# Asked in this order. Required pieces come before their own close-ups so a
# full outfit is on record before the 14-image budget is spent on details.
GARMENT_SLOTS: tuple[GarmentSlot, ...] = (
    GarmentSlot(
        key="shirt_front",
        ask="Add image of shirt/top's Front",
        label="Shirt/top front",
        fidelity=(
            "This is the master front of the shirt/kameez. Match its length, "
            "width, neckline, front opening, panel seams, base fabric color, "
            "print, and every motif exactly. Do not redesign the front."
        ),
    ),
    GarmentSlot(
        key="shirt_back",
        ask="Add image of shirt/top's Back",
        label="Shirt/top back",
        fidelity=(
            "This is the master back. Match the back neck, yoke, motifs, "
            "fabric, and length. Do not copy the front design onto the back "
            "unless this photo shows the same design."
        ),
    ),
    GarmentSlot(
        key="shirt_neck",
        ask="Add image of shirt/top's Neck",
        label="Shirt/top neck",
        fidelity=(
            "This fixes the neckline shape and the neck embroidery. Keep the "
            "depth, width, and finish of the neck exactly as photographed."
        ),
    ),
    GarmentSlot(
        key="shirt_neck_detail",
        ask="Add image of shirt/top's neck detailing (if any)",
        label="Shirt/top neck detailing",
        fidelity=(
            "Close-up of neck handwork only. Treat it as a magnification of "
            "the neck photo: same stitches, beads, stones, and spacing. Do not "
            "add motifs that appear only because the crop is tight, and do not "
            "spread this work past the area shown in the full neck photo."
        ),
        many=True,
        required=False,
    ),
    GarmentSlot(
        key="sleeve_right",
        ask="Add right sleeve",
        label="Right sleeve",
        fidelity=(
            "Full right sleeve, including the cuff if it is in frame. This "
            "sleeve stays on the woman's right arm. Do not mirror it onto the "
            "left arm unless the left-sleeve photo is identical."
        ),
    ),
    GarmentSlot(
        key="sleeve_left",
        ask="Add left sleeve",
        label="Left sleeve",
        fidelity=(
            "Full left sleeve, including the cuff if it is in frame. This "
            "sleeve stays on the woman's left arm."
        ),
    ),
    GarmentSlot(
        key="border",
        ask="Add Border",
        label="Border",
        fidelity=(
            "This is the border piece (hem, side panel, or daman) as it really "
            "is. Match its width, repeat, colors, lace, and where it sits on "
            "the garment. Do not invent a second border style."
        ),
    ),
    GarmentSlot(
        key="border_detail",
        ask="Add border detailing (if any)",
        label="Border detailing",
        fidelity=(
            "Close-up of the border's embroidery, lace, stones, crystals, "
            "hangings, or tassels. Use it only to sharpen the border already "
            "shown in the full border photo. Do not lengthen the border or "
            "add extra hanging pieces."
        ),
        many=True,
        required=False,
    ),
    GarmentSlot(
        key="shirt_patch",
        ask="Add shirt's detailing patch (if any)",
        label="Shirt detailing patch",
        fidelity=(
            "A local patch of shirt handwork. Reproduce this patch only where "
            "the photos show it. Do not tile it across empty fabric."
        ),
        many=True,
        required=False,
    ),
    GarmentSlot(
        key="trouser",
        ask="Add Trouser/bottom",
        label="Trouser/bottom",
        fidelity=(
            "Full trouser or bottom. Match the cut, rise, width, length, "
            "fabric, color, and any embroidery at the hem or ankle. Do not "
            "change it into a different style of pant, lehenga, or skirt."
        ),
    ),
    GarmentSlot(
        key="trouser_detail",
        ask="Add trouser/bottom detailing (if any)",
        label="Trouser/bottom detailing",
        fidelity=(
            "Close-up of trouser handwork. Apply it only at the location shown "
            "on the full trouser. Do not add embroidery to a plain trouser."
        ),
        many=True,
        required=False,
    ),
    GarmentSlot(
        key="dupatta",
        ask="Add dupatta",
        label="Dupatta",
        fidelity=(
            "Full dupatta, including both borders and the pallu if visible. "
            "Match the fabric, print, border width, and end design. Drape this "
            "same dupatta; do not substitute a different scarf."
        ),
    ),
    GarmentSlot(
        key="dupatta_detail",
        ask="Add dupatta detailing (if any)",
        label="Dupatta detailing",
        fidelity=(
            "Close-up of dupatta embroidery, border, tassels, or pallu work. "
            "Use it to sharpen the dupatta already shown, not to design a new one."
        ),
        many=True,
        required=False,
    ),
)


def image_caption(index: int, label: str, fidelity: str) -> str:
    """Short label placed immediately before each reference image."""
    return f"Image {index} — {label}. {fidelity}"


def build_prompt(references: list[dict]) -> str:
    """
    Build the full generation prompt for one outfit.

    references: dicts with keys index (int), label (str), fidelity (str),
    and filename (str). Order is the order the images are attached.
    """
    if not references:
        raise ValueError("At least one reference image is required to build the prompt.")

    catalog_lines = ["REFERENCE CATALOG"]
    for item in references:
        catalog_lines.append(
            f"Image {item['index']} — {item['label']} (file: {item['filename']})\n"
            f"{item['fidelity']}"
        )
    catalog = "\n\n".join(catalog_lines)

    provided = {item["label"] for item in references}
    missing = [
        slot.label
        for slot in GARMENT_SLOTS
        if slot.label not in provided
    ]
    if missing:
        missing_block = (
            "PIECES WITH NO PHOTO\n"
            + "\n".join(f"- {name}" for name in missing)
            + "\nNo photograph was uploaded for these pieces. Do not design a "
            "replacement. If that piece is already visible inside another "
            "uploaded photo, use only what that photo shows. If it is not "
            "visible anywhere, omit it."
        )
    else:
        missing_block = (
            "PIECES WITH NO PHOTO\n"
            "Every requested piece has at least one photo. Still do not add "
            "any further component that is not visible in those photos."
        )

    sections = [
        """TASK
Assemble one real outfit from the labeled reference photos, then photograph an adult Pakistani woman wearing that exact outfit. The garment in the final picture must be the same garment as the references: the same cut and the same surface detail. Study every reference before composing the picture.""",
        catalog,
        missing_block,
        """HOW TO READ THE PHOTOS
Use each full-piece photo as the map of shape, placement, and color. Use each close-up only as a magnification of work that already exists on that piece. A tight crop does not mean the motif repeats everywhere. Where two photos of the same piece overlap, they must agree; do not average them into a new design. Keep the right sleeve on the right arm and the left sleeve on the left arm. Keep the front, back, neck, border, trouser, and dupatta as separate pieces of one outfit.""",
        """GARMENT FIDELITY
Do not add anything to the dress, and do not remove anything from it.
Preserve, at the same place and the same scale: base fabric color and weave; printed or woven motifs; embroidery thread colors and stitch direction; gota, lace, and piping; glasswork, stones, crystals, and beads; crystal hangings; tassels; neckline shape; cuff shape; border width and repeat; patch placement; shirt length and hem; trouser cut and length; dupatta size, border, and pallu.
Do not invent extra flowers, vines, borders, buttons, belts, jackets, underskirts, or a second dupatta. Do not change a plain area into embroidery. Do not drop small hangings because they are thin. Do not recolor the fabric to flatter the background. Do not simplify dense handwork into a smooth print. Do not mirror an asymmetric pattern. Gold, silver, and beadwork must sit on the cloth the way the photos show, not melt into the print.""",
        """THE WOMAN
Photograph one adult Pakistani woman. She is an original person, not a copy or lookalike of any real influencer, actress, model, or public figure.
She must look photographed, not illustrated: South Asian features, a natural Pakistani skin tone with pores, fine facial hair, and slight unevenness, a realistic brown iris, distinct eyelashes, natural lips, and no plastic smoothing. Hands show five separate fingers with knuckles and nails. Feet and ankles are complete and match the body. Her expression is calm and natural. Hair is styled for Pakistani formal wear and does not cover the neckline, cuffs, or dupatta border.
Jewellery and footwear are not part of the garment photos. Add only what a Pakistani woman would wear with this outfit: earrings, a light necklace, and bangles or rings that leave the embroidery visible, plus footwear that suits the dress (formal sandals, heels, or khussa). Jewellery and shoes must not cover neck work, cuff work, or the trouser hem.""",
        """POSE, LIGHT, AND BACKGROUND
Full-length portrait, head to toe, with a small margin above the head and below the feet. The body is in a relaxed standing pose, weight on one leg, so the shirt, both sleeves, trouser, and dupatta read clearly. Use a slight three-quarter turn only if the front embroidery stays fully visible. Arms are away from the body enough to show the sleeves and cuffs. Drape the supplied dupatta in a customary Pakistani way, with the pallu visible, without hiding the front panel.
Background is minimal and uncluttered: a plain studio wall or a quiet interior surface. Choose a neutral color that suits the dress and does not shift the fabric color. Lighting is soft and even, with gentle shadows that show embroidery relief, bead sparkle, and lace edges. No harsh color gel, no sunset wash, no smoke, no furniture, no other people, no text, no logo, no watermark, no collage, and no frame.""",
        """SHARPNESS
This picture is a 4K fashion photograph. When it is zoomed, the neck work, border, patches, sleeve cuffs, trouser hem, and dupatta edge must still show individual stitches, stones, crystals, glass pieces, tassels, and lace holes. Detail comes from the reference photos, not from painted-looking texture. Do not blur the garment to make the face softer, and do not crop off the hem, cuffs, or feet.""",
        """OUTPUT
One single photograph. The dress is the outfit defined by the reference catalog and by no other source. The woman, pose, jewellery, footwear, background, and light follow the sections above.""",
    ]
    return "\n\n".join(sections)
