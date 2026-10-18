from pathlib import Path
import os
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Resolve the project directory from this script, not from a machine-specific path.
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results/final_analysis"
CSV = OUT / "representative_images_k15.csv"

# Set RDD2022_DATASET_ROOT if the dataset is stored somewhere else.
# Expected layout: <dataset root>/India/train/{images,annotations/xmls}
DATASET_ROOT = Path(os.environ.get(
    "RDD2022_DATASET_ROOT",
    str(BASE.parent / "RDD2022_India")
))
IMAGE_DIR = DATASET_ROOT / "India" / "train" / "images"
XML_DIR = DATASET_ROOT / "India" / "train" / "annotations" / "xmls"

if not CSV.exists():
    raise FileNotFoundError(f"Missing CSV: {CSV}")
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")
if not XML_DIR.is_dir():
    raise FileNotFoundError(f"Missing XML directory: {XML_DIR}")

df = pd.read_csv(CSV)
df["image_exists"] = (
    df["image_exists"].astype(str).str.lower().eq("true")
)
df = df.sort_values(
    ["cluster_id", "severity_score"],
    ascending=[True, False]
).groupby("cluster_id", group_keys=False).head(2)

if not df["image_exists"].all():
    raise ValueError("Some representative image files are missing.")

font = ImageFont.load_default()
audit_rows = []

def draw_annotated_image(filename):
    image_path = IMAGE_DIR / filename
    xml_path = XML_DIR / (Path(filename).stem + ".xml")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML annotation not found: {xml_path}")

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    root = ET.parse(xml_path).getroot()

    boxes_drawn = 0

    for obj in root.findall("object"):
        class_element = obj.find("name")
        box = obj.find("bndbox")

        if box is None:
            continue

        try:
            xmin = int(float(box.findtext("xmin")))
            ymin = int(float(box.findtext("ymin")))
            xmax = int(float(box.findtext("xmax")))
            ymax = int(float(box.findtext("ymax")))
        except (TypeError, ValueError):
            continue

        # Clip coordinates to the actual image dimensions.
        xmin = max(0, min(xmin, image.width - 1))
        xmax = max(0, min(xmax, image.width - 1))
        ymin = max(0, min(ymin, image.height - 1))
        ymax = max(0, min(ymax, image.height - 1))

        if xmax <= xmin or ymax <= ymin:
            continue

        damage_class = (
            class_element.text.strip()
            if class_element is not None and class_element.text
            else "damage"
        )

        color = "#00FF00"
        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline=color,
            width=max(2, image.width // 350)
        )

        label = damage_class
        text_box = draw.textbbox((xmin, ymin), label, font=font)
        label_height = text_box[3] - text_box[1]
        label_width = text_box[2] - text_box[0]

        label_y = max(0, ymin - label_height - 4)
        draw.rectangle(
            [xmin, label_y, xmin + label_width + 6,
             label_y + label_height + 4],
            fill=color
        )
        draw.text(
            (xmin + 3, label_y + 2),
            label,
            fill="black",
            font=font
        )
        boxes_drawn += 1

    return image, boxes_drawn, str(xml_path)

# Save individually annotated copies too.
annotated_dir = OUT / "representative_images_k15_annotated"
annotated_dir.mkdir(parents=True, exist_ok=True)

for part in range(3):
    cluster_ids = sorted(df["cluster_id"].unique())
    ids = cluster_ids[part * 5:(part + 1) * 5]

    if not ids:
        continue

    fig, axes = plt.subplots(
        len(ids), 2,
        figsize=(12, 3.6 * len(ids)),
        squeeze=False
    )

    for row_idx, cluster_id in enumerate(ids):
        group = df[df["cluster_id"] == cluster_id].head(2)

        for col_idx in range(2):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if col_idx >= len(group):
                continue

            item = group.iloc[col_idx]
            filename = item["filename"]

            image, box_count, xml_path = draw_annotated_image(filename)

            # Save the individual annotated image.
            individual_path = (
                annotated_dir /
                f"cluster_{cluster_id}_{Path(filename).stem}_boxed.png"
            )
            image.save(individual_path)

            audit_rows.append({
                "cluster_id": int(cluster_id),
                "filename": filename,
                "xml_path": xml_path,
                "boxes_drawn": box_count,
                "annotated_image": str(individual_path)
            })

            ax.imshow(image)
            ax.set_title(
                f"Cluster {cluster_id} | {filename}\n"
                f"Severity: {item['severity_score']:.3f} | "
                f"XML boxes: {box_count}",
                fontsize=9
            )

    fig.suptitle(
        f"K-Means K=15: XML-Annotated Representative Images "
        f"(Clusters {ids[0]}–{ids[-1]})",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = OUT / f"k15_representative_boxes_part{part + 1}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)

audit_path = OUT / "representative_bbox_audit_k15.csv"
pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

print("\nBounding-box audit:", audit_path)
print("Representative images processed:", len(audit_rows))
print("Total XML boxes drawn:", sum(r["boxes_drawn"] for r in audit_rows))
print("Individual boxed images:", annotated_dir)
