# -*- coding: utf-8 -*-
"""

YOLO prediction conversion for CVAT-assisted annotation.

Converts YOLO prediction labels into CVAT XML format,
allowing model-generated stone detections to be visually
reviewed and manually corrected before subsequent training.


Converts YOLO bounding-box labels into CVAT 1.1 XML format,
preserving image metadata and class information. Also generates
a ZIP archive for importing annotations into CVAT.
"""

from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import zipfile


IMG_DIR = Path(r"")                           # original images (what upload to CVAT)
LABELS_DIR = Path(r"")  # folder with *.txt (1 per image), use your labels dir
OUT_XML = Path(r"")           # CVAT 1.1 xml to create
OUT_ZIP = Path(r"")      # optional: zip containing annotations.xml
CLASSES_TXT = None  
DEFAULT_CLASSES = ["stone"]  # used if CLASSES_TXT is None or missing
# ======================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def load_classes():
    if CLASSES_TXT and Path(CLASSES_TXT).exists():
        lines = Path(CLASSES_TXT).read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]
    return DEFAULT_CLASSES

def yolo_to_xyxy(cx, cy, w, h, W, H):
    x_c = float(cx) * W
    y_c = float(cy) * H
    bw = float(w) * W
    bh = float(h) * H
    xtl = max(0.0, x_c - bw / 2)
    ytl = max(0.0, y_c - bh / 2)
    xbr = min(float(W), x_c + bw / 2)
    ybr = min(float(H), y_c + bh / 2)
    # round to 2 decimals (CVAT accepts floats)
    return round(xtl, 2), round(ytl, 2), round(xbr, 2), round(ybr, 2)

def main():
    classes = load_classes()
    class_map = {str(i): name for i, name in enumerate(classes)}

    # Collect images (recursive, case-insensitive)
    imgs = [p for p in IMG_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda p: p.as_posix().lower())

    # Build XML skeleton
    ann = ET.Element("annotations")
    ET.SubElement(ann, "version").text = "1.1"

    meta = ET.SubElement(ann, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    for name in classes:
        lbl = ET.SubElement(labels, "label")
        ET.SubElement(lbl, "name").text = name
        ET.SubElement(lbl, "attributes")  # empty

    # Per-image entries
    img_id = 0
    box_id = 0
    missing_image_sizes = 0

    for img_path in imgs:
        # path/name in XML should be relative for clarity
        try:
            rel = img_path.relative_to(IMG_DIR).as_posix()
        except Exception:
            rel = img_path.name

        # get size
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            missing_image_sizes += 1
            continue

        img_el = ET.SubElement(ann, "image", {
            "id": str(img_id),
            "name": rel,
            "width": str(W),
            "height": str(H),
        })

        # matching label file by basename
        lbl_path = LABELS_DIR / (img_path.stem + ".txt")
        if lbl_path.exists():
            try:
                lines = lbl_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = lbl_path.read_text(encoding="latin1").splitlines()

            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue  # skip malformed
                cls_id, cx, cy, w, h = parts[:5]
                label_name = class_map.get(cls_id, classes[0])  # fallback to first class
                xtl, ytl, xbr, ybr = yolo_to_xyxy(cx, cy, w, h, W, H)

                ET.SubElement(img_el, "box", {
                    "label": label_name,
                    "source": "auto",
                    "occluded": "0",
                    "xtl": str(xtl),
                    "ytl": str(ytl),
                    "xbr": str(xbr),
                    "ybr": str(ybr),
                    "z_order": "0",
                })
                box_id += 1

        img_id += 1

    # Write XML (pretty-ish)
    tree = ET.ElementTree(ann)
    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ", level=0)  # Python 3.9+
    tree.write(OUT_XML, encoding="utf-8", xml_declaration=True)

    # Also provide a zip with annotations.xml at root (some UIs prefer zip)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_XML, arcname="annotations.xml")

    print(f"Images processed: {len(imgs)}")
    print(f"Annotations written: {OUT_XML}")
    print(f"ZIP written: {OUT_ZIP}")
    if missing_image_sizes:
        print(f"WARNING: {missing_image_sizes} image(s) could not be opened for size; they were skipped.")

if __name__ == "__main__":
    main()
