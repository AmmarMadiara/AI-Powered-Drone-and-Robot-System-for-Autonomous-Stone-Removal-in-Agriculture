# -*- coding: utf-8 -*-
"""
YOLO dataset preparation and splitting.

Combines positive and optional negative images, splits the dataset
into training, validation, and testing subsets, and generates
the corresponding YOLO data.yaml configuration file.
"""

from pathlib import Path
import shutil
import random
import yaml
import sys

# ============== CONFIG ==============
POS_ROOT = Path(r"C:\YOLO_Traning_smaller\batches")  # contains images/, labels/, data.yaml
NEG_IMAGES_DIR = Path(r"C:\YOLO_Traning\TheNewNegativ")
OUTPUT_DIR = Path(r"C:\YOLO_Traning_smaller\Model768")

# Split ratios (must sum to 1.0)
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

# How many negatives to include:
# Option A (recommended): fraction relative to number of POSITIVE images
NEGATIVE_FRACTION =0  # e.g., 0.25 = 25% as many negatives as positives

# Option B: absolute cap (set to None to disable)
NEGATIVE_MAX = None  # e.g., 400

# Random seed for reproducibility
SEED = 42
# ====================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if str(src.resolve()) != str(dst.resolve()):
        shutil.copy2(src, dst)

def load_class_names(pos_root: Path):
    names = ["class0"]
    dy = pos_root / "data.yaml"
    if dy.exists():
        try:
            y = yaml.safe_load(dy.read_text(encoding="utf-8"))
            if isinstance(y, dict) and "names" in y:
                if isinstance(y["names"], list) and y["names"]:
                    names = y["names"]
                elif isinstance(y["names"], dict):
                    # id -> name map
                    keys = sorted(y["names"].keys(), key=lambda k: int(k))
                    names = [y["names"][k] for k in keys]
        except Exception:
            pass
    return names

def main():
    random.seed(SEED)
    # sanity checks
    pos_img_dir = POS_ROOT / "images"
    pos_lbl_dir = POS_ROOT / "labels"
    if not pos_img_dir.exists() or not pos_lbl_dir.exists():
        print("[ERROR] POS_ROOT must contain images/ and labels/:", POS_ROOT)
        sys.exit(1)
    if not NEG_IMAGES_DIR.exists():
        print("[ERROR] NEG_IMAGES_DIR not found:", NEG_IMAGES_DIR)
        sys.exit(1)

    # gather positives (image + label expected)
    pos_images = list_images(pos_img_dir)
    pos_pairs = []
    for img in pos_images:
        lbl = pos_lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pos_pairs.append((img, lbl))
    if not pos_pairs:
        print("[ERROR] Found no positive (image,label) pairs in:", POS_ROOT)
        sys.exit(1)

    # gather negatives
    neg_images = list_images(NEG_IMAGES_DIR)
    if not neg_images:
        print("[WARN] No negative images found in:", NEG_IMAGES_DIR)

    # decide how many negatives to include
    target_neg = int(len(pos_pairs) * NEGATIVE_FRACTION)
    if NEGATIVE_MAX is not None:
        target_neg = min(target_neg, int(NEGATIVE_MAX))
    target_neg = min(target_neg, len(neg_images))
    random.shuffle(neg_images)
    neg_images = neg_images[:target_neg]

    # split positives and negatives separately to keep similar distribution
    def split_list(items):
        n = len(items)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        n_test = n - n_train - n_val
        return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

    random.shuffle(pos_pairs)
    pos_train, pos_val, pos_test = split_list(pos_pairs)

    random.shuffle(neg_images)
    neg_train, neg_val, neg_test = split_list(neg_images)

    # prepare output dirs
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_pos(items, split):
        cnt = 0
        for img, lbl in items:
            dst_img = OUTPUT_DIR / "images" / split / img.name
            dst_lbl = OUTPUT_DIR / "labels" / split / (img.stem + ".txt")
            safe_copy(img, dst_img)
            safe_copy(lbl, dst_lbl)
            cnt += 1
        return cnt

    def copy_neg(items, split):
        cnt = 0
        for img in items:
            dst_img = OUTPUT_DIR / "images" / split / img.name
            dst_lbl = OUTPUT_DIR / "labels" / split / (img.stem + ".txt")
            # If a filename collision happens (rare), prefix with "neg_"
            if dst_img.exists():
                dst_img = dst_img.with_name("neg_" + dst_img.name)
                dst_lbl = dst_lbl.with_name("neg_" + dst_lbl.name)
            safe_copy(img, dst_img)
            # create an empty label file to mark as negative
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("", encoding="utf-8")
            cnt += 1
        return cnt

    stats = {}
    stats["pos_train"] = copy_pos(pos_train, "train")
    stats["pos_val"]   = copy_pos(pos_val, "val")
    stats["pos_test"]  = copy_pos(pos_test, "test")

    stats["neg_train"] = copy_neg(neg_train, "train")
    stats["neg_val"]   = copy_neg(neg_val, "val")
    stats["neg_test"]  = copy_neg(neg_test, "test")

    # write data.yaml
    names = load_class_names(POS_ROOT)
    data = {
        "names": names,
        "nc": len(names),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    (OUTPUT_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # print summary
    print("\n===== SUMMARY =====")
    print("Output dataset:", OUTPUT_DIR)
    print("Split ratios   :", SPLIT_RATIOS)
    print(f"Positives total: {len(pos_pairs)}")
    print(f"Negatives used : {len(neg_images)} (from {NEG_IMAGES_DIR})")
    print("--- By split ---")
    print(f"train: +{stats['pos_train']}  -{stats['neg_train']}")
    print(f"val  : +{stats['pos_val']}   -{stats['neg_val']}")
    print(f"test : +{stats['pos_test']}  -{stats['neg_test']}")
    print("Wrote data.yaml with train/val/test.")
    print("===================\n")

if __name__ == "__main__":
    main()
