# -*- coding: utf-8 -*-
"""
Stitch YOLO tile predictions back to original images,
drop duplicates using the INNER-WINDOW rule,
(optionally) merge, make overlays, and WRITE LABELS (TXT + CSV).

Assumes you created tiles with the preprocessor that wrote tiles_index.csv
including inner_x1..inner_y2.
"""

import os, glob, csv
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# ============== CONFIG ==============
# Where your original big images live (for overlays)
ORIG_DIR   = r"C:\Users\ammar\Downloads\Nybrodt_1_RGB-TEST\selected"

# tiles_index.csv from the preprocessor (has offsets + inner windows)
TILES_INDEX = r"C:\Users\ammar\Downloads\Nybrodt_1_RGB-TEST\selected\out\tiles_index.csv"

# Ultralytics output for the new run
PRED_DIR   = r"C:\Users\ammar\runs\detect\predict3"
LABEL_DIR  = os.path.join(PRED_DIR, "labels")  # usual place for *.txt

# Outputs (detections list per box)
# Outputs (detections list per box)
OUT_CSV    = r"C:\Users\ammar\runs\detect\detections_in_original_coords_inner_filtered.csv"
OUT_OVER   = r"C:\Users\ammar\runs\detect\overlay_inner_filtered"  # image overlays

# Optional post-merge across tiles (usually not needed after inner-window)
RUN_MERGE  = False    # set True to enable
MERGE_IOU  = 0.55

# --- NEW: label export config ---
OUT_LABELS_DIR = os.path.join(OUT_OVER, "labels")  # per-image YOLO txts
OUT_LABELS_CSV = os.path.join(PRED_DIR, "labels_in_original_coords.csv")  # combined CSV
MIN_CONF_FOR_LABEL = 0.0        # filter low-confidence if desired
INCLUDE_CONF_IN_LABELS = False  # True → write "class cx cy w h conf" in txt files
# ===================================


def read_tiles_index(csv_path):
    """
    Returns a map: crop_basename -> dict with
      src_image (basename), x, y, w, h, inner_x1..inner_y2 (floats)
    """
    df = pd.read_csv(csv_path)
    rows = {}
    for _, r in df.iterrows():
        crop = os.path.basename(str(r["tile_path"]))
        rows[crop] = dict(
            src_image=os.path.basename(str(r["src_image"])),
            x=int(r["x"]), y=int(r["y"]),
            w=int(r["w"]), h=int(r["h"]),
            ix1=float(r["inner_x1"]), iy1=float(r["inner_y1"]),
            ix2=float(r["inner_x2"]), iy2=float(r["inner_y2"]),
        )
    return rows


def get_img_size(img_path):
    with Image.open(img_path) as im:
        return im.size  # (W,H)


def parse_yolo_line(parts):
    """
    Ultralytics txt (pred): class x y w h conf (normalized)
    We'll be robust: if values look out of [0,1], try cls conf cx cy w h
    Returns: cls, cx, cy, w, h, conf
    """
    vals = list(map(float, parts))
    if len(vals) < 5:
        return None
    if len(vals) == 5:
        cls, cx, cy, w, h = vals[:5]
        conf = 1.0
    else:
        cls, cx, cy, w, h, conf = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]

    # sanity: if not normalized, fallback to "cls conf cx cy w h"
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        if len(vals) >= 6:
            cls, cx, cy, w, h, conf = vals[0], vals[2], vals[3], vals[4], vals[5], vals[1]
        else:
            return None
    return int(cls), float(cx), float(cy), float(w), float(h), float(conf)


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ub = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    denom = ua + ub - inter
    return inter/denom if denom > 0 else 0.0


def nms_xyxy(boxes, scores, iou_thr=0.55):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        keep_mask = []
        for j in rest:
            if iou_xyxy(boxes[i], boxes[j]) < iou_thr:
                keep_mask.append(j)
        idxs = np.array(keep_mask, dtype=int)
    return keep


def main():
    # --- load tile index
    index = read_tiles_index(TILES_INDEX)

    # --- collect label files
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    if not label_files:
        print(f"[ERROR] No label files in {LABEL_DIR}")
        return

    rows = []

    # --- stitch with inner-window filtering
    for lp in label_files:
        stem = os.path.splitext(os.path.basename(lp))[0]
        # find the tile image basename that matches this stem (any extension)
        # we only use the basename key stored in the index
        candidates = [k for k in index.keys() if os.path.splitext(k)[0] == stem]
        if not candidates:
            # Sometimes Ultralytics rewrites filenames; you can add more matching here if needed
            continue
        crop_name = candidates[0]
        info = index[crop_name]
        x_off, y_off = info["x"], info["y"]
        ix1, iy1, ix2, iy2 = info["ix1"], info["iy1"], info["ix2"], info["iy2"]
        orig_img = info["src_image"]

        # Need crop size to de-normalize (use the saved tile itself)
        possible_img_paths = [
            os.path.join(PRED_DIR, crop_name),
            os.path.join(PRED_DIR, "images", crop_name)
        ]
        crop_size = None
        for p in possible_img_paths:
            if os.path.isfile(p):
                crop_size = get_img_size(p)
                break
        # Fallback to TILE_SIZE from index
        if crop_size is None:
            crop_size = (info["w"], info["h"])
        W, H = crop_size

        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                parsed = parse_yolo_line(parts)
                if parsed is None:
                    continue
                cls, cx, cy, w, h, conf = parsed

                # de-normalize to crop pixels
                cx_px, cy_px = cx * W, cy * H
                bw_px, bh_px = w * W, h * H
                x1_c = cx_px - bw_px/2; y1_c = cy_px - bh_px/2
                x2_c = cx_px + bw_px/2; y2_c = cy_px + bh_px/2

                # project to original coords
                x1 = x_off + x1_c; y1 = y_off + y1_c
                x2 = x_off + x2_c; y2 = y_off + y2_c
                x_center = x_off + cx_px; y_center = y_off + cy_px

                # ---- INNER WINDOW FILTER (critical)
                if not (ix1 <= x_center < ix2 and iy1 <= y_center < iy2):
                    continue

                rows.append([
                    orig_img, crop_name, cls, conf,
                    round(x1,2), round(y1,2), round(x2,2), round(y2,2),
                    round(x_center,2), round(y_center,2), W, H
                ])

    # save filtered detections
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow([
            "original_image","crop_name","class","confidence",
            "x1","y1","x2","y2","x_center","y_center","crop_width","crop_height"
        ])
        w.writerows(rows)
    print(f"✅ Wrote inner-window filtered detections: {OUT_CSV}")
    print(f"   Total kept: {len(rows)}")

    # --- optional merge across tiles (usually small effect now)
    merged_df = None
    if RUN_MERGE:
        df = pd.DataFrame(rows, columns=[
            "original_image","crop_name","class","confidence",
            "x1","y1","x2","y2","x_center","y_center","crop_width","crop_height"
        ])
        merged_rows = []
        for (img, cls), g in df.groupby(["original_image","class"]):
            boxes = g[["x1","y1","x2","y2"]].to_numpy(float)
            scores = g["confidence"].to_numpy(float)
            if len(boxes) == 0:
                continue
            keep = nms_xyxy(boxes, scores, iou_thr=MERGE_IOU)
            for i in keep:
                x1,y1,x2,y2 = boxes[i]
                merged_rows.append([img, int(cls), float(scores[i]),
                                    round(x1,2), round(y1,2), round(x2,2), round(y2,2)])
        merged_df = pd.DataFrame(merged_rows, columns=[
            "original_image","class","confidence","x1","y1","x2","y2"
        ])
        mpath = os.path.join(PRED_DIR, "merged_after_inner_window.csv")
        merged_df.to_csv(mpath, index=False)
        print(f"🧩 Optional merge wrote: {mpath} (rows: {len(merged_df)})")

    # --- overlay + WRITE LABELS (TXT + CSV) ---
    os.makedirs(OUT_OVER, exist_ok=True)
    os.makedirs(OUT_LABELS_DIR, exist_ok=True)

    overlay_df = merged_df if RUN_MERGE else pd.read_csv(OUT_CSV)

    # collect rows for one combined CSV
    all_label_rows = []  # each: [original_image, class, confidence, x1,y1,x2,y2, cx_norm,cy_norm,w_norm,h_norm]

    for img_name, group in overlay_df.groupby("original_image"):
        full_path = os.path.join(ORIG_DIR, img_name)
        if not os.path.isfile(full_path):
            # if tiles_index stored absolute paths originally, try using that
            full_path = img_name if os.path.isfile(img_name) else full_path
        if not os.path.isfile(full_path):
            print(f"⚠️ Missing original: {img_name}")
            continue

        img = cv2.imread(full_path)
        if img is None:
            print(f"⚠️ Could not read: {img_name}")
            continue
        H, W = img.shape[:2]

        # build lines for YOLO txt and draw overlays
        label_lines = []

        # Use iterrows in both branches to avoid 'class' as attribute name issues
        if RUN_MERGE:
            # merged_df columns: original_image, class, confidence, x1,y1,x2,y2
            for _, r in group.iterrows():
                cls = int(r["class"])
                conf = float(r["confidence"])
                if conf < MIN_CONF_FOR_LABEL:
                    continue
                x1 = float(np.clip(r["x1"], 0, W - 1)); y1 = float(np.clip(r["y1"], 0, H - 1))
                x2 = float(np.clip(r["x2"], 0, W - 1)); y2 = float(np.clip(r["y2"], 0, H - 1))
                bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
                if bw <= 0 or bh <= 0:
                    continue

                # draw
                color = (0, 255, 0) if conf >= 0.5 else (0, 0, 255)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, f"{cls}:{conf:.2f}",
                            (int(x1), max(0, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # normalized YOLO
                cx_n = ((x1 + x2) / 2.0) / W
                cy_n = ((y1 + y2) / 2.0) / H
                bw_n = bw / W
                bh_n = bh / H
                if INCLUDE_CONF_IN_LABELS:
                    label_lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f} {conf:.6f}")
                else:
                    label_lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

                # for combined CSV (store both absolute + normalized)
                all_label_rows.append([img_name, cls, conf,
                                       round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                                       round(cx_n, 6), round(cy_n, 6), round(bw_n, 6), round(bh_n, 6)])
        else:
            # non-merged CSV columns: original_image, crop_name, class, confidence, x1,y1,x2,y2, ...
            for _, r in group.iterrows():
                cls = int(r["class"]) if "class" in r else -1
                conf = float(r["confidence"])
                if conf < MIN_CONF_FOR_LABEL:
                    continue
                x1 = float(np.clip(r["x1"], 0, W - 1)); y1 = float(np.clip(r["y1"], 0, H - 1))
                x2 = float(np.clip(r["x2"], 0, W - 1)); y2 = float(np.clip(r["y2"], 0, H - 1))
                bw = max(0.0, x2 - x1); bh = max(0.0, y2 - y1)
                if bw <= 0 or bh <= 0:
                    continue

                # draw
                color = (0, 255, 0) if conf >= 0.5 else (0, 0, 255)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, f"{cls}:{conf:.2f}",
                            (int(x1), max(0, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # normalized YOLO
                cx_n = ((x1 + x2) / 2.0) / W
                cy_n = ((y1 + y2) / 2.0) / H
                bw_n = bw / W
                bh_n = bh / H
                if INCLUDE_CONF_IN_LABELS:
                    label_lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f} {conf:.6f}")
                else:
                    label_lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

                # for combined CSV
                all_label_rows.append([img_name, cls, conf,
                                       round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                                       round(cx_n, 6), round(cy_n, 6), round(bw_n, 6), round(bh_n, 6)])

        # write overlay image
        out_path = os.path.join(OUT_OVER, f"overlay_{os.path.basename(img_name)}")
        cv2.imwrite(out_path, img)
        print(f"🟢 Saved overlay: {out_path}")

        # write YOLO label file (normalized to original size)
        stem = os.path.splitext(os.path.basename(img_name))[0]
        out_lab_path = os.path.join(OUT_LABELS_DIR, f"{stem}.txt")
        with open(out_lab_path, "w", encoding="utf-8") as lf:
            for line in label_lines:
                lf.write(line + "\n")
        print(f"🏷️  Saved labels: {out_lab_path} (objects: {len(label_lines)})")

    # after the loop, dump the combined CSV
    with open(OUT_LABELS_CSV, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["original_image","class","confidence",
                    "x1","y1","x2","y2",              # absolute pixels in original image
                    "cx_norm","cy_norm","w_norm","h_norm"])  # normalized to original W,H
        w.writerows(all_label_rows)

    print(f"📄 Wrote combined labels CSV: {OUT_LABELS_CSV} (rows: {len(all_label_rows)})")


if __name__ == "__main__":
    main()
