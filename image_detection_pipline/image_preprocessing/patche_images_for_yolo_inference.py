# -*- coding: utf-8 -*-
"""
Divides original images into overlapping patches, calculates image quality
metrics, and saves patch metadata for mapping detections back to the
original images. Optionally generates a filtered patch dataset.
"""


import csv
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# =============== USER CONFIG ===============
IN_DIR  = r""
OUT_DIR = r""
 # output root

TILE_SIZE = 640              # 640 or 1024 recommended
STRIDE    = TILE_SIZE // 2   # 50% overlap (good for small stones)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# ---- quality metrics (for analysis / optional filtered copy) ----
BLUR_VAR_MIN     = 20.0
INTENSITY_MIN    = 10.0
INTENSITY_MAX    = 245.0
EDGE_DENSITY_MIN = 0.005
GRAY_STD_MIN     = 3.0

KEEP_TOP_PCT       = 25.0   # only used if WRITE_FILTERED_COPY=True
MIN_KEEP_PER_IMAGE = 30     #    "
WRITE_FILTERED_COPY = False # True -> also write ...\tiles_filtered\ + tiles_filtered_index.csv
# ===========================================


# ---------- helpers ----------
def tile_positions_full(H, W, tile_size, stride):
    """Generate (x,y) starts covering full image; always add last row/col."""
    xs = list(range(0, max(W - tile_size, 0) + 1, stride))
    ys = list(range(0, max(H - tile_size, 0) + 1, stride))
    last_x = max(W - tile_size, 0)
    last_y = max(H - tile_size, 0)
    if not xs or xs[-1] != last_x: xs.append(last_x)
    if not ys or ys[-1] != last_y: ys.append(last_y)
    for y in ys:
        for x in xs:
            yield int(x), int(y)

def inner_window_for_tile(x, y, H, W, tile_size, stride):
    """
    Inner (dedup) window per tile.
    - Base margin m = (T - S)/2 each side.
    - Expand to edges at borders to avoid holes.
    If no overlap (S >= T), window == full tile.
    """
    if stride >= tile_size:
        return float(x), float(y), float(x + tile_size), float(y + tile_size)
    m = (tile_size - stride) / 2.0
    ix1, iy1 = x + m, y + m
    ix2, iy2 = x + tile_size - m, y + tile_size - m
    if x == 0: ix1 = x
    if y == 0: iy1 = y
    if x + tile_size >= W: ix2 = x + tile_size
    if y + tile_size >= H: iy2 = y + tile_size
    return float(ix1), float(iy1), float(ix2), float(iy2)

def compute_quality(tile_bgr):
    """Return (lap_var, mean_intensity, gray_std, edge_density)."""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_var = float(lap.var())
    mean_intensity = float(gray.mean())
    gray_std = float(gray.std())
    med = np.median(gray)
    low  = int(max(0,   0.66 * med))
    high = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, low, high)
    edge_density = float(edges.mean() / 255.0)
    return lap_var, mean_intensity, gray_std, edge_density

def texture_score(lap_var, edge_density, gray_std):
    # arbitrary but stable weighting for ranking
    return 0.6 * lap_var + 0.3 * (edge_density * 1000.0) + 0.1 * gray_std
# -------------------------------------------


def main():
    in_dir  = Path(IN_DIR)
    out_dir = Path(OUT_DIR)
    tiles_dir = out_dir / "tiles"                # FULL coverage tiles for inference
    tiles_dir.mkdir(parents=True, exist_ok=True)
    if WRITE_FILTERED_COPY:
        filt_dir = out_dir / "tiles_filtered"    # optional filtered copy
        filt_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not img_paths:
        print(f"[ERROR] No images found in {IN_DIR}")
        return

    # index for full tiles
    idx_csv = out_dir / "tiles_index.csv"
    widx = open(idx_csv, "w", newline="", encoding="utf-8")
    idx_writer = csv.writer(widx)
    idx_writer.writerow([
        "tile_path", "tile_name", "src_image",
        "src_W", "src_H",
        "x", "y", "w", "h",
        "inner_x1", "inner_y1", "inner_x2", "inner_y2",
        "lap_var", "mean_intensity", "edge_density", "gray_std", "score"
    ])

    # optional index for filtered copy
    if WRITE_FILTERED_COPY:
        fidx_csv = out_dir / "tiles_filtered_index.csv"
        wfidx = open(fidx_csv, "w", newline="", encoding="utf-8")
        fidx_writer = csv.writer(wfidx)
        fidx_writer.writerow([
            "tile_path", "tile_name", "src_image",
            "src_W", "src_H",
            "x", "y", "w", "h",
            "inner_x1", "inner_y1", "inner_x2", "inner_y2",
            "lap_var", "mean_intensity", "edge_density", "gray_std", "score"
        ])
    else:
        wfidx = fidx_writer = None

    total_written = 0
    total_filtered = 0

    for img_path in tqdm(img_paths, desc="Tiling full coverage"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        H, W = img.shape[:2]
        pos = list(tile_positions_full(H, W, TILE_SIZE, STRIDE))

        # pass 1: compute metrics for ranking (but we will still write ALL tiles to tiles_dir)
        per_tile = []
        for (x, y) in pos:
            crop = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if crop.shape[0] != TILE_SIZE or crop.shape[1] != TILE_SIZE:
                pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=img.dtype)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            lap_var, mean_intensity, gray_std, edge_density = compute_quality(crop)
            score = texture_score(lap_var, edge_density, gray_std)
            per_tile.append((x, y, crop, lap_var, mean_intensity, edge_density, gray_std, score))

        # write ALL tiles for inference (+ index with metrics)
        for (x, y, crop, lap_var, mean_intensity, edge_density, gray_std, score) in per_tile:
            tile_name = f"{img_path.stem}_x{x}_y{y}.jpg"
            save_path = tiles_dir / tile_name
            cv2.imwrite(str(save_path), crop)
            ix1, iy1, ix2, iy2 = inner_window_for_tile(x, y, H, W, TILE_SIZE, STRIDE)
            idx_writer.writerow([
                str(save_path), tile_name, str(img_path),
                W, H,
                x, y, TILE_SIZE, TILE_SIZE,
                round(ix1,2), round(iy1,2), round(ix2,2), round(iy2,2),
                round(lap_var,4), round(mean_intensity,4), round(edge_density,6), round(gray_std,4), round(score,4)
            ])
            total_written += 1

        # optional filtered copy (does NOT affect the full-coverage set)
        if WRITE_FILTERED_COPY:
            # hard filters
            ok = [t for t in per_tile
                  if (t[3] >= BLUR_VAR_MIN and
                      INTENSITY_MIN <= t[4] <= INTENSITY_MAX and
                      t[5] >= EDGE_DENSITY_MIN and
                      t[6] >= GRAY_STD_MIN)]
            # percentile threshold
            if ok:
                scores = np.array([t[7] for t in ok], dtype=float)
                pct = np.percentile(scores, 100.0 - KEEP_TOP_PCT)
                # enforce minimum kept per image
                keep = [t for t in ok if t[7] >= pct]
                if len(keep) < MIN_KEEP_PER_IMAGE:
                    # relax threshold a bit
                    pct2 = np.percentile(scores, max(0.0, 100.0 - (KEEP_TOP_PCT + 10.0)))
                    keep = [t for t in ok if t[7] >= pct2]
            else:
                keep = []

            for (x, y, crop, lap_var, mean_intensity, edge_density, gray_std, score) in keep:
                tile_name = f"{img_path.stem}_x{x}_y{y}.jpg"
                save_path = filt_dir / tile_name
                cv2.imwrite(str(save_path), crop)
                ix1, iy1, ix2, iy2 = inner_window_for_tile(x, y, H, W, TILE_SIZE, STRIDE)
                fidx_writer.writerow([
                    str(save_path), tile_name, str(img_path),
                    W, H,
                    x, y, TILE_SIZE, TILE_SIZE,
                    round(ix1,2), round(iy1,2), round(ix2,2), round(iy2,2),
                    round(lap_var,4), round(mean_intensity,4), round(edge_density,6), round(gray_std,4), round(score,4)
                ])
                total_filtered += 1

    widx.close()
    if wfidx: wfidx.close()

    print("\n[SUMMARY]")
    print(f"- Full-coverage tiles written: {total_written} → {OUT_DIR}\\tiles")
    print(f"- Index CSV: {OUT_DIR}\\tiles_index.csv")
    if WRITE_FILTERED_COPY:
        print(f"- Filtered copy written: {total_filtered} → {OUT_DIR}\\tiles_filtered")
        print(f"- Filtered index: {OUT_DIR}\\tiles_filtered_index.csv")


if __name__ == "__main__":
    main()
