"""
Divides original images into overlapping patches, evaluates image
quality and texture metrics, and retains the highest-scoring patches
for subsequent stone detection tasks.
"""

import os
import cv2
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= USER SETTINGS =================
IN_DIR  = r"C:\Users\ammar\Downloads\only_ones"
OUT_DIR = r"C:\Users\ammar\Downloads\only_ones/Nylenda_colliction_out"

TILE_SIZE = 640           # tile size (YOLO friendly: 512 or 640)
STRIDE = TILE_SIZE // 2   # 50% overlap
# STRIDE = TILE_SIZE      # uncomment for no overlap

# Quality thresholds (light filtering)
BLUR_VAR_MIN = 40.0       # lower blur cutoff
INTENSITY_MIN = 15.0      # drop very dark tiles
INTENSITY_MAX = 240.0     # drop very bright tiles
EDGE_DENSITY_MIN = 0.015  # drop tiles with extremely low edge activity
GRAY_STD_MIN = 5.0        # drop tiles with almost no local contrast

# Keep top X% based on texture score
KEEP_TOP_PCT = 25.0       # 25% means keep the best quarter of tiles
MIN_KEEP = 30             # always keep at least this many per source image

# Optional entropy filter
USE_ENTROPY = False
ENTROPY_RADIUS = 5
ENTROPY_MIN_MEAN = 2.0

# =================================================

if USE_ENTROPY:
    try:
        from skimage.filters.rank import entropy as sk_entropy
        from skimage.morphology import disk
        _ENTROPY_OK = True
    except Exception:
        print("[WARN] scikit-image not installed, disabling entropy filtering.")
        USE_ENTROPY = False
        _ENTROPY_OK = False

# ------------------- FUNCTIONS -------------------

def tile_image(img, tile_size, stride=None):
    if stride is None:
        stride = tile_size
    H, W = img.shape[:2]
    tiles, positions = [], []
    for y in range(0, max(H - tile_size, 0) + 1, stride):
        for x in range(0, max(W - tile_size, 0) + 1, stride):
            crop = img[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] == tile_size and crop.shape[1] == tile_size:
                tiles.append(crop)
                positions.append((x, y))
    return tiles, positions

def compute_quality_metrics(tile_bgr):
    """Return: lap_var, mean_intensity, gray_std, edge_density, gray"""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness / blur
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_var = float(lap.var())

    # Mean intensity
    mean_intensity = float(gray.mean())

    # Local contrast
    gray_std = float(gray.std())

    # Edge density
    med = np.median(gray)
    low = int(max(0, 0.66 * med))
    high = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, low, high)
    edge_density = float(edges.mean() / 255.0)

    return lap_var, mean_intensity, gray_std, edge_density, gray

def texture_score(lap_var, edge_density, gray_std):
    """Weighted texture score — higher is better (stones likely)."""
    return 0.6 * lap_var + 0.3 * (edge_density * 1000.0) + 0.1 * gray_std

def auto_threshold(scores, keep_top_percent=25.0, min_keep=30):
    if len(scores) == 0:
        return float("inf")
    pct = np.percentile(scores, 100.0 - keep_top_percent)
    if sum(s >= pct for s in scores) < min_keep:
        pct = np.percentile(scores, max(0.0, 100.0 - (keep_top_percent + 10.0)))
    return float(pct)

# ------------------- MAIN SCRIPT -------------------

def main():
    in_dir = Path(IN_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tiles_dir = out_dir / "all_tiles"
    all_tiles_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "tiles_index.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tile_path", "src_image", "x", "y",
            "lap_var", "mean_intensity", "edge_density", "gray_std", "score"
        ])

        total_tiles, total_kept = 0, 0

        img_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]])
        if not img_paths:
            print(f"[ERROR] No images found in {IN_DIR}")
            return

        for img_path in tqdm(img_paths, desc="Processing images"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Could not read image {img_path}")
                continue

            tiles, positions = tile_image(img, TILE_SIZE, STRIDE)
            metrics = []

            # Pass 1: compute metrics and drop hard fails
            for tile in tiles:
                lap_var, mean_intensity, gray_std, edge_density, gray = compute_quality_metrics(tile)

                # Blur filter
                if lap_var < BLUR_VAR_MIN:
                    metrics.append((None,))
                    continue

                # Dark/bright filter
                if mean_intensity < INTENSITY_MIN or mean_intensity > INTENSITY_MAX:
                    metrics.append((None,))
                    continue

                # Edge density filter
                if edge_density < EDGE_DENSITY_MIN:
                    metrics.append((None,))
                    continue

                # Local contrast filter
                if gray_std < GRAY_STD_MIN:
                    metrics.append((None,))
                    continue

                # Optional entropy filter
                if USE_ENTROPY:
                    g = gray.astype(np.uint8, copy=False)
                    ent = sk_entropy(g, disk(ENTROPY_RADIUS))
                    if float(ent.mean()) < ENTROPY_MIN_MEAN:
                        metrics.append((None,))
                        continue

                score = texture_score(lap_var, edge_density, gray_std)
                metrics.append((lap_var, mean_intensity, edge_density, gray_std, score))

            # Keep top X% scoring tiles
            scores = [m[4] for m in metrics if m is not None and len(m) == 5]
            thresh = auto_threshold(scores, KEEP_TOP_PCT, MIN_KEEP) if scores else float("inf")

            for tile, (x, y), m in zip(tiles, positions, metrics):
                total_tiles += 1
                if m is None or len(m) != 5:
                    continue
                lap_var, mean_intensity, edge_density, gray_std, score = m
                if score >= thresh:
                    fname = f"{img_path.stem}_x{x}_y{y}_s{int(score)}.jpg"
                    save_path = all_tiles_dir / fname
                    cv2.imwrite(str(save_path), tile)
                    writer.writerow([save_path, img_path, x, y, lap_var, mean_intensity, edge_density, gray_std, score])
                    total_kept += 1

        print(f"\nSummary:")
        print(f"- Total tiles processed: {total_tiles}")
        print(f"- Tiles kept after filtering: {total_kept} ({100*total_kept/max(1,total_tiles):.1f}%)")
        print(f"- Saved to: {all_tiles_dir}")
        print(f"- CSV log: {csv_path}")

if __name__ == "__main__":
    main()
