"""
HemoLens — Training Photo Lighting & Quality Analysis

Analyzes the raw training photos + nail crops to extract lighting stats
(brightness, contrast, color temperature, white balance, saturation, etc.)
that characterise the training distribution.

Produces a summary report with recommendations for end-users.

Usage:
    python analyze_training_photos.py
"""

import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW_DIR = Path("../data/raw")
PHOTO_DIR = RAW_DIR / "photo"
CROP_DIR = Path("../data/processed/nail_crops")
METADATA_CSV = RAW_DIR / "metadata.csv"


def analyze_image(img_bgr: np.ndarray) -> dict:
    """Extract lighting & quality features from an image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape

    return {
        # Overall brightness
        "luma_mean": gray.mean(),
        "luma_std": gray.std(),
        "luma_median": np.median(gray),
        "luma_p5": np.percentile(gray, 5),
        "luma_p95": np.percentile(gray, 95),

        # Dynamic range
        "luma_range": np.percentile(gray, 95) - np.percentile(gray, 5),

        # Fraction of very dark / very bright pixels
        "black_frac": (gray < 20).mean(),
        "white_frac": (gray > 235).mean(),

        # RGB channel means (white-balance proxy)
        "R_mean": rgb[:, :, 0].mean(),
        "G_mean": rgb[:, :, 1].mean(),
        "B_mean": rgb[:, :, 2].mean(),

        # Color temperature proxy: R/B ratio
        "RB_ratio": rgb[:, :, 0].mean() / max(rgb[:, :, 2].mean(), 1),

        # Saturation
        "sat_mean": hsv[:, :, 1].mean(),
        "sat_std": hsv[:, :, 1].std(),

        # LAB lightness
        "lab_L_mean": lab[:, :, 0].mean(),
        "lab_L_std": lab[:, :, 0].std(),

        # LAB a* (red-green axis) — key for hemoglobin
        "lab_a_mean": lab[:, :, 1].mean(),
        "lab_a_std": lab[:, :, 1].std(),

        # LAB b* (yellow-blue axis)
        "lab_b_mean": lab[:, :, 2].mean(),
        "lab_b_std": lab[:, :, 2].std(),

        # Hue (dominant color)
        "hue_mean": hsv[:, :, 0].mean(),
        "hue_std": hsv[:, :, 0].std(),

        # Value channel
        "value_mean": hsv[:, :, 2].mean(),
        "value_std": hsv[:, :, 2].std(),

        # Image dimensions
        "width": w,
        "height": h,
    }


def print_stat_table(df, columns, title):
    """Print a nicely formatted stats table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Feature':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'P5':>8} {'Median':>8} {'P95':>8} {'Max':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for c in columns:
        vals = df[c]
        print(f"  {c:<20} {vals.mean():>8.1f} {vals.std():>8.1f} {vals.min():>8.1f} "
              f"{vals.quantile(0.05):>8.1f} {vals.median():>8.1f} "
              f"{vals.quantile(0.95):>8.1f} {vals.max():>8.1f}")


def main():
    # --- 1) Analyze full raw photos ---
    print("Analyzing raw training photos...")
    photo_stats = []
    for img_path in sorted(PHOTO_DIR.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: could not read {img_path.name}")
            continue
        stats = analyze_image(img)
        stats["filename"] = img_path.name
        stats["patient_id"] = img_path.stem
        photo_stats.append(stats)

    df_photos = pd.DataFrame(photo_stats)
    print(f"  Analyzed {len(df_photos)} raw photos")

    # --- 2) Analyze nail crops ---
    print("Analyzing nail crops...")
    crop_stats = []
    for img_path in sorted(CROP_DIR.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stats = analyze_image(img)
        stats["filename"] = img_path.name
        # extract patient ID from "123_nail0.jpg"
        stats["patient_id"] = img_path.stem.split("_")[0]
        crop_stats.append(stats)

    df_crops = pd.DataFrame(crop_stats)
    print(f"  Analyzed {len(df_crops)} nail crops")

    # --- 3) Summary stats for raw photos ---
    print_stat_table(df_photos,
        ["luma_mean", "luma_std", "luma_range", "black_frac", "white_frac",
         "R_mean", "G_mean", "B_mean", "RB_ratio",
         "sat_mean", "hue_mean", "width", "height"],
        "RAW PHOTO STATISTICS (full images)")

    # --- 4) Summary stats for nail crops ---
    print_stat_table(df_crops,
        ["luma_mean", "luma_std", "luma_range", "black_frac", "white_frac",
         "R_mean", "G_mean", "B_mean", "RB_ratio",
         "sat_mean", "sat_std", "hue_mean", "hue_std",
         "lab_L_mean", "lab_a_mean", "lab_b_mean",
         "value_mean", "value_std"],
        "NAIL CROP STATISTICS")

    # --- 5) Check the existing color_features.csv for nail ROI stats ---
    color_csv = Path("../data/processed/color_features.csv")
    if color_csv.exists():
        df_color = pd.read_csv(color_csv)
        print(f"\n{'='*70}")
        print(f"  EXTRACTED COLOR FEATURES (from {len(df_color)} patients)")
        print(f"{'='*70}")
        for prefix, label in [("nail_", "NAIL BED"), ("skin_", "SKIN")]:
            cols = [c for c in df_color.columns if c.startswith(prefix)]
            key_cols = [c for c in cols if any(k in c for k in ["rgb_R_mean", "rgb_G_mean", "rgb_B_mean",
                                                                 "lab_L_mean", "lab_A_mean", "lab_B_lab_mean",
                                                                 "hsv_S_mean", "hsv_H_mean", "hsv_V_mean"])]
            if key_cols:
                print(f"\n  {label} ROI features:")
                print(f"  {'Feature':<30} {'Mean':>8} {'Std':>8} {'P5':>8} {'Median':>8} {'P95':>8}")
                print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                for c in sorted(key_cols):
                    v = df_color[c]
                    print(f"  {c:<30} {v.mean():>8.1f} {v.std():>8.1f} "
                          f"{v.quantile(0.05):>8.1f} {v.median():>8.1f} {v.quantile(0.95):>8.1f}")

    # --- 6) Compute "ideal" ranges and print recommendations ---
    print(f"\n{'='*70}")
    print(f"  RECOMMENDED PHOTO CONDITIONS (based on training distribution)")
    print(f"{'='*70}")

    # Nail crop brightness range (P5 to P95 of luma_mean across crops)
    luma_lo = df_crops["luma_mean"].quantile(0.05)
    luma_hi = df_crops["luma_mean"].quantile(0.95)
    luma_med = df_crops["luma_mean"].median()

    # RGB balance
    r_med = df_crops["R_mean"].median()
    g_med = df_crops["G_mean"].median()
    b_med = df_crops["B_mean"].median()

    # Saturation
    sat_lo = df_crops["sat_mean"].quantile(0.05)
    sat_hi = df_crops["sat_mean"].quantile(0.95)
    sat_med = df_crops["sat_mean"].median()

    # R/B ratio (color temp proxy)
    rb_lo = df_crops["RB_ratio"].quantile(0.05)
    rb_hi = df_crops["RB_ratio"].quantile(0.95)
    rb_med = df_crops["RB_ratio"].median()

    # Dynamic range
    dr_med = df_crops["luma_range"].median()

    # Black / white fraction
    bf_95 = df_crops["black_frac"].quantile(0.95)
    wf_95 = df_crops["white_frac"].quantile(0.95)

    # LAB a* (the clinically critical channel for Hb)
    la_lo = df_crops["lab_a_mean"].quantile(0.05)
    la_hi = df_crops["lab_a_mean"].quantile(0.95)

    print(f"""
  The model was trained on clinical photos with these nail-bed characteristics:

  BRIGHTNESS (grayscale luminance 0-255):
    Median: {luma_med:.0f}    Range (P5-P95): {luma_lo:.0f} – {luma_hi:.0f}
    → Nail beds were moderately bright, well-lit but not washed out.
    → Avoid very dark (<{luma_lo:.0f}) or very bright (>{luma_hi:.0f}) conditions.

  COLOR BALANCE (RGB means):
    R: {r_med:.0f}   G: {g_med:.0f}   B: {b_med:.0f}
    R/B ratio: {rb_med:.2f}  (range {rb_lo:.2f} – {rb_hi:.2f})
    → Photos are warm-toned (R > G > B), consistent with skin under
      neutral to warm white light. Avoid blue/cool LED or strong
      colored lighting.

  SATURATION (HSV S channel, 0-255):
    Median: {sat_med:.0f}   Range (P5-P95): {sat_lo:.0f} – {sat_hi:.0f}
    → Moderate saturation. Avoid desaturated (gray) or hyper-saturated
      conditions. Flash can wash out saturation.

  DYNAMIC RANGE (P95-P5 of luminance):
    Median: {dr_med:.0f}
    → Moderate contrast. Avoid harsh directional light (creates
      highlights/shadows on the nail).

  SHADOWS / HIGHLIGHTS:
    <{bf_95*100:.1f}% dark pixels,  <{wf_95*100:.1f}% bright pixels (at P95)
    → Very few pure-black or pure-white pixels in training data.

  LAB a* CHANNEL (red-green axis — the key signal for hemoglobin):
    Range (P5-P95): {la_lo:.0f} – {la_hi:.0f}
    → This is the channel the model relies on most. Any lighting that
      shifts a* (colored light, harsh flash, nail polish) will degrade
      predictions.

  ─────────────────────────────────────────────────────────────────────
  USER RECOMMENDATIONS:
  ─────────────────────────────────────────────────────────────────────

  1. LIGHTING: Use diffuse, neutral-white indoor lighting (e.g. overhead
     room light or daylight from a window, but NOT direct sunlight).
     The training photos were taken under standard clinical lighting —
     fluorescent or LED panels with neutral/warm white (~4000-5500K).

  2. AVOID: Flash, colored LED strips, blue-tinted screens, direct
     sunlight, very dim rooms. These shift the color balance away from
     the training distribution and cause the model to hallucinate.

  3. DISTANCE: Hold the camera 10-20 cm from the fingernail. Fill the
     frame with the hand/fingers, similar to how the training photos
     were composed (resolution ~{int(df_photos['width'].median())}×{int(df_photos['height'].median())} pixels).

  4. NAIL CONDITION: No nail polish, no artificial nails. The model
     reads the natural nail bed color. Keep nails clean and unpainted.

  5. SKIN TONE: The training set appears to cover a range of skin tones
     (R/B ratio range {rb_lo:.2f}–{rb_hi:.2f}), but performance is
     unvalidated on very dark or very light skin — expect degradation
     at the extremes.

  6. STABILITY: Keep the hand still. The multi-frame capture averages
     30 frames; movement causes blur and inconsistent crop regions,
     leading to high variance and unreliable results.
""")

    # --- 7) Export numeric ranges as JSON for potential in-app use ---
    import json
    ranges = {
        "brightness": {
            "ideal_min": round(float(luma_lo), 1),
            "ideal_max": round(float(luma_hi), 1),
            "median": round(float(luma_med), 1),
            "unit": "grayscale 0-255",
        },
        "saturation": {
            "ideal_min": round(float(sat_lo), 1),
            "ideal_max": round(float(sat_hi), 1),
            "median": round(float(sat_med), 1),
            "unit": "HSV S channel 0-255",
        },
        "color_temp_proxy_RB_ratio": {
            "ideal_min": round(float(rb_lo), 2),
            "ideal_max": round(float(rb_hi), 2),
            "median": round(float(rb_med), 2),
        },
        "dynamic_range_luma": {
            "median": round(float(dr_med), 1),
        },
        "lab_a_range": {
            "p5": round(float(la_lo), 1),
            "p95": round(float(la_hi), 1),
        },
        "max_black_frac_p95": round(float(bf_95), 4),
        "max_white_frac_p95": round(float(wf_95), 4),
        "photo_resolution_median": {
            "width": int(df_photos["width"].median()),
            "height": int(df_photos["height"].median()),
        },
    }

    out_path = Path("training_photo_profile.json")
    with open(out_path, "w") as f:
        json.dump(ranges, f, indent=2)
    print(f"  Numeric ranges exported to: {out_path}")


if __name__ == "__main__":
    main()
