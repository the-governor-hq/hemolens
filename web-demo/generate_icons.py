"""
Generate PWA icons for HemoLens web demo.

Design: styled-components–inspired 💅 fingernail icon.
Coral-to-rose gradient background with a clean white stylised finger + nail.

Usage:
    python generate_icons.py            # outputs to web-demo/icons/
    python generate_icons.py --out ./icons --sizes 192 512
"""

import argparse
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow is required:  pip install Pillow")
    raise SystemExit(1)

# ── Palette ──
GRADIENT_TOP = (255, 95, 109)       # coral
GRADIENT_BOT = (255, 45, 85)        # rose
WHITE        = (255, 255, 255)
NAIL_PINK    = (255, 200, 200)      # soft pink nail bed
NAIL_TIP     = (255, 240, 240)      # lighter tip (french manicure hint)
CUTICLE      = (255, 255, 255, 100) # subtle lunula


def _lerp(a, b, t):
    return int(a + (b - a) * t)


def _gradient_bg(draw, s, c_top, c_bot):
    for y in range(s):
        t = y / max(s - 1, 1)
        color = tuple(_lerp(c_top[i], c_bot[i], t) for i in range(3))
        draw.line([(0, y), (s - 1, y)], fill=color)


def _rounded_mask(s, radius):
    mask = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=radius, fill=255)
    return mask


def _circle_mask(s, margin):
    mask = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([margin, margin, s - margin, s - margin], fill=255)
    return mask


def _rounded_rect_pts(cx, cy, w, h, r, steps_per_corner=16):
    """Generate points for a rounded rectangle (clockwise from top-left)."""
    hw, hh = w / 2, h / 2
    r = min(r, hw, hh)
    pts = []
    # corners: TL, TR, BR, BL
    corners = [
        (cx - hw + r, cy - hh + r, math.pi, 1.5 * math.pi),
        (cx + hw - r, cy - hh + r, 1.5 * math.pi, 2 * math.pi),
        (cx + hw - r, cy + hh - r, 0, 0.5 * math.pi),
        (cx - hw + r, cy + hh - r, 0.5 * math.pi, math.pi),
    ]
    for ccx, ccy, a_start, a_end in corners:
        for i in range(steps_per_corner + 1):
            a = a_start + (a_end - a_start) * (i / steps_per_corner)
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    return pts


def _draw_finger(draw, cx, cy, scale):
    """
    Draw a stylised finger pointing up-right (like the 💅 / styled-components logo).
    The finger is a rounded elongated shape, tilted ~30°, with a nail on top.
    """
    angle = math.radians(-25)  # tilt counter-clockwise
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rot(x, y):
        """Rotate point around (cx, cy)."""
        dx, dy = x - cx, y - cy
        return (cx + dx * cos_a - dy * sin_a,
                cy + dx * sin_a + dy * cos_a)

    # ── Finger body ──
    finger_w = scale * 0.32
    finger_h = scale * 0.95
    finger_r = finger_w * 0.48  # roundedness
    finger_cy = cy + scale * 0.05  # shift down slightly

    body_pts = _rounded_rect_pts(cx, finger_cy, finger_w, finger_h, finger_r)
    body_pts = [rot(x, y) for x, y in body_pts]
    draw.polygon(body_pts, fill=WHITE)

    # ── Fingernail ──
    # Sits on the upper ~45% of the finger, slightly narrower
    nail_w = finger_w * 0.82
    nail_h = finger_h * 0.48
    nail_cy = finger_cy - finger_h * 0.22
    nail_r_top = nail_w * 0.50     # very round top
    nail_r_bot = nail_w * 0.12     # flatter bottom

    # Build nail shape: rounded top, flatter bottom
    # We'll use an elliptical top + straight-ish bottom
    nail_pts = []
    hw, hh = nail_w / 2, nail_h / 2

    # Top arc (semicircle-ish)
    steps = 32
    for i in range(steps + 1):
        t = i / steps
        a = math.pi + t * math.pi  # pi → 2pi (top half)
        rx = hw
        ry = hw * 0.85  # slightly elliptical
        x = cx + rx * math.cos(a)
        y = (nail_cy - hh + ry) + ry * math.sin(a)
        nail_pts.append((x, y))

    # Bottom edge (nearly straight, slight curve)
    for i in range(steps + 1):
        t = i / steps
        x = cx + hw - t * nail_w
        # subtle upward bow
        bow = -nail_h * 0.04 * math.sin(t * math.pi)
        y = nail_cy + hh + bow
        nail_pts.append((x, y))

    nail_pts = [rot(x, y) for x, y in nail_pts]
    draw.polygon(nail_pts, fill=NAIL_PINK)

    # ── Lunula (half-moon at cuticle) ──
    luna_r = nail_w * 0.30
    luna_cy_pos = nail_cy + hh - luna_r * 0.3
    luna_pts = []
    for i in range(steps + 1):
        t = i / steps
        a = math.pi + t * math.pi
        x = cx + luna_r * math.cos(a)
        y = luna_cy_pos + luna_r * 0.5 * math.sin(a)
        luna_pts.append((x, y))
    luna_pts = [rot(x, y) for x, y in luna_pts]
    draw.polygon(luna_pts, fill=CUTICLE)

    # ── Nail tip highlight (french tip) ──
    tip_h = nail_h * 0.15
    tip_cy_pos = nail_cy - hh + nail_w * 0.85 * 0.5
    tip_pts = []
    # Top arc of nail (reuse top portion)
    for i in range(steps + 1):
        t = i / steps
        a = math.pi + t * math.pi
        rx = hw * 0.95
        ry = hw * 0.78
        x = cx + rx * math.cos(a)
        y = (nail_cy - hh + ry) + ry * math.sin(a)
        tip_pts.append((x, y))
    # Bottom of tip strip
    for i in range(steps + 1):
        t = i / steps
        x = cx + hw * 0.95 - t * nail_w * 0.95
        y = nail_cy - hh + tip_h + nail_w * 0.85 * 0.12
        tip_pts.append((x, y))
    tip_pts = [rot(x, y) for x, y in tip_pts]
    draw.polygon(tip_pts, fill=NAIL_TIP + (180,))


def draw_hemolens_icon(size: int, *, maskable: bool = False) -> Image.Image:
    """Draw HemoLens icon: gradient bg + stylised fingernail."""
    ss = 4
    s = size * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = s // 2, s // 2

    # ── Gradient background ──
    _gradient_bg(draw, s, GRADIENT_TOP, GRADIENT_BOT)

    if maskable:
        mask = _rounded_mask(s, int(s * 0.22))
    else:
        mask = _circle_mask(s, int(s * 0.02))
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)

    # ── Fingernail ──
    scale = s * 0.62
    _draw_finger(draw, cx, cy, scale)

    # ── Downsample ──
    img = img.resize((size, size), Image.LANCZOS)
    return img


# ── Favicon (ICO) helper ──

def make_favicon(icon_img: Image.Image, out_path: Path):
    """Save a multi-size .ico file."""
    sizes = [16, 32, 48]
    frames = [icon_img.resize((s, s), Image.LANCZOS) for s in sizes]
    frames[0].save(str(out_path), format="ICO", sizes=[(s, s) for s in sizes],
                   append_images=frames[1:])


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Generate HemoLens PWA icons")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <script_dir>/icons/)")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[72, 96, 128, 144, 152, 192, 384, 512],
                        help="Icon sizes to generate")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else script_dir / "icons"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    for sz in args.sizes:
        img = draw_hemolens_icon(sz)
        fname = f"icon-{sz}x{sz}.png"
        img.save(out_dir / fname, "PNG", optimize=True)
        generated.append(fname)

        # Maskable variant for 192 and 512
        if sz in (192, 512):
            mimg = draw_hemolens_icon(sz, maskable=True)
            mfname = f"icon-{sz}x{sz}-maskable.png"
            mimg.save(out_dir / mfname, "PNG", optimize=True)
            generated.append(mfname)

    # Apple touch icon (180×180)
    apple = draw_hemolens_icon(180)
    apple.save(out_dir / "apple-touch-icon.png", "PNG", optimize=True)
    generated.append("apple-touch-icon.png")

    # Favicon .ico
    base = draw_hemolens_icon(256)
    make_favicon(base, out_dir / "favicon.ico")
    generated.append("favicon.ico")

    # 32×32 PNG favicon
    fav32 = draw_hemolens_icon(32)
    fav32.save(out_dir / "favicon-32x32.png", "PNG", optimize=True)
    generated.append("favicon-32x32.png")

    print(f"Generated {len(generated)} icons in {out_dir}/")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
