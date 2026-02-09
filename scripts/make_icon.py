"""Generate favicon.ico + PNGs from the AIPG logo SVG (or PNG fallback).
   Renders from vector via pycairo for pixel-perfect icons at every size.
   Run from repo root: python scripts/make_icon.py"""
import io
import os
import re
import sys
import xml.etree.ElementTree as ET

# Windows exe + taskbar: multiple sizes so Explorer and taskbar get exact rasters.
WINDOWS_ICO_SIZES = [(256, 256), (48, 48), (32, 32), (24, 24), (20, 20), (16, 16)]
# Web favicon: 16 and 32 are standard; 48 for high-DPI.
FAVICON_ICO_SIZES = [(48, 48), (32, 32), (16, 16)]


def _hex_to_rgba(h):
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, 1.0


def _draw_path_data(ctx, d):
    """Parse SVG path 'd' attribute and replay as cairo commands."""
    tokens = re.findall(
        r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d
    )
    i = 0
    cmd = "M"
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
        if cmd == "M":
            ctx.move_to(float(tokens[i]), float(tokens[i + 1]))
            i += 2
            cmd = "L"
        elif cmd == "L":
            ctx.line_to(float(tokens[i]), float(tokens[i + 1]))
            i += 2
        elif cmd == "Q":
            # Quadratic bezier -> cubic approximation
            px, py = ctx.get_current_point()
            qx, qy = float(tokens[i]), float(tokens[i + 1])
            ex, ey = float(tokens[i + 2]), float(tokens[i + 3])
            ctx.curve_to(
                px + 2 / 3 * (qx - px), py + 2 / 3 * (qy - py),
                ex + 2 / 3 * (qx - ex), ey + 2 / 3 * (qy - ey),
                ex, ey,
            )
            i += 4
        elif cmd in ("Z", "z"):
            ctx.close_path()
        else:
            break


def render_svg(svg_path, size=1024):
    """Render an SVG to a Pillow RGBA image using pycairo."""
    import cairo
    from PIL import Image

    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    vb = root.get("viewBox", "0 0 500 500").split()
    vw, vh = float(vb[2]), float(vb[3])

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surface)
    ctx.scale(size / vw, size / vh)

    # Draw filled paths from the FILL group
    defs = root.find(".//svg:defs", ns)
    fill_group = defs.find('.//svg:g[@id="Layer1_0_FILL"]', ns)
    for path_el in fill_group.findall("svg:path", ns):
        fill = path_el.get("fill", "#000")
        if fill == "none":
            continue
        d = path_el.get("d", "")
        ctx.new_path()
        _draw_path_data(ctx, d)
        ctx.set_source_rgba(*_hex_to_rgba(fill))
        ctx.fill()

    # Draw strokes
    stroke_el = defs.find('.//svg:path[@id="Layer1_0_1_STROKES"]', ns)
    if stroke_el is not None:
        d = stroke_el.get("d", "")
        ctx.new_path()
        _draw_path_data(ctx, d)
        ctx.set_source_rgba(*_hex_to_rgba(stroke_el.get("stroke", "#000")))
        ctx.set_line_width(float(stroke_el.get("stroke-width", "1")))
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.stroke()

    # Convert cairo surface -> Pillow Image (BGRA -> RGBA)
    buf = surface.get_data()
    img = Image.frombuffer("RGBA", (size, size), bytes(buf), "raw", "BGRA", 0, 1)
    return img


def load_png(png_path):
    """Load a PNG and prepare for icon generation."""
    from PIL import Image

    img = Image.open(png_path)
    if img.mode not in ("RGBA",):
        img = img.convert("RGBA")
    return img


def save_ico_at_sizes(img, resample, out_path, sizes):
    """Write ICO with exact pixel dimensions for each size."""
    resized = []
    for w, h in sizes:
        if (img.width, img.height) == (w, h):
            resized.append(img)
        else:
            resized.append(img.resize((w, h), resample))
    first, rest = resized[0], resized[1:]
    first.save(
        out_path, format="ICO",
        sizes=[(r.width, r.height) for r in resized],
        append_images=rest,
    )


def main():
    try:
        from PIL import Image
    except ImportError:
        print("Run: pip install pillow", file=sys.stderr)
        sys.exit(1)

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    svg_path = os.path.join(repo_root, "assets", "aipg-logo.svg")
    png_path = os.path.join(repo_root, "assets", "aipg-logo-1242.png")

    # Prefer SVG (vector -> pixel-perfect at any size), fall back to PNG
    img = None
    if os.path.isfile(svg_path):
        try:
            img = render_svg(svg_path, size=1024)
            print(f"Rendered {svg_path} -> 1024x1024")
        except Exception as e:
            print(f"SVG render failed ({e}), falling back to PNG")

    if img is None:
        if os.path.isfile(png_path):
            img = load_png(png_path)
            print(f"Using {png_path} ({img.width}x{img.height})")
        else:
            fallback = os.path.join(repo_root, "inference_worker", "web", "static", "logo.png")
            if os.path.isfile(fallback):
                img = load_png(fallback)
                print(f"Using fallback {fallback} ({img.width}x{img.height})")
            else:
                print("No logo source found", file=sys.stderr)
                sys.exit(1)

    # Upscale if source is smaller than largest icon
    if max(img.size) < 256:
        img = img.resize((256, 256), resample)

    # 1) favicon.ico at project root — full Windows sizes for exe + taskbar + tkinter
    root_ico = os.path.join(repo_root, "favicon.ico")
    save_ico_at_sizes(img, resample, root_ico, WINDOWS_ICO_SIZES)
    print(f"Created {root_ico}")

    # 2) favicon.ico in web/static — smaller sizes for browser tabs
    static_dir = os.path.join(repo_root, "inference_worker", "web", "static")
    save_ico_at_sizes(img, resample, os.path.join(static_dir, "favicon.ico"), FAVICON_ICO_SIZES)
    print(f"Created {os.path.join(static_dir, 'favicon.ico')}")

    # 3) Pixel-perfect PNGs for modern browsers
    for w, h in [(32, 32), (16, 16)]:
        out = os.path.join(static_dir, f"favicon-{w}x{h}.png")
        img.resize((w, h), resample).save(out, format="PNG")
        print(f"Created {out}")


if __name__ == "__main__":
    main()
