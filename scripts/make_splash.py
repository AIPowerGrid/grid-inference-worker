"""Create splash.png for PyInstaller --splash (shown during exe extraction).
   Run from repo root: python scripts/make_splash.py"""
import os
import sys

def main():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("pip install pillow", file=sys.stderr)
        sys.exit(1)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_root, "splash.png")

    w, h = 400, 180
    img = Image.new("RGB", (w, h), (26, 26, 46))  # #1a1a2e
    draw = ImageDraw.Draw(img)

    font_paths = ["arial.ttf", "Arial.ttf"]
    if os.name == "nt":
        font_paths.insert(0, os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "arial.ttf"))
    font_large = font_small = ImageFont.load_default()
    for fp in font_paths:
        try:
            font_large = ImageFont.truetype(fp, 22)
            font_small = ImageFont.truetype(fp, 14)
            break
        except (OSError, TypeError):
            continue

    draw.text((w // 2, 70), "Grid Inference Worker", fill="#eee", anchor="mm", font=font_large)
    draw.text((w // 2, 110), "Starting…", fill="#94a3b8", anchor="mm", font=font_small)

    img.save(out_path)
    print(f"Created {out_path}")


if __name__ == "__main__":
    main()
