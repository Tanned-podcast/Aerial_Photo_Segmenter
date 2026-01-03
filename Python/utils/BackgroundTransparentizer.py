"""
BackgroundTransparentizer

Usage:
    python BackgroundTransparentizer.py /path/to/input_dir /path/to/output_dir

What it does:
    - Walks the input directory (recursively)
    - For every .png image, sets pixels with RGB == (0,0,0) to fully transparent
    - Saves processed PNGs to the output directory while preserving subdirectory structure

Requirements:
    - Pillow (pip install pillow)

"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - friendly error
    print("Pillow is required: pip install pillow", file=sys.stderr)
    raise


def make_transparent(in_path: Path, out_path: Path) -> bool:
    """Open in_path PNG, set RGB (0,0,0) pixels to transparent, save to out_path.

    Returns True if processed and saved, False on error.
    """
    try:
        with Image.open(in_path) as im:
            im = im.convert("RGBA")
            pixels = im.getdata()

            new_pixels = []
            for px in pixels:
                r, g, b, a = px
                if (r, g, b) == (0, 0, 0):
                    new_pixels.append((r, g, b, 0))
                else:
                    new_pixels.append((r, g, b, a))

            im.putdata(new_pixels)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, "PNG")
        return True
    except Exception as e:
        print(f"Failed to process {in_path}: {e}", file=sys.stderr)
        return False


def iter_png_files(input_dir: Path, recursive: bool = True):
    pattern = "**/*.png" if recursive else "*.png"
    yield from input_dir.glob(pattern)


def main(argv: list[str] | None = None) -> int:

    input_dir = Path(r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Result_Segmentation\20260103_1810\PredMasks")
    output_dir = Path(r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Pred_MaskPNG")
    overwrite = True

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    count_total = 0
    count_ok = 0

    for png_path in iter_png_files(input_dir):
        if not png_path.is_file():
            continue

        rel = png_path.relative_to(input_dir)
        out_path = output_dir.joinpath(rel)

        if out_path.exists() and not overwrite:
            print(f"Skipping (exists): {out_path}")
            continue

        count_total += 1
        ok = make_transparent(png_path, out_path)
        if ok:
            count_ok += 1
            print(f"Saved: {out_path}")

    print(f"Done. Processed {count_ok}/{count_total} images.")
    return 0


main()
