"""
Generate GCP files from square-polygon FGBs and matching georeferenced images.

For each FGB (assumed to contain a single square polygon with vertices ordered
as: left-bottom(0), right-bottom(1), right-top(2), left-top(3)),
this script finds a same-named image in `image_dir`, projects the polygon
corner coordinates to the image CRS, converts world coordinates to pixel
coordinates using GDAL geotransform, and writes a per-polygon GCP file
containing four `-gcp px py lon lat` tokens (lon/lat in EPSG:4326).

Output examples (one line per file):
-gcp 0 5522 136.694253 37.058802 -gcp 6458 0 136.7289022 37.082446 -gcp 0 0 136.694253 37.082446 -gcp 6458 5522 136.7289022 37.058802

Usage:
- edit the default directories below or call from command line:
    python MaskGCPGenerator.py <vector_dir> <image_dir> <output_dir>

NOTE: run inside QGIS Python environment where PyQGIS and GDAL are available.
"""

import os
import sys
import re
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem
)
from PIL import Image

# --------------------- USER CONFIG ---------------------
# Default directories (edit as needed)
DEFAULT_VECTOR_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\SquarePolygons"
DEFAULT_IMAGE_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Pred_MaskPNG"
DEFAULT_OUTPUT_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\GCP"

IMAGE_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.jp2')
# -------------------------------------------------------


def find_images(image_dir, base_name):
    """Return list of image paths whose filename contains base_name or vice-versa."""
    candidates = []
    base_name_lower = base_name.lower()
    for root, _, files in os.walk(image_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTS:
                name_low = name.lower()
                if base_name_lower == name_low or base_name_lower in name_low or name_low in base_name_lower:
                    candidates.append(os.path.join(root, f))
    return candidates


# def world_to_pixel(ds, x, y):
#     gt = ds.GetGeoTransform()
#     ok, inv_gt = gdal.InvGeoTransform(gt)
#     if not ok:
#         raise RuntimeError("Could not invert geotransform for dataset")
#     px, py = gdal.ApplyGeoTransform(inv_gt, x, y)
#     # Return as integers (round to nearest pixel)
#     return int(round(px)), int(round(py))


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def generate_gcps(vector_dir, image_dir, output_dir):
    ensure_dir(output_dir)

    vector_files = [f for f in os.listdir(vector_dir) if f.lower().endswith('.fgb')]
    if not vector_files:
        print("No .fgb files found in vector_dir")
        return

    for fgb in vector_files:
        fgb_path = os.path.join(vector_dir, fgb)
        base_name = os.path.splitext(fgb)[0]
        print(f"Processing {fgb} -> looking for images containing '{base_name}'")

        vlayer = QgsVectorLayer(fgb_path, base_name, 'ogr')
        if not vlayer.isValid():
            print(f"  ERROR: Failed to load vector layer {fgb}")
            continue

        feats = list(vlayer.getFeatures())
        if not feats:
            print(f"  WARNING: {fgb} contains no features; skipping")
            continue

        feat = feats[0]
        geom = feat.geometry()
        try:
            pts = geom.asPolygon()[0]
        except Exception as e:
            print(f"  ERROR: Couldn't read polygon geometry for {fgb}: {e}")
            continue

        if len(pts) < 4:
            print(f"  WARNING: polygon has fewer than 4 vertices; skipping")
            continue

        # Find candidate images
        images = find_images(image_dir, base_name)
        if not images:
            print(f"  WARNING: No matching image found for {base_name}; skipping")
            continue

        image_path = images[0]
        print(f"  Using image: {image_path}")

        img = Image.open(image_path)
        if img is None:
            print(f"  ERROR: Could not open image {image_path}; skipping")
            continue

        # # Prepare CRS transforms: vector layer CRS -> raster CRS (for pixel mapping)
        # raster_wkt = ds.GetProjection()
        # if not raster_wkt:
        #     print(f"  ERROR: Image {image_path} has no projection WKT; skipping")
        #     continue

        # raster_crs = QgsCoordinateReferenceSystem()
        # if not raster_crs.createFromWkt(raster_wkt):
        #     print(f"  ERROR: Could not create raster CRS from WKT; skipping")
        #     continue

        # # Vector->Raster transform
        # xform_to_raster = QgsCoordinateTransform(vlayer.crs(), raster_crs, QgsProject.instance())
        # # Vector->WGS84 (EPSG:4326) for lon/lat output
        # wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
        # xform_to_wgs84 = QgsCoordinateTransform(vlayer.crs(), wgs84, QgsProject.instance())

        gcps = []
        img_width = img.size[0]-1
        img_height = img.size[1]-1
        img_vertices = [
            (0, img_height),        # Left-Bottom
            (img_width, img_height),# Right-Bottom
            (img_width, 0),        # Right-Top
            (0, 0)                 # Left-Top
        ]
        # Use first 4 vertices, assuming ordering 0:LB, 1:RB, 2:RT, 3:LT
        for i in range(4):
            pt = pts[i]
            # # Point transform to raster CRS to compute pixel coords
            # pt_raster = xform_to_raster.transform(pt)
            # px, py = world_to_pixel(ds, pt_raster.x(), pt_raster.y())

            # # Point transform to WGS84 for lon/lat
            # pt_wgs = xform_to_wgs84.transform(pt)
            w, h = img_vertices[i]
            lon = pt.x()
            lat = pt.y()
            gcps.append((w, h, lon, lat))

        # Build output line
        tokens = []
        for w, h, lon, lat in gcps:
            tokens.append(f"-gcp {w} {h} {lon:.6f} {lat:.6f}")
        line = ' '.join(tokens)

        img_base_name = os.path.splitext(image_path)[0].split("\\")[-1]
        out_path = os.path.join(output_dir, f"{img_base_name}.gcp")
        with open(out_path, 'w', encoding='utf-8') as fp:
            fp.write(line + '\n')

        print(f"  Wrote GCPs to {out_path}")

    print("Done: all GCP files generated")

vec_dir = DEFAULT_VECTOR_DIR
img_dir = DEFAULT_IMAGE_DIR
out_dir = DEFAULT_OUTPUT_DIR

generate_gcps(vec_dir, img_dir, out_dir)
