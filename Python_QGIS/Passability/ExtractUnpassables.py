#!/usr/bin/env python3
"""
Extract damage polygons B that reduce remaining road width < 4.0m for each road polygon A.
Compatible with QGIS 3.40 LTR.

Usage:
 - Edit the path variables below, then run in QGIS Python console or as a standalone Python script
   with QGIS Python environment (OSGeo4W or qgis standalone).
"""

import os
import glob
import csv

from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    QgsProject,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsVectorFileWriter,
    QgsGeometry,
    QgsFeatureRequest,
    QgsSpatialIndex,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
)
from qgis.PyQt.QtCore import QVariant

# -----------------------------
# === USER CONFIGURATION ===
# -----------------------------
# Path to road polygon layer A (file path or layer URI loaded in the project)
road_layer_path = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RoadBuffer\RoadBuffer_ALLAREA_MultiWidth_min_DRM_wajima_ONLYurban_NOTsunami_SegAdjusted.gpkg"
# Directory containing many damage polygon files (shapefiles, gpkg layers, geojson, etc.)
damages_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskBBox"
# Output path for result GeoPackage and layer name
output_gpkg = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Result_QGIS\remaining_width_under4_test.gpkg"  # update
output_layer_name = "remaining_under4"
# Output CSV (summary per A feature)
output_csv = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Result_QGIS\remaining_width_summary.csv"  # update

# Threshold
THRESHOLD = 4.0

# mapping from R22_005 to road width in meters
R22_WIDTH_MAP = {
    1: 13.0,
    2: 9.25,
    3: 4.25,
    4: 3.0,
}

# -----------------------------
# Load road layer A
# -----------------------------
layerA = QgsVectorLayer(road_layer_path, "roads_A", "ogr")
if not layerA.isValid():
    raise RuntimeError(f"Unable to load road layer A: {road_layer_path}")

# -----------------------------
# Load all damage layers B from directory
# -----------------------------
def list_vector_files(dirpath):
    exts = ("*.gpkg", "*.shp", "*.geojson", "*.json", "*.sqlite")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dirpath, e)))
    return files

damage_files = list_vector_files(damages_dir)
if not damage_files:
    raise RuntimeError(f"No damage vector files found in {damages_dir}")

# Collect all B features into a list, with source filename and width_m attribute
B_features = []  # list of dicts: { 'geom': QgsGeometry, 'width_m': float, 'source': str, 'orig_feat': QgsFeature }
for fpath in damage_files:
    # For gpkg there might be multiple layers; open all layers from datasource, but QgsVectorLayer with file alone loads first/default layer.
    # We'll attempt two ways: load as single layer (works for shapefile, geojson), and for gpkg, try to extract layers via ogr if needed.
    layer = QgsVectorLayer(fpath, os.path.basename(fpath), "ogr")
    if not layer.isValid():
        print(f"Warning: could not load {fpath}, skipping.")
        continue

    # Check attribute existence
    has_width = "width_m" in [field.name() for field in layer.fields()]
    if not has_width:
        print(f"Warning: 'width_m' not in attributes of {fpath}. Skipping layer.")
        continue

    for feat in layer.getFeatures():
        try:
            w = feat["width_m"]
            if w is None:
                continue
            geom = QgsGeometry(feat.geometry())

            B_features.append({
                "geom": geom,
                "width_m": float(w),
                "source": os.path.basename(fpath),
                "orig_feat": feat,
            })
        except Exception as e:
            print(f"Warning reading feature in {fpath}: {e}")
            continue

if not B_features:
    raise RuntimeError("No damage features with 'width_m' found in given directory.")

# -----------------------------
# Build spatial index for B
# -----------------------------
index = QgsSpatialIndex()
for i, b in enumerate(B_features):
    # create a minimal feature for adding to index
    f = QgsFeature()
    f.setGeometry(b["geom"])
    f.setId(i)
    index.addFeature(f)

# -----------------------------
# Prepare output features list
# -----------------------------
out_feats = []  # each item will be (geom, width_m, remaining_width, source)
# For CSV summary: list of rows per A feature
csv_rows = []  # dicts: { 'A_fid', 'R22_005', 'B_files': [...], 'remaining_widths': [...] }

# Iterate A features
for a_feat in layerA.getFeatures():
    # read R22_005
    r22 = a_feat["R22_005"]
    if r22 is None:
        print(f"Warning: feature {a_feat.id()} missing R22_005, skipping.")
        continue
    # Convert to int if possible
    try:
        r22_int = int(r22)
    except Exception:
        print(f"Warning: R22_005 value '{r22}' in feature {a_feat.id()} not an integer, skipping.")
        continue
    road_width = R22_WIDTH_MAP.get(r22_int)
    if road_width is None:
        print(f"Warning: R22_005={r22_int} not in mapping for feature {a_feat.id()}, skipping.")
        continue

    # geometry (transformed to target_crs)
    a_geom = QgsGeometry(a_feat.geometry())

    # find candidate B features by bbox
    bbox = a_geom.boundingBox()
    candidate_ids = index.intersects(bbox)

    found_B_files = []
    found_remaining = []

    for cid in candidate_ids:
        b = B_features[cid]
        b_geom = b["geom"]
        # Determine inclusion by checking if any vertex of B lies within A
        contains_vertex = False
        for pt in b_geom.vertices():
            # pt is a QgsPoint or QgsPointXY; create point geometry for containment test
            try:
                pt_geom = QgsGeometry.fromPointXY(pt)
            except Exception:
                try:
                    pt_geom = QgsGeometry.fromPoint(pt)
                except Exception:
                    continue
            if a_geom.contains(pt_geom):
                contains_vertex = True
                break
        if not contains_vertex:
            continue

        # Inclusion determined solely by vertex containment (no intersection checks)
        remaining_width = road_width - b["width_m"]
        # collect CSV info
        found_B_files.append(b["source"])
        found_remaining.append(remaining_width)
        # if threshold met -> record to output layer (duplicate entries allowed if same B intersects multiple A)
        if remaining_width < THRESHOLD:
            out_feats.append({
                "geom": b_geom,  # stored in target_crs
                "width_m": b["width_m"],
                "remaining_width": remaining_width,
                "source": b["source"],
            })

    csv_rows.append({
        "A_fid": a_feat.id(),
        "R22_005": r22_int,
        "B_files": found_B_files,
        "remaining_widths": found_remaining,
    })

# -----------------------------
# Write output layer (GeoPackage)
# -----------------------------
if out_feats:
    # create fields for output
    fields = QgsFields()
    fields.append(QgsField("width_m", QVariant.Double))
    fields.append(QgsField("remaining_w", QVariant.Double))
    fields.append(QgsField("source", QVariant.String))

    # Determine geometry type: use polygon
    geom_type = QgsWkbTypes.Polygon
    # Use CRS of road layer A for output
    crs_out = layerA.crs()
    transform_ctx = QgsProject.instance().transformContext()

    # Simpler/robust approach: create memory layer, add features, then write to gpkg.
    mem_layer = QgsVectorLayer("Polygon?crs={}".format(crs_out.authid()), "tmp", "memory")
    prov = mem_layer.dataProvider()
    prov.addAttributes(fields)
    mem_layer.updateFields()

    feat_id = 0
    feats_to_add = []
    for entry in out_feats:
        feat = QgsFeature()
        feat.setGeometry(entry["geom"])
        feat.setFields(mem_layer.fields())
        feat["width_m"] = float(entry["width_m"])
        feat["remaining_w"] = float(entry["remaining_width"])
        feat["source"] = entry["source"]
        feats_to_add.append(feat)
        feat_id += 1

    prov.addFeatures(feats_to_add)
    mem_layer.updateExtents()

    # Write to GeoPackage
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = output_layer_name
    options.fileEncoding = "UTF-8"
    res, err_msg = QgsVectorFileWriter.writeAsVectorFormatV2(mem_layer, output_gpkg, transform_ctx, options)
    if res != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"Failed to write output layer to {output_gpkg}: {err_msg}")

else:
    print("No features met remaining_width < threshold; no output layer created.")

# -----------------------------
# Write CSV summary (per A feature block)
# -----------------------------
with open(output_csv, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    for r in csv_rows:
        # header lines for the A feature
        writer.writerow(["A_fid", r["A_fid"]])
        writer.writerow(["R22_005", r["R22_005"]])
        writer.writerow([])
        # then list B files and remaining widths per line
        writer.writerow(["B_files", "Remaining_Widths"])
        if r["B_files"]:
            for fname, rem in zip(r["B_files"], r["remaining_widths"]):
                writer.writerow([fname, "{:.3f}".format(rem)])
        else:
            writer.writerow(["(none)", ""])  # no B files for this A
        # blank line between A blocks
        writer.writerow([])

print("Done. Output layer (if any) written to:", output_gpkg)
print("CSV summary written to:", output_csv)
