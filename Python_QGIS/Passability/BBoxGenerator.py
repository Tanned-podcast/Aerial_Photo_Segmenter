# PyQGIS script for QGIS 3.40 LTR
# - Computes principal road angle per damage polygon
# - Rotates polygon to align with road, computes bounding box width
# - Saves per-feature H polygons to a single GeoPackage
# - Appends results to a CSV
#
# Configuration parameters below.

from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsFields, QgsField,
    QgsWkbTypes, QgsProject, QgsVectorFileWriter, QgsRectangle, QgsFeatureRequest,
    QgsVectorLayer, QgsCoordinateTransformContext, QgsMemoryProviderUtils, QgsDistanceArea
)
import processing
from PyQt5.QtCore import QVariant
import math, os, csv
import glob

# ----------------------
# === Configuration ===
# ----------------------
ROAD_LAYER_PATH = r'C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RoadBufferBound\RoadBufferBound_ALLAREA_MultiWidth_min_DRM_wajima_ONLYurban_NOTsunami_SegAdjusted.gpkg'  # line layer (A)
masks_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskVector"  # list of polygon layers (one or many)
EPSILON = 0.000001              # buffer distance epsilon (same CRS units as input; small)
OUTPUT_CSV = r'C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\output.csv'
output_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskBBox"
BUFFER_SEGMENTS = 8        # buffer resolution
# ----------------------

# Load road layer
road_layer = QgsVectorLayer(ROAD_LAYER_PATH, 'road', 'ogr')
if not road_layer.isValid():
    raise RuntimeError('Failed to open road layer: {}'.format(ROAD_LAYER_PATH))
crs = road_layer.crs()

# Distance calculator for accurate meter measurements using GRS80 ellipsoid
dist_calc = QgsDistanceArea()
dist_calc.setSourceCrs(crs, QgsProject.instance().transformContext())
dist_calc.setEllipsoid('GRS80')

# Prepare output memory layer for H polygons
# h_layer is for all the edits, h_single is for exporting to a file
fields = QgsFields()
fields.append(QgsField('angle_deg_clockwise', QVariant.Double))
fields.append(QgsField('width_m', QVariant.Double))
geom_type = QgsWkbTypes.Polygon
h_layer = QgsVectorLayer(f'Polygon?crs={crs.authid()}', 'H_mem', 'memory')
pr = h_layer.dataProvider()
pr.addAttributes(fields)
h_layer.updateFields()

# Prepare CSV (append, add header if not exists)
csv_exists = os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
if not csv_exists:
    csv_writer.writerow(['angle_deg', 'width_m'])

# Find mask vector files
mask_files = sorted(glob.glob(os.path.join(masks_dir, '*.gpkg')))
if not mask_files:
    print('ERROR: no raster files found in', masks_dir)

saved_count = 0

# Process each damage layer and feature
for mask_file in mask_files:
    # load layer and its geometry
    dmg_layer = QgsVectorLayer(mask_file, 'damage', 'ogr')
    if not dmg_layer.isValid():
        print(f'Warning: failed to open damage layer {mask_file}, skipping.')
        continue
    
    feat = next(dmg_layer.getFeatures())
    geom = feat.geometry()
    if geom is None or geom.isEmpty():
        print(f'Warning: file {mask_file} has empty geometry; skipping.')
        continue

    centroid = geom.centroid().asPoint()

    # C: small buffer around B (in-memory geometry)
    c_geom = geom.buffer(EPSILON, BUFFER_SEGMENTS)

    # Create a temporary memory layer for C (single feature) for processing.clip
    c_mem = QgsVectorLayer(f'Polygon?crs={crs.authid()}', 'C_mem', 'memory')
    c_pr = c_mem.dataProvider()
    c_f = QgsFeature()
    c_f.setGeometry(c_geom)
    c_pr.addFeatures([c_f])
    c_mem.updateExtents()

    # D: clip road_layer by C
    params = {
        'INPUT': road_layer,
        'OVERLAY': c_mem,
        'OUTPUT': 'memory:'
    }
    res = processing.run('native:clip', params)
    d_layer = res['OUTPUT']

    # Gather vertices from D
    pts = []
    feat_angles = []
    dfeats = list(d_layer.getFeatures())

    # collect vertices to make sure there are vertices in lines
    for dfeat in dfeats:
        dgeom = dfeat.geometry()
        if dgeom.isEmpty():
            continue
        # iterate vertices
        for v in dgeom.vertices():
            pts.append((v.x(), v.y()))

    if len(pts) < 2:
        # fallback: if no clipped lines, set angle 0 and continue
        theta = 0.0
        print(f'Info: no intersection for file {mask_file}; using angle 0.')
    else:
        for dfeat in dfeats:
            dgeom = dfeat.geometry()
            if dgeom.isEmpty():
                print("dgeom is empty, skipping")
                continue

            angle_rad = dgeom.interpolateAngle(EPSILON)  # ラジアン
            feat_angles.append(angle_rad)

        theta = sum(feat_angles) / len(feat_angles) if feat_angles else 0.0
        theta = math.degrees(theta)  # degreeに変換

    # Define E such that rotating polygon CCW by E aligns road horizontally:
    # E = -theta (radians). We'll store angle in degrees (angle_deg).
    angle_deg = -theta

    # 4: Rotate B (ccw by E_deg) -> F
    f_geom = QgsGeometry(geom)  # copy
    f_geom.rotate(angle_deg, centroid)  # rotate CCW by angle_deg (negative => CW)

    # 5: Bounding box of F -> G (axis-aligned)
    bbox = f_geom.boundingBox()
    # compute width in meters using ellipsoidal distance (GRS80)
    p1 = QgsPointXY(bbox.xMinimum(), bbox.yMinimum())
    p2 = QgsPointXY(bbox.xMaximum(), bbox.yMinimum())
    line_geom = QgsGeometry.fromPolylineXY([p1, p2])
    width_m = dist_calc.measureLength(line_geom)

    # create bbox polygon geometry (G)
    rect_geom = QgsGeometry.fromRect(QgsRectangle(bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum()))

    # 7: Rotate G clockwise by E (i.e., rotate by -angle_deg) to get H
    h_geom = QgsGeometry(rect_geom)
    h_geom.rotate(-angle_deg, centroid)

    angle_deg_clockwise = -angle_deg  # store clockwise angle

    # Add H feature to a temporary single-feature memory layer and save it (prevents accumulation)
    hfeat = QgsFeature()
    hfeat.setGeometry(h_geom)
    hfeat.setAttributes([float(angle_deg_clockwise), float(width_m)])

    # create single feature memory layer (only current H)
    h_single = QgsVectorLayer(f'Polygon?crs={crs.authid()}', 'H_single', 'memory')
    h_single_pr = h_single.dataProvider()
    h_single_pr.addAttributes(fields)
    h_single.updateFields()
    h_single_pr.addFeatures([hfeat])
    h_single.updateExtents()

    # Append entry to CSV (width in meters)
    csv_writer.writerow([float(angle_deg_clockwise), float(width_m)])

    base_name = os.path.splitext(os.path.basename(mask_file))[0]
    out_fp = os.path.join(output_dir, f"{base_name}_bbox.gpkg")
    os.makedirs(output_dir, exist_ok=True)

    print('    -> saving to:', out_fp)
    try:
        res = processing.run('native:savefeatures', {'INPUT': h_single, 'OUTPUT': out_fp})
    except Exception as e:
        print(f'    -> failed to save file {mask_file}: {e}')
        continue

    if res and res.get('OUTPUT'):
        print('    -> saved:', res['OUTPUT'])
        saved_count += 1
    else:
        print('    -> failed to save file', mask_file)

# close CSV
csv_file.close()

if saved_count == 0:
    print('No clipped features were saved. Exiting.')
else:
    print(f'Done. Saved {saved_count} clipped layers to directory: {output_dir}')
