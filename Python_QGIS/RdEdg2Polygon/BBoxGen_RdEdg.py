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
    QgsVectorLayer, QgsCoordinateTransformContext, QgsMemoryProviderUtils, QgsDistanceArea, QgsGeometryUtilsBase
)
import processing
from PyQt5.QtCore import QVariant
import math, os, csv
import glob, datetime

# ----------------------
# === Configuration ===
# ----------------------
ROAD_LAYER_PATH = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RdEdg\wajima_rdedg_edited.gpkg"  # line layer (A)
masks_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\MaskVector_Clipped\GT_wajima"  # list of polygon layers (one or many)
EPSILON = 0.000003
EPSILON_ANGLE = 0.000001
OUTPUT_CSV = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_QGIS\output_GT_wajima.csv"
output_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\MaskBBox\GT_wajima"
BUFFER_SEGMENTS = 8        # buffer resolution
# ----------------------

# Collect error logs
errors =[]

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
# coordinates of the polygon edge that is most nearly perpendicular to road segments in D
fields.append(QgsField('perp_vertexID_start', QVariant.Int))
fields.append(QgsField('perp_vertexID_end', QVariant.Int))
fields.append(QgsField('perp_x1', QVariant.Double))
fields.append(QgsField('perp_y1', QVariant.Double))
fields.append(QgsField('perp_x2', QVariant.Double))
fields.append(QgsField('perp_y2', QVariant.Double))
# how far (in degrees) this best edge is from exact 90° (0.0 = perfect perp)
fields.append(QgsField('perp_err_deg', QVariant.Double))
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
        msg = f'Warning: failed to open damage layer {mask_file}, skipping.'
        errors.append(msg)
        print(msg)
        continue

    # Break multipart to singlepart to process each feature individually
    params_single = {
        'INPUT': dmg_layer,
        'OUTPUT': 'memory:'
    }
    res_single = processing.run('native:multiparttosingleparts', params_single)
    dmg_single = res_single['OUTPUT']

    for fid, feat in enumerate(dmg_single.getFeatures()):
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            msg = f'Warning: file {mask_file} has empty geometry; skipping.'
            errors.append(msg)
            print(msg)
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
            msg = f'Info: no intersection for file {mask_file}; skipping.'
            errors.append(msg)
            print(msg)
            continue
        else:
            for dfeat in dfeats:
                dgeom = dfeat.geometry()
                if dgeom.isEmpty():
                    msg = f"dgeom is empty for {mask_file}, id {dfeat.id()}, skipping"
                    errors.append(msg)
                    print(msg)
                    continue


                angle_rad = dgeom.interpolateAngle(EPSILON_ANGLE)  # ラジアン
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

        # create bbox polygon geometry (G)
        rect_geom = QgsGeometry.fromRect(QgsRectangle(bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(), bbox.yMaximum()))

        # 7: Rotate G clockwise by E (i.e., rotate by -angle_deg) to get H
        h_geom = QgsGeometry(rect_geom)
        h_geom.rotate(-angle_deg, centroid)

        angle_deg_clockwise = -angle_deg  # store clockwise angle

        # Determine polygon edge of H most nearly perpendicular to the road segments in D
        perp_x1 = perp_y1 = perp_x2 = perp_y2 = None
        perp_err = None
        if theta and not h_geom.isEmpty():
            poly = h_geom.asPolygon()
            if poly:
                ring = poly[0]
                n = len(ring)
                if n >= 2:
                    best_angle_val = -1.0
                    best_coords = None
                    best_err = None
                    for i in range(n - 1):  # last point is same as first in closed ring
                        p1 = ring[i]
                        p2 = ring[i + 1]
                        dx = p2.x() - p1.x()
                        dy = p2.y() - p1.y()
                        # Calculate Angle CW with 0 deg = North: NO "math.atan2" (0 deg isn't North) !!!
                        edge_ang = math.degrees(QgsGeometryUtilsBase.lineAngle(p1.x(), p1.y(), p2.x(), p2.y()))

                        # For this edge, find the road segment angle that gives the maximum angle between them
                        # (i.e., closest to 90°). min_angle_between returns angle in [0,90].
                        def min_angle_between(a, b):
                            diff = abs(a - b) % 180
                            return diff if diff <= 180 - diff else 180 - diff

                        best_for_edge = 0.0
                        val = min_angle_between(edge_ang, theta)
                        if val > best_for_edge:
                            best_for_edge = val

                        # best_for_edge is in [0,90], closer to 90 => more perpendicular
                        if best_for_edge > best_angle_val:
                            best_angle_val = best_for_edge
                            best_points = [p1, p2]
                            best_coords = (p1.x(), p1.y(), p2.x(), p2.y())
                            best_err = 90.0 - best_for_edge
                            best_id_start = i
                            best_id_end = (i + 1) % (n - 1)

                    if best_coords:
                        perp_x1, perp_y1, perp_x2, perp_y2 = best_coords
                        line_geom = QgsGeometry.fromPolylineXY(best_points)
                        width_m = dist_calc.measureLength(line_geom)
                        perp_err = float(best_err)
                        perp_vertexID_start = best_id_start
                        perp_vertexID_end = best_id_end

        # Add H feature to a temporary single-feature memory layer and save it (prevents accumulation)
        hfeat = QgsFeature()
        hfeat.setGeometry(h_geom)
        hfeat.setAttributes([
            float(angle_deg_clockwise),
            float(width_m),
            None if perp_vertexID_start is None else perp_vertexID_start,
            None if perp_vertexID_end is None else perp_vertexID_end,
            None if perp_x1 is None else float(perp_x1),
            None if perp_y1 is None else float(perp_y1),
            None if perp_x2 is None else float(perp_x2),
            None if perp_y2 is None else float(perp_y2),
            None if perp_err is None else float(perp_err),
        ])

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
        out_fp = os.path.join(output_dir, f"{base_name}_bbox_f{fid}.gpkg")
        os.makedirs(output_dir, exist_ok=True)

        print('    -> saving to:', out_fp)
        try:
            res = processing.run('native:savefeatures', {'INPUT': h_single, 'OUTPUT': out_fp})
        except Exception as e:
            msg = f'    -> failed to save file {mask_file}: {e}'
            errors.append(msg)
            print(msg)
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

    # If any errors collected, write them to a log file in output_dir
    if errors:
        err_fp = os.path.join(output_dir, f"processing_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(err_fp, 'w', encoding='utf-8') as ef:
                ef.write(f"Processing errors for run: {datetime.datetime.now().isoformat()}\n")
                ef.write(f"Source masks dir: {masks_dir}\n")
                ef.write("Errors:\n")
                for e in errors:
                    ef.write(e + '\n')
            print('Wrote error log to:', err_fp)
        except Exception as e:
            print('Failed to write error log file:', e)
