"""
original_road_width_from_rectangles.py
PyQGIS script that:
- Processes all polygon files in a directory (Layer A files).
- For each feature: compute centroid; check inclusion in Layer C polygons.
- If any centroid not in Layer C -> skip entire Layer A file (record filename).
- Otherwise, for each feature: build long line D through centroid at angle = angle_deg_clockwise + 90,
  find intersections with Layer B (road edges), select nearest 2 points, create line E,
  compute geodesic length using QgsDistanceArea (GRS80), write length to new attribute 'original_road_width'.
- Output: for each processed Layer A, produce a new line layer (only segment E for each feature),
  and write a skip-list text file of skipped layer filenames.
"""

import os
import math, datetime, glob

from qgis.core import (
    QgsVectorLayer, QgsProject, QgsFeature, QgsGeometry, QgsPointXY,
    QgsDistanceArea, QgsField, QgsFields, QgsVectorFileWriter, QgsWkbTypes,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsSpatialIndex,
    QgsFeatureRequest, QgsRectangle
)
import processing
from PyQt5.QtCore import QVariant

errors = []  # Collect error logs

# ----------------------------
# Utility functions
# ----------------------------

def list_layer_files_in_dir(directory, exts=(".shp", ".gpkg", ".geojson", ".json")):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(directory, f))]

def choose_utm_crs_from_lonlat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return QgsCoordinateReferenceSystem(f"EPSG:{epsg}")

def angle_to_direction_vector_deg_from_north_clockwise(angle_deg_clockwise):
    # angle_deg_clockwise is degrees clockwise from north.
    # For vector (dx, dy) in x-east, y-north:
    rad = math.radians(angle_deg_clockwise)
    dx = math.sin(rad)
    dy = math.cos(rad)
    return dx, dy

def extract_points_from_geometry(geom):
    """Return list of QgsPointXY from intersection geometry (points or line vertices)."""
    pts = []
    if geom is None or geom.isEmpty():
        return pts
    geomType = geom.type()
    if geomType == QgsWkbTypes.PointGeometry:
        try:
            mpts = geom.asMultiPoint()
            if mpts:
                pts.extend([QgsPointXY(p) for p in mpts])
            else:
                p = geom.asPoint()
                pts.append(QgsPointXY(p))
        except Exception:
            pass
    elif geomType == QgsWkbTypes.LineGeometry or geomType == QgsWkbTypes.PolygonGeometry:
        # extract vertex points of resulting geometry (could be line/linestring)
        try:
            ml = geom.asMultiPolyline()
            for line in ml:
                for p in line:
                    pts.append(QgsPointXY(p))
        except Exception:
            try:
                line = geom.asPolyline()
                for p in line:
                    pts.append(QgsPointXY(p))
            except Exception:
                pass
    return pts

# ----------------------------
# Main processing function
# ----------------------------
def process_directory_layersA(dir_A, path_layerB, path_layerC, out_dir, long_line_length_m=20000):
    """
    dir_A: directory containing LayerA files (each file treated as separate layer)
    path_layerB: path to road edge layer (line)
    path_layerC: path to road area layer (polygon)
    out_dir: directory where to write output E layers
    long_line_length_m: total length of D in meters (default 20 km)
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # Load layer B and C
    layerB = QgsVectorLayer(path_layerB, "layerB_roads", "ogr")
    layerC = QgsVectorLayer(path_layerC, "layerC_roadareas", "ogr")
    if not layerB.isValid() or not layerC.isValid():
        raise RuntimeError("LayerB or LayerC failed to load. Check paths.")

    # Build spatial indices for layerB and layerC (in their native CRS)
    indexB = QgsSpatialIndex()
    geom_dict_B = {}
    for f in layerB.getFeatures():
        geom = f.geometry()
        if geom:
            indexB.addFeature(f)
            # copy geometry safely (QgsGeometry.clone() may not be available)
            geom_dict_B[f.id()] = QgsGeometry.fromWkt(geom.asWkt())

    indexC = QgsSpatialIndex()
    geom_dict_C = {}
    for f in layerC.getFeatures():
        geom = f.geometry()
        if geom:
            indexC.addFeature(f)
            geom_dict_C[f.id()] = QgsGeometry.fromWkt(geom.asWkt())

    layer_files = list_layer_files_in_dir(dir_A)
    print(f"Found {len(layer_files)} layer files in {dir_A}")

    ctx = QgsProject.instance().transformContext()

    for file_path in layer_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing mask file: {base_name}")

        ### Deciding which layerB and layerC to use ###
        current_geom_dictB = geom_dict_B
        current_indexB = indexB
        current_geom_dictC = geom_dict_C
        current_indexC = indexC
        
        # Check if mask_file name contains "Node_{ID}""
        if 'Node' in base_name:
            import re
            node_match = re.search(r'Node(\d+)', base_name)
            if node_match:
                node_id = node_match.group(1)
                print(f'    -> Found Node ID in file name: {node_id}')
                # Search for files containing Node{node_id} in the NODE_ROAD_DIR
                node_road_files = glob.glob(os.path.join(NODE_ROAD_DIR, f"*Node{node_id}*"))
                if node_road_files:
                    node_road_file = node_road_files[0]  # Use the first matching file
                    node_polygon_layer = QgsVectorLayer(node_road_file, f'road_node_{node_id}', 'ogr')
                    if node_polygon_layer.isValid():
                        # Convert polygon layer to line layer if necessary
                        if node_polygon_layer.geometryType() == QgsWkbTypes.PolygonGeometry:
                            params_poly2line = {
                                'INPUT': node_polygon_layer,
                                'OUTPUT': 'memory:'
                            }
                            res_poly2line = processing.run('native:polygonstolines', params_poly2line)
                            node_line_layer = res_poly2line['OUTPUT']

                            # Build spatial indices for layerB and layerC (in their native CRS)
                            node_indexB = QgsSpatialIndex()
                            node_geom_dict_B = {}
                            for f in node_line_layer.getFeatures():
                                node_geom = f.geometry()
                                if node_geom:
                                    node_indexB.addFeature(f)
                                    # copy geometry safely (QgsGeometry.clone() may not be available)
                                    node_geom_dict_B[f.id()] = QgsGeometry.fromWkt(node_geom.asWkt())

                            node_indexC = QgsSpatialIndex()
                            node_geom_dict_C = {}
                            for f in node_polygon_layer.getFeatures():
                                node_geom = f.geometry()
                                if node_geom:
                                    node_indexC.addFeature(f)
                                    node_geom_dict_C[f.id()] = QgsGeometry.fromWkt(node_geom.asWkt())

                                    current_geom_dictB = node_geom_dict_B
                                    current_indexB = node_indexB
                                    current_geom_dictC = node_geom_dict_C
                                    current_indexC = node_indexC

                            print(f'    -> Using Node-specific road layer (converted from polygon): {node_road_file}')
                        else:
                            msg = f'Warning: node road layer {node_road_file} is not polygon, using default road layer'
                            errors.append(msg)
                            print(msg)
                    else:
                        msg = f'Warning: failed to open node road layer {node_road_file}, using default road layer'
                        errors.append(msg)
                        print(msg)
                else:
                    msg = f'Warning: no node road file matching Node{node_id} found in {NODE_ROAD_DIR}, using default road layer'
                    errors.append(msg)
                    print(msg)

        ##### Deciding which layerB and layerC to use #####


        print(f"Processing LayerA: {file_path} ...")
        layerA = QgsVectorLayer(file_path, base_name, "ogr")
        if not layerA.isValid():
            msg = f"  -> Failed to open {file_path}, skipping."
            errors.append(msg)
            print(msg)
            continue

        featA = next(layerA.getFeatures())
        if not featA:
            msg = f"  -> No features in {file_path}, skipping."
            errors.append(msg)
            print(msg)
            continue

        print(f"layerA.crs(), {layerA.crs()}, layerC.crs(), {layerC.crs()}")

        # Pre-check: all centroids must be within some polygon of layerC.
        transform_A_to_C = QgsCoordinateTransform(layerA.crs(), layerC.crs(), ctx)

        cen = featA.geometry().centroid()
        if cen is None or cen.isEmpty():
            msg = f"  -> Feature {featA.id()} centroid empty, skipping layer."
            errors.append(msg)
            print(msg)
            continue
        # create a safe copy of centroid geometry
        cen_inC = QgsGeometry.fromWkt(cen.asWkt())
        try:
            cen_inC.transform(transform_A_to_C)
        except Exception as e:
            msg = f"  -> Transform to LayerC CRS failed for {file_path}, skipping: {e}"
            errors.append(msg)
            print(msg)
            continue

        bbox = cen_inC.boundingBox()
        candidate_ids = current_indexC.intersects(bbox)
        contained_flag = False
        for cid in candidate_ids:
            poly_geom = current_geom_dictC.get(cid)
            if poly_geom and poly_geom.contains(cen_inC):
                contained_flag = True
                continue
        if not contained_flag:
            msg = f"  -> For file {file_path}, Feature {featA.id()} centroid not in any LayerC polygon, skipping whole layer."
            errors.append(msg)
            print(msg)
            continue

        # Prepare output memory layer for E lines
        fields = QgsFields()
        fields.append(QgsField("orig_fid", QVariant.Int))
        fields.append(QgsField("original_road_width_m", QVariant.Double))

        crs_out = layerA.crs()
        mem_layer = QgsVectorLayer(f"LineString?crs={crs_out.authid()}", f"{base_name}_E", "memory")
        prov = mem_layer.dataProvider()
        prov.addAttributes(fields)
        mem_layer.updateFields()

        # Ensure 'original_road_width' attribute exists in layerA; add if necessary
        if "original_road_width_m" not in [fld.name() for fld in layerA.fields()]:
            caps = layerA.dataProvider().capabilities()
            try:
                layerA.dataProvider().addAttributes([QgsField("original_road_width_m", QVariant.Double)])
                layerA.updateFields()
            except Exception as e:
                msg = f"  -> Failed to add 'original_road_width_m' field to LayerA {file_path}: {e}"
                errors.append(msg)
                print(msg)
                continue

        # Ensure 'remaining_road_width_m' attribute exists in layerA; add if necessary
        if "remaining_road_width_m" not in [fld.name() for fld in layerA.fields()]:
            caps = layerA.dataProvider().capabilities()
            try:
                layerA.dataProvider().addAttributes([QgsField("remaining_road_width_m", QVariant.Double)])
                layerA.updateFields()
            except Exception as e:
                msg = f"  -> Failed to add 'remaining_road_width_m' field to LayerA {file_path}: {e}"   
                errors.append(msg)
                print(msg)
                continue

        # For metric computations, decide a metric CRS (use UTM based on centroid lon/lat)
        # We'll pick the UTM zone from the first feature centroid (assume consistent area)
        first_cen = featA.geometry().centroid().asPoint()
        # If layerA CRS is geographic, get lon/lat directly; otherwise transform to EPSG:4326
        if layerA.crs().isGeographic():
            lon, lat = first_cen.x(), first_cen.y()
        else:
            # transform to 4326
            trans_to4326 = QgsCoordinateTransform(layerA.crs(), QgsCoordinateReferenceSystem("EPSG:4326"), ctx)
            p = QgsGeometry.fromPointXY(QgsPointXY(first_cen))
            p.transform(trans_to4326)
            lon, lat = p.asPoint().x(), p.asPoint().y()

        utm_crs = choose_utm_crs_from_lonlat(lon, lat)
        # Coordinate transforms:
        trans_A_to_metric = QgsCoordinateTransform(layerA.crs(), utm_crs, ctx)
        trans_metric_to_A = QgsCoordinateTransform(utm_crs, layerA.crs(), ctx)
        trans_B_to_metric = QgsCoordinateTransform(layerB.crs(), utm_crs, ctx)
        trans_metric_to_B = QgsCoordinateTransform(utm_crs, layerB.crs(), ctx)

        # Prepare distance measuring with GRS80 ellipsoid using LayerA CRS as source CRS
        darea = QgsDistanceArea()
        darea.setEllipsoid("GRS80")
        darea.setSourceCrs(layerA.crs(), ctx)

        half_len = long_line_length_m / 2.0

        # Process each feature
        layerA.startEditing()

        geom = featA.geometry()
        cen_geom = geom.centroid()
        cen_point = cen_geom.asPoint()

        # --- compute angle from perp_x1, perp_y1, perp_x2, perp_y2 attributes (lon,lat)
        px1 = featA.attribute("perp_x1")
        py1 = featA.attribute("perp_y1")
        px2 = featA.attribute("perp_x2")
        py2 = featA.attribute("perp_y2")
        if px1 is None or py1 is None or px2 is None or py2 is None:
            msg = f"  -> File {file_path} Feature {featA.id()} missing perp_* attributes, skipping this feature."
            errors.append(msg)
            print(msg)
            continue
        try:
            lon1 = float(px1); lat1 = float(py1); lon2 = float(px2); lat2 = float(py2)
        except Exception as e:
            msg = f"  -> File {file_path} Feature {featA.id()} perp_* attributes not numeric, skipping: {e}"
            errors.append(msg)
            print(msg)
            continue
        # create point geometries in EPSG:4326 and transform to metric CRS (utm_crs)
        p1_4326 = QgsGeometry.fromPointXY(QgsPointXY(lon1, lat1))
        p2_4326 = QgsGeometry.fromPointXY(QgsPointXY(lon2, lat2))
        trans_4326_to_metric = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:4326"), utm_crs, ctx)
        try:
            p1_4326.transform(trans_4326_to_metric)
            p2_4326.transform(trans_4326_to_metric)
        except Exception as e:
            msg = f"  -> File {file_path} Transform perp points to metric CRS failed for feature {featA.id()}, skipping: {e}"
            errors.append(msg)
            print(msg)
            continue
        p1m = p1_4326.asPoint()
        p2m = p2_4326.asPoint()
        dx_p = p2m.x() - p1m.x()
        dy_p = p2m.y() - p1m.y()
        if abs(dx_p) < 1e-9 and abs(dy_p) < 1e-9:
            msg = f"  -> File {file_path} Feature {featA.id()} perp points identical, skipping."
            errors.append(msg)
            print(msg)
            continue
        # angle in degrees clockwise from north = degrees(atan2(dx, dy))
        angle_X = (math.degrees(math.atan2(dx_p, dy_p)) + 360.0) % 360.0
        print(f"  -> Feature {featA.id()} computed angle_X from perp points: {angle_X:.4f} deg")

        # transform centroid to metric CRS
        cen_metric = QgsGeometry.fromWkt(cen_geom.asWkt())
        try:
            cen_metric.transform(trans_A_to_metric)
        except Exception as e:
            msg = f"  -> File {file_path} Transform centroid to metric CRS failed for feature {featA.id()}, skipping: {e}"
            errors.append(msg)
            print(msg)
            continue
        cen_m = cen_metric.asPoint()

        # direction vector (unit)
        dx, dy = angle_to_direction_vector_deg_from_north_clockwise(angle_X)

        # construct endpoints in metric CRS
        p_plus = QgsPointXY(cen_m.x() + dx * half_len, cen_m.y() + dy * half_len)
        p_minus = QgsPointXY(cen_m.x() - dx * half_len, cen_m.y() - dy * half_len)
        lineD_metric = QgsGeometry.fromPolylineXY([p_minus, p_plus])

        # Transform D into layerB CRS to query indexB
        D_in_B = QgsGeometry.fromWkt(lineD_metric.asWkt())
        try:
            D_in_B.transform(trans_metric_to_B)
        except Exception as e:
            msg = f"  -> File {file_path} Transform D to LayerB CRS failed for feature {featA.id()}, skipping: {e}"
            errors.append(msg)
            print(msg)
            continue
        bboxB = D_in_B.boundingBox()
        candidate_ids_B = current_indexB.intersects(bboxB)

        intersection_points_metric = []
        # Use D_in_B (line D reprojected into LayerB CRS) to intersect with geomB (which is in LayerB CRS).
        # Then transform resulting intersection points back to metric CRS for consistent distance calculations.
        D_for_intersect = QgsGeometry.fromWkt(D_in_B.asWkt())
        for bid in candidate_ids_B:
            geomB = current_geom_dictB.get(bid)
            if geomB is None:
                continue
            try:
                inter = D_for_intersect.intersection(geomB)

            except Exception as e:
                msg = f"  -> File {file_path} intersection failed for feature {bid}: {e}, skipping"
                errors.append(msg)
                print(msg)
                continue
            pts = extract_points_from_geometry(inter)
            # pts are in LayerB CRS; transform each to metric CRS and append
            for pt in pts:
                pt_geom_b = QgsGeometry.fromPointXY(pt)
                try:
                    pt_geom_b.transform(trans_B_to_metric)
                    pt_metric = QgsPointXY(pt_geom_b.asPoint())
                    intersection_points_metric.append(pt_metric)
                except Exception:
                    continue

        if len(intersection_points_metric) < 2:
            msg = f"  -> File {file_path} Feature {featA.id()} intersection points < 2, skipping this feature."
            errors.append(msg)
            print(msg)
            continue

        # compute distances from centroid to intersection points (metric)
        dist_pt_pairs = []
        for pt in intersection_points_metric:
            dd = math.hypot(pt.x() - cen_m.x(), pt.y() - cen_m.y())
            dist_pt_pairs.append((dd, pt))
        dist_pt_pairs.sort(key=lambda x: x[0])
        # take 2 nearest distinct points
        p1_metric = dist_pt_pairs[0][1]
        p2_metric = dist_pt_pairs[1][1]

        # transform points back to LayerA CRS for storage & geodesic distance
        p1_geom_metric = QgsGeometry.fromPointXY(p1_metric)
        p2_geom_metric = QgsGeometry.fromPointXY(p2_metric)
        try:
            p1_geom_metric.transform(trans_metric_to_A)
            p2_geom_metric.transform(trans_metric_to_A)
        except Exception as e:
            msg = f"  -> File {file_path} Transform intersection point back to LayerA CRS failed for feature {featA.id()}, skipping: {e}"
            errors.append(msg)
            print(msg)
            continue
        p1_a = QgsPointXY(p1_geom_metric.asPoint())
        p2_a = QgsPointXY(p2_geom_metric.asPoint())

        # Create line E in LayerA CRS
        lineE = QgsGeometry.fromPolylineXY([p1_a, p2_a])

        # Compute geodesic length (meters) using QgsDistanceArea with GRS80
        try:
            length_m = darea.measureLine([p1_a, p2_a])
        except Exception:
            # fallback: use planar distance in metric projection (reproject to utm and measure)
            # transform p1_a,p2_a to metric and compute planar distance
            p1m = p1_geom_metric.asPoint()
            p2m = p2_geom_metric.asPoint()
            length_m = math.hypot(p1m.x() - p2m.x(), p1m.y() - p2m.y())

        # Update layerA attribute 'original_road_width'
        try:
            f_idx = layerA.fields().indexFromName("original_road_width_m")
            layerA.changeAttributeValue(featA.id(), f_idx, float(length_m))
        except Exception as e:
            msg = f"  -> File {file_path} Failed to set 'original_road_width_m' on feature {featA.id()}: {e}"
            errors.append(msg)
            print(msg)
            continue

        # Update layerA attribute 'remaining_road_width_m'
        try:
            f_idx = layerA.fields().indexFromName("remaining_road_width_m")
            layerA.changeAttributeValue(featA.id(), f_idx, float(length_m) - float(featA.attribute("width_m")))
        except Exception as e:
            msg = f"  -> File {file_path} Failed to set 'remaining_road_width_m' on feature {featA.id()}: {e}"
            errors.append(msg)
            print(msg)
            continue

        # Add feature to E output layer
        out_feat = QgsFeature()
        out_feat.setFields(mem_layer.fields())
        out_feat.setAttribute("orig_fid", int(featA.id()))
        out_feat.setAttribute("original_road_width_m", float(length_m))
        out_feat.setGeometry(lineE)
        prov.addFeatures([out_feat])

        # commit edits on layerA
        layerA.commitChanges()

        # Write mem_layer to GPKG (same base name, suffix _E)
        out_path = os.path.join(out_dir, f"{base_name}_E.gpkg")
        try:
            QgsVectorFileWriter.writeAsVectorFormat(mem_layer, out_path, "utf-8", crs_out, "GPKG")
            print(f"  -> Wrote E layer to: {out_path}")
        except Exception as e:
            msg = f"  -> Failed to write E layer for {base_name}. Error: {e}"
            errors.append(msg)
            print(msg)
        
    # If any errors collected, write them to a log file in output_dir
    if errors:
        err_fp = os.path.join(out_dir, f"processing_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(err_fp, 'w', encoding='utf-8') as ef:
                ef.write(f"Processing errors for run: {datetime.datetime.now().isoformat()}\n")
                ef.write("Errors:\n")
                for e in errors:
                    ef.write(e + '\n')
            print('Wrote error log to:', err_fp)
        except Exception as e:
            print('Failed to write error log file:', e)


dir_A = r"C:\Users\kyohe\Aerial_Photo_Segmenter\Fails0117\MaskBBox"
path_layerB = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RdEdg\wajima_rdedg_edited_dissolved.gpkg"
path_layerC = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RdEdg\wajima_rdedg_edited_dissolved_polygon.gpkg"
NODE_ROAD_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RoadPolygon_Linkwise\wajima"
out_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Passability_WidthLine\Pred_wajima"
long_line_length = 100  

process_directory_layersA(dir_A, path_layerB, path_layerC, out_dir, long_line_length)
