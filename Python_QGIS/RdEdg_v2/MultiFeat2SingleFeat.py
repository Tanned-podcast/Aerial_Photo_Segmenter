# -*- coding: utf-8 -*-
# 1レイヤに複数フィーチャがある場合に、フィーチャごとに別レイヤとして GeoPackage に保存するスクリプト
# 今回は道路ポリゴンの各フィーチャを別レイヤに分割する例
import datetime, traceback
from qgis.core import QgsVectorLayer
import processing
import os

# —— パラメータ —————————————————
input_gpkg = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RdEdg\wajima_roadpolygon_linkwise.gpkg"
out_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RoadPolygon_Linkwise\wajima"
intersection_point_layer_path = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RdEdg\wajima_road_nodes_edited.gpkg"  # 交差点のポイントレイヤ
base_name = input_gpkg.split("\\")[-1].split(".")[0]  # レイヤ名を生成するベース
overwrite = True
# ————————————————————————————

errors = []  # collect error messages during processing
saved_count = 0

input_layer = QgsVectorLayer(input_gpkg, "input_layer", "ogr")
crs = input_layer.crs()  # 元レイヤの CRS

# Load intersection point layer
intersection_layer = QgsVectorLayer(intersection_point_layer_path, "intersection_points", "ogr")
if not intersection_layer.isValid():
    msg = f"Warning: failed to open intersection point layer {intersection_point_layer_path}, all features will be treated as Link"
    errors.append(msg)
    print(msg)
    intersection_layer = None

# Extract and save each feature individually (one file per feature)
for feat in input_layer.getFeatures():
    fid = feat.id()
    geom = feat.geometry()
    
    # Determine if this polygon is a Node polygon by checking if it contains any intersection points
    is_node = False
    if intersection_layer is not None:
        for point_feat in intersection_layer.getFeatures():
            point_geom = point_feat.geometry()
            if geom.contains(point_geom):
                is_node = True
                break
    
    # Generate filename based on whether it's a Node or Link polygon
    feature_type = "Node" if is_node else "Link"
    out_fp = os.path.join(out_dir, f"{base_name}_{feature_type}{fid}.gpkg")

    # Remove existing file so we can overwrite cleanly
    if os.path.exists(out_fp):
        try:
            os.remove(out_fp)
        except Exception:
            pass

    print(f'  -> extracting feature id {fid}')
    expr = f"$id = {fid}"
    try:
        res_ext = processing.run('native:extractbyexpression', {'INPUT': input_layer, 'EXPRESSION': expr, 'OUTPUT': 'memory:'})
        single = res_ext.get('OUTPUT')
        if single is None:
            msg = f"Extraction returned no layer for {base_name} feature {fid}"
            print(f'    -> {msg}, skipping')
            errors.append(msg)
            continue
    except Exception as e:
        msg = f"Failed to extract feature {fid} from {base_name}: {e}"
        print(f'    -> {msg}')
        errors.append(msg)
        errors.append(traceback.format_exc())
        continue

    print('    -> saving to:', out_fp)
    try:
        res = processing.run('native:savefeatures', {'INPUT': single, 'OUTPUT': out_fp})
    except Exception as e:
        msg = f"Failed to save feature {fid} from {base_name} to {out_fp}: {e}"
        print(f'    -> {msg}')
        errors.append(msg)
        errors.append(traceback.format_exc())
        continue

    if res and res.get('OUTPUT'):
        print('    -> saved:', res['OUTPUT'])
        saved_count += 1
    else:
        msg = f"Save returned no output for {base_name} feature {fid}"
        print('    ->', msg)
        errors.append(msg)

    if saved_count == 0:
        print('No clipped features were saved. Exiting.')
    else:
        print(f'Done. Saved {saved_count} clipped layers to directory: {out_dir}')

    # If any errors collected, write them to a log file in output_dir
    if errors:
        err_fp = os.path.join(out_dir, f"processing_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(err_fp, 'w', encoding='utf-8') as ef:
                ef.write(f"Processing errors for run: {datetime.datetime.now().isoformat()}\n")
                ef.write(f"Source dir: {input_gpkg}\n")
                ef.write("Errors:\n")
                for e in errors:
                    ef.write(e + '\n')
            print('Wrote error log to:', err_fp)
        except Exception as e:
            print('Failed to write error log file:', e)
