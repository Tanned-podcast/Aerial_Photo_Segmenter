"""
1番目に実行、まずは被害ラスターをベクターとして書き出す
Clip and aggregate polygonized raster masks by a road polygon using PyQGIS.

Usage (run inside QGIS Python console or a QGIS-enabled Python):
    python ClipMasksbyRoadPolygon.py \
        --roads /path/to/roads.gpkg \
        --masks_dir /path/to/mask_tiffs \
        --output /path/to/out.gpkg \
        [--mask-value 1] [--value-field DN] [--layer-name clipped_masks]

This script will:
 - polygonize each GeoTIFF (GDAL polygonize)
 - keep only polygons where the raster value equals --mask-value
 - clip those polygons by the road polygon layer
 - merge all clipped results into a single vector layer and write to the output

Note: run inside QGIS Python environment (processing must be available).
"""

import os
import glob
import datetime
import traceback

from qgis.core import QgsVectorLayer
import processing


masks_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260327Data_TimeCalc\MaskTIFFs\GT"
output_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260327Data_TimeCalc\MaskVector_Unclipped\GT"


import os
from datetime import datetime
from pathlib import Path

# 処理開始時間の記録
resultspath = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260327Data_TimeCalc\TimeCalc_Result\MaskPolygonizer"
region = "GT"
os.makedirs(resultspath, exist_ok=True)  # 結果保存フォルダがなければ作成
starttime = datetime.now().strftime('%Y%m%d_%H%M%S')
startdate = datetime.now().strftime('%Y%m%d')



def polygonize_raster(raster_path, value_field="DN", errors=None):
    """Polygonize a raster to a temporary GeoPackage and return a QgsVectorLayer (or path)."""
    import tempfile
    import uuid
    tmp_dir = tempfile.gettempdir()
    out_path = os.path.join(tmp_dir, f"poly_{uuid.uuid4().hex}.gpkg")
    params = {
        'INPUT': raster_path,
        'BAND': 1,
        'FIELD': value_field,
        'EIGHT_CONNECTEDNESS': False,
        'EXTRA': '',
        'OUTPUT': out_path
    }
    try:
        res = processing.run('gdal:polygonize', params)
        out = res.get('OUTPUT')
        if isinstance(out, QgsVectorLayer):
            return out
        layer = QgsVectorLayer(out, 'clipped', 'ogr')
        return layer
    except Exception as e:
        msg = f"Polygonize failed for {raster_path}: {e}"
        print('  ->', msg)
        if errors is not None:
            errors.append(msg)
            errors.append(traceback.format_exc())
        return None
    
def main(masks_dir, output_dir, value_field='DN'):

    # Find raster files
    raster_files = sorted(glob.glob(os.path.join(masks_dir, '*.tif')) + glob.glob(os.path.join(masks_dir, '*.tiff')))
    if not raster_files:
        print('ERROR: no raster files found in', masks_dir)
        return

    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    errors = []  # collect error messages during processing

    for rpath in raster_files:
        base = os.path.splitext(os.path.basename(rpath))[0]
        print('Polygonizing:', rpath)
        poly = polygonize_raster(rpath, value_field=value_field, errors=errors)

        # If polygonize failed, skip
        if poly is None:
            print(f'  -> polygonize failed for {rpath}, skipping')
            continue

        # Extract and save each feature individually (one file per feature)
        for feat in poly.getFeatures():
            fid = feat.id()
            out_fp = os.path.join(output_dir, f"{base}_clipped_f{fid}.gpkg")

            # Remove existing file so we can overwrite cleanly
            if os.path.exists(out_fp):
                try:
                    os.remove(out_fp)
                except Exception:
                    pass

            print(f'  -> extracting feature id {fid}')
            expr = f"$id = {fid}"
            try:
                res_ext = processing.run('native:extractbyexpression', {'INPUT': poly, 'EXPRESSION': expr, 'OUTPUT': 'memory:'})
                single = res_ext.get('OUTPUT')
                if single is None:
                    msg = f"Extraction returned no layer for {base} feature {fid}"
                    print(f'    -> {msg}, skipping')
                    errors.append(msg)
                    continue
            except Exception as e:
                msg = f"Failed to extract feature {fid} from {base}: {e}"
                print(f'    -> {msg}')
                errors.append(msg)
                errors.append(traceback.format_exc())
                continue

            print('    -> saving to:', out_fp)
            try:
                res = processing.run('native:savefeatures', {'INPUT': single, 'OUTPUT': out_fp})
            except Exception as e:
                msg = f"Failed to save feature {fid} from {base} to {out_fp}: {e}"
                print(f'    -> {msg}')
                errors.append(msg)
                errors.append(traceback.format_exc())
                continue

            if res and res.get('OUTPUT'):
                print('    -> saved:', res['OUTPUT'])
                saved_count += 1
            else:
                msg = f"Save returned no output for {base} feature {fid}"
                print('    ->', msg)
                errors.append(msg)

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

main(masks_dir, output_dir)

# 計算終了時間の取得とフォーマット
finishtime = datetime.now().strftime('%Y%m%d_%H%M%S')
datepath=str(Path(resultspath + f"\calc_time_{region}_{startdate}.txt"))

# ファイルを新規作成し、日付を書き込む
with open(datepath, 'w', encoding='utf-8') as f:
    f.write(starttime)
    f.write(finishtime)

print("Calculation Finished in ", finishtime)
