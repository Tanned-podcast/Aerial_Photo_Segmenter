"""
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
import sys
import glob

from qgis.core import QgsVectorLayer
import processing


roads_path = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RoadBuffer\RoadBuffer_ALLAREA_MultiWidth_min_DRM_wajima_ONLYurban_NOTsunami_SegAdjusted.gpkg"
masks_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskTIFFs"
output_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskVector"


def polygonize_raster(raster_path, value_field="DN"):
    """Polygonize a raster to a temporary GeoPackage and return a QgsVectorLayer."""
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
    res = processing.run('gdal:polygonize', params)
    return res['OUTPUT']


def clip_by_roads(input_layer, roads_layer):
    params = {
        'INPUT': input_layer,
        'OVERLAY': roads_layer,
        'OUTPUT': 'memory:'
    }
    res = processing.run('native:clip', params)
    out = res['OUTPUT']
    if isinstance(out, QgsVectorLayer):
        return out
    layer = QgsVectorLayer(out, 'clipped', 'ogr')
    return layer


def merge_layers(layers, target_crs, output_path):
    params = {
        'LAYERS': layers,
        'CRS': target_crs,
        'OUTPUT': output_path
    }
    res = processing.run('native:mergevectorlayers', params)
    return res['OUTPUT']


def main(roads_path, masks_dir, output_dir, value_field='DN', layer_name='clipped_vector_masks'):

    # Load roads layer
    roads = QgsVectorLayer(roads_path, 'roads', 'ogr')
    if not roads.isValid():
        print('ERROR: roads layer failed to load:', roads_path)
        sys.exit(1)

    # Find raster files
    raster_files = sorted(glob.glob(os.path.join(masks_dir, '*.tif')) + glob.glob(os.path.join(masks_dir, '*.tiff')))
    if not raster_files:
        print('ERROR: no raster files found in', masks_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0

    for rpath in raster_files:
        print('Polygonizing:', rpath)
        poly = polygonize_raster(rpath, value_field=value_field)

        # Clip by roads polygon
        clipped = clip_by_roads(poly, roads)

        # Skip empty layers
        if clipped.featureCount() == 0:
            print('  -> no features after clipping, skipping')
            continue

        print('  -> clipped features:', clipped.featureCount())

        base = os.path.splitext(os.path.basename(rpath))[0]

        # Extract and save each feature individually (one file per feature)
        for feat in clipped.getFeatures():
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
                res_ext = processing.run('native:extractbyexpression', {'INPUT': clipped, 'EXPRESSION': expr, 'OUTPUT': 'memory:'})
                single = res_ext.get('OUTPUT')
                if single is None:
                    print(f'    -> extraction returned no layer for feature {fid}, skipping')
                    continue
            except Exception as e:
                print(f'    -> failed to extract feature {fid}: {e}')
                continue

            print('    -> saving to:', out_fp)
            try:
                res = processing.run('native:savefeatures', {'INPUT': single, 'OUTPUT': out_fp})
            except Exception as e:
                print(f'    -> failed to save feature {fid}: {e}')
                continue

            if res and res.get('OUTPUT'):
                print('    -> saved:', res['OUTPUT'])
                saved_count += 1
            else:
                print('    -> failed to save feature', fid)

    if saved_count == 0:
        print('No clipped features were saved. Exiting.')
    else:
        print(f'Done. Saved {saved_count} clipped layers to directory: {output_dir}')

main(roads_path, masks_dir, output_dir, layer_name='clipped_vector_masks')
