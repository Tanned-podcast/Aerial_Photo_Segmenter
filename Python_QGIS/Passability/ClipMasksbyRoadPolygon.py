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

from qgis.core import QgsVectorLayer, QgsProject
import processing


roads_path = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RoadBuffer\RoadBuffer_ALLAREA_MultiWidth_min_DRM_wajima_ONLYurban_NOTsunami_SegAdjusted.gpkg"
masks_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskTIFFs"
output_path = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskVector\ClippedMasks.gpkg"


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


def main(roads_path, masks_dir, output_path, value_field='DN', layer_name='clipped_vector_masks'):

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

    clipped_layers = []

    for rpath in raster_files:
        print('Polygonizing:', rpath)
        poly = polygonize_raster(rpath, value_field=value_field)

        # Clip by roads polygon
        clipped = clip_by_roads(poly, roads)


        # Skip empty layers
        if clipped.featureCount() == 0:
            print('  -> no features after clipping, skipping')
            continue

        clipped_layers.append(clipped)
        print('  -> clipped features:', clipped.featureCount())

    if not clipped_layers:
        print('No clipped features found across all rasters. Exiting.')
        sys.exit(0)

    # Merge all clipped layers
    print('Merging {} clipped layers into {}'.format(len(clipped_layers), output_path))
    merged = merge_layers(clipped_layers, roads.crs().authid(), output_path)

    print('Output written to:', merged)
    # Optionally add to current QGIS project
    try:
        out_layer = QgsVectorLayer(merged, layer_name, 'ogr')
        if out_layer.isValid():
            QgsProject.instance().addMapLayer(out_layer)
            print('Added output layer to the project as "{}"'.format(layer_name))
    except Exception:
        pass

main(roads_path, masks_dir, output_path, layer_name='clipped_vector_masks')
