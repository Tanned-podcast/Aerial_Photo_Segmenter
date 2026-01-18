"""
Clip aggregated road damage polygons by multiple road polygon layers using PyQGIS.

Usage (run inside QGIS Python console or a QGIS-enabled Python):
    python ClipMasksbyMultiRoadPolygon.py

This script will:
 - polygonize all GeoTIFF rasters in the masks directory
 - merge all polygonized layers into a single vector layer (Layer A)
 - load all road polygon files from the roads directory (each file is Layer B)
 - clip Layer A by each Layer B separately
 - save clipped results to the output directory

Note: run inside QGIS Python environment (processing must be available).
"""

import os
import sys
import glob
import datetime
import traceback

from qgis.core import QgsVectorLayer, QgsField, QgsFields
from qgis.PyQt.QtCore import QVariant
import processing


masks_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\MaskTIFFs\GT_suzu"
roads_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\RoadPolygon_Linkwise\suzu"
output_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\MaskVector_Clipped\GT_suzu"


fields = QgsFields()
fields.append(QgsField('filename', QVariant.String))

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
        return res.get('OUTPUT')
    except Exception as e:
        msg = f"Polygonize failed for {raster_path}: {e}"
        print('  ->', msg)
        if errors is not None:
            errors.append(msg)
            errors.append(traceback.format_exc())
        return None


def clip_by_roads(input_layer, roads_layer, context_label=None, errors=None):
    params = {
        'INPUT': input_layer,
        'OVERLAY': roads_layer,
        'OUTPUT': 'memory:'
    }
    try:
        res = processing.run('native:clip', params)
        out = res.get('OUTPUT')
        if isinstance(out, QgsVectorLayer):
            return out
        layer = QgsVectorLayer(out, 'clipped', 'ogr')
        return layer
    except Exception as e:
        label = context_label or 'unknown'
        msg = f"Clipping failed for {label}: {e}"
        print('  ->', msg)
        if errors is not None:
            errors.append(msg)
            errors.append(traceback.format_exc())
        return None


def merge_layers(layers, target_crs, output_path=None):
    """Merge multiple layers into a single layer. If output_path is None, returns a memory layer."""
    if output_path is None:
        output_path = 'memory:'
    params = {
        'LAYERS': layers,
        'CRS': target_crs,
        'OUTPUT': output_path
    }
    res = processing.run('native:mergevectorlayers', params)
    if output_path == 'memory:':
        return res['OUTPUT']
    return res['OUTPUT']


def main(masks_dir, roads_dir, output_dir, value_field='DN'):
    """
    Main processing flow:
    1. Polygonize all rasters in masks_dir
    2. Merge all polygonized layers into Layer A
    3. Load all road polygon files from roads_dir (each is Layer B)
    4. Clip Layer A by each Layer B and save results
    """

    # Find raster files
    raster_files = sorted(glob.glob(os.path.join(masks_dir, '*.tif')) + glob.glob(os.path.join(masks_dir, '*.tiff')))
    if not raster_files:
        print('ERROR: no raster files found in', masks_dir)

    # Find road polygon files (gpkg and shp)
    road_files = sorted(glob.glob(os.path.join(roads_dir, '*.gpkg')) + glob.glob(os.path.join(roads_dir, '*.shp')))
    if not road_files:
        print('ERROR: no road polygon files found in', roads_dir)

    print(f'Found {len(raster_files)} raster files')
    print(f'Found {len(road_files)} road polygon files')

    os.makedirs(output_dir, exist_ok=True)

    errors = []
    
    # Step 1: Polygonize all rasters
    print('\n=== STEP 1: Polygonizing all raster files ===')
    polygonized_layers = []
    for rpath in raster_files:
        base = os.path.splitext(os.path.basename(rpath))[0]
        print(f'Polygonizing: {base}')
        poly = polygonize_raster(rpath, value_field=value_field, errors=errors)
        
        if poly is None:
            print(f'  -> polygonize failed for {rpath}, skipping')
            errors.append(f'Polygonize failed for {rpath}')
            continue
        
        # Load as QgsVectorLayer if it's a path
        if isinstance(poly, str):
            layer = QgsVectorLayer(poly, base, 'ogr')
            if not layer.isValid():
                print(f'  -> failed to load polygonized layer for {base}')
                errors.append(f'Failed to load polygonized layer: {base}')
                continue
        else:
            layer = poly
        
        # Dissolve the polygonized layer
        print(f'  -> Dissolving polygons')
        try:
            res_dissolve = processing.run('native:dissolve', {
                'INPUT': layer,
                'FIELD': '',
                'OUTPUT': 'memory:'
            })
            dissolved_layer = res_dissolve.get('OUTPUT')
            if dissolved_layer is None:
                if isinstance(dissolved_layer, str):
                    layer = QgsVectorLayer(dissolved_layer, base, 'ogr')
                else:
                    layer = dissolved_layer
            else:
                if isinstance(dissolved_layer, str):
                    layer = QgsVectorLayer(dissolved_layer, base, 'ogr')
                else:
                    layer = dissolved_layer
            print(f'  -> Dissolved successfully, features: {layer.featureCount()}')
        except Exception as e:
            msg = f"Dissolve failed for {base}: {e}"
            print(f'  -> {msg}')
            errors.append(msg)
            errors.append(traceback.format_exc())
            # Continue with undissolved layer if dissolve fails
        
        # Add filename field to the layer
        layer.startEditing()

        caps = layer.dataProvider().capabilities()
        if caps & layer.dataProvider().AddAttributes:
            layer.dataProvider().addAttributes(fields)
            layer.updateFields()
        else:
            msg = f'  -> cannot add attributes to layer {base}, skipping filename field'
            errors.append(msg)
            print(msg)
            continue

        
        # Set filename field value for all features
        for feat in layer.getFeatures():
            feat['filename'] = base
            layer.updateFeature(feat)
        
        layer.commitChanges()
        
        print(f'  -> polygonized successfully, features: {layer.featureCount()}')
        polygonized_layers.append(layer)
    
    if not polygonized_layers:
        print('ERROR: No layers were polygonized successfully')
        return
    
    # Step 2: Merge all polygonized layers into Layer A
    print(f'\n=== STEP 2: Merging {len(polygonized_layers)} polygonized layers ===')
    target_crs = polygonized_layers[0].crs()
    
    try:
        layer_A = merge_layers(polygonized_layers, target_crs, output_path='memory:')
        if isinstance(layer_A, str):
            layer_A = QgsVectorLayer(layer_A, 'merged_masks', 'ogr')
        
        print(f'  -> Merged successfully, total features in Layer A: {layer_A.featureCount()}')
    except Exception as e:
        msg = f"Failed to merge polygonized layers: {e}"
        print(f'ERROR: {msg}')
        errors.append(msg)
        errors.append(traceback.format_exc())
        (1)
    
    # Step 3 & 4: For each road polygon file (Layer B), clip Layer A
    print(f'\n=== STEP 3 & 4: Clipping Layer A by each road polygon (Layer B) ===')
    clipped_count = 0
    
    for road_file in road_files:
        road_base = os.path.splitext(os.path.basename(road_file))[0]
        print(f'\nProcessing road layer: {road_base}')
        
        # Load road layer (Layer B)
        roads_layer = QgsVectorLayer(road_file, road_base, 'ogr')
        if not roads_layer.isValid():
            msg = f'Failed to load road layer: {road_file}'
            print(f'  -> ERROR: {msg}')
            errors.append(msg)
            continue
        
        print(f'  -> Loaded road layer, features: {roads_layer.featureCount()}')
        
        # Clip Layer A by this road layer
        try:
            clipped = clip_by_roads(layer_A, roads_layer, context_label=road_base, errors=errors)
            
            if clipped is None:
                print(f'  -> Clipping failed, skipping')
                errors.append(f'Clipping failed for {road_base}')
                continue
            
            if clipped.featureCount() == 0:
                print(f'  -> No features after clipping, skipping')
                errors.append(f'No features after clipping for {road_base}')
                continue
            
            print(f'  -> Clipped successfully, features: {clipped.featureCount()}')
            
            # Extract and save each feature individually (one file per feature)
            for feat in clipped.getFeatures():
                fid = feat.id()
                # Get filename from the feature's filename field
                filename_idx = clipped.fields().indexFromName('filename')
                filename_value = feat.attribute(filename_idx) if filename_idx != -1 else 'unknown'
                output_filename = f"{filename_value}_{road_base}_clipped.gpkg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Remove existing file if it exists
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except Exception as e:
                        print(f'  -> Warning: could not remove existing file: {e}')
                
                print(f'  -> Extracting feature (filename: {filename_value})')
                expr = f"$id = {fid}"
                try:
                    res_ext = processing.run('native:extractbyexpression', {'INPUT': clipped, 'EXPRESSION': expr, 'OUTPUT': 'memory:'})
                    single = res_ext.get('OUTPUT')
                    if single is None:
                        msg = f"Extraction returned no layer for {road_base} feature {fid}"
                        print(f'    -> {msg}, skipping')
                        errors.append(msg)
                        continue
                except Exception as e:
                    msg = f"Failed to extract feature {fid} from {road_base}: {e}"
                    print(f'    -> {msg}')
                    errors.append(msg)
                    errors.append(traceback.format_exc())
                    continue
                
                print(f'    -> Saving to: {output_filename}')
                try:
                    res = processing.run('native:savefeatures', {'INPUT': single, 'OUTPUT': output_path})
                    if res and res.get('OUTPUT'):
                        print(f'    -> Saved successfully: {res["OUTPUT"]}')
                        clipped_count += 1
                    else:
                        msg = f"Save returned no output for {road_base} feature {fid}"
                        print(f'    -> {msg}')
                        errors.append(msg)
                except Exception as e:
                    msg = f"Failed to save feature {fid} from {road_base} to {output_path}: {e}"
                    print(f'    -> {msg}')
                    errors.append(msg)
                    errors.append(traceback.format_exc())
        
        except Exception as e:
            msg = f"Unexpected error processing {road_base}: {e}"
            print(f'  -> ERROR: {msg}')
            errors.append(msg)
            errors.append(traceback.format_exc())
    
    # Summary
    print(f'\n=== PROCESSING COMPLETE ===')
    if clipped_count == 0:
        print('WARNING: No clipped results were saved.')
    else:
        print(f'Successfully saved {clipped_count} clipped layers to: {output_dir}')
    
    # Write error log if there are any errors
    if errors:
        err_fp = os.path.join(output_dir, f"processing_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(err_fp, 'w', encoding='utf-8') as ef:
                ef.write(f"Processing errors for run: {datetime.datetime.now().isoformat()}\n")
                ef.write(f"Masks dir: {masks_dir}\n")
                ef.write(f"Roads dir: {roads_dir}\n")
                ef.write("Errors:\n")
                for e in errors:
                    ef.write(e + '\n')
            print(f'Wrote error log to: {err_fp}')
        except Exception as e:
            print(f'Failed to write error log file: {e}')


main(masks_dir, roads_dir, output_dir)
