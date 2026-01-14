# PyQGIS script: filter and merge polygons where remaining_road_width_m > 4
# Run in QGIS Python Console or a PyQGIS-enabled script

import os, datetime
from qgis.core import (
    QgsVectorLayer, QgsFields, QgsField, QgsFeature, QgsGeometry,
    QgsVectorFileWriter, QgsProject, QgsCoordinateTransform,
    QgsCoordinateTransformContext, QgsWkbTypes
)
from qgis.PyQt.QtCore import QVariant

errors = []  # collect error messages

def merge_filtered_polygons(input_dir, output_gpkg, out_dir, output_layer_name, threshold):
    """
    Scan input_dir for vector files, extract features with remaining_road_width_m > threshold,
    merge into a single layer and write to output_gpkg (GeoPackage).
    """
    found_features = []  # tuples: (QgsGeometry, dict(attributes), source_crs)
    fields_map = QgsFields()      # name -> QgsField (first-seen field definition)
    dest_crs = None
    geom_wkb_type = QgsWkbTypes.MultiPolygon

    def process_layer(vlayer, path):
        nonlocal dest_crs
        if not vlayer or not vlayer.isValid():
            msg = f"Invalid layer: {path}"
            errors.append(msg)
            print(msg)
            return
        # find attribute
        if vlayer.fields().indexFromName('remaining_road_width_m') == -1:
            msg = f"No 'remaining_road_width_m' field found in {path}."
            errors.append(msg)
            print(msg)
            return
        if dest_crs is None:
            dest_crs = vlayer.crs()
        # collect fields definitions (skip 'fid' which can cause read errors)
        for f in vlayer.fields():
            fname = f.name()
            if fname.lower() == 'fid':
                continue
            if fname not in fields_map:
                fields_map.append(QgsField(f))
        # ensure filename field present to record source file name
        if 'filename' not in fields_map:
            fields_map.append(QgsField('filename', QVariant.String))
        # iterate features
        for feat in vlayer.getFeatures():
            val = feat['remaining_road_width_m']
            if val is None:
                msg = f"val is None for file {path} feature ID {feat.id()}"
                errors.append(msg)
                print(msg)
                continue
            
            try:
                if True:
                # if float(val) > float(threshold):
                # if float(val) <= float(threshold):

                    geom = QgsGeometry.fromWkt(feat.geometry().asWkt())
                    # build attributes dict but skip 'fid' to avoid read errors
                    attrs = {f.name(): feat[f.name()] for f in vlayer.fields() if f.name().lower() != 'fid'}
                    # add source filename (basename) so output layer records origin

                    attrs['filename'] = os.path.basename(path)
                    found_features.append((geom, attrs, vlayer.crs()))
            except Exception:
                msg = f"feature register failed for file {path} feature ID {feat.id()}"
                errors.append(msg)
                print(msg)
                continue

    # walk directory and open vector files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            lower = file.lower()
            path = os.path.join(root, file)
            print("Processing file:", path)

            vlayer = QgsVectorLayer(path, os.path.splitext(file)[0], 'ogr')
            process_layer(vlayer, path)

    if not found_features:
        print("No features found with remaining_road_width_m >", threshold)
        return

    # build final memory layer
    if dest_crs is None:
        dest_crs = found_features[0][2]  # fallback
    mem_layer = QgsVectorLayer(f"MultiPolygon?crs={dest_crs.authid()}", output_layer_name, "memory")
    mem_dp = mem_layer.dataProvider()

    # set fields (use order of insertion)
    mem_dp.addAttributes(fields_map)
    mem_layer.updateFields()

    print(fields_map)
    # prepare coordinate transform context
    transform_context = QgsProject.instance().transformContext()

    # create and add features (transform geometry if needed)
    feat_list = []
    for geom, attrs_dict, src_crs in found_features:
        # transform geometry to dest_crs if necessary
        if src_crs != dest_crs:
            xform = QgsCoordinateTransform(src_crs, dest_crs, transform_context)
            try:
                geom.transform(xform)
            except Exception:
                pass
        new_feat = QgsFeature()
        new_feat.setGeometry(geom)
        # set attributes in target field order
        attr_values = [attrs_dict.get(f.name(), None) for f in mem_layer.fields()]
        new_feat.setAttributes(attr_values)
        feat_list.append(new_feat)

    mem_dp.addFeatures(feat_list)
    mem_layer.updateExtents()

    # write to GeoPackage
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = output_layer_name

    err = QgsVectorFileWriter.writeAsVectorFormatV2(mem_layer, output_gpkg, transform_context, options)
    # err is a tuple (error_code, error_message) in some QGIS versions or an enum; handle generically
    if isinstance(err, tuple):
        code, msg = err
        if code == QgsVectorFileWriter.NoError:
            print(f"Saved {len(feat_list)} features to {output_gpkg} as layer '{output_layer_name}'")
        else:
            raise RuntimeError(f"Error saving GeoPackage: {msg}")
    else:
        # try older return semantics (0 success)
        if err == QgsVectorFileWriter.NoError:
            print(f"Saved {len(feat_list)} features to {output_gpkg} as layer '{output_layer_name}'")
        else:
            raise RuntimeError("Error writing GeoPackage")
        
    # If any errors collected, write them to a log file in output_dir
    if errors:
        err_fp = os.path.join(out_dir, f"processing_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(err_fp, 'w', encoding='utf-8') as ef:
                ef.write(f"Processing errors for run: {datetime.datetime.now().isoformat()}\n")
                ef.write(f"Source output: {output_gpkg}\n")
                ef.write("Errors:\n")
                for e in errors:
                    ef.write(e + '\n')
            print('Wrote error log to:', err_fp)
        except Exception as e:
            print('Failed to write error log file:', e)


threshold = 4.0  # threshold for remaining_road_width_m
input_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\MaskBBox\Pred_wajima"

out_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_QGIS\Pred_wajima"

output_gpkg = rf"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_QGIS\Pred_wajima\Pred_wajima_overall.gpkg"
# output_gpkg = rf"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_QGIS\Pred_wajima\Pred_wajima_filtered_passables_thres{int(threshold)}m.gpkg"
# output_gpkg = rf"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_QGIS\Pred_wajima\Pred_wajima_filtered_unpassables_thres{int(threshold)}m.gpkg"

output_layer_name='overall'
# output_layer_name='filtered_passables'
# output_layer_name='filtered_unpassables'

merge_filtered_polygons(input_dir, output_gpkg, out_dir=out_dir, output_layer_name=output_layer_name, threshold=threshold)
    