# PyQGIS script: filter and merge polygons where remaining_road_width_m > 4
# Run in QGIS Python Console or a PyQGIS-enabled script

import os
from qgis.core import (
    QgsVectorLayer, QgsFields, QgsField, QgsFeature, QgsGeometry,
    QgsVectorFileWriter, QgsProject, QgsCoordinateTransform,
    QgsCoordinateTransformContext, QgsWkbTypes
)

def merge_filtered_polygons(input_dir, output_gpkg, output_layer_name, threshold):
    """
    Scan input_dir for vector files, extract features with remaining_road_width_m > threshold,
    merge into a single layer and write to output_gpkg (GeoPackage).
    """
    found_features = []  # tuples: (QgsGeometry, dict(attributes), source_crs)
    fields_map = {}      # name -> QgsField (first-seen field definition)
    dest_crs = None
    geom_wkb_type = QgsWkbTypes.MultiPolygon

    def process_layer(vlayer):
        nonlocal dest_crs
        if not vlayer or not vlayer.isValid():
            print("Invalid layer:", vlayer)
            return
        # find attribute
        if vlayer.fields().indexFromName('remaining_road_width_m') == -1:
            print("No 'remaining_road_width_m' field found.")
            return
        if dest_crs is None:
            dest_crs = vlayer.crs()
        # collect fields definitions
        for f in vlayer.fields():
            if f.name() not in fields_map:
                fields_map[f.name()] = QgsField(f)
        # iterate features
        for feat in vlayer.getFeatures():
            val = feat['remaining_road_width_m']
            if val is None:
                print("val is None")
                continue
            
            try:
                if float(val) > float(threshold):
                    geom = QgsGeometry.fromWkt(feat.geometry().asWkt())
                    attrs = {f.name(): feat[f.name()] for f in vlayer.fields()}
                    found_features.append((geom, attrs, vlayer.crs()))
            except Exception:
                print("feature register failed")
                continue

    # walk directory and open vector files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            lower = file.lower()
            path = os.path.join(root, file)
            print("Processing file:", path)

            vlayer = QgsVectorLayer(path, os.path.splitext(file)[0], 'ogr')
            process_layer(vlayer)

    if not found_features:
        print("No features found with remaining_road_width_m >", threshold)
        return

    # build final memory layer
    if dest_crs is None:
        dest_crs = found_features[0][2]  # fallback
    mem_layer = QgsVectorLayer(f"MultiPolygon?crs={dest_crs.authid()}", output_layer_name, "memory")
    mem_dp = mem_layer.dataProvider()

    # set fields (use order of insertion)
    ordered_fields = list(fields_map.values())
    mem_dp.addAttributes(ordered_fields)
    mem_layer.updateFields()

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


threshold = 4.0  # threshold for remaining_road_width_m
input_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RdEdg\MaskBBox"
output_gpkg = rf"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RdEdg\Result\filtered_unpassables_thres{int(threshold)}m.gpkg"
output_layer_name='filtered_unpassables'

merge_filtered_polygons(input_dir, output_gpkg, output_layer_name='filtered_unpassables', threshold=threshold)
    