"""
2番目に実行、道路被害マスク（ベクター）の近くにある道路縁だけを抽出。これを後で一筆書きにして道路領域を作る
PyQGIS script: Road buffer, merge damage masks (GeoPackage), select intersecting buffers, and clip edge lines

日本語の説明:
- レイヤA: 道路中心線（LineString）
- ディレクトリX: 複数の .gpkg（GeoPackage）道路被害マスク
- レイヤE: 道路縁線（LineString/Polygon）

処理:
1) レイヤA を個別バッファ（dissolve=False）→ メモリレイヤ B
2) ディレクトリX の全 .gpkg を読み込みマージ → メモリレイヤ C
3) B のうち C と "intersect（優先）" または "contains" するポリゴンのみ抽出 → メモリレイヤ D
4) レイヤE を D でクリップ → 最終出力

使い方（QGIS Python コンソール内で実行することを想定）:
- 例:
    run_from_paths(
        path_layer_a=r"C:/data/roads_centerlines.shp",
        dir_masks=r"C:/data/masks_gpkg",
        path_layer_e=r"C:/data/roads_edges.shp",
        output_path=r"C:/data/edges_clipped.shp",
        buffer_distance=5,
        prefer_contains=False
    )

注意: スクリプトは QGIS の Python 環境（PyQGIS）内で実行してください。
"""

from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsFeature,
    QgsGeometry,
    QgsSpatialIndex,
    QgsVectorFileWriter,
    QgsWkbTypes,
    QgsFields,
    QgsField,
    QgsFeatureRequest,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessingFeatureSourceDefinition,
)
import processing
import os
import glob
import typing


def load_vector_layer(path_or_layer: typing.Union[str, QgsVectorLayer], name_hint: str = None) -> QgsVectorLayer:
    """Load a vector layer either from an existing QgsVectorLayer or from a file path."""
    if isinstance(path_or_layer, QgsVectorLayer):
        return path_or_layer
    if not os.path.exists(path_or_layer):
        raise FileNotFoundError(f"File not found: {path_or_layer}")
    layer = QgsVectorLayer(path_or_layer, name_hint or os.path.basename(path_or_layer), "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Failed to load layer: {path_or_layer}")
    return layer


def create_memory_layer_from_layer(source_layer: QgsVectorLayer, name: str = "memory_out") -> QgsVectorLayer:
    """Create an empty memory layer with same fields and geometry type as source_layer."""
    geom_type = QgsWkbTypes.displayString(source_layer.wkbType())
    crs = source_layer.crs().authid()
    # Use OGR-style data source definition for memory provider
    uri = f"{source_layer.wkbType()}?crs={crs}"
    mem = QgsVectorLayer(f"{source_layer.wkbType()}?crs={crs}", name, "memory")
    mem_data = mem.dataProvider()
    mem_data.addAttributes(source_layer.fields())
    mem.updateFields()
    return mem


def buffer_features_individually(layer_a: QgsVectorLayer, distance: float, segments: int = 8) -> QgsVectorLayer:
    """Step 1: Buffer each feature individually (dissolve=False) -> memory layer B"""
    print("[Step 1] Buffering features individually...")

    params = {
        'INPUT': layer_a,
        'DISTANCE': distance,
        'SEGMENTS': segments,
        'END_CAP_STYLE': 0,  # Round
        'JOIN_STYLE': 0,  # Round
        'MITER_LIMIT': 2,
        'DISSOLVE': False,
        'OUTPUT': 'memory:'
    }
    res = processing.run('native:buffer', params)
    layer_b = res['OUTPUT']
    layer_b.setName('layer_B_buffers')
    QgsProject.instance().addMapLayer(layer_b, False)
    return layer_b


def merge_GeoPackages_to_memory(dir_gpkg: str, target_crs: QgsCoordinateReferenceSystem = None) -> QgsVectorLayer:
    """Step 2: Merge all .gpkg files in given directory into single memory layer C"""
    print("[Step 2] Scanning directory for .gpkg files...")
    gpkg_paths = glob.glob(os.path.join(dir_gpkg, '*.gpkg'))
    if not gpkg_paths:
        raise FileNotFoundError(f"No .gpkg files found in directory: {dir_gpkg}")

    print(f"Found {len(gpkg_paths)} .gpkg files. Loading...")
    layers = []
    for p in gpkg_paths:
        l = QgsVectorLayer(p, os.path.basename(p), 'ogr')
        if not l.isValid():
            print(f"Warning: failed to load {p}")
            continue
        layers.append(l)

    if not layers:
        raise RuntimeError("No valid .gpkg layers could be loaded.")

    # Use processing merge (native:mergevectorlayers) to merge into memory
    params = {
        'LAYERS': layers,
        'CRS': target_crs.authid() if target_crs is not None else layers[0].crs().authid(),
        'OUTPUT': 'memory:'
    }
    merged = processing.run('native:mergevectorlayers', params)
    layer_c = merged['OUTPUT']
    layer_c.setName('layer_C_merged_masks')
    QgsProject.instance().addMapLayer(layer_c, False)
    print("Merged .gpkg files into memory layer C.")
    return layer_c


def select_buffers_intersecting_masks(layer_b: QgsVectorLayer, layer_c: QgsVectorLayer, prefer_contains: bool = False) -> QgsVectorLayer:
    """Step 3: From B, select polygons that contain OR intersect features from C -> memory layer D"""
    print("[Step 3] Selecting buffer polygons that intersect/contain mask features...")

    # Build spatial index on layer_c
    features_c = list(layer_c.getFeatures())
    if not features_c:
        raise RuntimeError("Layer C has no features.")

    index = QgsSpatialIndex()
    fid_feature_map = {}
    for f in features_c:
        index.insertFeature(f)
        fid_feature_map[f.id()] = f

    # Prepare output memory layer D: same fields as B (we only need geometry and maybe ID)
    fields = layer_b.fields()
    layer_d = QgsVectorLayer(f"Polygon?crs={layer_b.crs().authid()}", 'layer_D_selected_buffers', 'memory')
    layer_d_data = layer_d.dataProvider()
    layer_d_data.addAttributes(fields)
    layer_d.updateFields()

    out_feats = []
    for feat_b in layer_b.getFeatures():
        geom_b = feat_b.geometry()
        bbox = geom_b.boundingBox()
        candidate_ids = index.intersects(bbox)
        match_found = False
        for fid in candidate_ids:
            f_c = fid_feature_map.get(fid)
            if f_c is None:
                continue
            geom_c = f_c.geometry()
            # Prefer intersects (more inclusive). If prefer_contains=True, check contains instead.
            if prefer_contains:
                if geom_b.contains(geom_c):
                    match_found = True
                    break
            else:
                if geom_b.intersects(geom_c):
                    match_found = True
                    break
        if match_found:
            new_feat = QgsFeature()
            new_feat.setGeometry(geom_b)
            new_feat.setFields(layer_d.fields())
            # copy attributes if same fields exist
            for i, field in enumerate(layer_b.fields()):
                try:
                    new_feat.setAttribute(i, feat_b.attribute(i))
                except Exception:
                    pass
            out_feats.append(new_feat)

    if not out_feats:
        print("No buffer polygons matched mask features; layer D will be empty.")
    layer_d_data.addFeatures(out_feats)
    layer_d.updateExtents()
    QgsProject.instance().addMapLayer(layer_d, False)
    print(f"Selected {len(out_feats)} buffer polygons into layer D.")
    return layer_d


def clip_edges_by_d(layer_e: QgsVectorLayer, layer_d: QgsVectorLayer, output_path: str = None) -> QgsVectorLayer:
    """Step 4: Clip layer E by layer D -> output (memory or written file)"""
    print("[Step 4] Clipping layer E by D...")

    params = {
        'INPUT': layer_e,
        'OVERLAY': layer_d,
        'OUTPUT': 'memory:'
    }
    res = processing.run('native:clip', params)
    clipped = res['OUTPUT']
    clipped.setName('edges_clipped')
    QgsProject.instance().addMapLayer(clipped, False)

    if output_path:
        # Save to disk
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = 'GPKG'
        error = QgsVectorFileWriter.writeAsVectorFormatV2(clipped, output_path, QgsProject.instance().transformContext(), options)
        if error[0] == QgsVectorFileWriter.NoError:
            print(f"Clipped result saved to {output_path}")
        else:
            print(f"Failed to save clipped result to {output_path}: {error}")

    return clipped


def run_from_paths(path_layer_a: str, dir_masks: str, path_layer_e: str, output_path: str = None, buffer_distance: float = 5.0, prefer_contains: bool = False, buffer_segments: int = 8):
    """Convenience function to run full pipeline given file paths.

    Note: All path-based layers are loaded via OGR and expected to be valid.
    """
    layer_a = load_vector_layer(path_layer_a, 'layer_A_roads')
    layer_e = load_vector_layer(path_layer_e, 'layer_E_edges')

    # Step 1
    layer_b = buffer_features_individually(layer_a, buffer_distance, segments=buffer_segments)

    # Step 2
    layer_c = merge_GeoPackages_to_memory(dir_masks, target_crs=layer_b.crs())

    # Step 3
    layer_d = select_buffers_intersecting_masks(layer_b, layer_c, prefer_contains=prefer_contains)

    # Step 4
    clipped = clip_edges_by_d(layer_e, layer_d, output_path=output_path)

    print("Processing finished.")
    return {
        'B': layer_b,
        'C': layer_c,
        'D': layer_d,
        'clipped': clipped
    }



example_dir_masks = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskPolygon"
example_layer_a = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Roadline\DRM_wajima_ONLYurban_NOTsunami_SegAdjusted.gpkg"
example_layer_e = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RdEdg\wajima_kibanchizu_rdedg.gpkg"
example_output = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RdEdg\wajima_kibanchizu_rdedg_clipped.gpkg"
buffer_distance = 0.0004


run_from_paths(
        path_layer_a=example_layer_a,
        dir_masks=example_dir_masks,
        path_layer_e=example_layer_e,
        output_path=example_output,
        buffer_distance=buffer_distance,
        prefer_contains=True
    )
