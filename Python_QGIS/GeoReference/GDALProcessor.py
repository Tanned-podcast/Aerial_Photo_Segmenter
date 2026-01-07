# 必ず OsGeo4W のシェル（OSGeo4W Shell / QGIS OSGeo4W Shell）から実行してください。そうしないと `from osgeo import gdal` が失敗する可能性があります。

"""
バッチ処理スクリプト:
- 画像ディレクトリ内の全画像について
- 同名の GCP ファイルを GCP ディレクトリから探す（拡張子は任意）
- GCP を画像に適用し、gdal.Warp で地理参照付き GeoTIFF を出力する

使い方（例）:
  python GDALProcessor.py --img_dir /path/to/images --gcp_dir /path/to/gcps --out_dir /path/to/out

GCP ファイルはテキストで、各行に少なくとも 4 個の数値が含まれれば読み込みます。
行の並びは自動判定します（pixel line lon lat もしくは lon lat pixel line 等）。
"""

from osgeo import gdal
import os
import glob
import re
import argparse
import sys

# サポートする画像拡張子
IMG_EXTS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.jp2', '.bmp'}


def find_gcp_file(gcp_dir, stem):
    """同名のファイルを gcp_dir から探す（拡張子は任意）。見つかればパスを返す。"""
    pattern = os.path.join(gcp_dir, stem + '.*')
    matches = glob.glob(pattern)
    return matches[0] if matches else None


FLOAT_RE = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')


def read_gcps_from_file(path):
    """GCP テキストファイルを読み、gdal.GCP のリストを返す。
    - ファイル内に "-gcp x y lon lat" の形式があればそれをそのまま読み込む（画像サイズに依存しない）
    - そうでなければ、行ごとに数値を抽出して先頭 4 個を pixel,line,lon,lat として扱う
    """
    gcps = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
            # まず "-gcp x y lon lat" のパターンを探す
            gcp_pattern = re.compile(
                r'-gcp\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
            )
            matches = gcp_pattern.findall(txt)
            if matches:
                for m in matches:
                    px = float(m[0]); ln = float(m[1]); lon = float(m[2]); lat = float(m[3])
                    gcps.append(gdal.GCP(lon, lat, 0, px, ln))
                return gcps
    except Exception as e:
        print(f"[ERROR] GCP ファイルの読み込みで例外: {path} -> {e}")
    return gcps


def process_one_image(in_path, gcp_path, out_path, gcp_srs='EPSG:4326', dst_srs=None, overwrite=False):
    print(f"Processing: {os.path.basename(in_path)}")
    if os.path.exists(out_path) and not overwrite:
        print(f"  SKIP: {out_path} already exists (use --overwrite to replace).")
        return

    ds = gdal.Open(in_path)
    if ds is None:
        print(f"  ERROR: Failed to open {in_path}")
        return

    gcps = read_gcps_from_file(gcp_path)
    if not gcps:
        print(f"  WARNING: No valid GCPs parsed from {gcp_path}; skipping.")
        ds = None
        return

    # 一時ファイルを作成して、まず gdal.Translate で GCP を埋め込む
    import tempfile
    tmp_path = None
    try:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
        tmp_path = tmpf.name
        tmpf.close()

        translate_opts = gdal.TranslateOptions(
            format='GTiff',
            GCPs=gcps,
            # creationOptions=['TILED=YES', 'COMPRESS=LZW']
        )
        trans_ds = gdal.Translate(tmp_path, ds, options=translate_opts)
        if trans_ds is None:
            print(f"  ERROR: gdal.Translate failed for {in_path}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            ds = None
            return
        trans_ds = None

        # いずれの方法でも、GCP の座標参照系 (GCP SRS) を明示的にセットしておく
        try:
            tmp_up = gdal.Open(tmp_path, gdal.GA_Update)
            if tmp_up is not None:
                tmp_up.SetGCPs(gcps, gcp_srs)
                tmp_up = None
        except Exception:
            # 無理に失敗を止めない（後続の Warp で失敗したらログで分かる）
            pass
        
        # WarpOptions を作成
        warp_opts = gdal.WarpOptions(
            format='GTiff',           # 出力フォーマット
            dstSRS=dst_srs if dst_srs else gcp_srs,       # 目標空間参照系
            resampleAlg='near',       # 最近傍補間 (nearest neighbor)
            polynomialOrder=1         # 補間次数 (order 1)
        )

        out_ds = gdal.Warp(out_path, tmp_path, options=warp_opts)
        if out_ds is None:
            print(f"  ERROR: gdal.Warp failed for {in_path}")
        else:
            print(f"  OK -> {out_path}")
            out_ds = None

    finally:
        # 一時ファイルのクリーンアップ
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if ds is not None:
            ds = None


def main():

    img_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Pred_MaskPNG"
    gcp_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\GCP"
    out_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\MaskTIFFs"
    dst_srs = "EPSG:4612"
    gcp_srs = "EPSG:4612"
    overwrite = True

    if not os.path.isdir(img_dir):
        print(f"ERROR: img_dir not found: {img_dir}")
        sys.exit(1)
    if not os.path.isdir(gcp_dir):
        print(f"ERROR: gcp_dir not found: {gcp_dir}")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    # 画像一覧取得
    files = sorted(os.listdir(img_dir))
    img_files = [f for f in files if os.path.splitext(f.lower())[1] in IMG_EXTS]

    if not img_files:
        print("No images found in img_dir.")
        return

    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        in_path = os.path.join(img_dir, fname)
        gcp_file = find_gcp_file(gcp_dir, stem)
        if not gcp_file:
            print(f"SKIP: No GCP file found for {fname} (expected {stem}.*) )")
            continue
        out_fname = stem + '.tif'  # 出力は GeoTIFF
        out_path = os.path.join(out_dir, out_fname)
        process_one_image(in_path, gcp_file, out_path, gcp_srs, dst_srs, overwrite)


if __name__ == '__main__':
    main()
