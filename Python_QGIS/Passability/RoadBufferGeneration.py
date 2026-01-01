#QGIS GUIを立ち上げずにローカルでPyQGISを使うためのコード
#絶対にOSGeo4WのCLIから実行すること。そうでないとPyQt5のDLLが見つからないエラーが出る。
import os
import sys
from pathlib import Path

QGIS_ROOT = r"C:\Program Files\QGIS 3.40.12"

#qgis.coreとpluginsに繋ぐパス
sys.path.append(QGIS_ROOT+r"\apps\qgis-ltr\python")
sys.path.append(QGIS_ROOT+r"\apps\qgis-ltr\python\plugins")

# 5. Import
from qgis.core import QgsApplication, QgsVectorLayer
import math

QgsApplication.setPrefixPath(QGIS_ROOT, True)
qgs = QgsApplication([], False)
qgs.initQgis()
print("PyQGIS OK!")

print("QGIS is working")

from qgis import processing
print("processing is working")

from plugins.processing.core.Processing import Processing
print("plugins.processing.core.Processing is working")

from qgis.analysis import QgsNativeAlgorithms
# **processing用の、QGISnativeアルゴリズムプロバイダを追加する**
# これをしないと、bufferとかの基本的なアルゴリズムが読み込めず「見つかりません」となり使えない
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

def main(road_layer_path=None, output_folder=None, dissolve=False):
    
    try:
        # 入力パラメータの設定
        road_layer_path = road_layer_path
        
        # 幅員階級ごとのバッファ幅の設定
        buffer_widths = {
            1: 13,  # 幅員階級1: 13m
            2: 5.5,   # 幅員階級2: 9m
            3: 3,   # 幅員階級3: 5m
            4: 2,   # 幅員階級4: 3m
        }
        
        # 出力ディレクトリの設定（絶対パスに変更）
        output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"出力ディレクトリを作成しました: {output_folder}")        
        
        # 出力パスの作成
        output_path = str(Path(output_folder+"/RoadBuffer_ALLAREA_MultiWidth_min_"+road_layer_path.split("\\")[-1].split(".")[0]+".gpkg"))
        
        # 入力レイヤーの読み込み
        road_layer = QgsVectorLayer(road_layer_path, "Road Centerline", "ogr")
        if not road_layer.isValid():
            raise Exception("道路中心線レイヤーの読み込みに失敗しました")        
        print("loading input layers done")

        #バッファ処理をメートル単位でするためにはEPSG4612ではダメ：日本測地系2011の7系に直す
        #再投影するCRSの設定
        original_crs = road_layer.crs().authid()
        temp_crs = 'EPSG:6675'
        

        # 再投影（メートル投影系へ）　CRSのDBにアクセスできませんっていうエラー出るけど正常に動くので問題なし
        print("reprojecting roadline...")
        reproj = processing.run("native:reprojectlayer", {
            'INPUT': road_layer,
            'TARGET_CRS': temp_crs,
            'OUTPUT': 'memory:'
        })['OUTPUT']
        print("roadline reprojected")

        # 幅員階級ごとにバッファを生成
        buffer_layers = []
        for width_class, buffer_width in buffer_widths.items():
            print(f"幅員階級 {width_class} のバッファを生成中...")
            
            # 幅員階級でフィルタリング
            expression = f'"R22_005" = {width_class}'
            filtered = processing.run("native:extractbyexpression", {
                'INPUT': reproj,
                'EXPRESSION': expression,
                'OUTPUT': 'memory:'
            })['OUTPUT']
            
            
            # バッファ処理
            #バッファは片側，余裕持たす
            buf = processing.run("native:buffer", {
                'INPUT': filtered,
                'DISTANCE': buffer_width,
                'SEGMENTS': 5,
                'END_CAP_STYLE': 0,
                'JOIN_STYLE': 0,
                'MITER_LIMIT': 2,
                'DISSOLVE': dissolve,
                'OUTPUT': 'memory:'
            })['OUTPUT']
            
            buffer_layers.append(buf)
            print(f"幅員階級 {width_class} のバッファ生成完了")

        # 幅員階級ごとに分かれたバッファレイヤーをマージ
        print("バッファレイヤーをマージ中...")
        merged = processing.run("native:mergevectorlayers", {
            'LAYERS': buffer_layers,
            'OUTPUT': 'memory:'
        })['OUTPUT']

        # 再投影（EPSG:4612 に戻す）
        print("reprojecting buffer...")
        final = processing.run("native:reprojectlayer", {
            'INPUT': merged,
            'TARGET_CRS': original_crs,
            'OUTPUT': output_path
        })['OUTPUT']
        print("buffer reprojected")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
    
    finally:
        # QGISアプリケーションの終了
        qgs.exitQgis()
        print("qgis app exited")

if __name__ == '__main__':
    road_layer_path = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\Roadline\DRM_wajima_ONLYurban_NOTsunami_SegAdjusted.gpkg"  # 適切なパスに変更
    output_folder = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\RoadBuffer"  # 適切なパスに変更
    main(road_layer_path, output_folder, dissolve=True)
    print("program finished")