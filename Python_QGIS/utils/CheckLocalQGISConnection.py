#QGIS GUIを立ち上げずにローカルでPyQGISを使うためのコード
#絶対にOSGeo4WのCLIから実行すること。そうでないとPyQt5のDLLが見つからないエラーが出る。
import os
import sys

QGIS_ROOT = r"C:\Program Files\QGIS 3.40.12"

#qgis.coreとpluginsに繋ぐパス
sys.path.append(QGIS_ROOT+r"\apps\qgis-ltr\python")
sys.path.append(QGIS_ROOT+r"\apps\qgis-ltr\python\plugins")

# 5. Import
from qgis.core import QgsApplication

QgsApplication.setPrefixPath(QGIS_ROOT, True)
qgs = QgsApplication([], False)
qgs.initQgis()
print("PyQGIS OK!")

print("QGIS is working")

import qgis.processing
print("qgis.processing is working")

from plugins.processing.core.Processing import Processing
print("plugins.processing.core.Processing is working")

qgs.exitQgis()
print("QGIS exited successfully")
