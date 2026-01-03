# 色付きマスク画像を白黒マスク画像に変換するスクリプト
from pathlib import Path

import numpy as np
from PIL import Image

# 読み込み元ディレクトリを指定
# 例: input_dir = r"../../Sandbox/SAM_Test/mask/cvat_masks"
input_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20251209Data\TrainVal\mask"  # 必要に応じてパスを書き換えてください
out_root = Path(r"C:\Users\kyohe\Aerial_Photo_Segmenter\Sandbox\SegCode_Test\ColorMask2Binary")

# 画像ファイルを取得（png/jpg/jpeg）
img_paths = [
    p for p in Path(input_dir).glob("*")
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
]

if not img_paths:
    raise FileNotFoundError(f"画像ファイルが見つかりません: {input_dir}")

for img_path in img_paths:
    # 画像を読み込み
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    # 画素値が0なら0のまま、それ以外(1以上)を1に変換
    binary = (arr > 0).astype(np.uint8)

    # バイナリ画像を保存
    out_path = out_root / f"{img_path.stem}_binary.png"
    Image.fromarray(binary * 255).save(out_path)

    print(f"Saved binary mask: {out_path}")
