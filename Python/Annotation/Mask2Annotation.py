from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class COCOLicense:
    id: int = 0
    name: str = ""
    url: str = ""


@dataclass
class COCOInfo:
    description: str = ""
    url: str = ""
    version: str = ""
    year: str = ""
    contributor: str = ""
    date_created: str = ""


@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str = ""


@dataclass
class COCOImage:
    id: int
    width: int
    height: int
    file_name: str
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: int = 0


@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: Dict
    area: float
    bbox: List[float]
    iscrowd: int = 1
    attributes: Dict = None

def mask_to_rle(mask: np.ndarray) -> Dict:
    """
    2値マスク(0/1)を、COCOの非圧縮RLE形式に変換する。

    COCO仕様に合わせて、(height, width) を
    「左上から右方向へ走査し、行ごとに下へ進む」順序で
    フラット化した走査順に対して run-length を計算する。
    """
    assert mask.ndim == 2
    h, w = mask.shape

    #**重要**
    #CVATでは転置した状態で読み込まれてしまうので、対策としてあらかじめこちらで転置
    mask = mask.T

    # 行方向(C-order)で 1D にする
    pixels = mask.reshape(-1)

    counts: List[int] = []
    prev = 0
    count = 0

    for p in pixels:
        if p == prev:
            count += 1
        else:
            counts.append(count)
            count = 1
            prev = int(p)
    counts.append(count)

    # 最初の要素は「0の連続数」である必要がある
    if pixels[0] == 1:
        counts = [0] + counts

    return {"counts": counts, "size": [int(h), int(w)]}


def compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """2値マスク(0/1)から [x_min, y_min, width, height] のbboxを算出。"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, 0, 0, 0
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return x_min, y_min, int(x_max - x_min + 1), int(y_max - y_min + 1)


def build_categories_from_values(values: List[int]) -> List[COCOCategory]:
    """
    画像中に現れる画素値のリストから COCO categories を作成。
    0/1 のみの場合は、例に合わせて Background / Debris とする。
    それ以外は Class{v} という名前でカテゴリを作る。
    """
    uniq = sorted(set(values))
    categories: List[COCOCategory] = []

    if uniq == [0, 1]:
        categories.append(COCOCategory(id=1, name="Background", supercategory=""))
        categories.append(COCOCategory(id=2, name="Debris", supercategory=""))
    else:
        for v in uniq:
            categories.append(
                COCOCategory(id=int(v) + 1, name=f"Class{int(v)}", supercategory="")
            )
    return categories


def create_coco_from_masks(input_dir: Path, output_json: Path) -> None:
    png_paths = sorted(
        p for p in input_dir.glob("*.png") if p.is_file()
    )
    if not png_paths:
        raise FileNotFoundError(f"PNG画像が見つかりません: {input_dir}")

    images: List[COCOImage] = []
    annotations: List[COCOAnnotation] = []

    # まず全画像のユニーク画素値を集めてカテゴリを決定
    all_values: List[int] = []
    for p in png_paths:
        img = Image.open(p).convert("L")
        arr = np.array(img)
        all_values.extend(np.unique(arr).tolist())

    categories = build_categories_from_values(all_values)
    # 画素値 v -> category_id の対応
    if set(all_values) == {0, 1}:
        value_to_cat = {0: 1, 1: 2}
    else:
        value_to_cat = {int(v): int(v) + 1 for v in set(all_values)}

    ann_id = 1

    for img_id, p in enumerate(png_paths, start=1):
        img = Image.open(p).convert("L")
        arr = np.asarray(img, order = "C")
        h, w = arr.shape

        images.append(
            COCOImage(
                id=img_id,
                width=int(w),
                height=int(h),
                file_name=p.name,
            )
        )

        #グレースケール画像に含まれるクラスのIDをすべて取ってくる
        unique_vals = sorted(np.unique(arr).tolist())
        #各クラスに対して
        for v in unique_vals:
            # 背景(画素値0など)はスキップし、前景クラスのみRLEを出力する
            if v not in value_to_cat or int(v) == 0:
                continue

            #クラスIDと画素値が同じところだけ1、他は0で取得　???
            mask = (arr == v).astype(np.uint8) 
            area = float(mask.sum())
            if area == 0:
                continue

            x, y, bw, bh = compute_bbox(mask)
            if bw == 0 or bh == 0:
                continue

            rle = mask_to_rle(mask)
            ann = COCOAnnotation(
                id=ann_id,
                image_id=img_id,
                category_id=value_to_cat[int(v)],
                segmentation=rle,
                area=area,
                bbox=[float(x), float(y), float(bw), float(bh)],
                iscrowd=1,
                attributes={"occluded": False},
            )
            annotations.append(ann)
            ann_id += 1

    coco = {
        "licenses": [asdict(COCOLicense())],
        "info": asdict(COCOInfo()),
        "categories": [asdict(c) for c in categories],
        "images": [asdict(im) for im in images],
        "annotations": [asdict(a) for a in annotations],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "グレースケールPNG画像(画素値がクラスID)から "
            "COCO形式アノテーションJSONを生成します。"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="入力PNG画像が格納されているディレクトリ",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="出力するCOCO形式JSONファイルのパス",
    )
    return parser.parse_args()


def main() -> None:
    # args = parse_args()
    # input_dir = Path(args.input_dir)
    # output_json = Path(args.output_json)
    root = "../../Sandbox/SAM_Test"
    input_dir = Path(root + "/mask/cvat_masks")
    output_json = Path(root + "/annotation.json")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"入力ディレクトリが存在しません: {input_dir}")

    create_coco_from_masks(input_dir, output_json)
    print(f"Annotation JSON created in {output_json}")


if __name__ == "__main__":
    main()


