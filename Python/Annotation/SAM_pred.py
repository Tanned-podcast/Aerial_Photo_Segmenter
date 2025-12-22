"""
SAM3（Segment Anything Model v3）を用いて、
テキストプロンプトベースの Semantic Segmentation 用アノテーション（2値マスク）を自動生成するスクリプト。

想定用途：
- 入力：航空写真ディレクトリ（jpg / jpeg / png）
- 出力：
    - 2値マスク（0: Background, 255: Debris） … 人が見やすいマスク
    - CVAT用クラスIDマスク（0: Background, 1: Debris） … CVAT「Segmentation masks」でそのままインポート可能
    - labelmap.txt … クラス定義

実行例:
    python SAM_pred.py --input_dir input_images --output_dir output_masks

事前条件:
- SAM3 がローカルにインストール済みであること
- PyTorch + CUDA が使用可能な環境であること
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List
import sys

import numpy as np
from PIL import Image

import torch

from dotenv import load_dotenv
load_dotenv("../../.env")

sys.path.append("../../sam3")


# SAM3 用のインポート
try:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from sam3.visualization_utils import plot_results
except ImportError as e:  # pragma: no cover - 実行環境依存
    Sam3Model = None
    Sam3Processor = None
    _import_error = e


DEFAULT_PROMPT = "dog"
HF_TOKEN = os.getenv("HF_TOKEN")

def check_cuda() -> torch.device:
    """CUDA が利用可能かをチェックし、利用できる GPU デバイスを返す。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA が利用できません。GPU / CUDA の設定を確認してください。")
    return torch.device("cuda")


def load_sam3_model(device: torch.device):
    """
    SAM3 モデルとプロセッサをロードして eval モードに設定する。
    
    Returns:
        tuple: (model, processor) のタプル
    """
    if Sam3Processor is None:
        raise ImportError(
            "SAM3 のインポートに失敗しました。'transformers' パッケージがインストールされているか確認してください。\n"
            f"元のエラー: {_import_error}"
        )

    # モデルとプロセッサをロード
    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model = model)
    model.eval()
    return model, processor


def run_sam3_inference(
    model,
    processor,
    image_path: Path,
    prompt: str,
) -> Iterable[torch.Tensor]:
    """
    SAM3 によるテキストプロンプト付きマスク推論を実行する。

    Args:
        model: Sam3Model インスタンス
        processor: Sam3Processor インスタンス
        image_path: 入力画像のパス
        prompt: テキストプロンプト

    Returns:
        (H, W) 形状のマスクのリスト（torch.Tensor）
    """
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    print("image loaded")

    # GitHub版SAM3のAPIに合わせて推論を実行
    # Sam3Processorが画像とテキストを統合して処理する    
    inf_state = processor.set_image(image)
    processor.reset_all_prompts(inf_state)
    inf_state = processor.set_text_prompt(state=inf_state, prompt=prompt)

    # 可視化（確認用）
    plot_results(image, inf_state)

    print("processor complete")

    # マスクのリストを返す（各マスクは (H, W) の bool テンソル）
    # Sam3Processor から返されたマスクを list[torch.Tensor(H, W)] に整形
    raw_masks = inf_state.get("masks", [])
    if isinstance(raw_masks, torch.Tensor):
        # 形状: (N, 1, H, W) or (N, H, W)
        if raw_masks.ndim == 4:
            masks = [m.squeeze(0) for m in raw_masks]
        elif raw_masks.ndim == 3:
            masks = [raw_masks[i] for i in range(raw_masks.shape[0])]
        else:
            masks = [raw_masks]
    elif isinstance(raw_masks, (list, tuple)):
        masks = []
        for mask in raw_masks:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            masks.append(mask)
    else:
        masks = []

    print("run_sam3_inf complete")

    return masks


def merge_masks_to_binary(masks: Iterable[torch.Tensor]) -> np.ndarray:
    """
    複数のマスクを OR 統合して 1 枚の 2値マスク (numpy.bool_ 配列, shape: (H, W)) を作成する。

    - Debris と判断されるマスク群をすべて OR 統合
    - マスクが空の場合は、すべて False の配列を返す
    """
    masks_list: List[torch.Tensor] = list(masks)

    if len(masks_list) == 0:
        raise ValueError("SAM3 からマスクが返されませんでした。プロンプトやモデル設定を確認してください。")

    # 1枚目のサイズを基準とする
    first = masks_list[0]
    if first.is_floating_point():
        base = first > 0.5
    else:
        base = first != 0

    merged = base.clone()

    for m in masks_list[1:]:
        if m.shape != merged.shape:
            raise ValueError("SAM3 から返されたマスクの形状が一致しません。前処理やモデル設定を確認してください。")
        if m.is_floating_point():
            merged |= m > 0.5
        else:
            merged |= m != 0

    binary_np = merged.detach().cpu().numpy().astype(bool)

    print("merge masks to binary complete")

    return binary_np


def save_visual_mask(binary_mask: np.ndarray, out_path: Path) -> None:
    """
    人が見やすいように、0/255 の 2値 PNG として保存する。
    - 255: Debris（道路上の瓦礫） → 白
    -   0: Background            → 黒
    """
    vis_mask = (binary_mask.astype(np.uint8) * 255)  # 0 or 255
    img = Image.fromarray(vis_mask, mode="L")
    img.save(out_path, format="PNG")
    
    print("save visual masks complete")


def save_cvat_id_mask(binary_mask: np.ndarray, out_path: Path) -> None:
    """
    CVAT「Segmentation masks」インポート向けに、クラスIDベースのマスクを保存する。
    - 0: Background
    - 1: Debris
    """
    id_mask = binary_mask.astype(np.uint8)  # 0 or 1
    img = Image.fromarray(id_mask, mode="L")
    img.save(out_path, format="PNG")

    print("save cvat id mask complete")


def create_labelmap(labelmap_path: Path) -> None:
    """
    CVAT 用のラベル定義ファイルを作成する。
    CVAT 側の仕様に合わせて、必要に応じてフォーマットを調整してください。
    """
    if labelmap_path.exists():
        return

    with labelmap_path.open("w", encoding="utf-8") as f:
        # シンプルにクラスID: クラス名 形式で定義
        f.write("0: Background\n")
        f.write("1: Debris\n")


def process_images(
    input_dir: Path,
    output_dir: Path,
    model,
    processor,
    device: torch.device,
    prompt: str,
) -> None:
    """入力ディレクトリ内の全画像に対してマスク推論を行い、結果を保存する。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # CVAT 用マスク出力ディレクトリ
    cvat_dir = output_dir / "cvat_masks"
    cvat_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    )

    if not image_files:
        print(f"入力ディレクトリ {input_dir} に対象画像が見つかりませんでした。")
        return

    print(f"処理開始: {len(image_files)} 枚の画像")

    for img_path in image_files:
        stem = img_path.stem
        vis_out = output_dir / f"{stem}.png"       # 0/255 マスク
        cvat_out = cvat_dir / f"{stem}.png"        # 0/1 クラスIDマスク

        print(f"  画像処理中: {img_path.name}")

        try:
            # SAM3 推論
            masks = run_sam3_inference(model, processor, img_path, prompt)

            # マスク統合（OR）
            binary_mask = merge_masks_to_binary(masks)

            # 保存
            save_visual_mask(binary_mask, vis_out)
            save_cvat_id_mask(binary_mask, cvat_out)
        except Exception as e:
            print(f"  エラー: {img_path.name} の処理中にエラーが発生しました: {e}")
            continue

    # labelmap.txt を cvat_masks 直下に作成
    create_labelmap(cvat_dir / "labelmap.txt")

    print("全ての画像の処理が完了しました。")
    print(f"- 2値マスク (0/255): {output_dir}")
    print(f"- CVAT 用クラスIDマスク (0/1) と labelmap.txt: {cvat_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3 を用いて Semantic Segmentation 用アノテーションマスクを生成するスクリプト"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="入力画像ディレクトリ（jpg / jpeg / png）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="出力マスクディレクトリ（2値マスク + CVAT用マスク）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"テキストプロンプト（デフォルト: '{DEFAULT_PROMPT}'）",
    )
    return parser.parse_args()


def main() -> None:

    input_dir = r"../../Sandbox/SAM_Test/img"
    output_dir = r"../../Sandbox/SAM_Test/mask"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prompt = DEFAULT_PROMPT

    # CUDA チェック
    device = check_cuda()

    # SAM3 モデルとプロセッサをロード
    model, processor = load_sam3_model(device)

    # 推論
    process_images(Path(input_dir), Path(output_dir), model, processor, device, prompt)


if __name__ == "__main__":
    main()
