"""
DeepLabV3-ResNet101 セマンティックセグメンテーションモデルに対する
瓦礫（Debris）クラスに着目した CAM 系 XAI 可視化・確率マップ可視化・および両者の関係分析可視化

要件:
- PyTorch + torchvision.models.segmentation.deeplabv3_resnet101
- 2クラス (0: background, 1: debris)
- pytorch-grad-cam を使用して5種類のXAI手法を実装
- 各テスト画像に対して7サブプロットのPNGを出力
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings
import seaborn as sns
import japanize_matplotlib
sns.set() # seabornの設定を無効化
sns.reset_orig() # seabornの設定をリセットしてmatplotlibのデフォルトに戻す

# matplotlibのスタイルを明示的に設定（seabornの影響を排除）
plt.style.use('default')  # matplotlibのデフォルトスタイルを使用
#seabornのフォントは必ずjapanize_matplotlibのデフォルトのやつに合わせること　でないと文字化け
sns.set(font='IPAexGothic')

# pytorch-grad-cam
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM
)
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# torchvision
from torch.types import Device
import torchvision.transforms as T
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

warnings.filterwarnings('ignore')


class DeepLabV3Wrapper(torch.nn.Module):
    """
    DeepLabV3モデルをラップして、CAM計算用に出力を統一する
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        # DeepLabV3は辞書形式で出力する場合がある
        if isinstance(output, dict):
            return output['out']
        return output


def load_model(model_weight_path: str, num_classes: int = 2, device: torch.device = None, use_pretrained = False) -> torch.nn.Module:
    """
    学習済みDeepLabV3-ResNet101モデルをロードする
    
    Args:
        model_weight_path: 学習済み重みファイルのパス
        num_classes: クラス数（デフォルト: 2）
        device: デバイス
    
    Returns:
        ロードされたモデル（evalモード）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if use_pretrained:
        # Prepare model (use torchvision's deeplabv3_resnet101)
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    else:
        # モデル構築
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        # ヘッドを置き換え
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        # 重みをロード
        state_dict = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(state_dict)


    
    # evalモードに設定
    model.eval()
    model.to(device)
    
    # ラッパーで包む
    wrapped_model = DeepLabV3Wrapper(model)
    
    return wrapped_model


def preprocess_image(image_path: str, img_size: Tuple[int, int] = (512, 512)) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    画像を前処理してテンソルに変換する
    
    Args:
        image_path: 画像ファイルのパス
        img_size: リサイズサイズ (H, W)
    
    Returns:
        (preprocessed_tensor, original_image_array, original_size)
    """
    # 画像を読み込み
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)
    
    # 前処理（ImageNet正規化）
    transform = T.Compose([
        T.Resize(img_size, interpolation=Image.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # テンソルに変換
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    
    # 元画像をnumpy配列に変換（可視化用）
    img_array = np.array(img.resize(img_size, Image.BILINEAR)) / 255.0  # (H, W, 3), [0, 1]
    
    return img_tensor, img_array, original_size


def get_probability_map(model: torch.nn.Module, img_tensor: torch.Tensor, 
                       debris_class_id: int = 1, device: torch.device = None) -> np.ndarray:
    """
    モデル推論を行い、瓦礫クラスの確率マップを取得する
    
    Args:
        model: 学習済みモデル
        img_tensor: 前処理済み画像テンソル (1, 3, H, W)
        debris_class_id: 瓦礫クラスのID（デフォルト: 1）
        device: デバイス
    
    Returns:
        瓦礫クラスの確率マップ (H, W)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # 推論
        logits = model(img_tensor)  # (1, C, H, W)
        
        # Softmaxを適用
        probs = F.softmax(logits, dim=1)  # (1, C, H, W)
        
        # 瓦礫クラスのみ抽出
        prob_map = probs[0, debris_class_id].cpu().numpy()  # (H, W)
    
    return prob_map


def get_prediction_mask(model: torch.nn.Module, img_tensor: torch.Tensor, 
                       img_size: Tuple[int, int], device: torch.device = None) -> np.ndarray:
    """
    モデル推論を行い、予測マスクを生成する
    背景クラス(0) → 画素値0(黒)、瓦礫クラス(1) → 画素値255(白)
    
    Args:
        model: 学習済みモデル
        img_tensor: 前処理済み画像テンソル (1, 3, H, W)
        img_size: 画像サイズ (H, W)
        device: デバイス
    
    Returns:
        予測マスク (H, W) - 画素値は0または255
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # 推論
        logits = model(img_tensor)  # (1, C, H, W)
        
        # クラス予測（argmax）
        pred = torch.argmax(logits, dim=1)  # (1, H, W)
        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W)
        
        # 背景クラス(0) → 0、瓦礫クラス(1) → 255に変換
        pred_mask = np.where(pred == 1, 255, 0).astype(np.uint8)
    
    return pred_mask


def load_gt_mask(gt_mask_path: str, img_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Ground Truthマスクを読み込む
    
    Args:
        gt_mask_path: GTマスクファイルのパス
        img_size: リサイズサイズ (H, W)
    
    Returns:
        GTマスク (H, W) - 画素値は0または255、ファイルが存在しない場合はNone
    """
    if not os.path.exists(gt_mask_path):
        return None
    
    try:
        # マスクを読み込み
        mask = Image.open(gt_mask_path).convert('L')
        
        # リサイズ
        mask = mask.resize(img_size, Image.NEAREST)
        
        # numpy配列に変換
        mask_array = np.array(mask, dtype=np.uint8)
        
        # 2値化（127を閾値として、背景=0、瓦礫=255に変換）
        # 既に0/255の場合はそのまま、0-255の範囲の場合は閾値処理
        if mask_array.max() > 1:
            mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        else:
            mask_array = (mask_array * 255).astype(np.uint8)
        
        return mask_array
    except Exception as e:
        print(f"警告: GTマスクの読み込みに失敗しました ({gt_mask_path}): {e}")
        return None


def normalize_map(map_array: np.ndarray) -> np.ndarray:
    """
    マップを[0, 1]範囲に正規化する
    
    Args:
        map_array: 入力マップ
    
    Returns:
        正規化されたマップ
    """
    min_val = map_array.min()
    max_val = map_array.max()
    if max_val - min_val > 1e-8:
        return (map_array - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(map_array)


def create_4class_heatmap(cam_norm: np.ndarray, prob_norm: np.ndarray) -> np.ndarray:
    """
    「見ている × 信じている」4分類ヒートマップを作成する
    
    Args:
        cam_norm: 正規化済みCAMマップ (H, W)
        prob_norm: 正規化済み確率マップ (H, W)
    
    Returns:
        4分類ヒートマップ (H, W) - 値は0, 1, 2, 3
    """
    heatmap = np.zeros_like(cam_norm, dtype=np.int32)
    
    # 閾値: 0.5
    threshold = 0.5
    
    # 見ている × 信じている: 0 (赤)
    mask = (cam_norm >= threshold) & (prob_norm >= threshold)
    heatmap[mask] = 0
    
    # 見ていない × 信じている: 1 (橙)
    mask = (cam_norm < threshold) & (prob_norm >= threshold)
    heatmap[mask] = 1
    
    # 見ている × 信じていない: 2 (青)
    mask = (cam_norm >= threshold) & (prob_norm < threshold)
    heatmap[mask] = 2
    
    # 見ていない × 信じていない: 3 (灰)
    mask = (cam_norm < threshold) & (prob_norm < threshold)
    heatmap[mask] = 3
    
    return heatmap


def visualize_xai_results(
    original_img: np.ndarray,
    cam_map: np.ndarray,
    prob_map: np.ndarray,
    cam_norm: np.ndarray,
    prob_norm: np.ndarray,
    evidence_map: np.ndarray,
    heatmap_4class: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    output_path: str,
    xai_method_name: str,
    alpha: float = 0.4
):
    """
    9サブプロットの可視化結果をPNGとして保存する
    
    Args:
        original_img: 元画像 (H, W, 3), [0, 1]
        cam_map: CAMマップ (H, W)
        prob_map: 確率マップ (H, W)
        cam_norm: 正規化済みCAMマップ (H, W)
        prob_norm: 正規化済み確率マップ (H, W)
        evidence_map: CAM × 確率の積マップ (H, W)
        heatmap_4class: 4分類ヒートマップ (H, W)
        pred_mask: 予測マスク (H, W) - 画素値0または255
        gt_mask: Ground Truthマスク (H, W) - 画素値0または255、Noneの場合は表示しない
        output_path: 出力ファイルパス
        xai_method_name: XAI手法名
        alpha: αブレンドの透明度
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    # 共通のカラーマップとスケール
    vmin_common = min(cam_norm.min(), prob_norm.min(), evidence_map.min())
    vmax_common = max(cam_norm.max(), prob_norm.max(), evidence_map.max())
    

    fontsize = 36
    labelsize = 24
    legendsize = 24

    # 1. 元画像
    axes[0].imshow(original_img)
    axes[0].set_title('元画像', fontsize=fontsize)
    axes[0].axis('off')

    # 2. 予測マスク
    axes[1].imshow(pred_mask, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('予測マスク', fontsize=fontsize)
    axes[1].axis('off')
    
    # 3. Ground Truthマスク
    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title('Ground Truthマスク', fontsize=fontsize)
    else:
        axes[2].text(0.5, 0.5, 'GTマスク\n未取得', 
                    ha='center', va='center', fontsize=fontsize, 
                    transform=axes[8].transAxes)
        axes[2].set_title('Ground Truthマスク', fontsize=fontsize)
    axes[2].axis('off')
    
    # 4. 元画像 + CAM（αブレンド）
    # show_cam_on_imageのimage_weightは元画像の重み（CAMの重みは1-image_weight）
    cam_overlay = show_cam_on_image(original_img, cam_norm, use_rgb=True, image_weight=1.0-alpha)
    axes[3].imshow(cam_overlay)
    axes[3].set_title(f'元画像 &\n Grad-CAM++', fontsize=fontsize)
    axes[3].axis('off')
    
    # 5. 元画像 + 確率マップ（αブレンド）
    prob_overlay = show_cam_on_image(original_img, prob_norm, use_rgb=True, image_weight=1.0-alpha)
    axes[4].imshow(prob_overlay)
    axes[4].set_title('元画像 &\n瓦礫クラス確率マップ', fontsize=fontsize)
    axes[4].axis('off')
    
    # 6. 「見ている × 信じている」4分類ヒートマップ
    colors = ['red', 'orange', 'blue', 'gray']
    cmap_custom = ListedColormap(colors)
    im7 = axes[5].imshow(heatmap_4class, cmap=cmap_custom, vmin=0, vmax=3)
    axes[5].set_title('4分類ヒートマップ', fontsize=fontsize)
    axes[5].axis('off')

    # 7. 瓦礫クラス CAM（正規化済）
    im2 = axes[6].imshow(cam_norm, cmap='jet', vmin=0, vmax=1)
    axes[6].set_title(f'Grad-CAM++', fontsize=fontsize)
    axes[6].axis('off')
    plt.colorbar(im2, ax=axes[6], fraction=0.046, pad=0.04).ax.tick_params(labelsize=labelsize)
    
    # 8. 瓦礫クラス確率マップ（正規化済）
    im3 = axes[7].imshow(prob_norm, cmap='jet', vmin=0, vmax=1)
    axes[7].set_title('瓦礫クラス確率マップ', fontsize=fontsize)
    axes[7].axis('off')
    plt.colorbar(im3, ax=axes[7], fraction=0.046, pad=0.04).ax.tick_params(labelsize=labelsize)

    # # 9. CAM × 確率の積マップ（evidence map）
    # im6 = axes[8].imshow(evidence_map, cmap='jet', vmin=vmin_common, vmax=vmax_common)
    # axes[8].set_title('GradCAM++ × 瓦礫クラス確率マップ', fontsize=fontsize)
    # axes[8].axis('off')
    # plt.colorbar(im6, ax=axes[8], fraction=0.046, pad=0.04).ax.tick_params(labelsize=14)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='CAMの値大 × クラス確率大'),
        Patch(facecolor='orange', label='CAMの値小 × クラス確率大'),
        Patch(facecolor='blue', label='CAMの値大 × クラス確率小'),
        Patch(facecolor='gray', label='CAMの値小 × クラス確率小')
    ]
    axes[8].legend(
        handles=legend_elements, 
        loc='upper center', 
        fontsize=legendsize,
        title="4分類ヒートマップの凡例",
        title_fontsize=legendsize)
    axes[8].axis('off')

    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_pixel_level_confusion(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, int]:
    """
    予測マスクとGTマスクからピクセル単位の混同行列を計算する
    
    Args:
        pred_mask: 予測マスク (H, W) - 画素値は0または255
        gt_mask: Ground Truthマスク (H, W) - 画素値は0または255
    
    Returns:
        混同行列の要素（TN, FP, FN, TP）を含む辞書
    """
    # 0/1形式に変換（0=背景、1=瓦礫）
    pred_binary = (pred_mask > 127).astype(int)
    gt_binary = (gt_mask > 127).astype(int)
    
    # フラット化
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    # 混同行列を計算（sklearnを使用）
    cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
    
    # TN, FP, FN, TPを抽出
    # cm[0,0] = TN (背景を背景と予測)
    # cm[0,1] = FP (背景を瓦礫と予測)
    # cm[1,0] = FN (瓦礫を背景と予測)
    # cm[1,1] = TP (瓦礫を瓦礫と予測)
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    
    return {
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'total_pixels': int(len(pred_flat))
    }


def compute_metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    """
    混同行列から各種メトリクスを計算する
    
    Args:
        tn: True Negative
        fp: False Positive
        fn: False Negative
        tp: True Positive
    
    Returns:
        メトリクスの辞書
    """
    eps = 1e-8
    
    # 全体の精度
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    # 瓦礫クラス（クラス1）のメトリクス
    precision_debris = tp / (tp + fp + eps)
    recall_debris = tp / (tp + fn + eps)
    f1_debris = 2 * precision_debris * recall_debris / (precision_debris + recall_debris + eps)
    
    # 背景クラス（クラス0）のメトリクス
    precision_background = tn / (tn + fn + eps)
    recall_background = tn / (tn + fp + eps)
    f1_background = 2 * precision_background * recall_background / (precision_background + recall_background + eps)
    
    # IoU（Intersection over Union）
    iou_debris = tp / (tp + fp + fn + eps)
    iou_background = tn / (tn + fp + fn + eps)
    mean_iou = (iou_debris + iou_background) / 2.0
    
    return {
        'accuracy': accuracy,
        'precision_debris': precision_debris,
        'recall_debris': recall_debris,
        'f1_debris': f1_debris,
        'precision_background': precision_background,
        'recall_background': recall_background,
        'f1_background': f1_background,
        'iou_debris': iou_debris,
        'iou_background': iou_background,
        'mean_iou': mean_iou
    }


def save_confusion_matrix_csv(
    overall_cm: np.ndarray,
    per_image_results: List[Dict],
    output_path: str
):
    """
    混同行列と評価結果をCSVファイルに保存する
    
    Args:
        overall_cm: データセット全体の混同行列 (2x2)
        per_image_results: 各画像の評価結果のリスト
        output_path: 出力CSVファイルのパス
    """
    # データセット全体のメトリクスを計算
    tn = int(overall_cm[0, 0])
    fp = int(overall_cm[0, 1])
    fn = int(overall_cm[1, 0])
    tp = int(overall_cm[1, 1])
    
    overall_metrics = compute_metrics_from_confusion(tn, fp, fn, tp)
    
    # データセット全体の結果をDataFrameにまとめる
    overall_data = {
        'image_name': ['DATASET_OVERALL'],
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'TP': [tp],
        'total_pixels': [tn + fp + fn + tp],
        **{k: [v] for k, v in overall_metrics.items()}
    }
    
    # 各画像の結果をDataFrameにまとめる
    if per_image_results:
        per_image_df = pd.DataFrame(per_image_results)
        
        # 全体の結果と結合
        overall_df = pd.DataFrame(overall_data)
        result_df = pd.concat([overall_df, per_image_df], ignore_index=True)
    else:
        result_df = pd.DataFrame(overall_data)
    
    # CSVに保存
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n推論結果CSVを保存しました: {output_path}")
    
    # 混同行列を別のCSVにも保存
    cm_df = pd.DataFrame(
        overall_cm,
        index=['GT_背景', 'GT_瓦礫'],
        columns=['Pred_背景', 'Pred_瓦礫']
    )
    cm_output_path = output_path.replace('.csv', '_confusion_matrix.csv')
    cm_df.to_csv(cm_output_path, encoding='utf-8-sig')
    print(f"混同行列CSVを保存しました: {cm_output_path}")
    
    # コンソールに結果を表示
    print("\n" + "="*80)
    print("データセット全体の評価結果")
    print("="*80)
    print(f"混同行列:")
    print(cm_df)
    print(f"\n全体精度 (Accuracy): {overall_metrics['accuracy']:.4f}")
    print(f"瓦礫クラス - 精度 (Precision): {overall_metrics['precision_debris']:.4f}")
    print(f"瓦礫クラス - 再現率 (Recall): {overall_metrics['recall_debris']:.4f}")
    print(f"瓦礫クラス - F1スコア: {overall_metrics['f1_debris']:.4f}")
    print(f"瓦礫クラス - IoU: {overall_metrics['iou_debris']:.4f}")
    print(f"平均IoU (mIoU): {overall_metrics['mean_iou']:.4f}")
    print("="*80)


def process_single_image(
    model: torch.nn.Module,
    cam_method,
    image_path: str,
    output_dir: str,
    xai_method_name: str,
    target_layer,
    gt_mask_dir: Optional[str],
    img_size: Tuple[int, int],
    debris_class_id: int = 1,
    device: torch.device = None,
    alpha: float = 0.4
) -> Optional[Dict]:
    """
    単一画像に対してXAI可視化を実行する
    
    Args:
        model: 学習済みモデル
        cam_method: CAM手法のインスタンス
        image_path: 画像ファイルのパス
        output_dir: 出力ディレクトリ
        xai_method_name: XAI手法名
        target_layer: ターゲットレイヤ
        gt_mask_dir: Ground Truthマスクディレクトリ（Noneの場合は読み込まない）
        img_size: 画像サイズ (H, W)
        debris_class_id: 瓦礫クラスのID
        device: デバイス
        alpha: αブレンドの透明度
    """
    try:
        # 画像前処理
        img_tensor, img_array, original_size = preprocess_image(image_path, img_size)
        
        # 確率マップ取得
        prob_map = get_probability_map(model, img_tensor, debris_class_id, device)
        
        # 予測マスク取得
        pred_mask = get_prediction_mask(model, img_tensor, img_size, device)
        
        img_tensor.to(device)

        # 予測マスクを0/1形式に変換（255→1、0→0）してCAM計算に使用
        # SemanticSegmentationTargetは0/1のバイナリマスクを期待（torch.Tensor形式）
        mask_for_cam = np.array((pred_mask / 255.0).astype(np.float32))  # (H, W), 0.0 or 1.0

        # CAM計算
        target = [SemanticSegmentationTarget(debris_class_id, mask_for_cam)]
        grayscale_cam = cam_method(input_tensor=img_tensor, targets=target)
        cam_map = grayscale_cam[0]  # (H, W)
        
        # 正規化
        cam_norm = normalize_map(cam_map)
        prob_norm = normalize_map(prob_map)
        # cam_norm = cam_map
        # prob_norm = prob_map


        # Evidence map (CAM × 確率)
        evidence_map = cam_norm * prob_norm
        
        # 4分類ヒートマップ
        heatmap_4class = create_4class_heatmap(cam_norm, prob_norm)
        
        # 画像名を取得（出力ファイル名とGTマスク検索に使用）
        image_name = Path(image_path).stem
        
        # Ground Truthマスクの読み込み
        gt_mask = None
        if gt_mask_dir is not None:
            # 同じ名前のマスクファイルを探す（複数の拡張子を試す）
            mask_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
            for ext in mask_extensions:
                gt_mask_path = os.path.join(gt_mask_dir, f"{image_name}{ext}")
                if os.path.exists(gt_mask_path):
                    gt_mask = load_gt_mask(gt_mask_path, img_size)
                    break
                # 大文字の拡張子も試す
                gt_mask_path = os.path.join(gt_mask_dir, f"{image_name}{ext.upper()}")
                if os.path.exists(gt_mask_path):
                    gt_mask = load_gt_mask(gt_mask_path, img_size)
                    break
        
        # 出力ファイル名
        output_path = os.path.join(output_dir, f"{image_name}_{xai_method_name}.png")
        
        # 可視化
        visualize_xai_results(
            original_img=img_array,
            cam_map=cam_map,
            prob_map=prob_map,
            cam_norm=cam_norm,
            prob_norm=prob_norm,
            evidence_map=evidence_map,
            heatmap_4class=heatmap_4class,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            output_path=output_path,
            xai_method_name=xai_method_name,
            alpha=alpha
        )
        
        print(f"✓ {Path(image_path).name} -> {xai_method_name}")
        
        # GTマスクがある場合、評価結果を返す
        if gt_mask is not None:
            confusion_result = compute_pixel_level_confusion(pred_mask, gt_mask)
            metrics = compute_metrics_from_confusion(
                confusion_result['TN'],
                confusion_result['FP'],
                confusion_result['FN'],
                confusion_result['TP']
            )
            
            result = {
                'image_name': image_name,
                **confusion_result,
                **metrics
            }
            return result
        else:
            return None
        
    except Exception as e:
        print(f"✗ エラー ({Path(image_path).name}, {xai_method_name}): {e}")
        return None


def main(
    model_weight_path: str,
    test_image_dir: str,
    output_root_dir: str,
    debris_class_id: int = 1,
    confidence_threshold: float = 0.5,
    alpha: float = 0.4,
    img_size: Tuple[int, int] = (512, 512),
    use_pretrained: bool = False,
    gt_mask_dir: Optional[str] = None,
):
    """
    メイン処理
    
    Args:
        model_weight_path: 学習済み重みファイルのパス
        test_image_dir: テスト画像ディレクトリ
        output_root_dir: 出力先ルートディレクトリ
        target_layer_A: CAM計算に使用する中間特徴レイヤ（例: "model.backbone.layer4[-1]"）
        debris_class_id: 瓦礫クラスのID（デフォルト: 1）
        confidence_threshold: 信頼度閾値（デフォルト: 0.5、現在は未使用）
        alpha: αブレンドの透明度（デフォルト: 0.4）
        img_size: 画像サイズ (H, W)（デフォルト: (512, 512)）
        use_pretrained: 事前学習済み重みを使用するか（デフォルト: False）
        gt_mask_dir: Ground Truthマスクディレクトリ（Noneの場合は読み込まない）
    """
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # モデルロード
    print(f"モデルをロード中: {model_weight_path}")
    model = load_model(model_weight_path, num_classes=2, device=device, use_pretrained=use_pretrained)
    print("モデルロード完了")
    

    target_layer = model.model.backbone.layer4[-1]


    # ターゲットレイヤの取得
    # target_layer_A を評価してレイヤオブジェクトを取得
    
    # try:
    #     target_layer = eval(target_layer_A)
    #     print(f"ターゲットレイヤ: {target_layer_A}")
    # except Exception as e:
    #     print(f"エラー: ターゲットレイヤ '{target_layer_A}' の取得に失敗しました: {e}")
    #     print("デフォルトの 'model.backbone.layer4[-1]' を使用します")
    #     target_layer = model.backbone.layer4[-1]
    
    # テスト画像の取得
    test_image_dir = Path(test_image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_image_dir.glob(f"*{ext}"))
    test_images = sorted(test_images)
    
    if len(test_images) == 0:
        print(f"警告: {test_image_dir} に画像が見つかりませんでした")
        return
    
    print(f"テスト画像数: {len(test_images)}")
    
    # 出力ディレクトリの作成
    output_root_dir = Path(output_root_dir)
    xai_methods = {
        # 'GradCAM': GradCAM,
        'GradCAMPlusPlus': GradCAMPlusPlus,
    }
    
    for method_name in xai_methods.keys():
        method_dir = output_root_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
    
    # 各XAI手法で処理
    for method_name, cam_class in xai_methods.items():
        print(f"\n{'='*60}")
        print(f"{method_name} で処理中...")
        print(f"{'='*60}")
        
        # CAM手法のインスタンス作成
        cam_method = cam_class(
            model=model,
            target_layers=[target_layer],
            # use_cuda=(device.type == 'cuda')
        )
        
        # 評価結果を保存するリスト
        per_image_results = []
        overall_cm = np.zeros((2, 2), dtype=np.int64)
        
        # 各画像を処理
        for img_path in test_images:
            result = process_single_image(
                model=model,
                cam_method=cam_method,
                image_path=str(img_path),
                output_dir=str(output_root_dir / method_name),
                xai_method_name=method_name,
                target_layer=target_layer,
                gt_mask_dir=gt_mask_dir,
                img_size=img_size,
                debris_class_id=debris_class_id,
                device=device,
                alpha=alpha
            )
            
            # 結果がある場合、集計に追加
            if result is not None:
                per_image_results.append(result)
                # 混同行列に加算
                overall_cm[0, 0] += result['TN']
                overall_cm[0, 1] += result['FP']
                overall_cm[1, 0] += result['FN']
                overall_cm[1, 1] += result['TP']
        
        # GTマスクがある場合、CSVに結果を保存
        if gt_mask_dir is not None and len(per_image_results) > 0:
            csv_output_path = output_root_dir / f"evaluation_results_{method_name}.csv"
            save_confusion_matrix_csv(overall_cm, per_image_results, str(csv_output_path))
    
    print(f"\n{'='*60}")
    print("処理完了!")
    print(f"出力先: {output_root_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # パラメータ設定（必要に応じて変更）
    MODEL_WEIGHT_PATH = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Weights\20260107_1428_epoch070.pth"
    TEST_IMAGE_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Test\img"
    GT_MASK_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Test\mask"  # Ground Truthマスクディレクトリ
    OUTPUT_ROOT_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_XAI"
    DEBRIS_CLASS_ID = 1
    CONFIDENCE_THRESHOLD = 0.5
    ALPHA = 0.4
    IMG_SIZE = (64, 64)
    USE_PRETRAINED = False

    print(f"use_pretrained = {USE_PRETRAINED}")
    
    main(
        model_weight_path=MODEL_WEIGHT_PATH,
        test_image_dir=TEST_IMAGE_DIR,
        output_root_dir=OUTPUT_ROOT_DIR,
        debris_class_id=DEBRIS_CLASS_ID,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        alpha=ALPHA,
        img_size=IMG_SIZE,
        use_pretrained=USE_PRETRAINED,
        gt_mask_dir=GT_MASK_DIR
    )
