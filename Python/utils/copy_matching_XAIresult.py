import pandas as pd
import shutil
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def extract_common_id(filename):
    """
    ファイル名から共通IDを抽出します。
    
    例: "b10_HouseCollapse_167_wajima_roadpolygon_linkwise_Link154_clipped_bbox.gpkg"
    → "b10_HouseCollapse_167"
    
    Args:
        filename (str): GPKGファイル名
        
    Returns:
        str: 抽出されたID（先頭から3番目のアンダースコアまで）
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return filename


def copy_matching_files(csv_path, source_dir, output_base_dir):
    """
    CSVファイルを読み込み、3つのグループに分けてPNGファイルをコピーします。
    
    Args:
        csv_path (str): CSVファイルのパス
        source_dir (str): 元のPNGファイルが格納されているディレクトリ
        output_base_dir (str): 出力先のベースディレクトリ
    """
    
    # パスをPathオブジェクトに変換
    csv_path = Path(csv_path)
    source_dir = Path(source_dir)
    output_base_dir = Path(output_base_dir)
    
    # CSVファイルの存在確認
    if not csv_path.exists():
        logger.error(f"CSVファイルが見つかりません: {csv_path}")
        return
    
    # ソースディレクトリの存在確認
    if not source_dir.exists():
        logger.error(f"ソースディレクトリが見つかりません: {source_dir}")
        return
    
    # CSVを読み込む
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"CSVファイルを読み込みました: {csv_path}")
    except Exception as e:
        logger.error(f"CSVファイルの読み込みに失敗しました: {e}")
        return
    
    # 3つのグループの定義
    groups = {
        'Pred_2': df[df['Pred_Passable'] == 2],
        'GT_2': df[df['GT_Passable'] == 2],
        'Correct_True': df[df['Correct'] == True]
    }
    
    # 各グループを処理
    for group_name, group_df in groups.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"グループ: {group_name} ({len(group_df)} 件)")
        logger.info(f"{'='*60}")
        
        # 出力ディレクトリを作成
        output_dir = output_base_dir / group_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"出力ディレクトリを作成しました: {output_dir}")
        
        # グループ内の各行を処理
        copied_count = 0
        skipped_count = 0
        
        for idx, row in group_df.iterrows():
            filename = row['filename']
            
            # 共通IDを抽出
            common_id = extract_common_id(filename)
            
            # PNGファイル名を生成
            png_filename = f"{common_id}_GradCAMPlusPlus.png"
            source_file = source_dir / png_filename
            
            # ファイルの存在確認とコピー
            if source_file.exists():
                try:
                    dest_file = output_dir / png_filename
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"✓ コピー: {png_filename}")
                    copied_count += 1
                except Exception as e:
                    logger.warning(f"✗ コピー失敗 ({png_filename}): {e}")
                    skipped_count += 1
            else:
                logger.warning(f"⚠ ファイルが見つかりません: {png_filename} (元CSVファイル: {filename})")
                skipped_count += 1
        
        # グループの処理結果をサマリー表示
        logger.info(f"\n[グループ '{group_name}' 処理結果]")
        logger.info(f"  コピー成功: {copied_count} 件")
        logger.info(f"  スキップ: {skipped_count} 件")
    
    logger.info(f"\n{'='*60}")
    logger.info("処理完了")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    # 設定例（必要に応じて変更してください）
    CSV_PATH = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_XAI\merged_QGIS_Seg.csv"  # CSVファイルのパス
    SOURCE_DIR = r"c:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_XAI\GradCAMPlusPlus"  # 元のPNGファイルのディレクトリ
    OUTPUT_BASE_DIR = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20260105Data\Result_XAI"  # 出力先のベースディレクトリ
    
    copy_matching_files(CSV_PATH, SOURCE_DIR, OUTPUT_BASE_DIR)
