"""
copy_matching_fgb.py

指定した画像ディレクトリ(A)のファイル名（拡張子を除く）を含む .fgb ファイルを
検索ディレクトリ(B)から抽出して出力ディレクトリ(C)にコピーするユーティリティ。

使い方の例:
    python copy_matching_fgb.py \
        --images-dir "C:/path/to/images" \
        --fgb-dir "C:/path/to/fgb_dir" \
        --out-dir "C:/path/to/out" \
        --recursive --dry-run

オプション:
    --recursive: Bの検索を再帰的に行う
    --dry-run: 実際にコピーせずに何が行われるかを表示する
    --no-overwrite: 既存のファイルを上書きしない

"""
from pathlib import Path
import shutil
from typing import Dict, List


def find_matching_fgb_files(images_dir: Path, fgb_dir: Path, recursive: bool = False) -> Dict[str, List[Path]]:
    """画像ファイルのstemごとに一致する .fgb ファイル一覧を返す。

    images_dir: 画像が入ったディレクトリ A
    fgb_dir: .fgb ファイルが入ったディレクトリ B
    recursive: Bを再帰的に検索するか
    """
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found or not a directory: {images_dir}")
    if not fgb_dir.exists() or not fgb_dir.is_dir():
        raise FileNotFoundError(f"FGB dir not found or not a directory: {fgb_dir}")

    image_files = [p for p in images_dir.iterdir() if p.is_file()]
    stems = {p.stem.lower().split(".")[0] for p in image_files}

    pattern = "**/*.fgb" if recursive else "*.fgb"
    fgb_files = list(fgb_dir.glob(pattern))

    matches: Dict[str, List[Path]] = {s: [] for s in stems}

    for f in fgb_files:
        name_lower = f.name.lower().split(".")[0]
        for s in stems:
            if s == name_lower:
                matches[s].append(f)

    # remove stems with no matches
    matches = {k: v for k, v in matches.items() if v}
    return matches


def copy_matches(matches: Dict[str, List[Path]], out_dir: Path, dry_run: bool = False, overwrite: bool = True) -> List[Path]:
    """matches に含まれるファイルを out_dir にコピーする。戻り値はコピーしたファイルのパス一覧。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []

    # To avoid copying the same file multiple times, track source absolute paths
    seen = set()

    for stem, files in matches.items():
        for src in files:
            src_abs = src.resolve()
            if src_abs in seen:
                continue
            seen.add(src_abs)
            dst = out_dir / src.name
            if dst.exists() and not overwrite:
                print(f"Skipping (exists): {dst}")
                continue
            if dry_run:
                print(f"Dry run: would copy {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
                print(f"Copied: {src} -> {dst}")
            copied.append(dst)

    return copied


def main():

    images_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20250105Data\TrainVal\img_1209testData"
    fgb_dir = r"C:\Users\kyohe\Aerial_Photo_Classifier\20251209Data\SquarePolygons\house_collapse\wajima_all"
    out_dir = r"C:\Users\kyohe\Aerial_Photo_Segmenter\20250105Data\SquarePolygons\wajima_all_1209TestData"
    recursive = True
    dry_run = False
    overwrite = True

    images_dir = Path(images_dir)
    fgb_dir = Path(fgb_dir)
    out_dir = Path(out_dir)

    matches = find_matching_fgb_files(images_dir, fgb_dir, recursive=recursive)

    if not matches:
        print("No matches found. Exiting.")
        return

    total_images = len({k for k in matches.keys()})
    total_matches = sum(len(v) for v in matches.values())
    print(f"Found matches for {total_images} image stems, total {total_matches} .fgb file(s)")

    copied = copy_matches(matches, out_dir, dry_run=dry_run, overwrite=overwrite)

    print(f"Done. Files copied: {len(copied)}")


main()
