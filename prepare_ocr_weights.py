import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "weights_manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def prepare_ocr_weights(
    ocr_dir: Path,
    manifest_path: Path,
    allow_download: bool = False,
) -> None:
    """Копирует/проверяет веса EasyOCR (ru) в локальный каталог ocr_dir по SHA-256 из манифеста."""
    if not manifest_path.is_file():
        raise RuntimeError(f"Манифест весов не найден: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models") or {}
    ocr_entries = [models.get("ocr_detector"), models.get("ocr_recognizer")]
    if any(not e or not e.get("file") or not e.get("sha256") for e in ocr_entries):
        raise RuntimeError("В weights_manifest.json отсутствуют записи ocr_detector / ocr_recognizer.")

    ocr_dir.mkdir(parents=True, exist_ok=True)
    default_cache = Path.home() / ".EasyOCR" / "model"

    for entry in ocr_entries:
        fname = entry["file"]
        expected_sha = entry["sha256"].lower()
        dst = ocr_dir / fname
        if not dst.is_file():
            cached = default_cache / fname
            if cached.is_file():
                print(f"Копирование {fname} из кэша {default_cache} -> {dst}")
                shutil.copy2(cached, dst)
            elif allow_download:
                print(f"Скачивание весов EasyOCR (ru) в {ocr_dir}...")
                import easyocr

                easyocr.Reader(
                    ["ru"],
                    gpu=False,
                    model_storage_directory=str(ocr_dir),
                    download_enabled=True,
                )
            else:
                raise RuntimeError(
                    f"Файл {dst} не найден, и отсутствует в {default_cache}. "
                    "Запустите с флагом --allow-download на машине с доступом в интернет."
                )

        actual_sha = _sha256_file(dst)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Критическая ошибка ИБ: SHA-256 файла {dst} ({actual_sha}) "
                f"не совпадает с манифестом ({expected_sha})."
            )
        print(f"OK: {dst.name} (sha256={actual_sha[:16]}...)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка и SHA-256 проверка офлайн-весов EasyOCR")
    parser.add_argument(
        "--ocr-dir",
        type=str,
        default=os.getenv("OCR_MODELS_DIR", str(PROJECT_DIR / "ocr_models")),
        help="Каталог для офлайн-весов EasyOCR (по умолчанию ocr_models/)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Путь к weights_manifest.json",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Разрешить скачивание весов EasyOCR из интернета, если их нет в локальном кэше",
    )
    args = parser.parse_args()
    prepare_ocr_weights(
        ocr_dir=Path(args.ocr_dir),
        manifest_path=Path(args.manifest),
        allow_download=args.allow_download,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        sys.exit(1)
