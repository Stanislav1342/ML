"""
update.py — Скрипт обновления проекта с GitHub без использования git.exe.

Предназначен для корпоративных рабочих станций, где запуск git.exe
ограничен политиками безопасности Windows.

Запуск:
    python update.py
"""

import io
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPO_ZIP_URL = "https://github.com/Stanislav1342/ML/archive/refs/heads/main.zip"
ALLOWED_URL_PREFIX = "https://github.com/Stanislav1342/ML/"
REQUEST_TIMEOUT_SEC = 60
CHUNK_SIZE = 65536
MAX_ARCHIVE_BYTES = 300 * 1024 * 1024
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024

PROTECTED_NAMES = {
    ".env",
    "AGENTS.md",
    "INTEGRATION_SPEC.md",
    "PRESENTATION_SPEECH.md",
    "PROJECT_SPECIFICATION.md",
    "PreAnnotator.py",
    "build_clean_yolo_dataset.py",
    "colab_train_gold_standard.py",
    "mine_300_samples.py",
    "best_clean_v2_gold.pt",
}

PROTECTED_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "env/",
    "Результаты/",
    "Старые веса/",
    "review_data/",
)


def _validate_download_url(url: str) -> None:
    """Проверяет допустимость схемы, домена и префикса URL перед загрузкой."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not url.startswith(ALLOWED_URL_PREFIX):
        raise ValueError(f"Недопустимый URL источника обновления: {url}")


def _is_protected_path(rel_path: str) -> bool:
    """Проверяет, относится ли файл к защищённым локальным или служебным путям."""
    if any(rel_path.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return True
    if PurePosixPath(rel_path).name in PROTECTED_NAMES:
        return True
    if rel_path.startswith("Test_PDFs/") and rel_path.lower().endswith(".pdf"):
        return True
    return False


def download_archive(url: str = REPO_ZIP_URL) -> bytes:
    """Загружает ZIP-архив из GitHub с ограничением размера и таймаута."""
    _validate_download_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "WaybillAudit-Updater/1.0"})

    buffer = io.BytesIO()
    total_read = 0
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:  # nosec B310
        content_length = resp.headers.get("Content-Length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"Размер архива превышает лимит {MAX_ARCHIVE_BYTES} байт.")

        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > MAX_ARCHIVE_BYTES:
                raise ValueError(f"Загрузка прервана: превышен лимит архива ({MAX_ARCHIVE_BYTES} байт).")
            buffer.write(chunk)

    return buffer.getvalue()


def extract_archive(zip_bytes: bytes, base_dir: Path) -> int:
    """Безопасно распаковывает файлы репозитория из ZIP с защитой от Zip Slip и Zip Bomb."""
    resolved_base = base_dir.resolve()
    updated_count = 0
    total_extracted = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            # Отклоняем символические ссылки внутри архива.
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise ValueError(f"Отклонена символическая ссылка в архиве: {member.filename}")

            norm_name = member.filename.replace("\\", "/")
            parts = norm_name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            rel_path = parts[1]

            pure_rel = PurePosixPath(rel_path)
            if pure_rel.is_absolute() or ".." in pure_rel.parts or ":" in rel_path.split("/")[0]:
                raise ValueError(f"Небезопасный путь в архиве (Zip Slip): {member.filename}")

            norm_rel = pure_rel.as_posix()
            if norm_rel in ("", ".") or _is_protected_path(norm_rel):
                continue

            dest_path = (resolved_base / pure_rel).resolve()
            try:
                dest_path.relative_to(resolved_base)
            except ValueError as exc:
                raise ValueError(f"Выход пути за пределы каталога проекта: {member.filename}") from exc

            if total_extracted + member.file_size > MAX_EXTRACTED_BYTES:
                raise ValueError(f"Превышен лимит распаковываемых данных ({MAX_EXTRACTED_BYTES} байт).")

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = dest_path.parent / f".{dest_path.name}.{os.getpid()}.tmp"
            try:
                with zf.open(member, "r") as src, open(tmp_path, "wb") as dst:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        total_extracted += len(chunk)
                        if total_extracted > MAX_EXTRACTED_BYTES:
                            raise ValueError(f"Превышен лимит распаковываемых данных ({MAX_EXTRACTED_BYTES} байт).")
                        dst.write(chunk)
                os.replace(tmp_path, dest_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

            updated_count += 1
            print(f"  [Обновлён] {rel_path}")

    return updated_count


def update_from_github(base_dir: Path = None) -> int:
    """Скачивает и применяет обновление из ветки main."""
    target_dir = base_dir if base_dir is not None else Path(__file__).resolve().parent
    print("=" * 64)
    print("  Обновление проекта с GitHub (без git.exe)")
    print("=" * 64)
    print(f"Источник: {REPO_ZIP_URL}")

    zip_bytes = download_archive(REPO_ZIP_URL)
    updated_count = extract_archive(zip_bytes, target_dir)

    print("=" * 64)
    print(f"  Обновление завершено успешно. Обновлено файлов: {updated_count}")
    print("  Локальные результаты, архивы весов и защищённые файлы сохранены.")
    print("=" * 64)
    return updated_count


def main() -> int:
    try:
        update_from_github()
        return 0
    except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, ValueError, OSError) as exc:
        print(f"Ошибка обновления: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
