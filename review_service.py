import argparse
from datetime import datetime, timedelta
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import uuid
import zipfile

import fitz
from PIL import Image, ImageDraw

from pipeline import BBox, CLASS_NAMES, Detection, Pipeline, Window, get_cyrillic_font

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("review_service")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "review_data"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SERVICE_VERSION = "1.4.1"

MAX_UPLOAD_BYTES = 600 * 1024 * 1024
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024
MAX_SESSION_PAGES = 3500
DEFAULT_MAX_PAGES = 3500
WINDOW_CROP_PADDING = 40
UPLOAD_CHUNK_BYTES = 1024 * 1024

MAX_KEEP_SESSIONS = 5
QUARANTINE_PENDING_MAX_DAYS = 60
QUARANTINE_REJECTED_MAX_DAYS = 14
COLLECTING_MAX_HOURS = 24
BATCH_MIN_FILES = 2
BATCH_MAX_FILES = 50
MAX_BATCH_EDITS = 200
BATCH_EDIT_ACTIONS = frozenset({"delete_box", "update_box", "add_box", "relabel_box", "fix_digit"})

ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")
FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")
ERR_PREFIX_RE = re.compile(r"^Окно #\d+:\s*")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

PROTECTED_LABELS = frozenset({
    "Окно аптеки",
    "Qr code",
    "Титульный лист",
    "Номер МЛ",
    "ФИО водителя",
})
EDITABLE_LABELS = frozenset({
    "Печать",
    "Плохая печать",
    "Печать экспедиции",
    "Подпись",
    "Расшифровка",
    "Расшифровка печать",
    "Рукописная цифра",
    "Печатная цифра",
    "Исправление",
})
DIGIT_LABELS = frozenset({"Рукописная цифра", "Печатная цифра"})
OPERATOR_ADD_LABELS = (
    "Печать",
    "Плохая печать",
    "Подпись",
    "Расшифровка",
    "Расшифровка печать",
    "Рукописная цифра",
    "Исправление",
    "Печать экспедиции",
)
STRUCTURAL_PAGE_LABELS = frozenset({"Титульный лист", "Номер МЛ", "ФИО водителя"})


class ServiceError(Exception):
    """Исключение бизнес-валидации или доступа HTTP API с HTTP-кодом."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _sanitize_log(value: Any, max_len: int = 200) -> str:
    """Очищает пользовательские строки от управляющих символов перед логированием."""
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:max_len]


def _validate_id(value: str, field_name: str = "id") -> str:
    """Проверяет идентификатор по строгому белому списку символов."""
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ServiceError(HTTPStatus.BAD_REQUEST, f"Некорректный формат поля '{field_name}'.")
    return value


def _validate_filename(value: str, field_name: str = "filename") -> str:
    """Проверяет односегментное имя файла."""
    if not isinstance(value, str) or not FILENAME_PATTERN.match(value) or value in (".", ".."):
        raise ServiceError(HTTPStatus.BAD_REQUEST, f"Некорректное имя файла '{field_name}'.")
    return value


def _safe_subpath(base_dir: Path, *parts: str) -> Path:
    """Гарантирует, что целевой путь находится строго внутри base_dir."""
    base_resolved = base_dir.resolve()
    target = base_resolved
    for part in parts:
        target = target / part
    target_resolved = target.resolve()
    if target_resolved != base_resolved and not target_resolved.is_relative_to(base_resolved):
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Недопустимый путь к ресурсу.")
    return target_resolved


def _is_loopback_host(host: str) -> bool:
    """Разрешает биндинг HTTP-сервера только на локальный loopback-интерфейс."""
    return host in ("127.0.0.1", "localhost", "::1")


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Атомарно записывает JSON-файл через временный файл и os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _read_json(path: Path) -> Any:
    """Читает JSON-файл в UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_file_stem(name: str, fallback: str = "document") -> str:
    """Формирует безопасное имя файла без спецсимволов."""
    stem = re.sub(r"[^\w\-]", "_", Path(name).stem).strip("_")
    return stem[:64] or fallback


def _safe_token(text: Any, fallback: str = "Unknown", max_len: int = 80) -> str:
    """Санитизирует текстовый токен (номер МЛ, ФИО) для имён файлов."""
    clean = re.sub(r"[^\w\-]+", "_", str(text or "")).strip("_")
    return clean[:max_len] or fallback


def _parse_iso_datetime(value: Optional[str], fallback_mtime: float) -> datetime:
    """Парсит ISO-дату или возвращает время модификации файла."""
    if value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_mtime)


def _parse_digit_value(raw_value: Any) -> int:
    """Парсит и проверяет целое число мест в диапазоне 1..99."""
    if isinstance(raw_value, bool) or raw_value is None:
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Укажите целое число мест от 1 до 99.")
    if isinstance(raw_value, int):
        val = raw_value
    elif isinstance(raw_value, str) and raw_value.strip().isdigit():
        val = int(raw_value.strip())
    else:
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Значение количества мест должно быть числом от 1 до 99.")
    if not (1 <= val <= 99):
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Количество мест должно быть в диапазоне от 1 до 99.")
    return val


def _clamp_crop_rect(bbox: List[float], width: int, height: int, pad: int = WINDOW_CROP_PADDING) -> List[int]:
    """Вычисляет прямоугольник обрезки окна с отступом и ограничением по границам листа."""
    x1, y1, x2, y2 = bbox
    cx1 = max(0, int(round(x1 - pad)))
    cy1 = max(0, int(round(y1 - pad)))
    cx2 = min(width, int(round(x2 + pad)))
    cy2 = min(height, int(round(y2 + pad)))
    return [cx1, cy1, max(cx1 + 1, cx2), max(cy1 + 1, cy2)]


def _detection_to_dict(det: Detection, elem_id: str, source: str = "model") -> Dict[str, Any]:
    """Преобразует Detection в сериализуемый словарь элемента."""
    return {
        "id": elem_id,
        "label": det.label,
        "confidence": round(float(det.confidence), 4),
        "bbox": [
            round(float(det.bbox.x1), 2),
            round(float(det.bbox.y1), 2),
            round(float(det.bbox.x2), 2),
            round(float(det.bbox.y2), 2),
        ],
        "ocr_text": det.ocr_text or "",
        "ocr_value": det.ocr_value,
        "cls_pred": det.cls_pred or "",
        "cls_conf": round(float(det.cls_conf or 0.0), 4),
        "editable": det.label not in PROTECTED_LABELS,
        "source": source,
    }


def _dict_to_detection(item: Dict[str, Any]) -> Detection:
    """Восстанавливает Detection из словаря элемента."""
    x1, y1, x2, y2 = item["bbox"]
    return Detection(
        label=item["label"],
        confidence=float(item.get("confidence", 1.0)),
        bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
        ocr_text=str(item.get("ocr_text") or ""),
        ocr_value=item.get("ocr_value"),
        cls_pred=str(item.get("cls_pred") or ""),
        cls_conf=float(item.get("cls_conf") or 0.0),
        source=str(item.get("source") or "model"),
    )


def _get_orig_elements(win: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Возвращает исходные элементы окна с учётом copy-on-write."""
    orig = win.get("original_elements")
    return orig if orig is not None else win.get("elements", [])


def _ensure_original_elements_saved(win: Dict[str, Any]) -> None:
    """Сохраняет исходный снимок элементов окна перед первой мутацией."""
    if win.get("original_elements") is None:
        win["original_elements"] = json.loads(json.dumps(win.get("elements", []), ensure_ascii=False))


def _rewrite_window_error_prefix(items: List[str], window_idx: int) -> List[str]:
    """Заменяет временный префикс 'Окно #1:' на реальный номер окна."""
    target_prefix = f"Окно #{window_idx + 1}: "
    return [
        ERR_PREFIX_RE.sub(target_prefix, msg, count=1) if ERR_PREFIX_RE.match(msg) else f"{target_prefix}{msg}"
        for msg in items
    ]


def _normalize_error_text(err: str) -> str:
    """Возвращает текст ошибки без префикса 'Окно #N:' для сопоставления."""
    return ERR_PREFIX_RE.sub("", str(err or "")).strip()


def _resolve_target_error(
    errors: List[str],
    error_message: Optional[str],
    error_index: Any,
) -> Optional[str]:
    """Находит целевую ошибку в списке по тексту (с префиксом или без) или по индексу."""
    if error_message is not None:
        msg_raw = str(error_message).strip()
        msg_norm = _normalize_error_text(msg_raw)
        for err in errors:
            if err == msg_raw or _normalize_error_text(err) == msg_norm:
                return err
    if isinstance(error_index, int) and not isinstance(error_index, bool):
        if 0 <= error_index < len(errors):
            return errors[error_index]
    return None


def _collect_window_error_element_ids(hints: List[Dict[str, Any]]) -> List[str]:
    """Собирает уникальные id элементов из неподтверждённых подсказок (кроме missing)."""
    seen = set()
    result: List[str] = []
    for h in hints:
        if h.get("kind") == "missing" or h.get("status") == "confirmed":
            continue
        for eid in h.get("related_element_ids") or []:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
    return result


def build_error_hints(
    errors: List[str],
    elements: List[Dict[str, Any]],
    return_split_y: Optional[float] = None,
    confirmed_errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Формирует структурированные подсказки для UI по списку ошибок окна."""
    confirmed_raw = set(confirmed_errors or [])
    confirmed_norm = {_normalize_error_text(e) for e in confirmed_raw}

    def _in_delivery(el: Dict[str, Any]) -> bool:
        if return_split_y is None:
            return True
        return ((el["bbox"][1] + el["bbox"][3]) / 2.0) < return_split_y

    hints: List[Dict[str, Any]] = []
    for err in errors:
        clean_msg = _normalize_error_text(err)
        lower = clean_msg.lower()
        is_confirmed = (err in confirmed_raw) or (clean_msg in confirmed_norm)

        code = "other"
        kind = "other"
        suggested_add: Optional[str] = None
        related_ids: List[str] = []

        if "отсутствует печать возврата" in lower:
            code, kind, suggested_add = "missing_return_stamp", "missing", "Печать"
        elif "плохая печать возврата" in lower:
            code, kind, suggested_add = "bad_return_stamp", "stamp", "Печать"
            related_ids = [
                el["id"] for el in elements
                if el["label"] == "Плохая печать"
                and (return_split_y is None or ((el["bbox"][1] + el["bbox"][3]) / 2.0 >= return_split_y))
            ]
        elif "отсутствует подпись" in lower:
            code, kind, suggested_add = "missing_signature", "missing", "Подпись"
        elif "отсутствует расшифровка" in lower:
            code, kind, suggested_add = "missing_decoding", "missing", "Расшифровка"
        elif "отсутствует печатное кол-во мест" in lower:
            code, kind, suggested_add = "missing_printed_digit", "missing", "Печатная цифра"
        elif "отсутствует рукописное кол-во мест" in lower:
            code, kind, suggested_add = "missing_hw_digit", "missing", "Рукописная цифра"
        elif "несовпадение цифр" in lower:
            code, kind = "digit_mismatch", "digit"
            related_ids = [el["id"] for el in elements if el["label"] in DIGIT_LABELS and _in_delivery(el)]
        elif "не удалось распознать печатную цифру" in lower:
            code, kind = "digit_unrecognized", "digit"
            unrec = [el["id"] for el in elements if el["label"] == "Печатная цифра" and el.get("ocr_value") is None and _in_delivery(el)]
            related_ids = unrec or [el["id"] for el in elements if el["label"] == "Печатная цифра" and _in_delivery(el)]
        elif "не удалось распознать рукописную цифру" in lower:
            code, kind = "digit_unrecognized", "digit"
            unrec = [el["id"] for el in elements if el["label"] == "Рукописная цифра" and el.get("ocr_value") is None and _in_delivery(el)]
            related_ids = unrec or [el["id"] for el in elements if el["label"] == "Рукописная цифра" and _in_delivery(el)]
        elif "две рукописные цифры" in lower:
            code, kind, suggested_add = "two_hw_digits", "correction", "Печать"
            related_ids = [el["id"] for el in elements if el["label"] == "Рукописная цифра" and _in_delivery(el)]
        elif "исправление" in lower:
            code, kind, suggested_add = "correction_no_second_stamp", "correction", "Печать"
            related_ids = [el["id"] for el in elements if el["label"] == "Исправление" and _in_delivery(el)]
        elif "плохая печать" in lower:
            code, kind, suggested_add = "bad_stamp", "stamp", "Печать"
            related_ids = [
                el["id"] for el in elements
                if el["label"] == "Плохая печать"
                and _in_delivery(el)
            ]
        elif "отсутствует печать" in lower:
            code, kind, suggested_add = "missing_stamp", "missing", "Печать"

        hints.append({
            "code": code,
            "message": clean_msg,
            "raw_error": err,
            "status": "confirmed" if is_confirmed else "pending",
            "kind": kind,
            "suggested_add_label": suggested_add,
            "related_element_ids": related_ids,
            "error_element_ids": list(related_ids),
        })
    return hints


def _compute_window_diff(orig_elements: Optional[List[Dict[str, Any]]], curr_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Вычисляет diff правок оператора внутри одного окна."""
    if orig_elements is None:
        return {"added": [], "deleted": [], "relabeled": [], "digit_fixed": [], "moved": [], "has_changes": False}

    orig_by_id = {el["id"]: el for el in orig_elements}
    curr_by_id = {el["id"]: el for el in curr_elements}

    deleted = [orig_by_id[eid] for eid in orig_by_id if eid not in curr_by_id]
    added = [curr_by_id[eid] for eid in curr_by_id if eid not in orig_by_id]
    relabeled = []
    digit_fixed = []
    moved = []

    for eid, curr_el in curr_by_id.items():
        if eid not in orig_by_id:
            continue
        orig_el = orig_by_id[eid]
        bbox_changed = orig_el.get("bbox") != curr_el.get("bbox")
        if orig_el["label"] != curr_el["label"]:
            relabeled.append({
                "id": eid,
                "from_label": orig_el["label"],
                "to_label": curr_el["label"],
                "bbox": curr_el["bbox"],
            })
        elif curr_el["label"] in DIGIT_LABELS and orig_el.get("ocr_value") != curr_el.get("ocr_value"):
            digit_fixed.append({
                "id": eid,
                "label": curr_el["label"],
                "from_value": orig_el.get("ocr_value"),
                "to_value": curr_el.get("ocr_value"),
                "bbox": curr_el["bbox"],
            })
        elif bbox_changed:
            moved.append({
                "id": eid,
                "label": curr_el["label"],
                "from_bbox": orig_el.get("bbox"),
                "to_bbox": curr_el.get("bbox"),
            })

    return {
        "added": added,
        "deleted": deleted,
        "relabeled": relabeled,
        "digit_fixed": digit_fixed,
        "moved": moved,
        "has_changes": bool(added or deleted or relabeled or digit_fixed or moved),
    }


def _validate_box_geometry(
    raw_bbox: Any,
    target_label: str,
    win_bbox: List[float],
    page_w: int,
    page_h: int,
) -> Tuple[float, float, float, float]:
    """Проверяет координаты и минимальные габариты рамки элемента внутри окна аптеки."""
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Поле 'bbox' должно содержать 4 координаты [x1, y1, x2, y2].")
    try:
        if any(isinstance(v, bool) for v in raw_bbox):
            raise ValueError("bool is not a valid coordinate")
        x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Координаты 'bbox' должны быть числами.")

    x1 = max(0.0, min(float(page_w), x1))
    y1 = max(0.0, min(float(page_h), y1))
    x2 = max(0.0, min(float(page_w), x2))
    y2 = max(0.0, min(float(page_h), y2))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    wx1, wy1, wx2, wy2 = [float(v) for v in win_bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if not (wx1 - 60.0 <= cx <= wx2 + 60.0 and wy1 - 60.0 <= cy <= wy2 + 60.0):
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Центр новой рамки находится вне границ текущего окна.")

    lim_x1 = max(0.0, wx1 - float(WINDOW_CROP_PADDING))
    lim_y1 = max(0.0, wy1 - float(WINDOW_CROP_PADDING))
    lim_x2 = min(float(page_w), wx2 + float(WINDOW_CROP_PADDING))
    lim_y2 = min(float(page_h), wy2 + float(WINDOW_CROP_PADDING))
    x1 = max(lim_x1, min(lim_x2, x1))
    x2 = max(lim_x1, min(lim_x2, x2))
    y1 = max(lim_y1, min(lim_y2, y1))
    y2 = max(lim_y1, min(lim_y2, y2))

    bw, bh = x2 - x1, y2 - y1
    if bw < 8.0 or bh < 8.0:
        raise ServiceError(HTTPStatus.BAD_REQUEST, "Выделенная область слишком мала (минимум 8x8 пикселей).")

    return x1, y1, x2, y2


def _render_annotated_page(page_img: Image.Image, page: Dict[str, Any]) -> Image.Image:
    """Отрисовывает актуальные рамки и подписи на изображении страницы."""
    annotated = page_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font_title = get_cyrillic_font(20)
    font_label = get_cyrillic_font(15)
    label_map = {"Рукописная цифра": "рц", "Печатная цифра": "пц"}

    for det in page.get("structural_detections", []):
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 100, 255), width=2)
        tag = det["label"] + (f" [{det['ocr_text']}]" if det.get("ocr_text") else "")
        draw.text((int(x1) + 2, max(2, int(y1) - 18)), tag, font=font_label, fill=(0, 100, 255))

    for win in page.get("windows", []):
        has_err = bool(win.get("errors") or win.get("warnings"))
        win_color = (200, 0, 0) if has_err else (50, 205, 50)
        wx1, wy1, wx2, wy2 = win["bbox"]
        draw.rectangle((wx1, wy1, wx2, wy2), outline=win_color, width=3 if has_err else 2)

        for el in win.get("elements", []):
            ex1, ey1, ex2, ey2 = el["bbox"]
            short = label_map.get(el["label"], el["label"])
            if el["label"] in DIGIT_LABELS:
                val = el.get("ocr_value")
                short += f" ({val if val is not None else '?'})"
            el_color = (220, 0, 0) if (has_err and el["label"] in DIGIT_LABELS) else (34, 150, 34)
            if el.get("source") == "operator":
                el_color = (0, 120, 215)
            draw.rectangle((ex1, ey1, ex2, ey2), outline=el_color, width=2)
            draw.text((int(ex1) + 2, max(2, int(ey1) - 16)), short, font=font_label, fill=el_color)

        if has_err:
            title_parts = [ERR_PREFIX_RE.sub("", e).strip() for e in (win.get("errors", []) + win.get("warnings", []))]
            title = " | ".join(title_parts)
            tb = font_title.getbbox(title)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            rx1 = max(0, int(wx1))
            ry1 = max(0, int(wy1))
            rx2 = min(annotated.width, rx1 + tw + 12)
            ry2 = min(annotated.height, ry1 + th + 8)
            draw.rectangle((rx1, ry1, rx2, ry2), fill=(200, 0, 0))
            draw.text((rx1 + 6, ry1 + 3), title, font=font_title, fill=(255, 255, 255))

    return annotated


def _to_label_studio_result(bbox: List[float], img_w: int, img_h: int, label: str, result_id: str) -> Dict[str, Any]:
    """Конвертирует бокс [x1, y1, x2, y2] в элемент разметки Label Studio (в процентах)."""
    x1 = float(max(0.0, min(img_w, bbox[0])))
    y1 = float(max(0.0, min(img_h, bbox[1])))
    x2 = float(max(x1, min(img_w, bbox[2])))
    y2 = float(max(y1, min(img_h, bbox[3])))
    return {
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": round((x1 / img_w) * 100.0, 4),
            "y": round((y1 / img_h) * 100.0, 4),
            "width": round(((x2 - x1) / img_w) * 100.0, 4),
            "height": round(((y2 - y1) / img_h) * 100.0, 4),
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "id": result_id[:12],
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "manual",
    }


def _to_yolo_line(bbox: List[float], img_w: int, img_h: int, cls_id: int) -> Optional[str]:
    """Конвертирует бокс [x1, y1, x2, y2] в строку формата YOLO."""
    x1 = float(max(0.0, min(img_w, bbox[0])))
    y1 = float(max(0.0, min(img_h, bbox[1])))
    x2 = float(max(x1, min(img_w, bbox[2])))
    y2 = float(max(y1, min(img_h, bbox[3])))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0 or img_w <= 0 or img_h <= 0:
        return None
    xc = min(1.0, max(0.0, (x1 + bw / 2.0) / img_w))
    yc = min(1.0, max(0.0, (y1 + bh / 2.0) / img_h))
    nw = min(1.0, max(0.0, bw / img_w))
    nh = min(1.0, max(0.0, bh / img_h))
    return f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


class ReviewService:
    """Сервис интерактивного разбора маршрутных листов и очереди дообучения."""

    def __init__(self, data_dir: Optional[Path] = None, pipeline: Optional[Pipeline] = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.sessions_dir = self.data_dir / "sessions"
        self.quarantine_dir = self.data_dir / "quarantine"
        self.exports_dir = self.data_dir / "exports"
        for folder in (
            self.sessions_dir,
            self.quarantine_dir / "pending",
            self.quarantine_dir / "approved",
            self.quarantine_dir / "rejected",
            self.exports_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        self._pipeline = pipeline
        self._pipeline_init_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._quarantine_lock = threading.Lock()
        self._sessions_cleanup_lock = threading.Lock()
        self._session_locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    @property
    def pipeline(self) -> Pipeline:
        """Ленивая и потокобезопасная загрузка моделей с абсолютными путями весов."""
        if self._pipeline is None:
            with self._pipeline_init_lock:
                if self._pipeline is None:
                    yolo_path = str(PROJECT_DIR / "Новые веса" / "best_clean.pt")
                    cls_path = str(PROJECT_DIR / "Новые веса" / "best_digit_classifier.pt")
                    self._pipeline = Pipeline(yolo_model_path=yolo_path, digit_classifier_path=cls_path)
        return self._pipeline

    def warmup(self) -> None:
        """Принудительно инициализирует ML-модели при старте сервера."""
        _ = self.pipeline

    def health_info(self) -> Dict[str, Any]:
        """Возвращает состояние сервиса и справочники допустимых классов."""
        return {
            "status": "ok",
            "version": SERVICE_VERSION,
            "models_loaded": self._pipeline is not None,
            "max_session_pages": MAX_SESSION_PAGES,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "class_names": list(CLASS_NAMES),
            "protected_labels": sorted(PROTECTED_LABELS),
            "editable_labels": sorted(EDITABLE_LABELS),
            "operator_add_labels": list(OPERATOR_ADD_LABELS),
            "digit_labels": sorted(DIGIT_LABELS),
        }

    def _session_dir(self, session_id: str) -> Path:
        sid = _validate_id(session_id, "session_id")
        return _safe_subpath(self.sessions_dir, sid)

    def _load_session(self, session_id: str) -> Dict[str, Any]:
        session_file = self._session_dir(session_id) / "session.json"
        if not session_file.exists():
            raise ServiceError(HTTPStatus.NOT_FOUND, f"Сессия '{_sanitize_log(session_id)}' не найдена.")
        return _read_json(session_file)

    def _save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        session_file = self._session_dir(session_id) / "session.json"
        _atomic_write_json(session_file, session_data)

    def _cleanup_sessions(self) -> int:
        """Удаляет старые завершённые сессии (лимит MAX_KEEP_SESSIONS) и зависшие collecting (>24ч)."""
        removed = 0
        now = datetime.now()
        collecting_max_age = timedelta(hours=COLLECTING_MAX_HOURS)
        with self._sessions_cleanup_lock:
            if not self.sessions_dir.exists():
                return 0
            entries: List[Tuple[str, str, Path]] = []
            active_count = 0
            for entry in self.sessions_dir.iterdir():
                if not entry.is_dir():
                    continue
                meta_path = entry / "session.json"
                if not meta_path.is_file():
                    continue
                try:
                    data = _read_json(meta_path)
                    status_val = data.get("status", "unknown")
                    created_at = str(data.get("created_at") or "")
                    if status_val == "collecting":
                        created_dt = _parse_iso_datetime(created_at, entry.stat().st_mtime)
                        if now - created_dt > collecting_max_age:
                            shutil.rmtree(entry, ignore_errors=True)
                            removed += 1
                    elif status_val in ("queued", "running"):
                        active_count += 1
                    else:
                        entries.append((created_at, entry.name, entry))
                except Exception:
                    continue

            entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
            keep_finished = max(0, MAX_KEEP_SESSIONS - active_count)
            for _, _, old_dir in entries[keep_finished:]:
                shutil.rmtree(old_dir, ignore_errors=True)
                removed += 1
        if removed:
            logger.info(f"Автоочистка сессий: удалено старых сессий={removed} (лимит={MAX_KEEP_SESSIONS}).")
        return removed

    def _cleanup_quarantine_locked(self) -> Dict[str, int]:
        """Очищает просроченные записи pending (>60д) и rejected (>14д). Вызывать под self._quarantine_lock."""
        now = datetime.now()
        removed = {"pending": 0, "rejected": 0}
        rules = (
            ("pending", timedelta(days=QUARANTINE_PENDING_MAX_DAYS)),
            ("rejected", timedelta(days=QUARANTINE_REJECTED_MAX_DAYS)),
        )
        for bucket, max_age in rules:
            bucket_dir = self.quarantine_dir / bucket
            if not bucket_dir.exists():
                continue
            for entry in list(bucket_dir.iterdir()):
                if not entry.is_dir():
                    continue
                meta_path = entry / "item.json"
                ref_dt: Optional[datetime] = None
                if meta_path.is_file():
                    try:
                        data = _read_json(meta_path)
                        ts_key = data.get("reviewed_at") or data.get("submitted_at") if bucket == "rejected" else data.get("submitted_at")
                        ref_dt = _parse_iso_datetime(ts_key, entry.stat().st_mtime)
                    except Exception:
                        ref_dt = datetime.fromtimestamp(entry.stat().st_mtime)
                else:
                    ref_dt = datetime.fromtimestamp(entry.stat().st_mtime)

                if now - ref_dt > max_age:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed[bucket] += 1

        if removed["pending"] or removed["rejected"]:
            logger.info(
                f"Автоочистка карантина: удалено pending={removed['pending']} (>60д), "
                f"rejected={removed['rejected']} (>14д)."
            )
        return removed

    @staticmethod
    def _refresh_session_metrics(session_data: Dict[str, Any]) -> None:
        """Пересчитывает агрегированную статистику по сессии и каждому МЛ."""
        pages = session_data.get("pages", [])
        pages_by_id = {p["page_id"]: p for p in pages}

        total_pages = len(pages)
        error_pages = 0
        pending_pages = 0
        confirmed_pages = 0
        fixed_clean_pages = 0
        dismissed_clean_pages = 0
        quarantined_pages = 0

        for page in pages:
            win_statuses = [w["review_status"] for w in page.get("windows", [])]
            has_curr_errors = any(bool(w.get("errors") or w.get("warnings")) for w in page.get("windows", []))
            has_orig_errors = any(bool(w.get("original_errors")) for w in page.get("windows", []))
            if has_curr_errors:
                error_pages += 1
            if "pending" in win_statuses or "fixed_with_errors" in win_statuses:
                pending_pages += 1
            elif has_orig_errors and "confirmed_error" in win_statuses:
                confirmed_pages += 1
            elif has_orig_errors and all(s in ("clean", "fixed_clean", "dismissed_clean") for s in win_statuses):
                if "fixed_clean" in win_statuses and "dismissed_clean" not in win_statuses:
                    fixed_clean_pages += 1
                else:
                    dismissed_clean_pages += 1
            if page.get("quarantine_item_id"):
                quarantined_pages += 1

        for wb in session_data.get("waybills", []):
            wb_pages = [pages_by_id[pid] for pid in wb.get("page_ids", []) if pid in pages_by_id]
            wb["total_pages"] = len(wb_pages)
            wb["error_pages"] = sum(
                1 for p in wb_pages if any(bool(w.get("errors") or w.get("warnings")) for w in p.get("windows", []))
            )
            wb["reviewed_pages"] = sum(
                1 for p in wb_pages
                if any(bool(w.get("original_errors")) for w in p.get("windows", []))
                and not any(w["review_status"] in ("pending", "fixed_with_errors") for w in p.get("windows", []))
            )

        session_data["summary"] = {
            "total_waybills": len(session_data.get("waybills", [])),
            "total_pages": total_pages,
            "error_pages": error_pages,
            "pending_pages": pending_pages,
            "confirmed_error_pages": confirmed_pages,
            "fixed_clean_pages": fixed_clean_pages,
            "dismissed_clean_pages": dismissed_clean_pages,
            "quarantined_pages": quarantined_pages,
        }

    def create_session_from_pdf(
        self,
        pdf_path: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        uploaded_temp_path: Optional[Path] = None,
        filename: str = "upload.pdf",
        max_pages: int = DEFAULT_MAX_PAGES,
    ) -> Dict[str, Any]:
        """Создаёт сессию разбора и запускает фоновую обработку PDF."""
        if not (1 <= max_pages <= MAX_SESSION_PAGES):
            raise ServiceError(HTTPStatus.BAD_REQUEST, f"Параметр max_pages должен быть от 1 до {MAX_SESSION_PAGES}.")

        session_id = f"ses_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        ses_dir = self.sessions_dir / session_id
        pages_dir = ses_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        local_pdf = ses_dir / "input.pdf"

        if uploaded_temp_path is not None:
            if not uploaded_temp_path.is_file() or uploaded_temp_path.stat().st_size == 0:
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Загруженный PDF-файл пуст.")
            if uploaded_temp_path.stat().st_size > MAX_UPLOAD_BYTES:
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Размер PDF превышает 600 МБ.")
            with open(uploaded_temp_path, "rb") as f:
                head = f.read(1024)
            if b"%PDF-" not in head:
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Загруженный файл не является корректным PDF.")
            shutil.move(str(uploaded_temp_path), str(local_pdf))
            source_name = Path(filename).name or "upload.pdf"
        elif pdf_bytes is not None:
            if len(pdf_bytes) == 0 or len(pdf_bytes) > MAX_UPLOAD_BYTES:
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Недопустимый размер PDF-файла.")
            if b"%PDF-" not in pdf_bytes[:1024]:
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Загруженный файл не является корректным PDF.")
            with open(local_pdf, "wb") as f:
                f.write(pdf_bytes)
            source_name = Path(filename).name or "upload.pdf"
        elif pdf_path:
            src = Path(pdf_path)
            if not src.exists() or not src.is_file():
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Указанный PDF-файл не найден.")
            if src.stat().st_size > MAX_UPLOAD_BYTES:
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Размер PDF превышает 600 МБ.")
            with open(src, "rb") as f:
                head = f.read(1024)
            if b"%PDF-" not in head:
                shutil.rmtree(ses_dir, ignore_errors=True)
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Файл не имеет заголовка PDF.")
            shutil.copy2(src, local_pdf)
            source_name = src.name
        else:
            shutil.rmtree(ses_dir, ignore_errors=True)
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Не передан pdf_path или PDF-содержимое.")

        safe_source_name = _sanitize_log(source_name, 120) or "document.pdf"
        initial_state = {
            "session_id": session_id,
            "source_filename": safe_source_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "processed_pages": 0,
            "total_pages": 0,
            "finalized": False,
            "finalized_at": None,
            "error": None,
            "waybills": [],
            "pages": [],
            "summary": {
                "total_waybills": 0,
                "total_pages": 0,
                "error_pages": 0,
                "pending_pages": 0,
                "confirmed_error_pages": 0,
                "fixed_clean_pages": 0,
                "dismissed_clean_pages": 0,
                "quarantined_pages": 0,
            },
        }
        self._save_session(session_id, initial_state)
        self._cleanup_sessions()

        worker = threading.Thread(
            target=self._run_session_worker,
            args=(session_id, local_pdf, pages_dir, max_pages),
            daemon=True,
        )
        worker.start()
        return self.get_session_status(session_id)

    def create_batch_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Создаёт пакетную сессию в статусе collecting для приёма нескольких PDF."""
        if not isinstance(payload, dict):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Тело запроса должно быть JSON-объектом.")
        files_total = payload.get("files_total")
        if (
            not isinstance(files_total, int)
            or isinstance(files_total, bool)
            or not (BATCH_MIN_FILES <= files_total <= BATCH_MAX_FILES)
        ):
            raise ServiceError(
                HTTPStatus.BAD_REQUEST,
                f"Поле 'files_total' должно быть целым числом от {BATCH_MIN_FILES} до {BATCH_MAX_FILES}.",
            )
        max_pages = payload.get("max_pages", DEFAULT_MAX_PAGES)
        if (
            not isinstance(max_pages, int)
            or isinstance(max_pages, bool)
            or not (1 <= max_pages <= MAX_SESSION_PAGES)
        ):
            raise ServiceError(
                HTTPStatus.BAD_REQUEST,
                f"Параметр max_pages должен быть от 1 до {MAX_SESSION_PAGES}.",
            )

        session_id = f"ses_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        ses_dir = self.sessions_dir / session_id
        (ses_dir / "pages").mkdir(parents=True, exist_ok=True)
        (ses_dir / "files").mkdir(parents=True, exist_ok=True)

        initial_state = {
            "session_id": session_id,
            "is_batch": True,
            "source_filename": f"Пакет: {files_total} файлов",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "collecting",
            "stage": "collecting",
            "progress": 0,
            "processed_pages": 0,
            "total_pages": 0,
            "files_total": files_total,
            "files_received": 0,
            "current_file_idx": 0,
            "current_filename": "",
            "max_pages": max_pages,
            "uploaded_files": [],
            "source_files": [],
            "finalized": False,
            "finalized_at": None,
            "error": None,
            "waybills": [],
            "pages": [],
            "summary": {
                "total_waybills": 0,
                "total_pages": 0,
                "error_pages": 0,
                "pending_pages": 0,
                "confirmed_error_pages": 0,
                "fixed_clean_pages": 0,
                "dismissed_clean_pages": 0,
                "quarantined_pages": 0,
            },
        }
        self._save_session(session_id, initial_state)
        self._cleanup_sessions()
        return {
            "session_id": session_id,
            "status": "collecting",
            "files_total": files_total,
            "files_received": 0,
        }

    def add_batch_file(
        self,
        session_id: str,
        uploaded_temp_path: Path,
        filename: str = "upload.pdf",
    ) -> Dict[str, Any]:
        """Добавляет очередной PDF-файл в пакетную сессию, находящуюся в статусе collecting."""
        if not uploaded_temp_path.is_file() or uploaded_temp_path.stat().st_size == 0:
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Загруженный PDF-файл пуст.")
        if uploaded_temp_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise ServiceError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Размер PDF превышает 600 МБ.")
        with open(uploaded_temp_path, "rb") as f:
            head = f.read(1024)
        if b"%PDF-" not in head:
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Загруженный файл не является корректным PDF.")

        safe_name = _sanitize_log(Path(filename).name or "upload.pdf", 120) or "upload.pdf"
        lock = self._get_session_lock(session_id)
        with lock:
            state = self._load_session(session_id)
            if state.get("status") != "collecting":
                raise ServiceError(HTTPStatus.CONFLICT, "Сессия уже запущена или завершена, добавление файлов невозможно.")
            files_total = int(state.get("files_total", 0))
            uploaded = list(state.get("uploaded_files") or [])
            if len(uploaded) >= files_total:
                raise ServiceError(HTTPStatus.CONFLICT, "Все файлы пакета уже загружены.")

            file_idx = len(uploaded)
            ses_dir = self._session_dir(session_id)
            files_dir = ses_dir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            rel_name = f"input_{file_idx:03d}.pdf"
            shutil.move(str(uploaded_temp_path), str(files_dir / rel_name))

            uploaded.append({"file_idx": file_idx, "filename": safe_name, "stored_file": rel_name})
            state["uploaded_files"] = uploaded
            state["files_received"] = len(uploaded)
            self._save_session(session_id, state)

            return {
                "session_id": session_id,
                "files_received": len(uploaded),
                "files_total": files_total,
                "status": "collecting",
            }

    @staticmethod
    def _format_batch_source_name(uploaded_files: List[Dict[str, Any]]) -> str:
        if not uploaded_files:
            return "Пакет: 0 файлов"
        if len(uploaded_files) == 1:
            return uploaded_files[0]["filename"]
        first = uploaded_files[0]["filename"]
        if len(uploaded_files) == 2:
            return f"Пакет: 2 файла ({first}, {uploaded_files[1]['filename']})"
        return f"Пакет: {len(uploaded_files)} файлов ({first}, ...)"

    def start_batch_session(self, session_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Переводит пакетную сессию из collecting в queued и запускает последовательную обработку файлов."""
        lock = self._get_session_lock(session_id)
        with lock:
            state = self._load_session(session_id)
            if state.get("status") != "collecting":
                raise ServiceError(HTTPStatus.CONFLICT, "Сессия уже запущена или завершена.")
            uploaded = list(state.get("uploaded_files") or [])
            if not uploaded:
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Не загружен ни один PDF-файл для анализа.")

            state["files_total"] = len(uploaded)
            state["files_received"] = len(uploaded)
            state["source_filename"] = self._format_batch_source_name(uploaded)
            state["status"] = "queued"
            state["stage"] = "queued"
            state["progress"] = 0
            self._save_session(session_id, state)

        worker = threading.Thread(
            target=self._run_batch_session_worker,
            args=(session_id,),
            daemon=True,
        )
        worker.start()
        return self.get_session_status(session_id)

    def _run_batch_session_worker(self, session_id: str) -> None:
        """Последовательно анализирует все PDF-файлы пакетной сессии без склейки МЛ между файлами."""
        lock = self._get_session_lock(session_id)
        current_fname = ""
        try:
            with lock:
                state = self._load_session(session_id)
                uploaded = list(state.get("uploaded_files") or [])
                max_pages = int(state.get("max_pages", DEFAULT_MAX_PAGES))
                state["status"] = "running"
                state["stage"] = "init"
                state["progress"] = 2
                self._save_session(session_id, state)

            ses_dir = self._session_dir(session_id)
            pages_dir = ses_dir / "pages"
            files_dir = ses_dir / "files"

            file_page_counts: List[int] = []
            for item in uploaded:
                current_fname = item["filename"]
                pdf_path = files_dir / item["stored_file"]
                try:
                    with fitz.open(pdf_path) as doc:
                        file_page_counts.append(len(doc))
                except Exception as exc:
                    raise RuntimeError(f"Файл {current_fname}: повреждённый или нечитаемый PDF") from exc

            planned_total = min(max_pages, sum(file_page_counts))
            if planned_total <= 0:
                raise RuntimeError("В загруженных PDF-файлах нет страниц для анализа.")

            with lock:
                cur = self._load_session(session_id)
                cur["total_pages"] = planned_total
                self._save_session(session_id, cur)

            waybills: List[Dict[str, Any]] = []
            pages: List[Dict[str, Any]] = []
            source_files_info: List[Dict[str, Any]] = []
            global_idx = 0
            wb_counter = 0
            pages_done_before = 0

            for f_idx, item in enumerate(uploaded):
                remaining_budget = max_pages - pages_done_before
                if remaining_budget <= 0:
                    break
                current_fname = item["filename"]
                pdf_path = files_dir / item["stored_file"]
                expected_for_file = min(remaining_budget, file_page_counts[f_idx])

                with lock:
                    cur = self._load_session(session_id)
                    cur["current_file_idx"] = f_idx + 1
                    cur["current_filename"] = current_fname
                    self._save_session(session_id, cur)

                def on_progress(done: int, total: int, stage: str, base_done: int = pages_done_before) -> None:
                    overall_done = min(planned_total, base_done + done)
                    ratio = overall_done / max(1, planned_total)
                    pct = int(5 + ratio * 90)
                    with lock:
                        cur_st = self._load_session(session_id)
                        cur_st["stage"] = stage
                        cur_st["progress"] = min(96, pct)
                        cur_st["processed_pages"] = overall_done
                        cur_st["total_pages"] = planned_total
                        self._save_session(session_id, cur_st)

                tmp_file_dir = pages_dir / f"_tmp_f{f_idx:03d}"
                tmp_file_dir.mkdir(parents=True, exist_ok=True)
                try:
                    try:
                        with self._inference_lock:
                            segments = self.pipeline.analyze_pdf_segments(
                                str(pdf_path),
                                str(tmp_file_dir),
                                max_pages=remaining_budget,
                                progress_callback=on_progress,
                                image_ext=".jpg",
                            )
                    except Exception as exc:
                        raise RuntimeError(f"Файл {current_fname}: ошибка при анализе PDF") from exc

                    file_pages_count = 0
                    for ctx, seg_pages in segments:
                        wb_page_ids: List[str] = []
                        for pr in seg_pages:
                            global_idx += 1
                            file_pages_count += 1
                            page_id = f"p{global_idx:04d}"
                            wb_page_ids.append(page_id)

                            src_img_path = Path(pr.image_path)
                            unique_img_name = f"f{f_idx:03d}_{src_img_path.name}"
                            dst_img_path = pages_dir / unique_img_name
                            shutil.move(str(src_img_path), str(dst_img_path))
                            with Image.open(dst_img_path) as im:
                                img_w, img_h = im.size

                            struct_dets = []
                            for d_idx, d in enumerate(pr.detections):
                                if d.label in STRUCTURAL_PAGE_LABELS or (d.label == "Qr code" and not pr.windows):
                                    struct_dets.append(_detection_to_dict(d, f"st_{global_idx}_{d_idx}", source="model"))

                            win_dicts = []
                            for w_idx, win in enumerate(pr.windows):
                                w_errs = [e for e in pr.errors if e.startswith(f"Окно #{w_idx + 1}:")]
                                w_warns = [w for w in pr.warnings if w.startswith(f"Окно #{w_idx + 1}:")]

                                elems = [
                                    _detection_to_dict(el, f"el_{global_idx}_{w_idx}_{e_idx}", source="model")
                                    for e_idx, el in enumerate(win.elements)
                                ]
                                wbbox = [
                                    round(float(win.bbox.x1), 2),
                                    round(float(win.bbox.y1), 2),
                                    round(float(win.bbox.x2), 2),
                                    round(float(win.bbox.y2), 2),
                                ]
                                hints = build_error_hints(w_errs, elems, win.return_split_y)
                                win_dicts.append({
                                    "window_idx": w_idx,
                                    "window_number": w_idx + 1,
                                    "bbox": wbbox,
                                    "crop_rect": _clamp_crop_rect(wbbox, img_w, img_h),
                                    "return_split_y": round(float(win.return_split_y), 2) if win.return_split_y is not None else None,
                                    "original_elements": None,
                                    "elements": elems,
                                "original_errors": list(w_errs),
                                "errors": list(w_errs),
                                "warnings": list(w_warns),
                                "confirmed_errors": [],
                                "dismissed_errors": [],
                                "error_hints": hints,
                                "error_element_ids": _collect_window_error_element_ids(hints),
                                    "review_status": "pending" if (w_errs or w_warns) else "clean",
                                    "review_meta": {
                                        "updated_by": None,
                                        "updated_at": None,
                                        "comment": "",
                                        "actions_count": 0,
                                    },
                                })

                            pages.append({
                                "page_id": page_id,
                                "page_num": pr.page_num,
                                "pdf_local_page_num": len(wb_page_ids),
                                "is_title_page": bool(any(d.label == "Титульный лист" for d in pr.detections) and not pr.windows),
                                "global_page_num": global_idx,
                                "pdf_page_index": pr.pdf_page_index,
                                "source_file_idx": f_idx,
                                "source_filename": current_fname,
                                "waybill_idx": wb_counter,
                                "ml_number": ctx.ml_number,
                                "driver_name": ctx.driver_name,
                                "image_file": unique_img_name,
                                "width": img_w,
                                "height": img_h,
                                "structural_detections": struct_dets,
                                "windows": win_dicts,
                                "quarantine_item_id": None,
                            })

                        start_pdf_page = seg_pages[0].pdf_page_index if seg_pages else 0
                        end_pdf_page = seg_pages[-1].pdf_page_index if seg_pages else 0
                        waybills.append({
                            "waybill_idx": wb_counter,
                            "source_file_idx": f_idx,
                            "source_filename": current_fname,
                            "ml_number": ctx.ml_number,
                            "driver_name": ctx.driver_name,
                            "page_ids": wb_page_ids,
                            "start_pdf_page": start_pdf_page,
                            "end_pdf_page": end_pdf_page,
                            "total_pages": len(wb_page_ids),
                            "error_pages": 0,
                            "reviewed_pages": 0,
                        })
                        wb_counter += 1
                finally:
                    shutil.rmtree(tmp_file_dir, ignore_errors=True)

                pages_done_before += expected_for_file if file_pages_count == 0 else file_pages_count
                source_files_info.append({"filename": current_fname, "pages": file_pages_count})

            with lock:
                final_state = self._load_session(session_id)
                final_state["status"] = "ready"
                final_state["stage"] = "done"
                final_state["progress"] = 100
                final_state["processed_pages"] = global_idx
                final_state["total_pages"] = global_idx
                final_state["source_files"] = source_files_info
                final_state["waybills"] = waybills
                final_state["pages"] = pages
                self._refresh_session_metrics(final_state)
                self._save_session(session_id, final_state)

            self._cleanup_sessions()
            logger.info(f"Пакетная сессия {session_id} готова: файлов={len(source_files_info)}, страниц={global_idx}, МЛ={len(waybills)}")
        except Exception as exc:
            logger.exception(f"Ошибка пакетной обработки сессии {session_id}: {exc}")
            err_msg = str(exc) if str(exc).startswith("Файл ") else (
                f"Файл {current_fname}: ошибка при анализе PDF." if current_fname else "Ошибка при анализе пакета PDF."
            )
            with lock:
                try:
                    err_state = self._load_session(session_id)
                except Exception:
                    err_state = {"session_id": session_id}
                err_state["status"] = "error"
                err_state["stage"] = "error"
                err_state["error"] = err_msg
                self._save_session(session_id, err_state)
            self._cleanup_sessions()

    def _run_session_worker(self, session_id: str, pdf_file: Path, pages_dir: Path, max_pages: int) -> None:
        """Выполняет ML-анализ PDF в фоне и сохраняет структуру сессии."""
        lock = self._get_session_lock(session_id)
        try:
            with lock:
                state = self._load_session(session_id)
                state["status"] = "running"
                state["stage"] = "init"
                state["progress"] = 2
                self._save_session(session_id, state)

            def on_progress(done: int, total: int, stage: str) -> None:
                ratio = done / max(1, total)
                if stage == "render":
                    pct = int(5 + ratio * 25)
                elif stage == "detect":
                    pct = int(30 + ratio * 25)
                else:
                    pct = int(55 + ratio * 40)
                with lock:
                    cur = self._load_session(session_id)
                    cur["stage"] = stage
                    cur["progress"] = min(96, pct)
                    cur["processed_pages"] = done
                    cur["total_pages"] = total
                    self._save_session(session_id, cur)

            with self._inference_lock:
                segments = self.pipeline.analyze_pdf_segments(
                    str(pdf_file),
                    str(pages_dir),
                    max_pages=max_pages,
                    progress_callback=on_progress,
                    image_ext=".jpg",
                )

            waybills: List[Dict[str, Any]] = []
            pages: List[Dict[str, Any]] = []
            global_idx = 0

            for wb_idx, (ctx, seg_pages) in enumerate(segments):
                wb_page_ids: List[str] = []
                for pr in seg_pages:
                    global_idx += 1
                    page_id = f"p{global_idx:04d}"
                    wb_page_ids.append(page_id)

                    src_img_path = Path(pr.image_path)
                    with Image.open(src_img_path) as im:
                        img_w, img_h = im.size

                    struct_dets = []
                    for d_idx, d in enumerate(pr.detections):
                        if d.label in STRUCTURAL_PAGE_LABELS or (d.label == "Qr code" and not pr.windows):
                            struct_dets.append(_detection_to_dict(d, f"st_{global_idx}_{d_idx}", source="model"))

                    win_dicts = []
                    for w_idx, win in enumerate(pr.windows):
                        w_errs = [e for e in pr.errors if e.startswith(f"Окно #{w_idx + 1}:")]
                        w_warns = [w for w in pr.warnings if w.startswith(f"Окно #{w_idx + 1}:")]

                        elems = [
                            _detection_to_dict(el, f"el_{global_idx}_{w_idx}_{e_idx}", source="model")
                            for e_idx, el in enumerate(win.elements)
                        ]
                        wbbox = [
                            round(float(win.bbox.x1), 2),
                            round(float(win.bbox.y1), 2),
                            round(float(win.bbox.x2), 2),
                            round(float(win.bbox.y2), 2),
                        ]
                        hints = build_error_hints(w_errs, elems, win.return_split_y)
                        win_dicts.append({
                            "window_idx": w_idx,
                            "window_number": w_idx + 1,
                            "bbox": wbbox,
                            "crop_rect": _clamp_crop_rect(wbbox, img_w, img_h),
                            "return_split_y": round(float(win.return_split_y), 2) if win.return_split_y is not None else None,
                            "original_elements": None,
                            "elements": elems,
                            "original_errors": list(w_errs),
                            "errors": list(w_errs),
                            "warnings": list(w_warns),
                            "confirmed_errors": [],
                            "dismissed_errors": [],
                            "error_hints": hints,
                            "error_element_ids": _collect_window_error_element_ids(hints),
                            "review_status": "pending" if (w_errs or w_warns) else "clean",
                            "review_meta": {
                                "updated_by": None,
                                "updated_at": None,
                                "comment": "",
                                "actions_count": 0,
                            },
                        })

                    pages.append({
                        "page_id": page_id,
                        "page_num": pr.page_num,
                        "pdf_local_page_num": len(wb_page_ids),
                        "is_title_page": bool(any(d.label == "Титульный лист" for d in pr.detections) and not pr.windows),
                        "global_page_num": global_idx,
                        "pdf_page_index": pr.pdf_page_index,
                        "waybill_idx": wb_idx,
                        "ml_number": ctx.ml_number,
                        "driver_name": ctx.driver_name,
                        "image_file": src_img_path.name,
                        "width": img_w,
                        "height": img_h,
                        "structural_detections": struct_dets,
                        "windows": win_dicts,
                        "quarantine_item_id": None,
                    })

                start_pdf_page = seg_pages[0].pdf_page_index if seg_pages else 0
                end_pdf_page = seg_pages[-1].pdf_page_index if seg_pages else 0
                waybills.append({
                    "waybill_idx": wb_idx,
                    "ml_number": ctx.ml_number,
                    "driver_name": ctx.driver_name,
                    "page_ids": wb_page_ids,
                    "start_pdf_page": start_pdf_page,
                    "end_pdf_page": end_pdf_page,
                    "total_pages": len(wb_page_ids),
                    "error_pages": 0,
                    "reviewed_pages": 0,
                })

            with lock:
                final_state = self._load_session(session_id)
                final_state["status"] = "ready"
                final_state["stage"] = "done"
                final_state["progress"] = 100
                final_state["processed_pages"] = global_idx
                final_state["total_pages"] = global_idx
                final_state["waybills"] = waybills
                final_state["pages"] = pages
                self._refresh_session_metrics(final_state)
                self._save_session(session_id, final_state)

            self._cleanup_sessions()
            logger.info(f"Сессия {session_id} готова: страниц={global_idx}, МЛ={len(waybills)}")
        except Exception as exc:
            logger.exception(f"Ошибка фоновой обработки сессии {session_id}: {exc}")
            with lock:
                try:
                    err_state = self._load_session(session_id)
                except Exception:
                    err_state = {"session_id": session_id}
                err_state["status"] = "error"
                err_state["stage"] = "error"
                err_state["error"] = "Ошибка при анализе PDF. Проверьте корректность файла и логи сервиса."
                self._save_session(session_id, err_state)
            self._cleanup_sessions()

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает список последних сессий разбора с датой создания."""
        items: List[Dict[str, Any]] = []
        if not self.sessions_dir.exists():
            return items
        for entry in self.sessions_dir.iterdir():
            meta_file = entry / "session.json"
            if not meta_file.is_file():
                continue
            try:
                data = _read_json(meta_file)
                created_at = data.get("created_at") or datetime.fromtimestamp(meta_file.stat().st_mtime).isoformat(timespec="seconds")
                items.append({
                    "session_id": data["session_id"],
                    "source_filename": data.get("source_filename", ""),
                    "created_at": created_at,
                    "status": data.get("status", "unknown"),
                    "progress": data.get("progress", 0),
                    "finalized": bool(data.get("finalized", False)),
                    "finalized_at": data.get("finalized_at"),
                    "summary": data.get("summary", {}),
                })
            except Exception:
                continue
        items.sort(key=lambda x: (x.get("created_at", ""), x.get("session_id", "")), reverse=True)
        return items[:limit]

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Возвращает статус и прогресс сессии."""
        data = self._load_session(session_id)
        return {
            "session_id": data["session_id"],
            "source_filename": data.get("source_filename", ""),
            "created_at": data.get("created_at", ""),
            "status": data.get("status", "unknown"),
            "stage": data.get("stage", ""),
            "progress": data.get("progress", 0),
            "processed_pages": data.get("processed_pages", 0),
            "total_pages": data.get("total_pages", 0),
            "files_total": data.get("files_total", 1),
            "files_received": data.get("files_received", 1 if not data.get("is_batch") else 0),
            "current_file_idx": data.get("current_file_idx", 1 if data.get("status") in ("running", "ready") and not data.get("is_batch") else 0),
            "current_filename": data.get("current_filename", data.get("source_filename", "") if not data.get("is_batch") else ""),
            "source_files": data.get("source_files", []),
            "finalized": bool(data.get("finalized", False)),
            "finalized_at": data.get("finalized_at"),
            "error": data.get("error"),
            "summary": data.get("summary", {}),
        }

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Удаляет завершённую сессию проверки с диска."""
        sid = _validate_id(session_id, "session_id")
        lock = self._get_session_lock(sid)
        with lock, self._sessions_cleanup_lock:
            ses_dir = _safe_subpath(self.sessions_dir, sid)
            session_file = ses_dir / "session.json"
            if not session_file.exists():
                raise ServiceError(HTTPStatus.NOT_FOUND, f"Сессия '{sid}' не найдена.")
            data = _read_json(session_file)
            if data.get("status") in ("queued", "running"):
                raise ServiceError(HTTPStatus.CONFLICT, "Нельзя удалить сессию во время выполнения анализа.")
            shutil.rmtree(ses_dir, ignore_errors=True)
            return {"deleted": True, "session_id": sid}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Возвращает полную структуру сессии (без служебных дубликатов original_elements)."""
        data = self._load_session(session_id)
        for page in data.get("pages", []):
            for win in page.get("windows", []):
                win.pop("original_elements", None)
        return data

    @staticmethod
    def _find_page_and_window(
        session_data: Dict[str, Any],
        page_id: str,
        window_idx: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        pid = _validate_id(page_id, "page_id")
        page = next((p for p in session_data.get("pages", []) if p["page_id"] == pid), None)
        if page is None:
            raise ServiceError(HTTPStatus.NOT_FOUND, f"Страница '{pid}' не найдена.")
        if window_idx is None:
            return page, None
        windows = page.get("windows", [])
        if not (0 <= window_idx < len(windows)):
            raise ServiceError(HTTPStatus.NOT_FOUND, f"Окно #{window_idx} не найдено на странице '{pid}'.")
        return page, windows[window_idx]

    def get_page_image_bytes(self, session_id: str, page_id: str, annotated: bool = False) -> Tuple[bytes, str]:
        """Возвращает изображение полной страницы (оригинал или с текущими рамками)."""
        data = self._load_session(session_id)
        page, _ = self._find_page_and_window(data, page_id)
        img_path = _safe_subpath(self._session_dir(session_id), "pages", _validate_filename(page["image_file"]))
        if not img_path.exists():
            raise ServiceError(HTTPStatus.NOT_FOUND, "Изображение страницы не найдено.")
        if not annotated:
            ctype = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            return img_path.read_bytes(), ctype
        with Image.open(img_path) as im:
            rendered = _render_annotated_page(im, page)
            buf = io.BytesIO()
            rendered.save(buf, format="PNG")
            return buf.getvalue(), "image/png"

    def get_window_image_bytes(self, session_id: str, page_id: str, window_idx: int) -> Tuple[bytes, str]:
        """Возвращает вырезанное изображение окна аптеки по crop_rect."""
        data = self._load_session(session_id)
        page, win = self._find_page_and_window(data, page_id, window_idx)
        img_path = _safe_subpath(self._session_dir(session_id), "pages", _validate_filename(page["image_file"]))
        if not img_path.exists():
            raise ServiceError(HTTPStatus.NOT_FOUND, "Изображение страницы не найдено.")
        cx1, cy1, cx2, cy2 = win["crop_rect"]
        with Image.open(img_path) as im:
            crop = im.convert("RGB").crop((cx1, cy1, cx2, cy2))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return buf.getvalue(), "image/png"

    def get_element_crop_bytes(self, session_id: str, page_id: str, element_id: str) -> Tuple[bytes, str]:
        """Возвращает квадратный кроп элемента (1.4 * max_side для цифр)."""
        eid = _validate_id(element_id, "element_id")
        data = self._load_session(session_id)
        page, _ = self._find_page_and_window(data, page_id)
        target_el = None
        for win in page.get("windows", []):
            target_el = next((e for e in win.get("elements", []) if e["id"] == eid), None)
            if target_el is not None:
                break
        if target_el is None:
            raise ServiceError(HTTPStatus.NOT_FOUND, f"Элемент '{eid}' не найден.")
        img_path = _safe_subpath(self._session_dir(session_id), "pages", _validate_filename(page["image_file"]))
        bbox = BBox(*[float(v) for v in target_el["bbox"]])
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            if target_el["label"] in DIGIT_LABELS:
                crop = Pipeline._crop_digit_square(rgb, bbox)
            else:
                crop = rgb.crop((
                    max(0, int(bbox.x1 - 8)),
                    max(0, int(bbox.y1 - 8)),
                    min(rgb.width, int(bbox.x2 + 8)),
                    min(rgb.height, int(bbox.y2 + 8)),
                ))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=95)
            return buf.getvalue(), "image/jpeg"

    def _recompute_window_logic(self, win: Dict[str, Any], window_idx: int) -> None:
        """Мгновенно пересчитывает бизнес-логику по окну на основе текущих элементов."""
        tmp_win = Window(
            bbox=BBox(*[float(v) for v in win["bbox"]]),
            elements=[_dict_to_detection(e) for e in win["elements"]],
            return_split_y=win.get("return_split_y"),
        )
        errors, warnings = self.pipeline._apply_business_logic(
            [tmp_win],
            img_path=None,
            detect_return_splits=False,
        )
        rewritten_errors = _rewrite_window_error_prefix(errors, window_idx)
        dismissed_raw = list(win.get("dismissed_errors") or [])
        dismissed_norm = {_normalize_error_text(e) for e in dismissed_raw}
        filtered_errors = [
            e for e in rewritten_errors
            if _normalize_error_text(e) not in dismissed_norm
        ]

        win["errors"] = filtered_errors
        win["warnings"] = _rewrite_window_error_prefix(warnings, window_idx)

        curr_norm = {_normalize_error_text(e) for e in filtered_errors}
        confirmed_kept = [
            e for e in (win.get("confirmed_errors") or [])
            if _normalize_error_text(e) in curr_norm
        ]
        win["confirmed_errors"] = confirmed_kept
        win["dismissed_errors"] = dismissed_raw

        win["error_hints"] = build_error_hints(
            win["errors"],
            win["elements"],
            win.get("return_split_y"),
            confirmed_errors=confirmed_kept,
        )
        win["error_element_ids"] = _collect_window_error_element_ids(win["error_hints"])

        has_curr = bool(win["errors"] or win["warnings"])
        diff = _compute_window_diff(win.get("original_elements"), win["elements"])
        if has_curr:
            confirmed_norm = {_normalize_error_text(e) for e in confirmed_kept}
            all_confirmed = bool(win["errors"]) and not win["warnings"] and all(
                _normalize_error_text(e) in confirmed_norm for e in win["errors"]
            )
            if all_confirmed:
                win["review_status"] = "confirmed_error"
            else:
                win["review_status"] = "fixed_with_errors" if diff["has_changes"] else "pending"
        else:
            if dismissed_raw:
                win["review_status"] = "dismissed_clean"
            else:
                win["review_status"] = "fixed_clean" if diff["has_changes"] else "clean"

    def _apply_single_element_edit(
        self,
        elements: List[Dict[str, Any]],
        edit: Dict[str, Any],
        win_bbox: List[float],
        page_w: int,
        page_h: int,
        img_path: Path,
        source: str = "operator",
    ) -> List[Dict[str, Any]]:
        """Применяет одну правку элемента (add_box/delete_box/update_box/relabel_box/fix_digit) к списку элементов."""
        action = edit.get("action")
        id_prefix = "adm" if source == "admin" else "op"

        if action == "fix_digit":
            eid = _validate_id(str(edit.get("element_id") or ""), "element_id")
            el = next((e for e in elements if e["id"] == eid), None)
            if el is None:
                raise ServiceError(HTTPStatus.NOT_FOUND, "Элемент цифры не найден в окне.")
            if el["label"] not in DIGIT_LABELS:
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Изменить значение можно только у цифровой рамки.")
            val = _parse_digit_value(edit.get("new_value"))
            el["ocr_value"] = val
            el["ocr_text"] = str(val)
            el["cls_pred"] = str(val)
            el["cls_conf"] = 1.0
            el["source"] = source
            return elements

        if action == "delete_box":
            eid = _validate_id(str(edit.get("element_id") or ""), "element_id")
            el = next((e for e in elements if e["id"] == eid), None)
            if el is None:
                raise ServiceError(HTTPStatus.NOT_FOUND, "Удаляемый элемент не найден в окне.")
            if el["label"] in PROTECTED_LABELS:
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"Системный класс '{el['label']}' защищён от удаления.")
            return [e for e in elements if e["id"] != eid]

        if action == "relabel_box":
            eid = _validate_id(str(edit.get("element_id") or ""), "element_id")
            new_label = str(edit.get("new_label") or "")
            if new_label not in EDITABLE_LABELS:
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"Недопустимый целевой класс '{_sanitize_log(new_label)}'.")
            el = next((e for e in elements if e["id"] == eid), None)
            if el is None:
                raise ServiceError(HTTPStatus.NOT_FOUND, "Элемент не найден в окне.")
            if el["label"] in PROTECTED_LABELS:
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"Системный класс '{el['label']}' нельзя переименовать.")
            el["label"] = new_label
            el["source"] = source
            if new_label in DIGIT_LABELS:
                raw_val = edit.get("new_value")
                if raw_val is not None and str(raw_val).strip() != "":
                    val = _parse_digit_value(raw_val)
                    el["ocr_value"] = val
                    el["ocr_text"] = str(val)
                    el["cls_pred"] = str(val)
                    el["cls_conf"] = 1.0
                elif el.get("ocr_value") is None:
                    with self._inference_lock:
                        ocr_text, cls_pred, cls_conf = self.pipeline.classify_digit(
                            str(img_path), BBox(*[float(v) for v in el["bbox"]])
                        )
                    el["ocr_text"] = ocr_text
                    el["cls_pred"] = cls_pred
                    el["cls_conf"] = round(float(cls_conf), 4)
                    el["ocr_value"] = int(ocr_text) if ocr_text.isdigit() else None
            else:
                el["ocr_text"] = ""
                el["ocr_value"] = None
                el["cls_pred"] = ""
                el["cls_conf"] = 0.0
            return elements

        if action == "add_box":
            new_label = str(edit.get("new_label") or "")
            if new_label not in EDITABLE_LABELS:
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"Класс '{_sanitize_log(new_label)}' запрещён для добавления.")
            x1, y1, x2, y2 = _validate_box_geometry(
                edit.get("bbox"), new_label, win_bbox, page_w, page_h
            )
            ocr_text, cls_pred, cls_conf, ocr_val = "", "", 0.0, None
            if new_label in DIGIT_LABELS:
                raw_val = edit.get("new_value")
                if raw_val is not None and str(raw_val).strip() != "":
                    ocr_val = _parse_digit_value(raw_val)
                    ocr_text = str(ocr_val)
                    cls_pred = str(ocr_val)
                    cls_conf = 1.0
                else:
                    with self._inference_lock:
                        ocr_text, cls_pred, cls_conf = self.pipeline.classify_digit(str(img_path), BBox(x1, y1, x2, y2))
                    ocr_val = int(ocr_text) if ocr_text.isdigit() else None
            elements.append({
                "id": f"{id_prefix}_{uuid.uuid4().hex[:10]}",
                "label": new_label,
                "confidence": 1.0,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "ocr_text": ocr_text,
                "ocr_value": ocr_val,
                "cls_pred": cls_pred,
                "cls_conf": round(float(cls_conf), 4),
                "editable": True,
                "source": source,
            })
            return elements

        if action == "update_box":
            eid = _validate_id(str(edit.get("element_id") or ""), "element_id")
            el = next((e for e in elements if e["id"] == eid), None)
            if el is None:
                raise ServiceError(HTTPStatus.NOT_FOUND, "Элемент не найден в окне.")
            if el["label"] in PROTECTED_LABELS:
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"Системный класс '{el['label']}' защищён от изменений.")

            target_label = str(edit.get("new_label") or el["label"])
            if target_label not in EDITABLE_LABELS:
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"Недопустимый целевой класс '{_sanitize_log(target_label)}'.")

            bbox_changed = edit.get("bbox") is not None
            if bbox_changed:
                x1, y1, x2, y2 = _validate_box_geometry(
                    edit.get("bbox"), target_label, win_bbox, page_w, page_h
                )
                new_bbox = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            else:
                new_bbox = list(el["bbox"])

            el["label"] = target_label
            el["bbox"] = new_bbox
            el["source"] = source

            if target_label in DIGIT_LABELS:
                raw_val = edit.get("new_value")
                if raw_val is not None and str(raw_val).strip() != "":
                    val = _parse_digit_value(raw_val)
                    el["ocr_value"] = val
                    el["ocr_text"] = str(val)
                    el["cls_pred"] = str(val)
                    el["cls_conf"] = 1.0
                elif bbox_changed or el.get("ocr_value") is None:
                    with self._inference_lock:
                        ocr_text, cls_pred, cls_conf = self.pipeline.classify_digit(
                            str(img_path), BBox(*[float(v) for v in new_bbox])
                        )
                    el["ocr_text"] = ocr_text
                    el["cls_pred"] = cls_pred
                    el["cls_conf"] = round(float(cls_conf), 4)
                    el["ocr_value"] = int(ocr_text) if ocr_text.isdigit() else None
            else:
                el["ocr_text"] = ""
                el["ocr_value"] = None
                el["cls_pred"] = ""
                el["cls_conf"] = 0.0
            return elements

        raise ServiceError(HTTPStatus.BAD_REQUEST, f"Неизвестное действие '{_sanitize_log(action)}'.")

    def _apply_edits_atomic(
        self,
        elements: List[Dict[str, Any]],
        edits: Any,
        win_bbox: List[float],
        page_w: int,
        page_h: int,
        img_path: Path,
        source: str = "operator",
    ) -> List[Dict[str, Any]]:
        """Атомарно применяет пакет правок на глубокой копии списка элементов."""
        if not isinstance(edits, list) or not (1 <= len(edits) <= MAX_BATCH_EDITS):
            raise ServiceError(
                HTTPStatus.BAD_REQUEST,
                f"Поле 'edits' должно быть списком от 1 до {MAX_BATCH_EDITS} правок.",
            )
        for idx, item in enumerate(edits):
            if not isinstance(item, dict) or item.get("action") not in BATCH_EDIT_ACTIONS:
                raise ServiceError(
                    HTTPStatus.BAD_REQUEST,
                    f"Некорректная правка #{idx + 1}: допустимы только add_box, delete_box, update_box, relabel_box, fix_digit.",
                )
        working = json.loads(json.dumps(elements, ensure_ascii=False))
        for item in edits:
            working = self._apply_single_element_edit(
                working, item, win_bbox, page_w, page_h, img_path, source=source
            )
        return working

    def apply_action(self, session_id: str, page_id: str, window_idx: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Применяет правку оператора в окне и возвращает пересчитанный статус."""
        if not isinstance(payload, dict):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Тело запроса должно быть JSON-объектом.")
        action = payload.get("action")
        if not action:
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Не указано действие 'action'.")

        actor = _sanitize_log(payload.get("user") or payload.get("actor") or "operator", 64) or "operator"
        comment = _sanitize_log(payload.get("comment") or "", 300)

        lock = self._get_session_lock(session_id)
        with lock:
            session_data = self._load_session(session_id)
            if session_data.get("status") != "ready":
                raise ServiceError(HTTPStatus.CONFLICT, "Сессия ещё обрабатывается.")
            page, win = self._find_page_and_window(session_data, page_id, window_idx)
            img_path = _safe_subpath(self._session_dir(session_id), "pages", _validate_filename(page["image_file"]))
            page_w = int(page["width"])
            page_h = int(page["height"])
            elements: List[Dict[str, Any]] = win["elements"]
            actions_delta = 1

            if action == "batch_edit":
                edits = payload.get("edits")
                new_elements = self._apply_edits_atomic(
                    elements, edits, win["bbox"], page_w, page_h, img_path, source="operator"
                )
                _ensure_original_elements_saved(win)
                win["elements"] = new_elements
                self._recompute_window_logic(win, window_idx)
                actions_delta = len(edits)

            elif action in BATCH_EDIT_ACTIONS:
                working = json.loads(json.dumps(elements, ensure_ascii=False))
                new_elements = self._apply_single_element_edit(
                    working, payload, win["bbox"], page_w, page_h, img_path, source="operator"
                )
                _ensure_original_elements_saved(win)
                win["elements"] = new_elements
                self._recompute_window_logic(win, window_idx)

            elif action == "confirm_real_error":
                err_msg = payload.get("error_message")
                err_idx = payload.get("error_index")
                if err_msg is not None or err_idx is not None:
                    target_err = _resolve_target_error(win.get("errors", []), err_msg, err_idx)
                    if target_err is None:
                        raise ServiceError(HTTPStatus.BAD_REQUEST, "Указанная ошибка не найдена в окне.")
                    confirmed = list(win.get("confirmed_errors") or [])
                    if _normalize_error_text(target_err) not in {_normalize_error_text(e) for e in confirmed}:
                        confirmed.append(target_err)
                    win["confirmed_errors"] = confirmed
                    win["dismissed_errors"] = [
                        e for e in (win.get("dismissed_errors") or [])
                        if _normalize_error_text(e) != _normalize_error_text(target_err)
                    ]
                    win["error_hints"] = build_error_hints(
                        win["errors"],
                        win["elements"],
                        win.get("return_split_y"),
                        confirmed_errors=confirmed,
                    )
                    win["error_element_ids"] = _collect_window_error_element_ids(win["error_hints"])
                    confirmed_norm = {_normalize_error_text(e) for e in confirmed}
                    all_confirmed = bool(win["errors"]) and not win["warnings"] and all(
                        _normalize_error_text(e) in confirmed_norm for e in win["errors"]
                    )
                    if all_confirmed:
                        win["review_status"] = "confirmed_error"
                    else:
                        diff = _compute_window_diff(win.get("original_elements"), win["elements"])
                        win["review_status"] = "fixed_with_errors" if diff["has_changes"] else "pending"
                else:
                    win["dismissed_errors"] = []
                    self._recompute_window_logic(win, window_idx)
                    win["confirmed_errors"] = list(win["errors"])
                    win["error_hints"] = build_error_hints(
                        win["errors"],
                        win["elements"],
                        win.get("return_split_y"),
                        confirmed_errors=win["confirmed_errors"],
                    )
                    win["error_element_ids"] = _collect_window_error_element_ids(win["error_hints"])
                    win["review_status"] = "confirmed_error" if (win["errors"] or win["warnings"]) else "clean"

            elif action in ("mark_valid", "confirm_no_error"):
                err_msg = payload.get("error_message")
                err_idx = payload.get("error_index")
                if err_msg is not None or err_idx is not None:
                    target_err = _resolve_target_error(win.get("errors", []), err_msg, err_idx)
                    if target_err is None:
                        raise ServiceError(HTTPStatus.BAD_REQUEST, "Указанная ошибка не найдена в окне.")
                    target_norm = _normalize_error_text(target_err)
                    win["errors"] = [e for e in win.get("errors", []) if _normalize_error_text(e) != target_norm]
                    win["confirmed_errors"] = [
                        e for e in (win.get("confirmed_errors") or [])
                        if _normalize_error_text(e) != target_norm
                    ]
                    dismissed = list(win.get("dismissed_errors") or [])
                    if target_norm not in {_normalize_error_text(e) for e in dismissed}:
                        dismissed.append(target_err)
                    win["dismissed_errors"] = dismissed
                    win["error_hints"] = build_error_hints(
                        win["errors"],
                        win["elements"],
                        win.get("return_split_y"),
                        confirmed_errors=win["confirmed_errors"],
                    )
                    win["error_element_ids"] = _collect_window_error_element_ids(win["error_hints"])
                    if not win["errors"] and not win["warnings"]:
                        win["review_status"] = "dismissed_clean"
                    else:
                        confirmed_norm = {_normalize_error_text(e) for e in win["confirmed_errors"]}
                        all_confirmed = bool(win["errors"]) and not win["warnings"] and all(
                            _normalize_error_text(e) in confirmed_norm for e in win["errors"]
                        )
                        if all_confirmed:
                            win["review_status"] = "confirmed_error"
                        else:
                            diff = _compute_window_diff(win.get("original_elements"), win["elements"])
                            win["review_status"] = "fixed_with_errors" if diff["has_changes"] else "pending"
                else:
                    win["errors"] = []
                    win["warnings"] = []
                    win["confirmed_errors"] = []
                    win["dismissed_errors"] = []
                    win["error_hints"] = []
                    win["error_element_ids"] = []
                    win["review_status"] = "dismissed_clean"

            elif action == "reset_window":
                win["elements"] = json.loads(json.dumps(_get_orig_elements(win), ensure_ascii=False))
                win["original_elements"] = None
                win["confirmed_errors"] = []
                win["dismissed_errors"] = []
                self._recompute_window_logic(win, window_idx)

            else:
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"Неизвестное действие '{_sanitize_log(action)}'.")

            meta = win["review_meta"]
            meta["updated_by"] = actor
            meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
            if comment:
                meta["comment"] = comment
            meta["actions_count"] = 0 if action == "reset_window" else int(meta.get("actions_count", 0)) + actions_delta

            self._refresh_session_metrics(session_data)
            self._save_session(session_id, session_data)

            waybill = session_data["waybills"][page["waybill_idx"]]
            win_response = dict(win)
            win_response.pop("original_elements", None)
            return {
                "window": win_response,
                "page_quarantine_item_id": page.get("quarantine_item_id"),
                "waybill": waybill,
                "session_summary": session_data["summary"],
            }

    @staticmethod
    def _render_lossless_page_png(ses_dir: Path, page: Dict[str, Any], dst_png: Path) -> Image.Image:
        """Рендерит эталонный ЧБ PNG без JPEG-сжатия из исходного PDF для датасета."""
        src_file_idx = page.get("source_file_idx")
        if src_file_idx is not None:
            cand_pdf = ses_dir / "files" / f"input_{int(src_file_idx):03d}.pdf"
            input_pdf = cand_pdf if cand_pdf.is_file() else (ses_dir / "input.pdf")
        else:
            input_pdf = ses_dir / "input.pdf"
        pdf_idx = int(page.get("pdf_page_index", 0))
        if input_pdf.is_file():
            try:
                with fitz.open(input_pdf) as doc:
                    if 0 <= pdf_idx < len(doc):
                        fitz_page = doc.load_page(pdf_idx)
                        render_dpi = 72 if (fitz_page.rect.width > 2500 or fitz_page.rect.height > 2500) else 200
                        pix = fitz_page.get_pixmap(dpi=render_dpi, colorspace=fitz.csGRAY)
                        img = Image.frombytes("L", [pix.width, pix.height], pix.samples).convert("RGB")
                        img.save(dst_png, format="PNG")
                        return img
            except Exception as exc:
                logger.warning(f"Не удалось перерендерить lossless PNG из input.pdf (p={pdf_idx}): {exc}")

        fallback_img = _safe_subpath(ses_dir, "pages", _validate_filename(page["image_file"]))
        with Image.open(fallback_img) as im:
            rgb = im.convert("RGB")
            rgb.save(dst_png, format="PNG")
            return rgb.copy()

    def _store_page_in_quarantine(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        page: Dict[str, Any],
        actor: str,
        actor_name: str,
        comment: str,
        allow_without_diff: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Записывает или обновляет страницу с правками разметки в карантин pending."""
        ses_dir = self._session_dir(session_id)
        page_id = page["page_id"]

        window_diffs = []
        changed_windows_clean = True
        has_any_change = False
        has_dismissed_clean = False
        changed_digit_elements: List[Tuple[int, Dict[str, Any], Optional[int]]] = []

        for w_idx, win in enumerate(page.get("windows", [])):
            if win.get("review_status") == "dismissed_clean":
                has_dismissed_clean = True
            diff = _compute_window_diff(win.get("original_elements"), win["elements"])
            if diff["has_changes"]:
                has_any_change = True
                if win.get("errors") or win.get("warnings"):
                    changed_windows_clean = False
                window_diffs.append({"window_idx": w_idx, "window_number": w_idx + 1, "diff": diff})

                orig_by_id = {e["id"]: e for e in _get_orig_elements(win)}
                for el in win["elements"]:
                    if el["label"] in DIGIT_LABELS and el.get("ocr_value") is not None:
                        orig_el = orig_by_id.get(el["id"])
                        if orig_el is None or orig_el.get("ocr_value") != el.get("ocr_value") or orig_el.get("label") != el["label"]:
                            old_val = orig_el.get("ocr_value") if orig_el else None
                            changed_digit_elements.append((w_idx, el, old_val))

        if has_dismissed_clean or (not has_any_change and not allow_without_diff):
            return None

        existing_id = page.get("quarantine_item_id")
        if existing_id and (self.quarantine_dir / "approved" / existing_id).is_dir():
            return {"item_id": existing_id, "status": "approved", "invariant_passed": changed_windows_clean, "digit_crops_count": 0, "skipped_already_approved": True}

        reuse_pending = bool(existing_id and (self.quarantine_dir / "pending" / existing_id).is_dir())
        item_id = existing_id if reuse_pending else f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        item_dir = self.quarantine_dir / "pending" / item_id
        if reuse_pending:
            shutil.rmtree(item_dir, ignore_errors=True)
        crops_dir = item_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        rgb_img = self._render_lossless_page_png(ses_dir, page, item_dir / "page.png")
        digit_crops = []
        try:
            for c_idx, (w_idx, el, old_val) in enumerate(changed_digit_elements, start=1):
                kind = "pc" if el["label"] == "Печатная цифра" else "rc"
                crop_name = f"crop_{c_idx:02d}_{kind}_val{el['ocr_value']}.jpg"
                crop_img = Pipeline._crop_digit_square(rgb_img, BBox(*[float(v) for v in el["bbox"]]))
                crop_img.save(crops_dir / crop_name, quality=95)
                digit_crops.append({
                    "file": crop_name,
                    "window_idx": w_idx,
                    "element_id": el["id"],
                    "label": el["label"],
                    "old_value": old_val,
                    "value": int(el["ocr_value"]),
                    "bbox": el["bbox"],
                })
        finally:
            rgb_img.close()

        item_data = {
            "item_id": item_id,
            "status": "pending",
            "session_id": session_id,
            "source_filename": session_data.get("source_filename", ""),
            "page_id": page_id,
            "page_num": page["page_num"],
            "ml_number": page["ml_number"],
            "driver_name": page["driver_name"],
            "width": page["width"],
            "height": page["height"],
            "submitted_by": actor,
            "submitted_by_name": actor_name,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "comment": comment,
            "invariant_passed": changed_windows_clean,
            "window_diffs": window_diffs,
            "digit_crops": digit_crops,
            "structural_detections": page.get("structural_detections", []),
            "windows": [
                {
                    "window_idx": w["window_idx"],
                    "window_number": w["window_number"],
                    "bbox": w["bbox"],
                    "crop_rect": w["crop_rect"],
                    "return_split_y": w.get("return_split_y"),
                    "original_elements": _get_orig_elements(w),
                    "edited_elements": w["elements"],
                    "original_errors": w["original_errors"],
                    "errors": w["errors"],
                    "warnings": list(w.get("warnings", [])),
                }
                for w in page.get("windows", [])
            ],
            "reviewed_by": None,
            "reviewed_at": None,
            "review_comment": "",
            "exported_at": None,
            "export_batch": None,
        }
        _atomic_write_json(item_dir / "item.json", item_data)
        page["quarantine_item_id"] = item_id
        return {
            "item_id": item_id,
            "status": "pending",
            "invariant_passed": changed_windows_clean,
            "digit_crops_count": len(digit_crops),
        }

    def submit_page_to_quarantine(self, session_id: str, page_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Сохраняет исправленную оператором страницу в карантин дообучения."""
        actor = _sanitize_log((payload or {}).get("user") or "operator", 64) or "operator"
        actor_name = _sanitize_log((payload or {}).get("user_name") or actor, 100) or actor
        comment = _sanitize_log((payload or {}).get("comment") or "", 300)

        lock = self._get_session_lock(session_id)
        with lock, self._quarantine_lock:
            self._cleanup_quarantine_locked()
            session_data = self._load_session(session_id)
            page, _ = self._find_page_and_window(session_data, page_id)
            if any(w.get("review_status") == "dismissed_clean" for w in page.get("windows", [])):
                raise ServiceError(
                    HTTPStatus.BAD_REQUEST,
                    "На странице есть окна со статусом 'Ошибки нет' (mark_valid) без исправленной разметки — в датасет YOLO отправлять нельзя.",
                )
            res = self._store_page_in_quarantine(session_id, session_data, page, actor, actor_name, comment)
            if res is None:
                raise ServiceError(HTTPStatus.BAD_REQUEST, "На странице нет исправлений разметки модели для отправки в карантин.")
            self._refresh_session_metrics(session_data)
            self._save_session(session_id, session_data)
            res["session_summary"] = session_data["summary"]
            return res

    def finalize_session(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Финализирует проверку и автоматически отправляет в карантин страницы с правками разметки."""
        actor = _sanitize_log((payload or {}).get("user") or "operator", 64) or "operator"
        actor_name = _sanitize_log((payload or {}).get("user_name") or actor, 100) or actor
        comment = _sanitize_log((payload or {}).get("comment") or "Автосбор при финализации проверки", 300)
        manual_ids = {
            str(pid)
            for pid in ((payload or {}).get("manual_markup_page_ids") or [])
            if isinstance(pid, str) and pid
        }

        lock = self._get_session_lock(session_id)
        with lock, self._quarantine_lock:
            self._cleanup_quarantine_locked()
            session_data = self._load_session(session_id)
            if session_data.get("status") != "ready":
                raise ServiceError(HTTPStatus.CONFLICT, "Сессия ещё обрабатывается.")

            quarantined_pages = 0
            quarantined_crops = 0
            skipped_dismissed = 0

            for page in session_data.get("pages", []):
                windows = page.get("windows", [])
                is_manual = page["page_id"] in manual_ids
                has_diff = any(_compute_window_diff(w.get("original_elements"), w["elements"])["has_changes"] for w in windows)

                if any(w.get("review_status") == "dismissed_clean" for w in windows):
                    if has_diff or is_manual:
                        skipped_dismissed += 1
                    continue

                if is_manual:
                    if any(w.get("review_status") in ("pending", "fixed_with_errors") for w in windows):
                        continue
                    page_comment = f"{comment} [РУЧНАЯ ПРАВКА]".strip() if comment else "[РУЧНАЯ ПРАВКА]"
                    res = self._store_page_in_quarantine(
                        session_id, session_data, page, actor, actor_name, page_comment, allow_without_diff=True
                    )
                else:
                    if not has_diff:
                        continue
                    if any(bool(w.get("errors") or w.get("warnings")) for w in windows):
                        continue
                    res = self._store_page_in_quarantine(session_id, session_data, page, actor, actor_name, comment)

                if res is not None and not res.get("skipped_already_approved"):
                    quarantined_pages += 1
                    quarantined_crops += int(res.get("digit_crops_count", 0))

            session_data["finalized"] = True
            session_data["finalized_at"] = datetime.now().isoformat(timespec="seconds")
            self._refresh_session_metrics(session_data)
            self._save_session(session_id, session_data)

            valid_waybills, error_waybills = self._classify_waybills(session_data)
            return {
                "finalized": True,
                "finalized_at": session_data["finalized_at"],
                "status": "finalized",
                "auto_quarantined_pages": quarantined_pages,
                "auto_quarantined_crops": quarantined_crops,
                "submitted_pages": quarantined_pages,
                "submitted_crops": quarantined_crops,
                "skipped_dismissed_pages": skipped_dismissed,
                "valid_waybills": len(valid_waybills),
                "error_waybills": len(error_waybills),
                "session_summary": session_data["summary"],
            }

    def _locate_quarantine_item(self, item_id: str) -> Tuple[str, Path]:
        qid = _validate_id(item_id, "item_id")
        for bucket in ("pending", "approved", "rejected"):
            cand = _safe_subpath(self.quarantine_dir / bucket, qid)
            if (cand / "item.json").is_file():
                return bucket, cand
        raise ServiceError(HTTPStatus.NOT_FOUND, f"Запись карантина '{qid}' не найдена.")

    def list_quarantine(self, status_filter: str = "pending") -> List[Dict[str, Any]]:
        """Возвращает список элементов карантина для админ-проверки."""
        buckets = ("pending", "approved", "rejected") if status_filter == "all" else (status_filter,)
        if any(b not in ("pending", "approved", "rejected") for b in buckets):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Недопустимый фильтр status (pending|approved|rejected|all).")

        items: List[Dict[str, Any]] = []
        with self._quarantine_lock:
            self._cleanup_quarantine_locked()
            for bucket in buckets:
                bucket_dir = self.quarantine_dir / bucket
                if not bucket_dir.exists():
                    continue
                for entry in bucket_dir.iterdir():
                    meta_path = entry / "item.json"
                    if not meta_path.is_file():
                        continue
                    data = _read_json(meta_path)
                    counts = {"added": 0, "deleted": 0, "relabeled": 0, "digit_fixed": 0}
                    for wd in data.get("window_diffs", []):
                        d = wd["diff"]
                        counts["added"] += len(d.get("added", []))
                        counts["deleted"] += len(d.get("deleted", []))
                        counts["relabeled"] += len(d.get("relabeled", []))
                        counts["digit_fixed"] += len(d.get("digit_fixed", []))
                    items.append({
                        "item_id": data["item_id"],
                        "status": data["status"],
                        "ml_number": data.get("ml_number", "Unknown"),
                        "driver_name": data.get("driver_name", "Unknown"),
                        "page_num": data.get("page_num", 1),
                        "source_filename": data.get("source_filename", ""),
                        "submitted_by": data.get("submitted_by", ""),
                        "submitted_by_name": data.get("submitted_by_name", ""),
                        "submitted_at": data.get("submitted_at", ""),
                        "invariant_passed": data.get("invariant_passed", False),
                        "admin_edited": bool(data.get("admin_edited", False)),
                        "diff_counts": counts,
                        "digit_crops_count": len(data.get("digit_crops", [])),
                        "exported_at": data.get("exported_at"),
                        "reviewed_by": data.get("reviewed_by"),
                        "reviewed_at": data.get("reviewed_at"),
                    })

        items.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
        return items

    def get_quarantine_item(self, item_id: str) -> Dict[str, Any]:
        """Возвращает детальную карточку записи карантина."""
        with self._quarantine_lock:
            _, item_dir = self._locate_quarantine_item(item_id)
            return _read_json(item_dir / "item.json")

    def get_quarantine_image_bytes(
        self,
        item_id: str,
        window_idx: Optional[int] = None,
        crop_file: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """Возвращает изображение страницы, окна или кропа цифры из карантина."""
        with self._quarantine_lock:
            _, item_dir = self._locate_quarantine_item(item_id)
            data = _read_json(item_dir / "item.json")
            if crop_file is not None:
                safe_crop = _validate_filename(crop_file, "crop_file")
                crop_path = _safe_subpath(item_dir / "crops", safe_crop)
                if not crop_path.is_file():
                    raise ServiceError(HTTPStatus.NOT_FOUND, "Кроп цифры не найден.")
                return crop_path.read_bytes(), "image/jpeg"

            page_path = item_dir / "page.png"
            if not page_path.is_file():
                raise ServiceError(HTTPStatus.NOT_FOUND, "Изображение карантина не найдено.")
            if window_idx is None:
                return page_path.read_bytes(), "image/png"

            windows = data.get("windows", [])
            if not (0 <= window_idx < len(windows)):
                raise ServiceError(HTTPStatus.NOT_FOUND, f"Окно #{window_idx} не найдено в записи карантина.")
            cx1, cy1, cx2, cy2 = windows[window_idx]["crop_rect"]
            with Image.open(page_path) as im:
                crop = im.convert("RGB").crop((cx1, cy1, cx2, cy2))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                return buf.getvalue(), "image/png"

    def decide_quarantine_item(self, item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Одобряет или отклоняет правку в карантине от имени администратора."""
        if not isinstance(payload, dict):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Некорректный формат запроса.")
        decision = str(payload.get("decision") or "").lower()
        if decision not in ("approve", "reject"):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Параметр 'decision' должен быть 'approve' или 'reject'.")
        admin_user = _sanitize_log(payload.get("admin_user") or "admin", 64) or "admin"
        comment = _sanitize_log(payload.get("comment") or "", 300)

        target_bucket = "approved" if decision == "approve" else "rejected"
        with self._quarantine_lock:
            self._cleanup_quarantine_locked()
            cur_bucket, src_dir = self._locate_quarantine_item(item_id)
            data = _read_json(src_dir / "item.json")
            data["status"] = target_bucket
            data["reviewed_by"] = admin_user
            data["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            data["review_comment"] = comment

            if cur_bucket != target_bucket:
                dst_dir = self.quarantine_dir / target_bucket / data["item_id"]
                if dst_dir.exists():
                    shutil.rmtree(dst_dir, ignore_errors=True)
                shutil.move(str(src_dir), str(dst_dir))
                _atomic_write_json(dst_dir / "item.json", data)
            else:
                _atomic_write_json(src_dir / "item.json", data)

            return {
                "item_id": data["item_id"],
                "status": data["status"],
                "reviewed_by": data["reviewed_by"],
                "reviewed_at": data["reviewed_at"],
            }

    def edit_quarantine_window(self, item_id: str, window_idx: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Атомарно редактирует разметку окна в pending-записи карантина от имени администратора."""
        if not isinstance(payload, dict):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "Тело запроса должно быть JSON-объектом.")
        admin_user = _sanitize_log(payload.get("admin_user") or "admin", 64) or "admin"
        admin_name = _sanitize_log(payload.get("admin_name") or admin_user, 100) or admin_user

        with self._quarantine_lock:
            bucket, item_dir = self._locate_quarantine_item(item_id)
            data = _read_json(item_dir / "item.json")
            if bucket != "pending" or data.get("status") != "pending":
                raise ServiceError(
                    HTTPStatus.CONFLICT,
                    "Редактировать разметку можно только у записей карантина в статусе pending.",
                )

            windows = data.get("windows", [])
            if not (0 <= window_idx < len(windows)):
                raise ServiceError(HTTPStatus.NOT_FOUND, f"Окно #{window_idx} не найдено в записи карантина.")

            page_path = item_dir / "page.png"
            if not page_path.is_file():
                raise ServiceError(HTTPStatus.NOT_FOUND, "Изображение карантина не найдено.")

            win = windows[window_idx]
            img_w = int(data["width"])
            img_h = int(data["height"])

            new_elements = self._apply_edits_atomic(
                win.get("edited_elements", []),
                payload.get("edits"),
                win["bbox"],
                img_w,
                img_h,
                page_path,
                source="admin",
            )
            win["edited_elements"] = new_elements

            tmp_win = Window(
                bbox=BBox(*[float(v) for v in win["bbox"]]),
                elements=[_dict_to_detection(e) for e in new_elements],
                return_split_y=win.get("return_split_y"),
            )
            errs, warns = self.pipeline._apply_business_logic(
                [tmp_win],
                img_path=None,
                detect_return_splits=False,
            )
            win["errors"] = _rewrite_window_error_prefix(errs, window_idx)
            win["warnings"] = _rewrite_window_error_prefix(warns, window_idx)

            window_diffs: List[Dict[str, Any]] = []
            changed_windows_clean = True
            changed_digit_elements: List[Tuple[int, Dict[str, Any], Optional[int]]] = []

            for w_i, w_obj in enumerate(windows):
                orig_els = w_obj.get("original_elements", [])
                ed_els = w_obj.get("edited_elements", [])
                diff = _compute_window_diff(orig_els, ed_els)
                if diff["has_changes"]:
                    if w_obj.get("errors") or w_obj.get("warnings"):
                        changed_windows_clean = False
                    window_diffs.append({
                        "window_idx": w_i,
                        "window_number": w_obj.get("window_number", w_i + 1),
                        "diff": diff,
                    })
                    orig_by_id = {e["id"]: e for e in orig_els}
                    for el in ed_els:
                        if el["label"] in DIGIT_LABELS and el.get("ocr_value") is not None:
                            orig_el = orig_by_id.get(el["id"])
                            if orig_el is None or orig_el.get("ocr_value") != el.get("ocr_value") or orig_el.get("label") != el["label"]:
                                old_val = orig_el.get("ocr_value") if orig_el else None
                                changed_digit_elements.append((w_i, el, old_val))

            crops_dir = item_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            for old_crop in crops_dir.glob("*.jpg"):
                old_crop.unlink(missing_ok=True)

            digit_crops = []
            with Image.open(page_path) as im:
                rgb_img = im.convert("RGB")
                try:
                    for c_idx, (w_i, el, old_val) in enumerate(changed_digit_elements, start=1):
                        kind = "pc" if el["label"] == "Печатная цифра" else "rc"
                        crop_name = f"crop_{c_idx:02d}_{kind}_val{el['ocr_value']}.jpg"
                        crop_img = Pipeline._crop_digit_square(rgb_img, BBox(*[float(v) for v in el["bbox"]]))
                        crop_img.save(crops_dir / crop_name, quality=95)
                        digit_crops.append({
                            "file": crop_name,
                            "window_idx": w_i,
                            "element_id": el["id"],
                            "label": el["label"],
                            "old_value": old_val,
                            "value": int(el["ocr_value"]),
                            "bbox": el["bbox"],
                        })
                finally:
                    rgb_img.close()

            data["window_diffs"] = window_diffs
            data["invariant_passed"] = changed_windows_clean
            data["digit_crops"] = digit_crops
            data["admin_edited"] = True
            data["admin_edited_by"] = admin_user
            data["admin_edited_by_name"] = admin_name
            data["admin_edited_at"] = datetime.now().isoformat(timespec="seconds")

            _atomic_write_json(item_dir / "item.json", data)
            return data

    def export_approved(self, include_exported: bool = False) -> Dict[str, Any]:
        """Экспортирует одобренные правки в датасет (PNG + YOLO txt + Label Studio JSON + кропы) и очищает выгруженные."""
        with self._quarantine_lock:
            approved_dir = self.quarantine_dir / "approved"
            selected: List[Tuple[Path, Dict[str, Any]]] = []
            for entry in sorted(approved_dir.iterdir()):
                meta_path = entry / "item.json"
                if not meta_path.is_file():
                    continue
                data = _read_json(meta_path)
                if not include_exported and data.get("exported_at"):
                    continue
                selected.append((entry, data))

            if not selected:
                self._cleanup_quarantine_locked()
                return {
                    "exported_pages": 0,
                    "exported_crops": 0,
                    "export_dir": None,
                    "export_zip": None,
                    "message": "Нет новых одобренных записей для выгрузки.",
                }

            now = datetime.now()
            batch_name = f"export_{now.strftime('%Y-%m-%d_%H%M%S')}"
            batch_dir = self.exports_dir / batch_name
            out_images = batch_dir / "images"
            out_labels = batch_dir / "labels"
            out_crops = batch_dir / "crops"
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            out_crops.mkdir(parents=True, exist_ok=True)

            ls_tasks: List[Dict[str, Any]] = []
            written_files: List[Tuple[Path, bool]] = []
            total_crops = 0
            exported_dirs: List[Path] = []

            for entry_dir, data in selected:
                page_src = entry_dir / "page.png"
                if not page_src.is_file():
                    continue
                safe_ml = _safe_token(data.get("ml_number", "Unknown"), "ML")
                img_stem = f"{safe_ml}_{data['item_id']}_p{int(data.get('page_num', 1)):03d}"
                img_name = f"{img_stem}.png"
                dst_img = out_images / img_name
                shutil.copy2(page_src, dst_img)
                written_files.append((dst_img, True))

                img_w = int(data["width"])
                img_h = int(data["height"])
                ls_results: List[Dict[str, Any]] = []
                yolo_lines: List[str] = []

                def _append_box(bbox: List[float], label: str, res_id: str) -> None:
                    cls_id = CLASS_TO_ID.get(label)
                    if cls_id is None:
                        return
                    ls_results.append(_to_label_studio_result(bbox, img_w, img_h, label, res_id))
                    yline = _to_yolo_line(bbox, img_w, img_h, cls_id)
                    if yline:
                        yolo_lines.append(yline)

                for s_idx, det in enumerate(data.get("structural_detections", [])):
                    _append_box(det["bbox"], det["label"], f"s{s_idx}_{det['id']}")

                for w_idx, win in enumerate(data.get("windows", [])):
                    _append_box(win["bbox"], "Окно аптеки", f"w{w_idx}")
                    for e_idx, el in enumerate(win.get("edited_elements", [])):
                        _append_box(el["bbox"], el["label"], f"e{w_idx}_{e_idx}_{el['id']}")

                dst_lbl = out_labels / f"{img_stem}.txt"
                dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
                written_files.append((dst_lbl, False))

                ls_tasks.append({
                    "data": {"image": f"/data/local-files/?d=images/{img_name}"},
                    "annotations": [{"result": ls_results}],
                    "meta": {
                        "item_id": data["item_id"],
                        "ml_number": data.get("ml_number"),
                        "driver_name": data.get("driver_name"),
                        "submitted_by": data.get("submitted_by"),
                        "reviewed_by": data.get("reviewed_by"),
                    },
                })

                for crop_meta in data.get("digit_crops", []):
                    crop_src = entry_dir / "crops" / _validate_filename(crop_meta["file"], "crop_file")
                    if not crop_src.is_file():
                        continue
                    cls_folder = str(int(crop_meta["value"]))
                    dst_cls_dir = out_crops / cls_folder
                    dst_cls_dir.mkdir(parents=True, exist_ok=True)
                    kind = "pc" if crop_meta["label"] == "Печатная цифра" else "rc"
                    dst_crop = dst_cls_dir / f"{safe_ml}_{data['item_id']}_{kind}_{uuid.uuid4().hex[:6]}.jpg"
                    shutil.copy2(crop_src, dst_crop)
                    written_files.append((dst_crop, True))
                    total_crops += 1

                exported_dirs.append(entry_dir)

            classes_file = batch_dir / "classes.txt"
            classes_file.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
            ann_file = batch_dir / "annotations_export.json"
            _atomic_write_json(ann_file, ls_tasks)
            written_files.append((classes_file, True))
            written_files.append((ann_file, True))

            for target_file, require_nonempty in written_files:
                if not target_file.is_file():
                    raise ServiceError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Ошибка записи экспорта: файл {target_file.name} не создан.")
                if require_nonempty and target_file.stat().st_size == 0:
                    raise ServiceError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Ошибка записи экспорта: файл {target_file.name} пуст.")

            zip_path = self.exports_dir / f"dataset_{now.strftime('%Y-%m-%d_%H-%M')}.zip"
            if zip_path.exists():
                zip_path = self.exports_dir / f"dataset_{now.strftime('%Y-%m-%d_%H-%M-%S')}.zip"
            if zip_path.exists():
                zip_path = self.exports_dir / f"dataset_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:4]}.zip"

            tmp_zip = self.exports_dir / f".{zip_path.name}.tmp"
            try:
                with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for item_path in sorted(batch_dir.rglob("*")):
                        if item_path.is_file():
                            zf.write(item_path, arcname=item_path.relative_to(batch_dir).as_posix())
                if not tmp_zip.is_file() or tmp_zip.stat().st_size == 0:
                    raise ServiceError(HTTPStatus.INTERNAL_SERVER_ERROR, "Ошибка записи ZIP-архива экспорта.")
                os.replace(tmp_zip, zip_path)
            except Exception:
                tmp_zip.unlink(missing_ok=True)
                raise

            for entry_dir in exported_dirs:
                shutil.rmtree(entry_dir, ignore_errors=True)

            self._cleanup_quarantine_locked()
            return {
                "exported_pages": len(ls_tasks),
                "exported_crops": total_crops,
                "export_batch": batch_name,
                "export_dir": str(batch_dir),
                "export_zip": str(zip_path),
                "annotations_file": str(ann_file),
                "labels_dir": str(out_labels),
            }

    @staticmethod
    def _build_waybill_export_names(session_data: Dict[str, Any]) -> Dict[int, str]:
        """Формирует уникальные имена МЛ формата {safe_ml}_{safe_driver}_{safe_source_stem}."""
        default_src = session_data.get("source_filename", "document")
        seen_counts: Dict[str, int] = {}
        names: Dict[int, str] = {}
        for wb in session_data.get("waybills", []):
            wb_src = wb.get("source_filename") or default_src
            src_stem = _safe_file_stem(wb_src, "document")
            safe_ml = _safe_token(wb.get("ml_number", "Unknown"), "Unknown_ML")
            safe_driver = _safe_token(wb.get("driver_name", "Unknown"), "Unknown_Driver")
            base_key = f"{safe_ml}_{safe_driver}"
            seen_counts[base_key] = seen_counts.get(base_key, 0) + 1
            suffix = f"_{seen_counts[base_key]}" if seen_counts[base_key] > 1 else ""
            names[int(wb["waybill_idx"])] = f"{base_key}{suffix}_{src_stem}"
        return names

    @staticmethod
    def _classify_waybills(session_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Разделяет маршрутные листы на валидные и требующие доработки по текущему состоянию."""
        pages_by_id = {p["page_id"]: p for p in session_data.get("pages", [])}
        valid_list: List[Dict[str, Any]] = []
        error_list: List[Dict[str, Any]] = []

        for wb in session_data.get("waybills", []):
            wb_errors: List[str] = []
            for pid in wb.get("page_ids", []):
                page = pages_by_id.get(pid)
                if not page:
                    continue
                for win in page.get("windows", []):
                    for msg in win.get("errors", []) + win.get("warnings", []):
                        m_win = re.match(r"^Окно #(\d+):\s*(.*)$", msg)
                        if m_win:
                            wb_errors.append(f"  Стр. {int(page['page_num']):02d}, Окно #{m_win.group(1)}: {m_win.group(2)}")
                        else:
                            wb_errors.append(f"  Стр. {int(page['page_num']):02d}, Окно #{int(win['window_number'])}: {msg}")
            entry = {"waybill": wb, "errors": wb_errors}
            if wb_errors:
                error_list.append(entry)
            else:
                valid_list.append(entry)
        return valid_list, error_list

    def build_report_text(self, session_id: str) -> Tuple[str, str]:
        """Формирует двухсекционный текстовый отчёт (сначала проблемные МЛ, затем валидные)."""
        data = self._load_session(session_id)
        wb_names = self._build_waybill_export_names(data)
        valid_list, error_list = self._classify_waybills(data)

        lines: List[str] = []
        source_files = data.get("source_files") or []
        if len(source_files) > 1:
            lines.append(f"ПАКЕТ ФАЙЛОВ ({len(source_files)}):")
            for sf in source_files:
                lines.append(f"  - {sf.get('filename', 'document.pdf')} ({int(sf.get('pages', 0))} стр.)")
            lines.append("")

        lines.append("МАРШРУТНЫЕ ЛИСТЫ НА ДОРАБОТКУ:")
        if error_list:
            for item in error_list:
                lines.append(wb_names[int(item["waybill"]["waybill_idx"])])
                lines.extend(item["errors"])
        else:
            lines.append("(нет)")

        lines.append("")
        lines.append("ВАЛИДНЫЕ МАРШРУТНЫЕ ЛИСТЫ:")
        if valid_list:
            for item in valid_list:
                lines.append(wb_names[int(item["waybill"]["waybill_idx"])])
        else:
            lines.append("(нет)")

        dl_name = f"report_{_safe_file_stem(data.get('source_filename', 'ml'))}.txt"
        return "\n".join(lines) + "\n", dl_name

    def build_export_zip_file(self, session_id: str) -> Tuple[Path, str]:
        """Нарезает чистые исходные PDF по МЛ и собирает ZIP с папками Архив/, Ошибки/ и отчётом."""
        data = self._load_session(session_id)
        if data.get("status") != "ready":
            raise ServiceError(HTTPStatus.CONFLICT, "Сессия ещё обрабатывается.")

        ses_dir = self._session_dir(session_id)
        is_batch = bool(data.get("is_batch"))
        default_input_pdf = ses_dir / "input.pdf"
        if not is_batch and not default_input_pdf.is_file():
            raise ServiceError(HTTPStatus.NOT_FOUND, "Исходный PDF сессии не найден.")

        wb_names = self._build_waybill_export_names(data)
        valid_list, error_list = self._classify_waybills(data)
        valid_indices = {int(item["waybill"]["waybill_idx"]) for item in valid_list}
        report_txt, report_name = self.build_report_text(session_id)
        src_stem = _safe_file_stem(data.get("source_filename", "ml"))

        tmp_fd, tmp_zip_str = tempfile.mkstemp(prefix=f"export_{session_id}_", suffix=".zip", dir=str(self.exports_dir))
        os.close(tmp_fd)
        tmp_zip_path = Path(tmp_zip_str)

        open_docs: Dict[str, fitz.Document] = {}
        try:
            with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(report_name, report_txt.encode("utf-8"))

                pages_by_id = {p["page_id"]: p for p in data.get("pages", [])}
                for wb in data.get("waybills", []):
                    wb_idx = int(wb["waybill_idx"])
                    folder = "Архив" if wb_idx in valid_indices else "Ошибки"
                    pdf_name = f"{wb_names[wb_idx]}.pdf"

                    src_file_idx = wb.get("source_file_idx")
                    if src_file_idx is not None:
                        pdf_path = ses_dir / "files" / f"input_{int(src_file_idx):03d}.pdf"
                    else:
                        pdf_path = default_input_pdf

                    if not pdf_path.is_file():
                        continue
                    doc_key = str(pdf_path)
                    if doc_key not in open_docs:
                        open_docs[doc_key] = fitz.open(pdf_path)
                    src_doc = open_docs[doc_key]

                    if "start_pdf_page" in wb and "end_pdf_page" in wb:
                        start_p = int(wb["start_pdf_page"])
                        end_p = int(wb["end_pdf_page"])
                        page_indices = list(range(start_p, end_p + 1))
                    else:
                        page_indices = [
                            int(pages_by_id[pid]["pdf_page_index"])
                            for pid in wb.get("page_ids", [])
                            if pid in pages_by_id
                        ]
                    if not page_indices:
                        continue

                    sub_doc = fitz.open()
                    try:
                        if page_indices == list(range(page_indices[0], page_indices[-1] + 1)):
                            sub_doc.insert_pdf(src_doc, from_page=page_indices[0], to_page=page_indices[-1])
                        else:
                            for pidx in page_indices:
                                sub_doc.insert_pdf(src_doc, from_page=pidx, to_page=pidx)
                        zf.writestr(f"{folder}/{pdf_name}", sub_doc.tobytes(deflate=True))
                    finally:
                        sub_doc.close()
        except Exception:
            tmp_zip_path.unlink(missing_ok=True)
            raise
        finally:
            for d in open_docs.values():
                try:
                    d.close()
                except Exception:
                    pass

        return tmp_zip_path, f"waybills_{src_stem}.zip"

    def build_archive_pdf_bytes(self, session_id: str) -> Tuple[bytes, str]:
        """Собирает обновлённый многостраничный PDF со свежими рамками после проверки."""
        data = self._load_session(session_id)
        ses_dir = self._session_dir(session_id)
        rendered_pages: List[Image.Image] = []
        for page in data.get("pages", []):
            img_path = _safe_subpath(ses_dir, "pages", _validate_filename(page["image_file"]))
            if not img_path.is_file():
                continue
            with Image.open(img_path) as im:
                rendered_pages.append(_render_annotated_page(im, page))

        if not rendered_pages:
            raise ServiceError(HTTPStatus.NOT_FOUND, "Нет страниц для сборки PDF-архива.")

        buf = io.BytesIO()
        rendered_pages[0].save(buf, format="PDF", save_all=True, append_images=rendered_pages[1:])
        dl_name = f"checked_{_safe_file_stem(data.get('source_filename', 'ml'))}.pdf"
        return buf.getvalue(), dl_name


def _make_handler(service: ReviewService, api_token: Optional[str]):
    """Создаёт класс обработчика HTTP-запросов для ReviewService."""

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

        def _send_json(self, status_code: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status_code: int, data: bytes, content_type: str, download_name: Optional[str] = None) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if download_name:
                quoted = urllib.parse.quote(download_name)
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quoted}")
            self.end_headers()
            self.wfile.write(data)

        def _send_file_stream(self, file_path: Path, content_type: str, download_name: str, unlink_after: bool = True) -> None:
            try:
                file_size = file_path.stat().st_size
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Cache-Control", "no-store")
                quoted = urllib.parse.quote(download_name)
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quoted}")
                self.end_headers()
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            finally:
                if unlink_after:
                    file_path.unlink(missing_ok=True)

        def _require_token(self, path: str) -> None:
            if not api_token or path == "/api/ml/health":
                return
            supplied = self.headers.get("X-ML-Token", "")
            if not hmac.compare_digest(supplied, api_token):
                raise ServiceError(HTTPStatus.UNAUTHORIZED, "Неверный или отсутствующий токен X-ML-Token.")

        def _parse_content_length(self, max_bytes: int) -> int:
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                raise ServiceError(HTTPStatus.LENGTH_REQUIRED, "Отсутствует заголовок Content-Length.")
            try:
                length = int(raw_len)
            except ValueError:
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Некорректный заголовок Content-Length.")
            if length < 0 or length > max_bytes:
                raise ServiceError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Размер запроса превышает лимит {max_bytes} байт.")
            return length

        def _read_body(self, max_bytes: int) -> bytes:
            length = self._parse_content_length(max_bytes)
            return self.rfile.read(length)

        def _stream_upload_to_temp(self) -> Path:
            length = self._parse_content_length(MAX_UPLOAD_BYTES)
            if length == 0:
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Загруженный файл пуст.")
            tmp_fd, tmp_name = tempfile.mkstemp(prefix=".upload_", suffix=".pdf", dir=str(service.sessions_dir))
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)
            remaining = length
            first_chunk = True
            try:
                with open(tmp_path, "wb") as dst:
                    while remaining > 0:
                        to_read = min(UPLOAD_CHUNK_BYTES, remaining)
                        chunk = self.rfile.read(to_read)
                        if not chunk:
                            raise ServiceError(HTTPStatus.BAD_REQUEST, "Неожиданный обрыв загрузки файла.")
                        if first_chunk:
                            first_chunk = False
                            if b"%PDF-" not in chunk[:1024]:
                                raise ServiceError(HTTPStatus.BAD_REQUEST, "Загруженный файл не является корректным PDF.")
                        dst.write(chunk)
                        remaining -= len(chunk)
                return tmp_path
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

        def _read_json_body(self) -> Dict[str, Any]:
            raw = self._read_body(MAX_JSON_BODY_BYTES)
            if not raw:
                return {}
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except Exception:
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Некорректный JSON в теле запроса.")
            if not isinstance(parsed, dict):
                raise ServiceError(HTTPStatus.BAD_REQUEST, "Ожидается JSON-объект.")
            return parsed

        def _dispatch(self, method: str) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)
            self._require_token(path)

            if method == "GET" and path == "/api/ml/health":
                return self._send_json(HTTPStatus.OK, service.health_info())

            if method == "GET" and path == "/api/ml/sessions":
                limit = int(qs.get("limit", ["20"])[0]) if qs.get("limit", ["20"])[0].isdigit() else 20
                return self._send_json(HTTPStatus.OK, {"sessions": service.list_sessions(limit=limit)})

            if method == "POST" and path == "/api/ml/sessions":
                body = self._read_json_body()
                max_pages = int(body.get("max_pages", DEFAULT_MAX_PAGES))
                res = service.create_session_from_pdf(pdf_path=body.get("pdf_path"), max_pages=max_pages)
                return self._send_json(HTTPStatus.ACCEPTED, res)

            if method == "POST" and path == "/api/ml/sessions/batch":
                body = self._read_json_body()
                return self._send_json(HTTPStatus.CREATED, service.create_batch_session(body))

            if method == "POST" and path == "/api/ml/sessions/upload":
                tmp_pdf = self._stream_upload_to_temp()
                try:
                    fname = qs.get("filename", ["upload.pdf"])[0]
                    mp_str = qs.get("max_pages", [str(DEFAULT_MAX_PAGES)])[0]
                    max_pages = int(mp_str) if mp_str.isdigit() else DEFAULT_MAX_PAGES
                    res = service.create_session_from_pdf(
                        uploaded_temp_path=tmp_pdf,
                        filename=fname,
                        max_pages=max_pages,
                    )
                finally:
                    tmp_pdf.unlink(missing_ok=True)
                return self._send_json(HTTPStatus.ACCEPTED, res)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/files", path)
            if method == "POST" and m:
                tmp_pdf = self._stream_upload_to_temp()
                try:
                    fname = qs.get("filename", ["upload.pdf"])[0]
                    res = service.add_batch_file(m.group(1), tmp_pdf, filename=fname)
                finally:
                    tmp_pdf.unlink(missing_ok=True)
                return self._send_json(HTTPStatus.OK, res)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/start", path)
            if method == "POST" and m:
                body = self._read_json_body()
                return self._send_json(HTTPStatus.ACCEPTED, service.start_batch_session(m.group(1), body))

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/status", path)
            if method == "GET" and m:
                return self._send_json(HTTPStatus.OK, service.get_session_status(m.group(1)))

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)", path)
            if method == "GET" and m:
                return self._send_json(HTTPStatus.OK, service.get_session(m.group(1)))
            if method == "DELETE" and m:
                return self._send_json(HTTPStatus.OK, service.delete_session(m.group(1)))

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/delete", path)
            if method == "POST" and m:
                return self._send_json(HTTPStatus.OK, service.delete_session(m.group(1)))

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/report", path)
            if method == "GET" and m:
                txt, fname = service.build_report_text(m.group(1))
                return self._send_bytes(HTTPStatus.OK, txt.encode("utf-8"), "text/plain; charset=utf-8", fname)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/export-zip", path)
            if method == "GET" and m:
                zip_path, dl_name = service.build_export_zip_file(m.group(1))
                return self._send_file_stream(zip_path, "application/zip", dl_name, unlink_after=True)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/archive-pdf", path)
            if method == "GET" and m:
                pdf_data, fname = service.build_archive_pdf_bytes(m.group(1))
                return self._send_bytes(HTTPStatus.OK, pdf_data, "application/pdf", fname)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/finalize", path)
            if method == "POST" and m:
                body = self._read_json_body()
                return self._send_json(HTTPStatus.OK, service.finalize_session(m.group(1), body))

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/pages/([^/]+)/image", path)
            if method == "GET" and m:
                annotated = qs.get("annotated", ["0"])[0] in ("1", "true", "yes")
                img_bytes, ctype = service.get_page_image_bytes(m.group(1), m.group(2), annotated=annotated)
                return self._send_bytes(HTTPStatus.OK, img_bytes, ctype)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/pages/([^/]+)/windows/(\d+)/image", path)
            if method == "GET" and m:
                img_bytes, ctype = service.get_window_image_bytes(m.group(1), m.group(2), int(m.group(3)))
                return self._send_bytes(HTTPStatus.OK, img_bytes, ctype)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/pages/([^/]+)/elements/([^/]+)/crop", path)
            if method == "GET" and m:
                img_bytes, ctype = service.get_element_crop_bytes(m.group(1), m.group(2), m.group(3))
                return self._send_bytes(HTTPStatus.OK, img_bytes, ctype)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/pages/([^/]+)/windows/(\d+)/action", path)
            if method == "POST" and m:
                body = self._read_json_body()
                res = service.apply_action(m.group(1), m.group(2), int(m.group(3)), body)
                return self._send_json(HTTPStatus.OK, res)

            m = re.fullmatch(r"/api/ml/sessions/([^/]+)/pages/([^/]+)/submit-quarantine", path)
            if method == "POST" and m:
                body = self._read_json_body()
                res = service.submit_page_to_quarantine(m.group(1), m.group(2), body)
                return self._send_json(HTTPStatus.OK, res)

            if method == "GET" and path == "/api/ml/admin/quarantine":
                st = qs.get("status", ["pending"])[0]
                return self._send_json(HTTPStatus.OK, {"items": service.list_quarantine(st)})

            m = re.fullmatch(r"/api/ml/admin/quarantine/([^/]+)", path)
            if method == "GET" and m:
                return self._send_json(HTTPStatus.OK, service.get_quarantine_item(m.group(1)))

            m = re.fullmatch(r"/api/ml/admin/quarantine/([^/]+)/image", path)
            if method == "GET" and m:
                img_bytes, ctype = service.get_quarantine_image_bytes(m.group(1))
                return self._send_bytes(HTTPStatus.OK, img_bytes, ctype)

            m = re.fullmatch(r"/api/ml/admin/quarantine/([^/]+)/windows/(\d+)/image", path)
            if method == "GET" and m:
                img_bytes, ctype = service.get_quarantine_image_bytes(m.group(1), window_idx=int(m.group(2)))
                return self._send_bytes(HTTPStatus.OK, img_bytes, ctype)

            m = re.fullmatch(r"/api/ml/admin/quarantine/([^/]+)/crops/([^/]+)", path)
            if method == "GET" and m:
                img_bytes, ctype = service.get_quarantine_image_bytes(m.group(1), crop_file=m.group(2))
                return self._send_bytes(HTTPStatus.OK, img_bytes, ctype)

            m = re.fullmatch(r"/api/ml/admin/quarantine/([^/]+)/windows/(\d+)/edit", path)
            if method == "POST" and m:
                body = self._read_json_body()
                return self._send_json(HTTPStatus.OK, service.edit_quarantine_window(m.group(1), int(m.group(2)), body))

            m = re.fullmatch(r"/api/ml/admin/quarantine/([^/]+)/decision", path)
            if method == "POST" and m:
                body = self._read_json_body()
                return self._send_json(HTTPStatus.OK, service.decide_quarantine_item(m.group(1), body))

            if method == "POST" and path == "/api/ml/admin/export-approved":
                body = self._read_json_body()
                include_exported = bool(body.get("include_exported", False))
                return self._send_json(HTTPStatus.OK, service.export_approved(include_exported=include_exported))

            raise ServiceError(HTTPStatus.NOT_FOUND, "Маршрут не найден.")

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_DELETE(self) -> None:
            self._handle("DELETE")

        def _handle(self, method: str) -> None:
            try:
                self._dispatch(method)
            except ServiceError as se:
                self._send_json(se.status_code, {"error": se.message})
            except Exception as exc:
                logger.exception("Внутренняя ошибка HTTP-обработчика: %s", exc)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Внутренняя ошибка ML-сервиса."})

    return RequestHandler


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    data_dir: Optional[Path] = None,
    preload: bool = False,
) -> None:
    """Запускает локальный многопоточный HTTP-сервер ML-модуля."""
    if not _is_loopback_host(host):
        raise ValueError("Из соображений ИБ разрешён запуск review_service только на loopback-адресе (127.0.0.1 / localhost).")
    api_token = os.getenv("ML_API_TOKEN") or None
    service = ReviewService(data_dir=data_dir)
    if preload:
        service.warmup()
    handler_cls = _make_handler(service, api_token)
    server = ThreadingHTTPServer((host, port), handler_cls)
    logger.info(f"ML Review Service запущен: http://{host}:{port} (data_dir={service.data_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Остановка ML Review Service...")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный ML-сервис разбора маршрутных листов и карантина дообучения")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Запустить локальный HTTP API-сервер")
    p_serve.add_argument("--host", type=str, default=DEFAULT_HOST, help="Хост (только loopback, по умолчанию 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT, help="Порт (по умолчанию 8765)")
    p_serve.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Каталог сессий и карантина")
    p_serve.add_argument("--preload", action="store_true", help="Предзагрузить модели сразу при старте")

    p_exp = sub.add_parser("export-approved", help="Выгрузить одобренные правки из карантина")
    p_exp.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Каталог сессий и карантина")
    p_exp.add_argument("--include-exported", action="store_true", help="Повторно включить ранее выгруженные записи")

    args = parser.parse_args()
    if args.command == "serve":
        serve(host=args.host, port=args.port, data_dir=Path(args.data_dir), preload=args.preload)
    elif args.command == "export-approved":
        svc = ReviewService(data_dir=Path(args.data_dir))
        print(json.dumps(svc.export_approved(include_exported=args.include_exported), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
