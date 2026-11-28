import argparse
from dataclasses import dataclass, field
import hashlib
import itertools
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import List, Optional, Tuple
import uuid

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from inference_backend import InferenceBackend, create_inference_backend, verify_weight_integrity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Печать",
    "Плохая печать",
    "Печать экспедиции",
    "Подпись",
    "Расшифровка печать",
    "Расшифровка",
    "Рукописная цифра",
    "Печатная цифра",
    "Исправление",
    "Qr code",
    "Окно аптеки",
    "Титульный лист",
    "Номер МЛ",
    "ФИО водителя",
]


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BBox
    ocr_text: str = ""
    ocr_value: int = None
    cls_pred: str = ""
    cls_conf: float = 0.0
    source: str = "model"


@dataclass
class Window:
    bbox: BBox
    elements: List[Detection] = field(default_factory=list)
    return_split_y: Optional[float] = None


@dataclass
class PageResult:
    page_num: int
    image_path: str
    detections: List[Detection]
    windows: List[Window]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pdf_page_index: int = 0


@dataclass
class MLContext:
    ml_number: str = "Unknown"
    driver_name: str = "Unknown"


def get_cyrillic_font(size: int = 18):
    """Возвращает шрифт с поддержкой кириллицы (Windows, macOS, Linux)."""
    font_candidates = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def cv2_imread(path_str):
    """Кросс-платформенное чтение изображений с поддержкой кириллицы в путях на Windows."""
    try:
        data = np.fromfile(str(path_str), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(str(path_str))


def cv2_imwrite(path_str, img, params: list = None) -> bool:
    """Кросс-платформенная запись изображений с поддержкой кириллицы в путях на Windows."""
    try:
        ext = Path(path_str).suffix
        success, buf = cv2.imencode(ext, img, params or [])
        if success:
            with open(str(path_str), "wb") as f:
                buf.tofile(f)
            return True
    except Exception:
        pass
    return cv2.imwrite(str(path_str), img)


def calculate_overlap_ratio(obj: BBox, window: BBox) -> float:
    """Вычисляет долю площади объекта obj, попадающую внутрь бокса window."""
    obj_area = obj.area
    if obj_area == 0:
        return 0.0

    inter_x1 = max(obj.x1, window.x1)
    inter_y1 = max(obj.y1, window.y1)
    inter_x2 = min(obj.x2, window.x2)
    inter_y2 = min(obj.y2, window.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return (inter_w * inter_h) / obj_area


def calculate_true_iou(a: BBox, b: BBox) -> float:
    """Вычисляет Intersection over Union (IoU) между двумя боксами."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def text_to_number(text: str):
    """Преобразует распознанную строку (число или числительное прописью) в int."""
    if not text:
        return None

    mapping = {
        "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
        "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
        "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
        "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
        "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
        "двадцать": 20,
    }

    t_clean = re.sub(r"[^\w\s]", "", text.lower().strip())
    for word, num in mapping.items():
        if word == t_clean or word in t_clean.split():
            return num

    digits = re.sub(r"\D", "", text)
    if digits:
        return int(digits)

    return None


class SpatialResolver:
    """
    Трёхуровневый пространственно-семантический резолвер привязки элементов к окнам аптек:
    1) геометрическое зонирование (линии Вороного);
    2) венгерский алгоритм (Kuhn-Munkres) для одиночных реквизитов;
    3) арифметический оптимизатор для пограничных рукописных цифр.
    """

    def __init__(self, boundary_margin: float = 25.0):
        self.boundary_margin = boundary_margin

    def filter_overlapping(self, elements: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
        """Подавляет вложенные дубликаты одного класса, оставляя более крупные боксы."""
        if not elements:
            return []
        elements = sorted(
            elements,
            key=lambda e: (e.bbox.x2 - e.bbox.x1) * (e.bbox.y2 - e.bbox.y1),
            reverse=True,
        )
        kept = []
        for e in elements:
            overlap = False
            for k in kept:
                ix1 = max(e.bbox.x1, k.bbox.x1)
                iy1 = max(e.bbox.y1, k.bbox.y1)
                ix2 = min(e.bbox.x2, k.bbox.x2)
                iy2 = min(e.bbox.y2, k.bbox.y2)
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    e_area = (e.bbox.x2 - e.bbox.x1) * (e.bbox.y2 - e.bbox.y1)
                    if e_area > 0 and (inter_area / e_area) > iou_thresh:
                        overlap = True
                        break
            if not overlap:
                kept.append(e)
        return kept

    def resolve(self, windows: List[Window], other_detections: List[Detection]) -> List[Window]:
        """Распределяет детекции по отсортированным сверху вниз окнам аптек."""
        if not windows:
            return windows

        windows.sort(key=lambda w: w.bbox.y1)
        num_windows = len(windows)
        for win in windows:
            win.elements = []

        if num_windows == 1:
            windows[0].elements = list(other_detections)
            return windows

        split_y = [
            (windows[k].bbox.y2 + windows[k + 1].bbox.y1) / 2.0
            for k in range(num_windows - 1)
        ]

        def get_zone_idx(cy: float) -> int:
            for idx, sy in enumerate(split_y):
                if cy < sy:
                    return idx
            return num_windows - 1

        filtered_dets = []
        for label_group in set(d.label for d in other_detections):
            group_elements = [d for d in other_detections if d.label == label_group]
            filtered_dets.extend(self.filter_overlapping(group_elements, iou_thresh=0.5))

        def match_hungarian(elements: List[Detection], max_penalty: float = 2.5):
            if not elements:
                return
            cost_matrix = np.zeros((len(elements), num_windows), dtype=float)
            for i, el in enumerate(elements):
                cy = (el.bbox.y1 + el.bbox.y2) / 2.0
                for j, win in enumerate(windows):
                    wy_center = (win.bbox.y1 + win.bbox.y2) / 2.0
                    wh = max(1.0, win.bbox.y2 - win.bbox.y1)
                    if win.bbox.y1 <= cy <= win.bbox.y2:
                        cost_matrix[i, j] = abs(cy - wy_center) / wh
                    else:
                        gap = min(abs(cy - win.bbox.y1), abs(cy - win.bbox.y2))
                        if gap <= self.boundary_margin:
                            cost_matrix[i, j] = 1.0 + (gap / self.boundary_margin)
                        else:
                            cost_matrix[i, j] = 999.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_idx = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < max_penalty:
                    windows[c].elements.append(elements[r])
                    assigned_idx.add(r)

            # Избыточные элементы (например, вторая печать при исправлении) относим к зоне Вороного.
            for r, el in enumerate(elements):
                if r not in assigned_idx:
                    cy = (el.bbox.y1 + el.bbox.y2) / 2.0
                    windows[get_zone_idx(cy)].elements.append(el)

        match_hungarian([d for d in filtered_dets if d.label == "Подпись"])
        match_hungarian([d for d in filtered_dets if d.label in ("Расшифровка", "Расшифровка печать")])
        match_hungarian([d for d in filtered_dets if d.label == "Печать"])

        for d in filtered_dets:
            if d.label in ("Qr code", "Плохая печать", "Печать экспедиции", "Исправление", "Печатная цифра"):
                cy = (d.bbox.y1 + d.bbox.y2) / 2.0
                windows[get_zone_idx(cy)].elements.append(d)

        hw_digits = [d for d in filtered_dets if d.label == "Рукописная цифра"]
        boundary_hw = []

        for hd in hw_digits:
            cy = (hd.bbox.y1 + hd.bbox.y2) / 2.0
            is_boundary = False
            for k, sy in enumerate(split_y):
                if abs(cy - sy) < self.boundary_margin:
                    boundary_hw.append((hd, k))
                    is_boundary = True
                    break
            if not is_boundary:
                windows[get_zone_idx(cy)].elements.append(hd)

        for k in range(num_windows - 1):
            digits_at_k = [hd for hd, s_k in boundary_hw if s_k == k]
            if not digits_at_k:
                continue

            win_k = windows[k]
            win_next = windows[k + 1]

            p_vals_k = [e.ocr_value for e in win_k.elements if e.label == "Печатная цифра" and e.ocr_value is not None]
            p_sum_k = sum(p_vals_k) if p_vals_k else None

            p_vals_next = [e.ocr_value for e in win_next.elements if e.label == "Печатная цифра" and e.ocr_value is not None]
            p_sum_next = sum(p_vals_next) if p_vals_next else None

            h_vals_k = [e.ocr_value for e in win_k.elements if e.label == "Рукописная цифра" and e.ocr_value is not None]
            h_sum_k = sum(h_vals_k)
            h_cnt_k = len([e for e in win_k.elements if e.label == "Рукописная цифра"])

            h_vals_next = [e.ocr_value for e in win_next.elements if e.label == "Рукописная цифра" and e.ocr_value is not None]
            h_sum_next = sum(h_vals_next)
            h_cnt_next = len([e for e in win_next.elements if e.label == "Рукописная цифра"])

            # Предохранитель от комбинаторного DoS O(2^N) при большом числе пограничных цифр.
            if len(digits_at_k) > 6:
                boundary_y = (win_k.bbox.y2 + win_next.bbox.y1) / 2.0
                win_k.elements.extend(d for d in digits_at_k if ((d.bbox.y1 + d.bbox.y2) / 2.0) <= boundary_y)
                win_next.elements.extend(d for d in digits_at_k if ((d.bbox.y1 + d.bbox.y2) / 2.0) > boundary_y)
                continue

            best_penalty = float("inf")
            best_assignment = None

            for choices in itertools.product([0, 1], repeat=len(digits_at_k)):
                b_k = [d for d, c in zip(digits_at_k, choices) if c == 0]
                b_next = [d for d, c in zip(digits_at_k, choices) if c == 1]

                tot_h_sum_k = h_sum_k + sum(d.ocr_value for d in b_k if d.ocr_value is not None)
                tot_h_sum_next = h_sum_next + sum(d.ocr_value for d in b_next if d.ocr_value is not None)
                tot_cnt_k = h_cnt_k + len(b_k)
                tot_cnt_next = h_cnt_next + len(b_next)

                pen = 0.0
                if p_sum_k is not None:
                    pen += -100.0 if tot_h_sum_k == p_sum_k else abs(tot_h_sum_k - p_sum_k) * 20.0
                if p_sum_next is not None:
                    pen += -100.0 if tot_h_sum_next == p_sum_next else abs(tot_h_sum_next - p_sum_next) * 20.0

                if p_sum_k and p_sum_k > 0 and tot_cnt_k == 0:
                    pen += 200.0
                if p_sum_next and p_sum_next > 0 and tot_cnt_next == 0:
                    pen += 200.0

                for d in b_k:
                    cy = (d.bbox.y1 + d.bbox.y2) / 2.0
                    if cy > split_y[k]:
                        pen += (cy - split_y[k]) * 0.5
                for d in b_next:
                    cy = (d.bbox.y1 + d.bbox.y2) / 2.0
                    if cy < split_y[k]:
                        pen += (split_y[k] - cy) * 0.5

                if pen < best_penalty:
                    best_penalty = pen
                    best_assignment = (b_k, b_next)

            if best_assignment:
                win_k.elements.extend(best_assignment[0])
                win_next.elements.extend(best_assignment[1])

        for win in windows:
            printed = [e for e in win.elements if e.label == "Печатная цифра"]
            handwritten = [e for e in win.elements if e.label == "Рукописная цифра"]
            other = [e for e in win.elements if e.label not in ("Печатная цифра", "Рукописная цифра")]
            win.elements = other + printed + handwritten

        return windows


class Pipeline:
    """Основной конвейер детекции, распознавания реквизитов и аудита маршрутных листов."""

    @staticmethod
    def _verify_weight_integrity(model_path: str, allow_untrusted: bool = False):
        """Проверяет SHA-256 контрольную сумму весов до загрузки PyTorch (защита от Pickle RCE)."""
        return verify_weight_integrity(model_path, allow_untrusted=allow_untrusted)

    def __init__(
        self,
        yolo_model_path: str = "Новые веса/best_clean.pt",
        digit_classifier_path: str = "Новые веса/best_digit_classifier.pt",
        allow_untrusted_weights: bool = False,
        backend: Optional[InferenceBackend] = None,
    ):
        self.backend: InferenceBackend = (
            backend
            if backend is not None
            else create_inference_backend(
                yolo_model_path=yolo_model_path,
                digit_classifier_path=digit_classifier_path,
                allow_untrusted_weights=allow_untrusted_weights,
            )
        )
        self.spatial_resolver = SpatialResolver(boundary_margin=25.0)

    @property
    def weights_version(self) -> str:
        return self.backend.weights_version

    @property
    def device(self) -> str:
        return getattr(self.backend, "detect_device", "remote")

    @staticmethod
    def _normalize_ml_number(text: str) -> str:
        """Нормализует OCR-строку номера МЛ (чинит искажения префикса М->П/Н/[/... и цифр)."""
        if not text:
            return "Unknown"
        lat_to_cyr = str.maketrans({
            "M": "М", "m": "м", "C": "С", "c": "с", "K": "К", "k": "к",
            "H": "Н", "h": "н", "P": "Р", "p": "р", "A": "А", "a": "а",
            "E": "Е", "e": "е", "O": "О", "o": "о", "T": "Т", "t": "т",
            "B": "В", "X": "Х", "x": "х",
        })
        digit_fix = str.maketrans({
            "З": "3", "з": "3", "О": "0", "о": "0", "Б": "6", "б": "6",
            "Д": "4", "д": "4", "Ч": "4", "ч": "4",
            "Е": "5", "е": "5", "Э": "5", "э": "5",
            "В": "8", "в": "8", "Т": "7", "т": "7",
            "I": "1", "l": "1", "|": "1",
        })
        norm = text.translate(lat_to_cyr)

        m = re.search(r"(?:[МПНШЩ\[\{\(\s]|^)СК\s*([0-9ЗзОоБбДдЕеЧчВвТтЭэIl|]{4,10})", norm, re.IGNORECASE)
        if m:
            raw_digits = m.group(1)
            if sum(ch.isdigit() for ch in raw_digits) >= 3:
                digits = re.sub(r"\D", "", raw_digits.translate(digit_fix))
                if digits:
                    return f"МСК{digits}"

        compact = re.sub(r"[^А-Яа-я0-9]", "", norm)
        m_gen = re.search(r"([А-Яа-я]{2,4})([0-9ЗзОоБбДдЕеЧчВвТтЭэ]{5,10})", compact)
        if m_gen:
            raw_digits = m_gen.group(2)
            if sum(ch.isdigit() for ch in raw_digits) >= 3:
                prefix = m_gen.group(1).upper()
                digits = re.sub(r"\D", "", raw_digits.translate(digit_fix))
                if digits:
                    return f"{prefix}{digits}"

        clean_fallback = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text).strip()
        return clean_fallback if clean_fallback else "Unknown"

    @staticmethod
    def _crop_digit_square(img: Image.Image, bbox: BBox) -> Image.Image:
        """Вырезает квадратный кроп цифры со стороной 1.4 * max_side."""
        w, h = img.size
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        max_side = max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
        size = int(max_side * 1.4)
        x1 = max(0, int(cx - size / 2))
        y1 = max(0, int(cy - size / 2))
        x2 = min(w, int(cx + size / 2))
        y2 = min(h, int(cy + size / 2))
        return img.crop((x1, y1, x2, y2))

    def classify_digit(self, img_path: str, bbox: BBox) -> Tuple[str, str, float]:
        """Классифицирует кроп цифры; возвращает (итоговый_класс_или_пусто, сырой_класс, confidence)."""
        try:
            img = Image.open(img_path).convert("RGB")
            crop_img = self._crop_digit_square(img, bbox)
            res = self.backend.classify_crops([crop_img])[0]
            top1_conf = float(res["confidence"])
            predicted_class = str(res["class_name"])
            if predicted_class == "unknown" or top1_conf < 0.58:
                return "", predicted_class, top1_conf
            return predicted_class, predicted_class, top1_conf
        except Exception as e:
            logger.error(f"Digit CLS Error: {e}")
            return "", "unknown", 0.0

    def _collect_training_crops(
        self,
        img_path: str,
        windows: List[Window],
        other_detections: List[Detection],
        errors: List[str],
        ml_number: str,
        page_num: int,
        crops_root: Path,
    ) -> int:
        """Сохраняет сложные и редкие кропы цифр в буфер дообучения по правилам отбора."""
        candidates: List[Tuple[Detection, bool]] = []
        seen_ids = set()

        for w_idx, win in enumerate(windows):
            win_errors = [e for e in errors if e.startswith(f"Окно #{w_idx + 1}:")]
            has_digit_issue = any(
                ("Несовпадение цифр" in e or "Не удалось распознать" in e) for e in win_errors
            )
            for el in win.elements:
                if el.label in ("Печатная цифра", "Рукописная цифра"):
                    seen_ids.add(id(el))
                    candidates.append((el, has_digit_issue))

        for d in other_detections:
            if d.label in ("Печатная цифра", "Рукописная цифра") and id(d) not in seen_ids:
                candidates.append((d, False))

        to_save: List[Tuple[Detection, Path]] = []
        for d, has_digit_issue in candidates:
            is_unrecognized = d.ocr_value is None
            is_low_conf = d.cls_conf < 0.60
            is_ten_plus = d.ocr_value is not None and d.ocr_value >= 10
            if not (has_digit_issue or is_unrecognized or is_low_conf or is_ten_plus):
                continue

            if is_unrecognized or d.cls_pred == "unknown" or d.cls_conf < 0.58:
                sub_name = "unknown"
            else:
                sub_name = re.sub(r"[^\w\-]", "_", str(d.cls_pred)).strip("_") or "unknown"
            to_save.append((d, crops_root / sub_name))

        if not to_save:
            return 0

        safe_ml = re.sub(r"[^\w\-]", "_", str(ml_number)).strip("_") or "Unknown_ML"
        page_img = Image.open(img_path).convert("RGB")
        saved_count = 0
        for d, dst_dir in to_save:
            dst_dir.mkdir(parents=True, exist_ok=True)
            kind = "pc" if d.label == "Печатная цифра" else "rc"
            crop_img = self._crop_digit_square(page_img, d.bbox)
            out_path = dst_dir / f"{safe_ml}_p{page_num:03d}_{kind}_{uuid.uuid4().hex[:8]}.jpg"
            crop_img.save(out_path, quality=95)
            saved_count += 1
        return saved_count

    def run_ocr(
        self,
        img_path: str,
        bbox: BBox,
        ocr_type: str,
        padding: int = 10,
        is_digit: bool = False,
        allowlist: str = None,
        upscale_min_height: int = 0,
        pil_img: Optional[Image.Image] = None,
    ) -> str:
        """Выполняет распознавание текста или цифры в области bbox."""
        try:
            if ocr_type in ("printed_digit_cls", "handwritten_digit_cls"):
                text, _, _ = self.classify_digit(img_path, bbox)
                return text

            img = pil_img if pil_img is not None else Image.open(img_path).convert("RGB")
            w, h = img.size
            x1 = max(0, bbox.x1 - padding)
            y1 = max(0, bbox.y1 - padding)
            x2 = min(w, bbox.x2 + padding)
            y2 = min(h, bbox.y2 + padding)
            crop_cv = np.array(img.crop((x1, y1, x2, y2)))
            if upscale_min_height and crop_cv.shape[0] < upscale_min_height:
                crop_cv = cv2.resize(crop_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            if ocr_type == "easyocr":
                effective_allowlist = "0123456789" if is_digit else allowlist
                items = self.backend.ocr_crops([(crop_cv, effective_allowlist)])[0]
                return " ".join(it["text"] for it in items if it.get("text")) if items else ""

            if ocr_type == "easyocr_driver":
                items = self.backend.ocr_crops([(crop_cv, allowlist)])[0]
                if not items:
                    return "Unknown"

                h_crop = crop_cv.shape[0]
                name_words = []
                for it in items:
                    bbox_pts = it["bbox"]
                    text = it["text"]
                    conf = float(it["confidence"])
                    clean = re.sub(r"[^а-яА-ЯёЁ]", "", text)
                    is_footnote = bool(
                        re.search(
                            r"(?:рекв|пека|пены|енъиз|ениз|позвол|пазвол|идент|ндент|лицо|выдав)",
                            clean,
                            flags=re.IGNORECASE,
                        )
                    )
                    y_center = (bbox_pts[0][1] + bbox_pts[2][1]) / 2.0
                    if is_footnote or (y_center > h_crop * 0.72 and conf < 0.7):
                        continue
                    x_center = (bbox_pts[0][0] + bbox_pts[1][0]) / 2.0
                    for word in text.split():
                        w_clean = re.sub(r"[^а-яА-ЯёЁ]", "", word)
                        if len(w_clean) >= 2:
                            name_words.append((x_center, w_clean))

                if not name_words:
                    fallback_text = items[0]["text"] if items else "Unknown"
                    # Санитизация для предотвращения Path Traversal при создании каталогов.
                    clean_fallback = re.sub(r"[^\w\s\-]", "", fallback_text).strip()
                    return clean_fallback or "Unknown"

                name_words.sort(key=lambda item: item[0])
                selected = [word for _, word in name_words[:3]]
                typos = {
                    "Апекс": "Алекс",
                    "Апександрович": "Александрович",
                    "Апександр": "Александр",
                    "Донис": "Денис",
                    "Анаольевич": "Анатольевич",
                }
                selected = [typos.get(word, word) for word in selected]
                return " ".join(selected)
        except Exception as e:
            logger.error(f"OCR Error ({ocr_type}): {e}")
            return ""
        return ""

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        generate_visuals: bool = True,
        collect_crops: bool = False,
        crops_dir: str = None,
        max_pages: int = 150,
    ):
        """Обрабатывает один PDF в изолированном временном каталоге с гарантированной очисткой."""
        pdf_file = Path(pdf_path)
        out_path = Path(output_dir)
        crops_root = (
            Path(crops_dir)
            if crops_dir
            else (Path.home() / "Documents" / "ML_Dataset Выборка" / "кропы" / "Дообучение")
        )
        logger.info(f"Обработка {pdf_file.name}")
        with tempfile.TemporaryDirectory(prefix=f"{pdf_file.stem}_") as tmp_dir:
            return self._process_pdf_internal(
                pdf_file,
                out_path,
                Path(tmp_dir),
                generate_visuals=generate_visuals,
                collect_crops=collect_crops,
                crops_root=crops_root,
                max_pages=max_pages,
            )

    def analyze_pdf_segments(
        self,
        pdf_path: str,
        work_dir: str,
        max_pages: int = 3500,
        progress_callback=None,
        image_ext: str = ".jpg",
    ) -> List[Tuple[MLContext, List[PageResult]]]:
        """Анализирует PDF и сохраняет постраничные изображения в work_dir для интерактивного разбора."""
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        return self._run_pdf_segments(
            Path(pdf_path),
            work_path,
            collect_crops=False,
            max_pages=max_pages,
            progress_callback=progress_callback,
            image_ext=image_ext,
        )

    def _process_pdf_internal(
        self,
        pdf_path: Path,
        output_dir: Path,
        temp_dir: Path,
        generate_visuals: bool = True,
        collect_crops: bool = False,
        crops_root: Path = None,
        max_pages: int = 150,
    ):
        segments = self._run_pdf_segments(
            pdf_path,
            temp_dir,
            collect_crops=collect_crops,
            crops_root=crops_root,
            max_pages=max_pages,
            image_ext=".png",
        )
        all_page_results = [pr for _, seg_pages in segments for pr in seg_pages]
        if generate_visuals:
            for seg_ctx, seg_pages in segments:
                self._generate_outputs(pdf_path.name, seg_pages, seg_ctx, output_dir)

        return all_page_results

    def _run_pdf_segments(
        self,
        pdf_path: Path,
        temp_dir: Path,
        collect_crops: bool = False,
        crops_root: Path = None,
        max_pages: int = 150,
        progress_callback=None,
        image_ext: str = ".png",
    ) -> List[Tuple[MLContext, List[PageResult]]]:
        ext_clean = ".jpg" if image_ext.lower() in (".jpg", ".jpeg") else ".png"
        chunk_size = 16

        segments: List[Tuple[MLContext, List[PageResult]]] = []
        ml_context = MLContext()
        page_results: List[PageResult] = []
        has_seen_windows = False
        table_page_num = 0
        t_render = 0.0
        t_yolo = 0.0
        t_digits = 0.0
        t_logic = 0.0

        with fitz.open(pdf_path) as doc:
            max_allowed_pages = max_pages
            if len(doc) > max_allowed_pages:
                logger.warning(
                    f"Документ {pdf_path.name} содержит {len(doc)} страниц (лимит: {max_allowed_pages}). "
                    f"Обрабатываются первые {max_allowed_pages} страниц."
                )
            total_to_render = min(len(doc), max_allowed_pages)
            logger.info(f"Потоковая обработка {total_to_render} страниц (чанками по {chunk_size})...")

            for chunk_start in range(0, total_to_render, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_to_render)
                chunk_paths: List[Path] = []
                chunk_grays: List[np.ndarray] = []
                chunk_bgr_imgs: List[np.ndarray] = []
                chunk_pil_imgs: List[Image.Image] = []

                t0 = time.perf_counter()
                for i in range(chunk_start, chunk_end):
                    page = doc.load_page(i)
                    # Защита от Pixel Flood: при аномальных габаритах листа понижаем DPI.
                    render_dpi = 72 if (page.rect.width > 2500 or page.rect.height > 2500) else 200
                    pix = page.get_pixmap(dpi=render_dpi, colorspace=fitz.csGRAY)
                    page_gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
                    bgr_img = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)

                    img_path = temp_dir / f"page_{i:04d}{ext_clean}"
                    if ext_clean == ".jpg":
                        cv2_imwrite(img_path, page_gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    else:
                        cv2_imwrite(img_path, bgr_img)

                    chunk_paths.append(img_path)
                    chunk_grays.append(page_gray)
                    chunk_bgr_imgs.append(bgr_img)
                    chunk_pil_imgs.append(Image.fromarray(page_gray).convert("RGB"))
                    if progress_callback is not None:
                        progress_callback(i + 1, total_to_render, "render")
                t_render += time.perf_counter() - t0

                t0 = time.perf_counter()
                chunk_results = self.backend.detect_pages(
                    chunk_pil_imgs,
                    imgsz=1024,
                    conf=0.12,
                    iou=0.45,
                )
                t_yolo += time.perf_counter() - t0
                if progress_callback is not None:
                    progress_callback(chunk_end, total_to_render, "detect")

                chunk_prepared = []
                chunk_digit_items = []

                for offset, page_dets in enumerate(chunk_results):
                    global_i = chunk_start + offset
                    img_path = str(chunk_paths[offset])
                    page_gray = chunk_grays[offset]
                    bgr_img = chunk_bgr_imgs[offset]
                    page_pil = chunk_pil_imgs[offset]
                    gh, gw = page_gray.shape
                    detections = []

                    for det in page_dets:
                        x1, y1, x2, y2 = det["bbox"]
                        cls_id = int(det["class_id"])
                        conf = float(det["confidence"])
                        label = det.get("class_name") or CLASS_NAMES[cls_id]

                        min_conf = 0.20
                        if label == "Печать":
                            min_conf = 0.40
                        elif label == "Расшифровка печать":
                            min_conf = 0.12
                        elif label == "Подпись":
                            min_conf = 0.13
                        elif label in ("Окно аптеки", "Qr code"):
                            min_conf = 0.25

                        if conf < min_conf:
                            continue

                        # Отсекаем фантомы "Печать" на пустой клетке, сохраняя уверенные бледные оттиски (conf >= 0.85).
                        if label == "Печать" and conf < 0.85:
                            rx1 = max(0, int(x1))
                            ry1 = max(0, int(y1))
                            rx2 = min(gw, int(x2))
                            ry2 = min(gh, int(y2))
                            roi = page_gray[ry1:ry2, rx1:rx2]
                            if roi.size > 0 and float(np.mean(roi < 240)) < 0.085:
                                continue

                        detections.append(Detection(label, conf, BBox(x1, y1, x2, y2)))

                    has_title = any(d.label == "Титульный лист" for d in detections)
                    ml_number_det = next((d for d in detections if d.label == "Номер МЛ"), None) if has_title else None
                    driver_det = next((d for d in detections if d.label == "ФИО водителя"), None) if has_title else None

                    page_ml_number = None
                    page_driver_name = None
                    page_pil = None
                    if has_title and (ml_number_det or driver_det):
                        page_pil = Image.fromarray(page_gray).convert("RGB")
                        if ml_number_det:
                            text = self.run_ocr(
                                img_path,
                                ml_number_det.bbox,
                                "easyocr",
                                padding=10,
                                upscale_min_height=60,
                                pil_img=page_pil,
                            )
                            normalized_ml = self._normalize_ml_number(text)
                            ml_number_det.ocr_text = normalized_ml
                            page_ml_number = normalized_ml

                        if driver_det:
                            fio_allowlist = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя "
                            driver_name = self.run_ocr(
                                img_path,
                                driver_det.bbox,
                                "easyocr_driver",
                                padding=10,
                                allowlist=fio_allowlist,
                                pil_img=page_pil,
                            )
                            driver_det.ocr_text = driver_name
                            page_driver_name = driver_name if driver_name else "Unknown"

                    windows = []
                    window_detections = [d for d in detections if d.label == "Окно аптеки"]
                    other_detections = [d for d in detections if d.label != "Окно аптеки"]

                    for wd in window_detections:
                        windows.append(Window(bbox=wd.bbox))
                    windows.sort(key=lambda w: w.bbox.y1)

                    # Отсекаем рукописные пометки за правой границей таблицы окна.
                    cleaned_other = []
                    for d in other_detections:
                        if d.label == "Рукописная цифра" and windows:
                            cy = (d.bbox.y1 + d.bbox.y2) / 2.0
                            matched_win = min(windows, key=lambda w: abs(cy - (w.bbox.y1 + w.bbox.y2) / 2.0))
                            if (d.bbox.x1 + d.bbox.x2) / 2.0 > matched_win.bbox.x2 + 10.0:
                                continue
                        cleaned_other.append(d)
                    other_detections = cleaned_other

                    # Cross-class NMS для совпадающих рамок Подпись/Расшифровка и Рукописная цифра.
                    sig_dec_boxes = [d for d in other_detections if d.label in ("Подпись", "Расшифровка")]
                    hw_candidates = [d for d in other_detections if d.label == "Рукописная цифра"]
                    non_hw = [d for d in other_detections if d.label != "Рукописная цифра"]

                    valid_hw = []
                    for hd in hw_candidates:
                        is_cross_duplicate = False
                        hd_w = hd.bbox.x2 - hd.bbox.x1
                        for s in sig_dec_boxes:
                            if calculate_true_iou(hd.bbox, s.bbox) > 0.75:
                                if hd_w > 120.0 or s.confidence >= hd.confidence:
                                    is_cross_duplicate = True
                                    break
                                if s in non_hw:
                                    non_hw.remove(s)
                        if not is_cross_duplicate:
                            valid_hw.append(hd)

                    # Горизонтальное слияние разорванных двузначных рукописных цифр; одиночные отсекаем ниже 0.30.
                    merged_hw = []
                    valid_hw.sort(key=lambda d: d.bbox.x1)
                    used_indices = set()
                    for idx1 in range(len(valid_hw)):
                        if idx1 in used_indices:
                            continue
                        d1 = valid_hw[idx1]
                        cy1 = (d1.bbox.y1 + d1.bbox.y2) / 2.0
                        merged = False
                        for idx2 in range(idx1 + 1, len(valid_hw)):
                            if idx2 in used_indices:
                                continue
                            d2 = valid_hw[idx2]
                            cy2 = (d2.bbox.y1 + d2.bbox.y2) / 2.0
                            x_gap = d2.bbox.x1 - d1.bbox.x2
                            comb_w = max(d1.bbox.x2, d2.bbox.x2) - min(d1.bbox.x1, d2.bbox.x1)
                            if -10.0 <= x_gap <= 25.0 and abs(cy1 - cy2) <= 25.0 and comb_w <= 125.0:
                                merged_bbox = BBox(
                                    min(d1.bbox.x1, d2.bbox.x1),
                                    min(d1.bbox.y1, d2.bbox.y1),
                                    max(d1.bbox.x2, d2.bbox.x2),
                                    max(d1.bbox.y2, d2.bbox.y2),
                                )
                                merged_hw.append(Detection("Рукописная цифра", max(d1.confidence, d2.confidence), merged_bbox))
                                used_indices.add(idx1)
                                used_indices.add(idx2)
                                merged = True
                                break
                        if not merged:
                            if d1.confidence >= 0.30:
                                merged_hw.append(d1)
                            used_indices.add(idx1)

                    other_detections = non_hw + merged_hw
                    digit_dets = [d for d in other_detections if d.label in ("Печатная цифра", "Рукописная цифра")]
                    if digit_dets:
                        if page_pil is None:
                            page_pil = Image.fromarray(page_gray).convert("RGB")
                        for d in digit_dets:
                            chunk_digit_items.append((d, self._crop_digit_square(page_pil, d.bbox)))

                    has_title_meta = bool(ml_number_det is not None or driver_det is not None)
                    chunk_prepared.append((
                        global_i, img_path, bgr_img, detections, windows, other_detections,
                        has_title, has_title_meta, page_ml_number, page_driver_name,
                    ))

                if chunk_digit_items:
                    t0 = time.perf_counter()
                    cls_results = self.backend.classify_crops([crop for _, crop in chunk_digit_items])
                    for (d, _), cls_res in zip(chunk_digit_items, cls_results):
                        top1_conf = float(cls_res["confidence"])
                        predicted_class = str(cls_res["class_name"])
                        if predicted_class == "unknown" or top1_conf < 0.58:
                            d.ocr_text = ""
                        else:
                            d.ocr_text = predicted_class
                        d.cls_pred = predicted_class
                        d.cls_conf = top1_conf
                        d.ocr_value = text_to_number(d.ocr_text)
                    t_digits += time.perf_counter() - t0

                t0 = time.perf_counter()
                for (
                    global_i, img_path, bgr_img, detections, windows, other_detections,
                    has_title, has_title_meta, page_ml_number, page_driver_name,
                ) in chunk_prepared:
                    # Новый МЛ начинается с титульника с реквизитами либо после страниц с окнами аптек.
                    is_new_ml = has_title and (has_title_meta or has_seen_windows)
                    if is_new_ml and page_results:
                        segments.append((ml_context, page_results))
                        ml_context = MLContext()
                        page_results = []
                        has_seen_windows = False
                        table_page_num = 0

                    if has_title:
                        if page_ml_number is not None:
                            ml_context.ml_number = page_ml_number
                        if page_driver_name is not None:
                            ml_context.driver_name = page_driver_name
                    if windows:
                        has_seen_windows = True

                    windows = self.spatial_resolver.resolve(windows, other_detections)
                    errors, warnings = self._apply_business_logic(windows, img_path=img_path, page_cv=bgr_img)

                    if windows:
                        table_page_num += 1
                        local_page_num = table_page_num
                    else:
                        local_page_num = 0

                    if collect_crops and crops_root is not None:
                        self._collect_training_crops(
                            img_path,
                            windows,
                            other_detections,
                            errors,
                            ml_context.ml_number,
                            local_page_num,
                            crops_root,
                        )

                    page_results.append(PageResult(
                        page_num=local_page_num,
                        image_path=img_path,
                        detections=detections,
                        windows=windows,
                        errors=errors,
                        warnings=warnings,
                        pdf_page_index=global_i,
                    ))
                    if progress_callback is not None:
                        progress_callback(global_i + 1, total_to_render, "analyze")
                t_logic += time.perf_counter() - t0

                del chunk_bgr_imgs, chunk_grays, chunk_results, chunk_digit_items, chunk_prepared

        if page_results:
            segments.append((ml_context, page_results))

        logger.info(
            f"Итого время ({total_to_render} стр.): "
            f"render={t_render:.1f}с, yolo={t_yolo:.1f}с, digits={t_digits:.1f}с, logic={t_logic:.1f}с"
        )
        return segments

    def _apply_business_logic(
        self,
        windows: List[Window],
        img_path: str = None,
        detect_return_splits: bool = True,
        page_cv: np.ndarray = None,
    ) -> Tuple[List[str], List[str]]:
        """Проверяет окна аптек и секции возврата «Получение» на соответствие регламенту."""
        errors = []
        warnings = []

        for w_idx, win in enumerate(windows):
            return_split_y = win.return_split_y
            win_w = max(1.0, win.bbox.x2 - win.bbox.x1)
            win_h = max(1.0, win.bbox.y2 - win.bbox.y1)
            is_return_candidate = (win_h / win_w) >= 0.40
            if is_return_candidate:
                qr_det = next((e for e in win.elements if e.label == "Qr code"), None)
                if qr_det is not None:
                    qr_cy = (qr_det.bbox.y1 + qr_det.bbox.y2) / 2.0
                    if ((qr_cy - win.bbox.y1) / win_h) > 0.18:
                        is_return_candidate = False

            if return_split_y is None and detect_return_splits and is_return_candidate:
                if page_cv is None and img_path and Path(img_path).exists():
                    try:
                        page_cv = cv2_imread(img_path)
                    except Exception:
                        page_cv = None
                if page_cv is not None:
                    try:
                        gh, gw = page_cv.shape[:2]
                        wx1, wy1 = max(0, int(win.bbox.x1)), max(0, int(win.bbox.y1))
                        wx2, wy2 = min(gw, int(win.bbox.x2)), min(gh, int(win.bbox.y2))
                        cw, ch = max(10, wx2 - wx1), max(10, wy2 - wy1)
                        sub_x1 = wx1
                        sub_x2 = min(gw, wx1 + int(cw * 0.32))
                        sub_y1 = min(gh, wy1 + int(ch * 0.33))
                        sub_y2 = min(gh, wy1 + int(ch * 0.86))
                        left_crop = page_cv[sub_y1:sub_y2, sub_x1:sub_x2]
                        if left_crop.size > 0 and float(np.mean(left_crop < 200)) >= 0.005:
                            scale = 0.5 if (left_crop.shape[0] >= 120 and left_crop.shape[1] >= 120) else 1.0
                            ocr_crop = (
                                cv2.resize(left_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                                if scale < 1.0
                                else left_crop
                            )
                            for item in self.backend.ocr_crops([(ocr_crop, None)])[0]:
                                bbox = item["bbox"]
                                text = item["text"]
                                conf = float(item["confidence"])
                                if re.search(r"п[оа0][лпда][уy][ч4]", text.lower()) and conf > 0.35:
                                    word_top_y = min(pt[1] for pt in bbox) / scale
                                    return_split_y = float(sub_y1) + word_top_y
                                    win.return_split_y = return_split_y
                                    break
                    except Exception as ex:
                        logger.debug(f"Ошибка проверки блока Получение: {ex}")

            if return_split_y is not None:
                delivery_elements = [
                    e for e in win.elements
                    if ((e.bbox.y1 + e.bbox.y2) / 2.0) < return_split_y
                    or (
                        e.label in ("Подпись", "Расшифровка", "Расшифровка печать")
                        and e.bbox.y1 < return_split_y - 25.0
                        and ((e.bbox.y1 + e.bbox.y2) / 2.0) < return_split_y + 40.0
                    )
                ]
                return_elements = [e for e in win.elements if (e.bbox.y1 + e.bbox.y2) / 2.0 >= return_split_y]

                ret_stamps = [e for e in return_elements if e.label in ("Печать", "Плохая печать")]
                ret_bad = [e for e in ret_stamps if e.label == "Плохая печать"]
                ret_good = [
                    e for e in ret_stamps
                    if e.label == "Печать" and not any(calculate_overlap_ratio(e.bbox, b.bbox) > 0.5 for b in ret_bad)
                ]

                if len(ret_stamps) == 0:
                    errors.append(f"Окно #{w_idx + 1}: Отсутствует печать возврата")
                elif len(ret_bad) > 0 and len(ret_good) == 0:
                    errors.append(f"Окно #{w_idx + 1}: Плохая печать возврата (отсутствует хорошая печать возврата)")

                win_elements_to_check = delivery_elements
            else:
                win_elements_to_check = win.elements

            labels = [e.label for e in win_elements_to_check]
            if "Печать экспедиции" in labels:
                continue
            if "Qr code" not in labels:
                continue

            has_correction = "Исправление" in labels
            has_bad_stamp = "Плохая печать" in labels
            bad_stamps = [e for e in win_elements_to_check if e.label == "Плохая печать"]
            distinct_good_stamps = [
                e for e in win_elements_to_check
                if e.label == "Печать" and not any(calculate_overlap_ratio(e.bbox, b.bbox) > 0.5 for b in bad_stamps)
            ]
            num_good_stamps = len(distinct_good_stamps)

            signatures = [e for e in win_elements_to_check if e.label == "Подпись"]
            decodings = [e for e in win_elements_to_check if e.label in ("Расшифровка", "Расшифровка печать")]

            if len(decodings) >= 2 and len(signatures) == 0:
                signatures.append(decodings.pop())
            if len(signatures) >= 2 and len(decodings) == 0:
                decodings.append(signatures.pop())

            if len(signatures) == 0:
                errors.append(f"Окно #{w_idx + 1}: Отсутствует подпись")
            if len(decodings) == 0:
                errors.append(f"Окно #{w_idx + 1}: Отсутствует расшифровка")

            printed = [e for e in win_elements_to_check if e.label == "Печатная цифра"]
            handwritten = [e for e in win_elements_to_check if e.label == "Рукописная цифра"]

            if "Печатная цифра" not in labels:
                errors.append(f"Окно #{w_idx + 1}: Отсутствует печатное кол-во мест")
            if "Рукописная цифра" not in labels and not has_correction:
                errors.append(f"Окно #{w_idx + 1}: Отсутствует рукописное кол-во мест")

            if printed and handwritten:
                _PRIORITY_SOURCES = ("operator", "admin")

                def _is_priority(det: Detection) -> bool:
                    return getattr(det, "source", "model") in _PRIORITY_SOURCES

                def filter_overlapping(elements: List[Detection]) -> List[Detection]:
                    if len(elements) > 1:
                        has_compact = any(
                            (e.bbox.x2 - e.bbox.x1) < 130.0 and (e.bbox.y2 - e.bbox.y1) < 130.0
                            for e in elements
                        )
                        if has_compact:
                            elements = [
                                e for e in elements
                                if _is_priority(e) or not ((e.bbox.x2 - e.bbox.x1) > 170.0 and (e.bbox.y2 - e.bbox.y1) > 140.0)
                            ]
                    elements = sorted(
                        elements,
                        key=lambda e: (1 if _is_priority(e) else 0, (e.bbox.x2 - e.bbox.x1) * (e.bbox.y2 - e.bbox.y1)),
                        reverse=True,
                    )
                    kept: List[Detection] = []
                    for e in elements:
                        if _is_priority(e):
                            kept.append(e)
                            continue
                        overlap = False
                        e_area = max(1.0, (e.bbox.x2 - e.bbox.x1) * (e.bbox.y2 - e.bbox.y1))
                        for k in kept:
                            ix1 = max(e.bbox.x1, k.bbox.x1)
                            iy1 = max(e.bbox.y1, k.bbox.y1)
                            ix2 = min(e.bbox.x2, k.bbox.x2)
                            iy2 = min(e.bbox.y2, k.bbox.y2)
                            if ix1 < ix2 and iy1 < iy2:
                                inter_area = (ix2 - ix1) * (iy2 - iy1)
                                if _is_priority(k):
                                    k_area = max(1.0, (k.bbox.x2 - k.bbox.x1) * (k.bbox.y2 - k.bbox.y1))
                                    if inter_area / min(e_area, k_area) > 0.40:
                                        overlap = True
                                        break
                                else:
                                    if inter_area / e_area > 0.40:
                                        overlap = True
                                        break
                        if not overlap:
                            kept.append(e)
                    return sorted(kept, key=lambda e: e.bbox.x1)

                printed = filter_overlapping(printed)
                handwritten = filter_overlapping(handwritten)

                # Схлопываем дублирующие записи формата "число + пропись/повтор" на одной строке.
                if len(handwritten) >= 2:
                    corrections = [e for e in win_elements_to_check if e.label == "Исправление"]
                    deduped_hw = []
                    hw_sorted = sorted(handwritten, key=lambda e: e.bbox.x1)
                    skip_idx = set()
                    for i_h in range(len(hw_sorted)):
                        if i_h in skip_idx:
                            continue
                        cur = hw_sorted[i_h]
                        deduped_hw.append(cur)
                        cur_cy = (cur.bbox.y1 + cur.bbox.y2) / 2.0
                        cur_cx = (cur.bbox.x1 + cur.bbox.x2) / 2.0
                        for j_h in range(i_h + 1, len(hw_sorted)):
                            if j_h in skip_idx:
                                continue
                            nxt = hw_sorted[j_h]
                            if _is_priority(cur) or _is_priority(nxt):
                                continue
                            nxt_cy = (nxt.bbox.y1 + nxt.bbox.y2) / 2.0
                            nxt_cx = (nxt.bbox.x1 + nxt.bbox.x2) / 2.0
                            has_correction_between = any(
                                cur_cx <= (c.bbox.x1 + c.bbox.x2) / 2.0 <= nxt_cx
                                and abs(((c.bbox.y1 + c.bbox.y2) / 2.0) - ((cur_cy + nxt_cy) / 2.0)) <= 80.0
                                for c in corrections
                            )
                            if has_correction_between:
                                continue
                            x_gap = nxt.bbox.x1 - cur.bbox.x2
                            if (
                                cur.ocr_value is not None
                                and cur.ocr_value == nxt.ocr_value
                                and -15.0 <= x_gap <= 100.0
                                and abs(cur_cy - nxt_cy) <= 40.0
                            ):
                                skip_idx.add(j_h)
                    handwritten = deduped_hw

                valid_hw = [e for e in handwritten if e.ocr_value is not None]
                if valid_hw:
                    handwritten = valid_hw

                p_values = [e.ocr_value for e in printed if e.ocr_value is not None]
                p_sum = sum(p_values) if p_values else None
                h_values = [e.ocr_value for e in handwritten if e.ocr_value is not None]
                h_sum = sum(h_values) if h_values else None

                if p_sum is not None and h_sum is not None:
                    if p_sum != h_sum:
                        errors.append(f"Окно #{w_idx + 1}: Несовпадение цифр (печатная: {p_sum}, рукописная: {h_sum})")
                else:
                    if p_sum is None:
                        errors.append(f"Окно #{w_idx + 1}: Не удалось распознать ПЕЧАТНУЮ цифру")
                    if h_sum is None:
                        errors.append(f"Окно #{w_idx + 1}: Не удалось распознать РУКОПИСНУЮ цифру")

            if has_correction:
                if num_good_stamps < 2:
                    errors.append(f"Окно #{w_idx + 1}: Исправление (отсутствует вторая заверяющая печать)")
            elif len(handwritten) >= 2:
                if num_good_stamps < 2:
                    errors.append(f"Окно #{w_idx + 1}: Исправление: две рукописные цифры (отсутствует вторая печать)")
            elif has_bad_stamp:
                if num_good_stamps < 1:
                    errors.append(f"Окно #{w_idx + 1}: Плохая печать (отсутствует хорошая печать)")
            else:
                if num_good_stamps < 1:
                    errors.append(f"Окно #{w_idx + 1}: Отсутствует печать")

        return errors, warnings

    def _generate_outputs(self, pdf_name: str, page_results: List[PageResult], ctx: MLContext, output_dir: Path):
        """Сохраняет визуализированный PDF-архив, PNG проблемных страниц и текстовый отчёт."""
        errors_dir = output_dir / "Ошибки"
        errors_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = output_dir / "Архив"
        archive_dir.mkdir(parents=True, exist_ok=True)

        has_errors = False
        report_lines = []
        safe_ml = re.sub(r"[^\w\-]", "_", str(ctx.ml_number)).strip("_") or "Unknown_ML"
        safe_driver = re.sub(r"[^\w\-]", "_", str(ctx.driver_name)).strip("_") or "Unknown_Driver"
        safe_pdf_name = re.sub(r"[^\w\.\-]", "_", Path(pdf_name).name).strip("_")
        base_folder_name = f"{safe_ml}_{safe_driver}"
        folder_name = base_folder_name
        suffix_idx = 2
        while (archive_dir / f"{folder_name}_{safe_pdf_name}").exists():
            folder_name = f"{base_folder_name}_{suffix_idx}"
            suffix_idx += 1

        archive_images = []
        font_title = get_cyrillic_font(20)
        font_label = get_cyrillic_font(16)

        def draw_smart_label(draw, text, target_bbox, color_rgb, font, drawn_labels, win_bbox=None, img_w=None):
            if hasattr(target_bbox, "x1"):
                bx1, by1, bx2, by2 = target_bbox.x1, target_bbox.y1, target_bbox.x2, target_bbox.y2
            elif isinstance(target_bbox, (tuple, list)) and len(target_bbox) == 4:
                bx1, by1, bx2, by2 = target_bbox
            else:
                bx1, by1 = target_bbox[0], target_bbox[1]
                bx2, by2 = bx1 + 20, by1 + 20

            _, wy1, _, wy2 = (
                (win_bbox.x1, win_bbox.y1, win_bbox.x2, win_bbox.y2)
                if hasattr(win_bbox, "x1")
                else (win_bbox if win_bbox else (0, 0, img_w or 2000, 3000))
            )

            max_w = img_w or 2000
            text_box = font.getbbox(text)
            tw = text_box[2] - text_box[0]
            th = text_box[3] - text_box[1]
            pad_x, pad_y = 3, 2
            lw, lh = tw + pad_x * 2, th + pad_y * 2
            fits_inside = by2 - by1 >= lh + 8 and bx2 - bx1 >= lw + 8

            candidates = [
                (bx1, by1 - lh - 2, -60),
                (bx1, by2 + 2, -40),
                (bx1 + 2, by1 + 2, -20 if fits_inside else 5000),
                (bx1 + 2, by2 - lh - 2, -10 if fits_inside else 5000),
                (bx2 - lw, by1 - lh - 2, 0),
                (bx2 - lw, by2 + 2, 0),
                (bx2 + 3, by1, 10),
                (bx1 - lw - 3, by1, 10),
                (bx1, by1 - lh * 2 - 4, 30),
                (bx1, by2 + lh + 4, 30),
            ]

            el_cx = (bx1 + bx2) / 2
            el_cy = (by1 + by2) / 2
            best_score = float("inf")
            best_rect = None

            for cand_x, cand_y, pref_bonus in candidates:
                rx1 = max(5, min(max_w - lw - 5, int(cand_x)))
                ry1 = int(cand_y)
                rx2 = rx1 + lw
                ry2 = ry1 + lh

                out_penalty = 0
                if win_bbox:
                    if ry1 < wy1 - 18:
                        out_penalty += (wy1 - 18 - ry1) * 300
                    if ry2 > wy2 + 18:
                        out_penalty += (ry2 - (wy2 + 18)) * 300

                c_cx = (rx1 + rx2) / 2
                c_cy = (ry1 + ry2) / 2
                dist = ((c_cx - el_cx) ** 2 + (c_cy - el_cy) ** 2) ** 0.5

                collision_count = 0
                overlap_area = 0
                for dx1, dy1, dx2, dy2 in drawn_labels:
                    ox1 = max(rx1, dx1)
                    oy1 = max(ry1, dy1)
                    ox2 = min(rx2, dx2)
                    oy2 = min(ry2, dy2)
                    if ox1 < ox2 and oy1 < oy2:
                        collision_count += 1
                        overlap_area += (ox2 - ox1) * (oy2 - oy1)

                score = (collision_count * 20000) + (overlap_area * 50) + out_penalty + (dist * 0.4) + pref_bonus
                if score < best_score:
                    best_score = score
                    best_rect = (rx1, ry1, rx2, ry2)

            rx1, ry1, rx2, ry2 = best_rect
            draw.rectangle((rx1, ry1, rx2, ry2), fill=color_rgb)
            r, g, b = color_rgb
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            text_color = (0, 0, 0) if luminance > 0.6 else (255, 255, 255)
            draw.text((rx1 + pad_x, ry1 + pad_y), text, font=font, fill=text_color)
            drawn_labels.append((rx1, ry1, rx2, ry2))

        window_elem_labels = {
            "Окно аптеки", "Печать", "Плохая печать", "Печать экспедиции",
            "Подпись", "Расшифровка печать", "Расшифровка", "Рукописная цифра",
            "Печатная цифра", "Исправление", "Qr code",
        }
        label_map = {"Рукописная цифра": "рц", "Печатная цифра": "пц"}

        for pr in page_results:
            base_img = Image.open(pr.image_path).convert("RGB")
            archive_img = base_img.copy()
            draw_archive = ImageDraw.Draw(archive_img)
            error_img = base_img.copy()
            draw_error = ImageDraw.Draw(error_img)

            page_has_errors = bool(pr.errors or pr.warnings)
            if page_has_errors:
                has_errors = True
                save_dir = errors_dir / folder_name
                save_dir.mkdir(parents=True, exist_ok=True)
                report_lines.append(f"\nМЛ: {safe_ml} | Водитель: {safe_driver} | Файл: {safe_pdf_name}")

            drawn_labels = []
            for w_idx, win in enumerate(pr.windows):
                window_errors = [e for e in pr.errors if e.startswith(f"Окно #{w_idx + 1}:")]
                window_warnings = [w for w in pr.warnings if w.startswith(f"Окно #{w_idx + 1}:")]
                has_win_error = bool(window_errors or window_warnings)
                win_color = (200, 0, 0) if has_win_error else (50, 205, 50)

                bad_labels = set()
                if has_win_error:
                    err_str = " ".join(window_errors + window_warnings).lower()
                    if "несовпадение цифр" in err_str or "не удалось распознать" in err_str:
                        bad_labels.update(["Печатная цифра", "Рукописная цифра"])
                    if "плохая печать" in err_str or "отсутствует хорошая печать" in err_str:
                        bad_labels.add("Плохая печать")
                    if "исправление" in err_str:
                        bad_labels.add("Исправление")
                    if "подпись" in err_str:
                        bad_labels.add("Подпись")
                    if "расшифровка" in err_str:
                        bad_labels.update(["Расшифровка", "Расшифровка печать"])

                draw_archive.rectangle((win.bbox.x1, win.bbox.y1, win.bbox.x2, win.bbox.y2), outline=win_color, width=2)
                for el in win.elements:
                    display_label = label_map.get(el.label, el.label)
                    label_text = f"{display_label}"
                    if el.label in ("Рукописная цифра", "Печатная цифра"):
                        label_text += f" ({el.ocr_value if el.ocr_value is not None else 'Не распознано'})"
                    elif el.ocr_text:
                        label_text += f" ({el.ocr_text})"
                    el_color = (200, 0, 0) if el.label in bad_labels else (50, 205, 50)
                    draw_archive.rectangle((el.bbox.x1, el.bbox.y1, el.bbox.x2, el.bbox.y2), outline=el_color, width=2)
                    draw_smart_label(
                        draw_archive,
                        label_text,
                        el.bbox,
                        el_color,
                        font_label,
                        drawn_labels,
                        win_bbox=win.bbox,
                        img_w=archive_img.width,
                    )

                if has_win_error:
                    draw_error.rectangle((win.bbox.x1, win.bbox.y1, win.bbox.x2, win.bbox.y2), outline=(220, 0, 0), width=3)
                    win_title = " | ".join(e.split(": ", 1)[-1] for e in (window_errors + window_warnings))
                    title_box = font_title.getbbox(win_title)
                    tw, th = title_box[2] - title_box[0], title_box[3] - title_box[1]
                    x_start = int(win.bbox.x1)
                    y_offset = int(win.bbox.y1) - 15
                    if y_offset - th < 0:
                        y_offset = int(win.bbox.y1) + th + 15
                    draw_error.rectangle(
                        (x_start, y_offset - th - 5, x_start + tw + 8, y_offset + 5),
                        fill=(255, 255, 255),
                        outline=(220, 0, 0),
                        width=1,
                    )
                    draw_error.text((x_start + 4, y_offset - th - 2), win_title, font=font_title, fill=(220, 0, 0))
                    for err in window_errors + window_warnings:
                        report_lines.append(f"  Стр. {pr.page_num:03d} | {err}")

            for d in pr.detections:
                if d.label not in window_elem_labels:
                    color = (0, 100, 255)
                    label_text = f"{d.label}"
                    if d.ocr_text:
                        label_text += f" [{d.ocr_text}]"
                    draw_archive.rectangle((d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2), outline=color, width=2)
                    draw_smart_label(
                        draw_archive,
                        label_text,
                        d.bbox,
                        color,
                        font_label,
                        drawn_labels,
                        win_bbox=None,
                        img_w=archive_img.width,
                    )

            for w_idx, win in enumerate(pr.windows):
                window_errors = [e for e in pr.errors if e.startswith(f"Окно #{w_idx + 1}:")]
                window_warnings = [w for w in pr.warnings if w.startswith(f"Окно #{w_idx + 1}:")]
                if window_errors or window_warnings:
                    win_title = " | ".join(e.split(": ", 1)[-1] for e in (window_errors + window_warnings))
                    bbox_t = font_title.getbbox(win_title)
                    tw = bbox_t[2] - bbox_t[0]
                    th = bbox_t[3] - bbox_t[1]
                    pad_x, pad_y = 6, 4
                    rx1 = max(0, int(win.bbox.x1))
                    ry1 = max(0, int(win.bbox.y1))
                    rx2 = min(archive_img.width, rx1 + tw + pad_x * 2)
                    ry2 = ry1 + th + pad_y * 2
                    draw_archive.rectangle((rx1, ry1, rx2, ry2), fill=(200, 0, 0))
                    draw_archive.text((rx1 + pad_x, ry1 + pad_y), win_title, font=font_title, fill=(255, 255, 255))

            archive_images.append(archive_img)
            if page_has_errors:
                out_img = save_dir / f"{pdf_name}_page_{pr.page_num:03d}.png"
                error_img.save(str(out_img))

        if archive_images:
            archive_pdf_path = archive_dir / f"{folder_name}_{safe_pdf_name}"
            archive_images[0].save(archive_pdf_path, save_all=True, append_images=archive_images[1:])

        if has_errors:
            logger.error(f"{safe_pdf_name} (стр.: {len(page_results)}): найдены ошибки! Водитель: {safe_driver}, МЛ: {safe_ml}")
            report_path = output_dir / "report.txt"
            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n".join(report_lines) + "\n")
        else:
            logger.info(f"{safe_pdf_name} (стр.: {len(page_results)}): всё отлично. Водитель: {safe_driver}, МЛ: {safe_ml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пайплайн проверки МЛ")
    parser.add_argument("input_dir", type=str, help="Папка с PDF файлами")
    parser.add_argument("--out", type=str, default="Результаты", help="Папка для результатов")
    parser.add_argument("--model", type=str, default="Новые веса/best_clean.pt", help="Путь к YOLO модели")
    parser.add_argument(
        "--digit_model",
        type=str,
        default="Новые веса/best_digit_classifier.pt",
        help="Путь к классификатору цифр",
    )
    parser.add_argument(
        "--allow-untrusted-weights",
        action="store_true",
        help="Разрешить запуск сторонних неверифицированных весов моделей",
    )
    parser.add_argument(
        "--collect-crops",
        action="store_true",
        help="Собирать сложные/редкие кропы цифр в папку Дообучение",
    )
    parser.add_argument(
        "--crops-dir",
        type=str,
        default=str(Path.home() / "Documents" / "ML_Dataset Выборка" / "кропы" / "Дообучение"),
        help="Папка для сбора кропов цифр на дообучение",
    )
    parser.add_argument("--max-pages", type=int, default=150, help="Максимальное число страниц на один PDF")

    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("Ошибки", "Архив"):
        sub_path = out_dir / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)
    report_file = out_dir / "report.txt"
    if report_file.exists():
        report_file.unlink()

    pipeline = Pipeline(args.model, args.digit_model, allow_untrusted_weights=args.allow_untrusted_weights)

    input_path = Path(args.input_dir)
    if input_path.is_file():
        pipeline.process_pdf(
            args.input_dir,
            args.out,
            collect_crops=args.collect_crops,
            crops_dir=args.crops_dir,
            max_pages=args.max_pages,
        )
    else:
        for f in input_path.glob("*.pdf"):
            pipeline.process_pdf(
                str(f),
                args.out,
                collect_crops=args.collect_crops,
                crops_dir=args.crops_dir,
                max_pages=args.max_pages,
            )
