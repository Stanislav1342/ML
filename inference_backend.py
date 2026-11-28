from abc import ABC, abstractmethod
import base64
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "weights_manifest.json"
DEFAULT_OCR_MODELS_DIR = PROJECT_DIR / "ocr_models"

KNOWN_WEIGHT_HASHES: Dict[str, str] = {
    "best_clean.pt": "0bdfcbbabf8a987a6f5a2cd2648f96213c65dd8225101f7b1021074ae17d2747",
    "best_clean_966.pt": "0bf3ac31e82d93c8d3a8dc71b0be66d8b95ff693e1926a95add8ea3cf286d7d7",
    "best_clean_966_2026-11-19.pt": "0bf3ac31e82d93c8d3a8dc71b0be66d8b95ff693e1926a95add8ea3cf286d7d7",
    "best_clean_v2_gold.pt": "d5bca87876bde838ac9fc0bb8e3a846d8b860266fec9719a9c2b17466d1b4333",
    "best_clean_prev.pt": "5d419954991aa03f78d1e73fd6ec23a02ce2731c5dae88a79fd8c5ba49b5db95",
    "best_digit_classifier.pt": "48196c721a78735ee05875d9c944e128c7fc503a5f6dc9c0173368d334c40f2a",
    "best_digit_classifier_41cls_2026-09-24.pt": "18f1847773f6a517f09cdb03d4db219fbd302807ec43a35b3c9bfafda799b874",
}


def verify_weight_integrity(model_path: str, allow_untrusted: bool = False) -> str:
    """Проверяет SHA-256 контрольную сумму весов до загрузки PyTorch (защита от Pickle RCE)."""
    path = Path(model_path)
    if not path.exists():
        return ""

    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    if path.name in KNOWN_WEIGHT_HASHES:
        if file_hash != KNOWN_WEIGHT_HASHES[path.name]:
            raise RuntimeError(
                f"Критическая ошибка ИБ: Хэш файла {path.name} ({file_hash}) не совпадает с доверенным! "
                "Загрузка модели заблокирована во избежание Pickle RCE."
            )
        return file_hash

    if file_hash in KNOWN_WEIGHT_HASHES.values():
        return file_hash

    if not allow_untrusted:
        raise RuntimeError(
            f"Критическая ошибка ИБ: Модель {path.name} (хэш: {file_hash}) отсутствует в реестре доверенных весов! "
            "Загрузка заблокирована во избежание Pickle RCE. "
            "Для принудительного использования сторонних весов укажите флаг --allow-untrusted-weights."
        )
    logger.warning(
        f"ВНИМАНИЕ [ИБ]: Модель {path.name} отсутствует в реестре доверенных весов. "
        "Принудительная загрузка разрешена параметром allow_untrusted."
    )
    return file_hash


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_weights_manifest(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Загружает JSON-манифест весов weights_manifest.json."""
    mpath = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not mpath.is_file():
        raise RuntimeError(f"Манифест весов не найден: {mpath}")
    return json.loads(mpath.read_text(encoding="utf-8"))


def verify_manifest_weights(
    manifest_path: Optional[str] = None,
    yolo_model_path: Optional[str] = None,
    digit_classifier_path: Optional[str] = None,
    ocr_models_dir: Optional[str] = None,
    allow_untrusted: bool = False,
    require_ocr_files: bool = True,
) -> Dict[str, Any]:
    """Проверяет SHA-256 всех 4 моделей по weights_manifest.json (fail-closed до загрузки)."""
    manifest = load_weights_manifest(manifest_path)
    models = manifest.get("models") or {}
    weights_ver = str(manifest.get("weights_version") or "unknown")

    det_entry = models.get("detector") or {}
    cls_entry = models.get("classifier") or {}
    ocr_det_entry = models.get("ocr_detector") or {}
    ocr_rec_entry = models.get("ocr_recognizer") or {}

    det_path = Path(yolo_model_path or (PROJECT_DIR / det_entry.get("file", "")))
    cls_path = Path(digit_classifier_path or (PROJECT_DIR / cls_entry.get("file", "")))

    def _check_entry(path: Path, expected_sha: str, label: str) -> str:
        if not path.is_file():
            raise RuntimeError(f"Критическая ошибка ИБ: файл весов {label} не найден: {path}")
        actual = _sha256_file(path)
        if expected_sha and actual != expected_sha.lower():
            if not allow_untrusted:
                raise RuntimeError(
                    f"Критическая ошибка ИБ: SHA-256 файла {path.name} ({actual}) "
                    f"не совпадает с манифестом ({expected_sha})."
                )
            logger.warning(f"ВНИМАНИЕ [ИБ]: SHA-256 {path.name} не совпадает с манифестом (allow_untrusted=True).")
        return actual

    det_sha = _check_entry(det_path, str(det_entry.get("sha256") or ""), "detector")
    cls_sha = _check_entry(cls_path, str(cls_entry.get("sha256") or ""), "classifier")

    ocr_dir = Path(ocr_models_dir or os.getenv("OCR_MODELS_DIR") or DEFAULT_OCR_MODELS_DIR)
    ocr_det_sha = ""
    ocr_rec_sha = ""
    if require_ocr_files:
        ocr_det_sha = _check_entry(
            ocr_dir / str(ocr_det_entry.get("file") or "craft_mlt_25k.pth"),
            str(ocr_det_entry.get("sha256") or ""),
            "ocr_detector",
        )
        ocr_rec_sha = _check_entry(
            ocr_dir / str(ocr_rec_entry.get("file") or "cyrillic_g2.pth"),
            str(ocr_rec_entry.get("sha256") or ""),
            "ocr_recognizer",
        )

    return {
        "weights_version": weights_ver,
        "models": {
            "detector": {
                "sha256": det_sha,
                "classes": int(det_entry.get("num_classes", 14)),
            },
            "classifier": {
                "sha256": cls_sha,
                "classes": int(cls_entry.get("num_classes", 39)),
            },
            "ocr": {
                "sha256": {
                    "detector": ocr_det_sha,
                    "recognizer": ocr_rec_sha,
                },
            },
        },
    }


def _encode_pil_base64(img: Image.Image, fmt: str = "png") -> str:
    """Кодирует PIL.Image в base64 (по умолчанию PNG без потерь)."""
    buf = io.BytesIO()
    fmt_low = (fmt or "png").strip().lower()
    if fmt_low in ("jpg", "jpeg"):
        img.convert("RGB").save(buf, format="JPEG", quality=92)
    else:
        img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_ndarray_png_base64(arr: np.ndarray) -> str:
    """Кодирует numpy-изображение в lossless PNG base64."""
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.uint8))
    else:
        pil_img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class InferenceBackend(ABC):
    """Абстрактный интерфейс ML-инференса (детектор, классификатор цифр, EasyOCR)."""

    @property
    @abstractmethod
    def weights_version(self) -> str:
        """Возвращает идентификатор текущей версии весов."""

    @abstractmethod
    def detect_pages(
        self,
        images: List[Image.Image],
        imgsz: int = 1024,
        conf: float = 0.12,
        iou: float = 0.45,
    ) -> List[List[Dict[str, Any]]]:
        """Детектирует объекты на списке страниц. Возвращает список списков словарей детекций."""

    @abstractmethod
    def classify_crops(self, crops: List[Image.Image]) -> List[Dict[str, Any]]:
        """Классифицирует список кропов цифр."""

    @abstractmethod
    def ocr_crops(
        self,
        crops: List[Tuple[np.ndarray, Optional[str]]],
    ) -> List[List[Dict[str, Any]]]:
        """Распознаёт текст на списке (crop_ndarray, allowlist)."""

    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Возвращает состояние бэкенда инференса."""


class LocalInferenceBackend(InferenceBackend):
    """Локальный инференс через PyTorch/Ultralytics/EasyOCR с ленивыми импортами."""

    def __init__(
        self,
        yolo_model_path: str = "Новые веса/best_clean.pt",
        digit_classifier_path: str = "Новые веса/best_digit_classifier.pt",
        allow_untrusted_weights: bool = False,
        detect_device: Optional[str] = None,
        cls_device: Optional[str] = None,
        easyocr_device: Optional[str] = None,
        ocr_models_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        require_offline_ocr: bool = False,
    ):
        ocr_dir = Path(ocr_models_dir or os.getenv("OCR_MODELS_DIR") or DEFAULT_OCR_MODELS_DIR)
        use_offline_ocr = bool(require_offline_ocr or ocr_dir.is_dir() or os.getenv("OCR_MODELS_DIR"))

        m_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
        if m_path.is_file():
            self.manifest_info = verify_manifest_weights(
                manifest_path=str(m_path),
                yolo_model_path=yolo_model_path,
                digit_classifier_path=digit_classifier_path,
                ocr_models_dir=str(ocr_dir),
                allow_untrusted=allow_untrusted_weights,
                require_ocr_files=use_offline_ocr,
            )
            det_sha = self.manifest_info["models"]["detector"]["sha256"]
            cls_sha = self.manifest_info["models"]["classifier"]["sha256"]
        else:
            det_sha = verify_weight_integrity(yolo_model_path, allow_untrusted=allow_untrusted_weights)
            cls_sha = verify_weight_integrity(digit_classifier_path, allow_untrusted=allow_untrusted_weights)
            self.manifest_info = {
                "weights_version": f"local:{det_sha[:12] or 'none'}-{cls_sha[:12] or 'none'}",
                "models": {
                    "detector": {"sha256": det_sha, "classes": 14},
                    "classifier": {"sha256": cls_sha, "classes": 39},
                    "ocr": {"sha256": {"detector": "", "recognizer": ""}},
                },
            }

        import easyocr
        import torch
        from ultralytics import YOLO

        def _resolve_device(raw: Optional[str], env_name: str) -> str:
            val = (raw if raw is not None else os.getenv(env_name, "auto")).strip().lower()
            if not val or val == "auto":
                if torch.cuda.is_available():
                    return "cuda"
                if torch.backends.mps.is_available():
                    return "mps"
                return "cpu"
            return val

        self.detect_device = _resolve_device(detect_device, "YOLO_DETECT_DEVICE")
        self.cls_device = _resolve_device(cls_device, "YOLO_CLS_DEVICE")
        self.easyocr_device = _resolve_device(easyocr_device, "EASYOCR_DEVICE")

        if (
            "cpu" in (self.detect_device, self.cls_device, self.easyocr_device)
            and not os.getenv("OMP_NUM_THREADS")
        ):
            cpu_total = os.cpu_count() or 4
            torch.set_num_threads(min(8, max(2, (cpu_total * 3) // 4)))

        easyocr_gpu = self.easyocr_device != "cpu"
        logger.info(f"Загрузка YOLOv8 ({yolo_model_path})...")
        self.yolo = YOLO(yolo_model_path)

        logger.info(
            f"Загрузка классификатора цифр ({digit_classifier_path}) и EasyOCR "
            f"(offline={use_offline_ocr}, dir={ocr_dir if use_offline_ocr else 'default'})..."
        )
        if use_offline_ocr:
            self.easyocr = easyocr.Reader(
                ["ru"],
                gpu=easyocr_gpu,
                model_storage_directory=str(ocr_dir),
                download_enabled=False,
            )
        else:
            self.easyocr = easyocr.Reader(["ru"], gpu=easyocr_gpu)

        self.actual_easyocr_device = str(getattr(self.easyocr, "device", self.easyocr_device))
        logger.info(
            f"Устройства LocalInferenceBackend: detect={self.detect_device}, "
            f"cls={self.cls_device}, easyocr_req={self.easyocr_device}, "
            f"easyocr_actual={self.actual_easyocr_device} (gpu={easyocr_gpu})"
        )

        self.digit_classifier = YOLO(digit_classifier_path)

        env_ver = os.getenv("WEIGHTS_VERSION", "").strip()
        self._weights_version = env_ver or str(self.manifest_info["weights_version"])

    @property
    def weights_version(self) -> str:
        return self._weights_version

    def detect_pages(
        self,
        images: List[Image.Image],
        imgsz: int = 1024,
        conf: float = 0.12,
        iou: float = 0.45,
    ) -> List[List[Dict[str, Any]]]:
        if not images:
            return []
        yolo_batch = 4 if self.detect_device == "cpu" else 16
        results = self.yolo(
            images,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            batch=yolo_batch,
            device=self.detect_device,
        )
        out: List[List[Dict[str, Any]]] = []
        for res in results:
            dets: List[Dict[str, Any]] = []
            if res.boxes is not None:
                names = res.names or {}
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].cpu().numpy())
                    score = float(box.conf[0].cpu().numpy())
                    dets.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class_id": cls_id,
                        "class_name": str(names.get(cls_id, cls_id)),
                        "confidence": score,
                    })
            out.append(dets)
        return out

    def classify_crops(self, crops: List[Image.Image]) -> List[Dict[str, Any]]:
        if not crops:
            return []
        cls_batch = min(64, len(crops))
        results = self.digit_classifier(
            crops,
            batch=cls_batch,
            verbose=False,
            device=self.cls_device,
        )
        out: List[Dict[str, Any]] = []
        for res in results:
            top1_idx = int(res.probs.top1)
            top1_conf = float(res.probs.top1conf)
            cls_name = str(res.names[top1_idx])
            out.append({
                "class_id": top1_idx,
                "class_name": cls_name,
                "confidence": top1_conf,
            })
        return out

    def ocr_crops(
        self,
        crops: List[Tuple[np.ndarray, Optional[str]]],
    ) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for crop_arr, allowlist in crops:
            kwargs: Dict[str, Any] = {}
            if allowlist:
                kwargs["allowlist"] = allowlist
            raw_items = self.easyocr.readtext(crop_arr, **kwargs)
            items: List[Dict[str, Any]] = []
            for bbox_pts, text, conf in raw_items:
                poly = [[float(pt[0]), float(pt[1])] for pt in bbox_pts]
                items.append({
                    "bbox": poly,
                    "text": str(text),
                    "confidence": float(conf),
                })
            out.append(items)
        return out

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "mode": "local",
            "weights_version": self._weights_version,
            "devices": {
                "detect": self.detect_device,
                "classify": self.cls_device,
                "ocr": self.actual_easyocr_device,
            },
            "models": self.manifest_info.get("models", {}),
        }


class HttpInferenceBackend(InferenceBackend):
    """Удалённый инференс по HTTP (JSON + base64) без зависимостей torch/ultralytics/easyocr."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        detect_path: Optional[str] = None,
        classify_path: Optional[str] = None,
        ocr_path: Optional[str] = None,
        health_path: Optional[str] = None,
        detect_batch_size: Optional[int] = None,
        cls_batch_size: Optional[int] = None,
        ocr_batch_size: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        retries: Optional[int] = None,
        trust_env: Optional[bool] = None,
        detect_image_format: Optional[str] = None,
        backoff_base: float = 0.5,
    ):
        raw_url = base_url if base_url is not None else os.getenv("INFERENCE_BASE_URL", "")
        self.base_url = raw_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("INFERENCE_BASE_URL не задан для HttpInferenceBackend.")

        self.api_key = (api_key if api_key is not None else os.getenv("INFERENCE_API_KEY", "")).strip()

        if extra_headers is not None:
            self.extra_headers = dict(extra_headers)
        else:
            raw_extra = os.getenv("INFERENCE_EXTRA_HEADERS_JSON", "").strip()
            self.extra_headers = json.loads(raw_extra) if raw_extra else {}

        self.detect_path = detect_path or os.getenv("INFERENCE_DETECT_PATH", "/detect")
        self.classify_path = classify_path or os.getenv("INFERENCE_CLASSIFY_PATH", "/classify")
        self.ocr_path = ocr_path or os.getenv("INFERENCE_OCR_PATH", "/ocr")
        self.health_path = health_path or os.getenv("INFERENCE_HEALTH_PATH", "/health")

        self.detect_batch_size = max(1, int(detect_batch_size or os.getenv("INFERENCE_DETECT_BATCH_SIZE", "8")))
        self.cls_batch_size = max(1, int(cls_batch_size or os.getenv("INFERENCE_CLS_BATCH_SIZE", "64")))
        self.ocr_batch_size = max(1, int(ocr_batch_size or os.getenv("INFERENCE_OCR_BATCH_SIZE", "16")))

        self.timeout_sec = float(timeout_sec if timeout_sec is not None else os.getenv("INFERENCE_TIMEOUT_SEC", "120"))
        self.retries = max(1, int(retries if retries is not None else os.getenv("INFERENCE_RETRIES", "3")))
        self.backoff_base = float(backoff_base)

        if trust_env is None:
            trust_env = os.getenv("INFERENCE_TRUST_ENV", "0").strip().lower() in ("1", "true", "yes")
        self._opener = (
            urllib.request.build_opener()
            if trust_env
            else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )

        fmt = (detect_image_format or os.getenv("INFERENCE_DETECT_IMAGE_FORMAT", "png")).strip().lower()
        self.detect_image_format = "jpeg" if fmt in ("jpg", "jpeg") else "png"
        self._weights_version = os.getenv("WEIGHTS_VERSION", "remote:unknown").strip() or "remote:unknown"

    @property
    def weights_version(self) -> str:
        return self._weights_version

    def _build_headers(self, has_body: bool) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if has_body:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for k, v in self.extra_headers.items():
            headers[str(k)] = str(v)
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout_sec: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = self._build_headers(has_body=(data is not None))
        eff_timeout = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        eff_retries = self.retries if retries is None else max(1, int(retries))

        last_exc: Optional[Exception] = None
        for attempt in range(eff_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self._opener.open(req, timeout=eff_timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    parsed = json.loads(raw) if raw else {}
                    if isinstance(parsed, dict) and parsed.get("weights_version"):
                        self._weights_version = str(parsed["weights_version"])
                    return parsed
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    body = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"HTTP {exc.code} от {url}: {body[:300]}") from exc
                last_exc = exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_exc = exc

            if attempt + 1 < eff_retries and self.backoff_base > 0:
                time.sleep(self.backoff_base * (2 ** attempt))

        raise RuntimeError(f"Ошибка запроса к {url} после {eff_retries} попыток: {last_exc}") from last_exc

    def detect_pages(
        self,
        images: List[Image.Image],
        imgsz: int = 1024,
        conf: float = 0.12,
        iou: float = 0.45,
    ) -> List[List[Dict[str, Any]]]:
        if not images:
            return []
        all_detections: List[List[Dict[str, Any]]] = []
        for start in range(0, len(images), self.detect_batch_size):
            batch = images[start:start + self.detect_batch_size]
            encoded = [_encode_pil_base64(im, self.detect_image_format) for im in batch]
            resp = self._request_json(
                "POST",
                self.detect_path,
                {"images": encoded, "imgsz": imgsz, "conf": conf, "iou": iou},
            )
            results = resp.get("results") or []
            for item in results:
                all_detections.append(list(item.get("detections") or []))
        return all_detections

    def classify_crops(self, crops: List[Image.Image]) -> List[Dict[str, Any]]:
        if not crops:
            return []
        all_results: List[Dict[str, Any]] = []
        for start in range(0, len(crops), self.cls_batch_size):
            batch = crops[start:start + self.cls_batch_size]
            encoded = [_encode_pil_base64(im, "png") for im in batch]
            resp = self._request_json("POST", self.classify_path, {"crops": encoded})
            all_results.extend(resp.get("results") or [])
        return all_results

    def ocr_crops(
        self,
        crops: List[Tuple[np.ndarray, Optional[str]]],
    ) -> List[List[Dict[str, Any]]]:
        if not crops:
            return []
        all_results: List[List[Dict[str, Any]]] = []
        for start in range(0, len(crops), self.ocr_batch_size):
            batch = crops[start:start + self.ocr_batch_size]
            payload_crops = [
                {"image": _encode_ndarray_png_base64(arr), "allowlist": allowlist}
                for arr, allowlist in batch
            ]
            resp = self._request_json("POST", self.ocr_path, {"crops": payload_crops})
            for item in (resp.get("results") or []):
                all_results.append(list(item.get("items") or []))
        return all_results

    def get_health(
        self,
        timeout_sec: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._request_json(
            "GET",
            self.health_path,
            timeout_sec=timeout_sec,
            retries=retries,
        )


def create_inference_backend(
    yolo_model_path: str = "Новые веса/best_clean.pt",
    digit_classifier_path: str = "Новые веса/best_digit_classifier.pt",
    allow_untrusted_weights: bool = False,
) -> InferenceBackend:
    """Создаёт локальный или HTTP бэкенд инференса в зависимости от env INFERENCE_MODE."""
    mode = os.getenv("INFERENCE_MODE", "local").strip().lower()
    if mode == "http":
        return HttpInferenceBackend()
    if mode == "local":
        return LocalInferenceBackend(
            yolo_model_path=yolo_model_path,
            digit_classifier_path=digit_classifier_path,
            allow_untrusted_weights=allow_untrusted_weights,
        )
    raise ValueError(f"Неизвестный INFERENCE_MODE='{mode}' (ожидается 'local' или 'http').")
