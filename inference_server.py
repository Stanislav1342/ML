import argparse
import base64
import hmac
import io
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
import litserve as ls
import numpy as np
from PIL import Image

from inference_backend import (
    LocalInferenceBackend,
    resolve_inference_devices,
    verify_manifest_weights,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("inference_server")


def _decode_base64_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _decode_base64_ndarray(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


class DetectLitAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.backend = LocalInferenceBackend(
            load_detector=True,
            load_classifier=False,
            load_ocr=False,
            require_offline_ocr=True,
        )

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        images = [_decode_base64_pil(b64) for b64 in (request.get("images") or [])]
        return {
            "images": images,
            "imgsz": int(request.get("imgsz", 1024)),
            "conf": float(request.get("conf", 0.12)),
            "iou": float(request.get("iou", 0.45)),
        }

    def predict(self, inputs: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        return self.backend.detect_pages(
            inputs["images"],
            imgsz=inputs["imgsz"],
            conf=inputs["conf"],
            iou=inputs["iou"],
        )

    def encode_response(self, output: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return {
            "weights_version": self.backend.weights_version,
            "model_sha256": self.backend.manifest_info["models"]["detector"]["sha256"],
            "device": self.backend.detect_device,
            "results": [{"detections": dets} for dets in output],
        }


class ClassifyLitAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.backend = LocalInferenceBackend(
            load_detector=False,
            load_classifier=True,
            load_ocr=False,
            require_offline_ocr=True,
        )

    def decode_request(self, request: Dict[str, Any]) -> List[Image.Image]:
        return [_decode_base64_pil(b64) for b64 in (request.get("crops") or [])]

    def predict(self, crops: List[Image.Image]) -> List[Dict[str, Any]]:
        return self.backend.classify_crops(crops)

    def encode_response(self, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "weights_version": self.backend.weights_version,
            "model_sha256": self.backend.manifest_info["models"]["classifier"]["sha256"],
            "device": self.backend.cls_device,
            "results": output,
        }


class OcrLitAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.backend = LocalInferenceBackend(
            load_detector=False,
            load_classifier=False,
            load_ocr=True,
            require_offline_ocr=True,
        )

    def decode_request(self, request: Dict[str, Any]) -> List[Tuple[np.ndarray, Optional[str]]]:
        out: List[Tuple[np.ndarray, Optional[str]]] = []
        for item in (request.get("crops") or []):
            arr = _decode_base64_ndarray(item["image"])
            out.append((arr, item.get("allowlist")))
        return out

    def predict(self, crops: List[Tuple[np.ndarray, Optional[str]]]) -> List[List[Dict[str, Any]]]:
        return self.backend.ocr_crops(crops)

    def encode_response(self, output: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return {
            "weights_version": self.backend.weights_version,
            "model_sha256": self.backend.manifest_info["models"]["ocr"]["sha256"],
            "device": self.backend.actual_easyocr_device,
            "results": [{"items": items} for items in output],
        }


def create_lit_server(api_token: Optional[str] = None) -> ls.LitServer:
    """Создаёт LitServer с 3 эндпоинтами инференса, /health и опциональной Bearer-авторизацией."""
    # Проверяем SHA-256 всех 4 моделей заранее (fail-closed до старта HTTP), без загрузки весов в главный процесс.
    manifest_info = verify_manifest_weights(require_ocr_files=True)
    devices = resolve_inference_devices()
    env_ver = os.getenv("WEIGHTS_VERSION", "").strip()
    health_payload = {
        "status": "ok",
        "mode": "local",
        "weights_version": env_ver or str(manifest_info["weights_version"]),
        "devices": devices,
        "models": manifest_info.get("models", {}),
    }

    server = ls.LitServer(
        [
            DetectLitAPI(api_path="/detect"),
            ClassifyLitAPI(api_path="/classify"),
            OcrLitAPI(api_path="/ocr"),
        ],
        accelerator="cpu",
        devices=1,
        workers_per_device=1,
        timeout=120,
        max_payload_size=128 * 1024 * 1024,
        healthcheck_path="/_litserve_health",
    )

    @server.app.get("/health")
    async def health():
        return JSONResponse(status_code=200, content=health_payload)

    if api_token:
        expected = f"Bearer {api_token}"

        @server.app.middleware("http")
        async def bearer_auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/_litserve_health"):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if not hmac.compare_digest(auth, expected):
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
            return await call_next(request)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-сервис инференса маршрутных листов на LitServe")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("INFERENCE_SERVER_HOST", "127.0.0.1"),
        help="Хост прослушивания (по умолчанию 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("INFERENCE_SERVER_PORT", "8000")),
        help="Порт прослушивания (по умолчанию 8000)",
    )
    args = parser.parse_args()

    token = os.getenv("INFERENCE_SERVER_TOKEN", "").strip() or None
    server = create_lit_server(api_token=token)
    logger.info(f"Запуск LitServe inference_server на http://{args.host}:{args.port} (auth={'on' if token else 'off'})")
    server.run(host=args.host, port=args.port, generate_client_file=False)


if __name__ == "__main__":
    main()
