# Развёртывание ML-модуля проверки маршрутных листов (CPU k8s ↔ LiteLLM ↔ GPU Mac)

## 1. Архитектура

```text
Веб-портал (k8s) ──► CPU-под review_service.py (:8765, INFERENCE_MODE=http, SQLite /data/state.db)
                           │
                           ▼ (JSON + base64, Authorization: Bearer)
                     LiteLLM Proxy (:4000, pass_through_endpoints /waybills-cv/*)
                           │
                           ▼
                     GPU-хост Mac: inference_server.py (:8000, LitServe + MPS + офлайн EasyOCR)
```

- **CPU-под (`Dockerfile`, `requirements-cpu.txt`):** `python:3.12-slim`, без `torch`/`ultralytics`/`easyocr`/`litserve`. Хранит состояние сессий, очередь задач (`jobs`) и метаданные карантина в SQLite (`<REVIEW_DATA_DIR>/state.db`, WAL) с изоляцией файлов по `jobs/<job_id>/`.
- **GPU-хост (`inference_server.py`, `requirements-gpu.txt`):** LitServe-сервер (`/detect`, `/classify`, `/ocr`, `/health`) с проверкой SHA-256 всех 4 моделей из `weights_manifest.json` до старта HTTP (`download_enabled=False` для EasyOCR).

## 2. Реестр моделей (`weights_manifest.json`, `weights_version: waybills-cv-v1.4.1`)

| Компонент | Файл | Классы | SHA-256 |
|---|---|---|---|
| Детектор макета | `Новые веса/best_clean.pt` | 14 | `0bdfcbbabf8a987a6f5a2cd2648f96213c65dd8225101f7b1021074ae17d2747` |
| Классификатор цифр | `Новые веса/best_digit_classifier.pt` | 39 | `48196c721a78735ee05875d9c944e128c7fc503a5f6dc9c0173368d334c40f2a` |
| EasyOCR detector | `ocr_models/craft_mlt_25k.pth` | — | `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17` |
| EasyOCR recognizer | `ocr_models/cyrillic_g2.pth` | — | `48d0f3b58f28aa64651ab1032cc2d498c4de25135829668e87c14e7a07529f29` |

## 3. Переменные окружения

### CPU-сервис (`review_service.py`)
| Переменная | По умолчанию | Описание |
|---|---|---|
| `INFERENCE_MODE` | `local` (`http` в Docker) | Режим инференса (`http` для k8s) |
| `INFERENCE_BASE_URL` | — | URL LiteLLM Proxy / `inference_server.py` |
| `INFERENCE_API_KEY` | — | Bearer-токен для `Authorization: Bearer ...` |
| `INFERENCE_TIMEOUT_SEC` | `120` | Таймаут HTTP-запросов к инференсу |
| `INFERENCE_RETRIES` | `3` | Число попыток при 5xx / сетевых ошибках |
| `INFERENCE_TRUST_ENV` | `0` | `0` — игнорировать системный HTTP_PROXY |
| `REVIEW_DATA_DIR` | `./review_data` (`/data` в Docker) | Каталог `state.db`, `jobs/`, `quarantine/` (PVC) |
| `ALLOW_NON_LOOPBACK` | `0` (`1` в Docker) | Разрешить бинд на `0.0.0.0` внутри контейнера |
| `ML_API_TOKEN` | — | Токен защиты `/api/ml/*` (заголовок `X-ML-Token`) |
| `MAX_PDF_PAGES` | `3500` | Максимальное число страниц PDF |
| `LARGE_PDF_POLICY` | `reject` | `reject` (HTTP 400) или `partial` (первые N стр.) |
| `JOB_WORKERS` | `2` | Число фоновых воркеров очереди SQLite |
| `MAX_JOB_ATTEMPTS` | `2` | Попыток до перевода задачи в `failed` после рестарта |

### GPU-сервис (`inference_server.py`)
| Переменная | По умолчанию | Описание |
|---|---|---|
| `INFERENCE_SERVER_HOST` | `127.0.0.1` | Хост прослушивания (`0.0.0.0` за файрволом) |
| `INFERENCE_SERVER_PORT` | `8000` | Порт LitServe |
| `INFERENCE_SERVER_TOKEN` | — | Bearer-токен (если задан, обязателен на `/detect`, `/classify`, `/ocr`) |
| `OCR_MODELS_DIR` | `./ocr_models` | Каталог офлайн-весов EasyOCR |
| `WEIGHTS_MANIFEST_PATH` | `./weights_manifest.json` | Путь к манифесту SHA-256 |
| `YOLO_DETECT_DEVICE` | `auto` | Устройство детектора (`mps`, `cpu`, `auto`) |
| `YOLO_CLS_DEVICE` | `auto` | Устройство классификатора (`mps`, `cpu`, `auto`) |
| `EASYOCR_DEVICE` | `auto` | Устройство EasyOCR (`mps`, `cpu`, `auto`) |

## 4. Сборка и запуск

### CPU-контейнер
```bash
docker build -t waybills-cpu:1.5.0 .
docker compose -f docker-compose.local.yml up --build
curl -s http://127.0.0.1:8765/api/ml/health
```

### GPU-сервис на Mac + LiteLLM
```bash
pip install -r requirements-gpu.txt
python prepare_ocr_weights.py
INFERENCE_SERVER_TOKEN=<secret> python inference_server.py --host 0.0.0.0 --port 8000
```
Конфигурация LiteLLM Proxy: см. `litellm_passthrough_example.yaml`.

## 5. Контрольная проверка (`verify_golden.py`)
```bash
# Локальный эталон:
python verify_golden.py --mode local
# Проверка через LitServe / LiteLLM:
python verify_golden.py --mode http --base-url http://127.0.0.1:8000 --api-key <secret>
```
Код возврата `0` означает полное совпадение `weights_version`, окон, элементов, цифр и бизнес-ошибок.
