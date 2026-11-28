"""Формирует обезличенный контрольный набор PDF и эталон golden_expected.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from inference_backend import LocalInferenceBackend
from pipeline import PageResult, Pipeline


def _anonymize_page_image(img_path: Path, pr: PageResult, out_pdf: Path) -> None:
    """Обезличивает страницу: закрашивает шапку, подвал, поля и ПДн внутри окон, оставляя элементы и линии окон."""
    with Image.open(img_path) as im:
        arr = np.array(im.convert("RGB"))
    h, w, _ = arr.shape

    keep_mask = np.zeros((h, w), dtype=bool)

    for win in pr.windows:
        wx1 = max(0, int(win.bbox.x1))
        wy1 = max(0, int(win.bbox.y1))
        wx2 = min(w, int(win.bbox.x2))
        wy2 = min(h, int(win.bbox.y2))

        # Сохраняем контурные линии самого окна (10 px по периметру)
        border = 10
        keep_mask[wy1:min(wy2, wy1 + border), wx1:wx2] = True
        keep_mask[max(wy1, wy2 - border):wy2, wx1:wx2] = True
        keep_mask[wy1:wy2, wx1:min(wx2, wx1 + border)] = True
        keep_mask[wy1:wy2, max(wx1, wx2 - border):wx2] = True

        # Сохраняем полосу якоря «Получение»
        if win.return_split_y is not None:
            sy = int(win.return_split_y)
            y_top = max(wy1, sy - 25)
            y_bot = min(wy2, sy + 30)
            keep_mask[y_top:y_bot, wx1:wx2] = True

        # Сохраняем все детектированные элементы (печати, подписи, расшифровки, цифры, QR, исправления)
        pad = 10
        for el in win.elements:
            ex1 = max(0, int(el.bbox.x1) - pad)
            ey1 = max(0, int(el.bbox.y1) - pad)
            ex2 = min(w, int(el.bbox.x2) + pad)
            ey2 = min(h, int(el.bbox.y2) + pad)
            keep_mask[ey1:ey2, ex1:ex2] = True

    arr[~keep_mask] = 255
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(out_pdf), format="PDF", resolution=200.0)


def serialize_page_result(pr: PageResult) -> Dict[str, Any]:
    """Сериализует PageResult в детерминированный словарь для эталона и сверки."""
    windows_out: List[Dict[str, Any]] = []
    for win in pr.windows:
        elems = [
            {
                "label": el.label,
                "bbox": [round(float(el.bbox.x1), 2), round(float(el.bbox.y1), 2), round(float(el.bbox.x2), 2), round(float(el.bbox.y2), 2)],
                "ocr_value": el.ocr_value,
                "confidence": round(float(el.confidence), 4),
            }
            for el in win.elements
        ]
        elems.sort(key=lambda e: (e["label"], e["bbox"][1], e["bbox"][0]))
        windows_out.append({
            "bbox": [round(float(win.bbox.x1), 2), round(float(win.bbox.y1), 2), round(float(win.bbox.x2), 2), round(float(win.bbox.y2), 2)],
            "return_split_y": round(float(win.return_split_y), 2) if win.return_split_y is not None else None,
            "elements": elems,
        })
    return {
        "page_num": int(pr.page_num),
        "windows_count": len(pr.windows),
        "errors": list(pr.errors),
        "warnings": list(pr.warnings),
        "windows": windows_out,
    }


def build_golden(input_dir: Path, output_dir: Path) -> None:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"В каталоге {input_dir} не найдено ни одного PDF.")

    backend = LocalInferenceBackend(require_offline_ocr=True)
    pipe = Pipeline(backend=backend)

    selected: Dict[str, tuple[Path, PageResult]] = {}

    with tempfile.TemporaryDirectory(prefix="golden_scan_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for pdf_idx, pdf_path in enumerate(pdfs):
            work_sub = tmp_path / f"src_{pdf_idx}"
            segments = pipe.analyze_pdf_segments(
                str(pdf_path),
                str(work_sub),
                max_pages=200,
                image_ext=".png",
                large_pdf_policy="partial",
            )
            for _, seg_pages in segments:
                for pr in seg_pages:
                    if not pr.windows:
                        continue
                    has_return = any(w.return_split_y is not None for w in pr.windows)
                    has_err = bool(pr.errors)
                    if "clean" not in selected and not has_err and not pr.warnings and len(pr.windows) >= 2:
                        selected["clean"] = (Path(pr.image_path), pr)
                    elif "return_block" not in selected and has_return:
                        selected["return_block"] = (Path(pr.image_path), pr)
                    elif "error" not in selected and has_err and not has_return:
                        selected["error"] = (Path(pr.image_path), pr)
                    elif "error_extra" not in selected and has_err and (Path(pr.image_path), pr) != selected.get("error"):
                        selected["error_extra"] = (Path(pr.image_path), pr)

                if all(k in selected for k in ("clean", "return_block", "error", "error_extra")):
                    break
            if all(k in selected for k in ("clean", "return_block", "error", "error_extra")):
                break

        samples_dir = output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        name_map = {
            "clean": "sample_01_clean.pdf",
            "error": "sample_02_error.pdf",
            "return_block": "sample_03_return_block.pdf",
            "error_extra": "sample_04_error_extra.pdf",
        }

        expected_samples: Dict[str, Any] = {}
        for key in ("clean", "error", "return_block", "error_extra"):
            if key not in selected:
                continue
            img_p, pr = selected[key]
            out_pdf = samples_dir / name_map[key]
            _anonymize_page_image(img_p, pr, out_pdf)

            # Повторно прогоняем обезличенный PDF, чтобы эталон точно соответствовал входному PDF
            verify_sub = tmp_path / f"verify_{key}"
            segs = pipe.analyze_pdf_segments(
                str(out_pdf),
                str(verify_sub),
                max_pages=10,
                image_ext=".png",
                large_pdf_policy="partial",
            )
            pages_ser = [serialize_page_result(p) for _, sp in segs for p in sp]
            expected_samples[name_map[key]] = {"pages": pages_ser}

    golden_doc = {
        "schema_version": 1,
        "weights_version": backend.weights_version,
        "samples": expected_samples,
    }
    out_json = output_dir / "golden_expected.json"
    out_json.write_text(json.dumps(golden_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Golden dataset создан: {len(expected_samples)} PDF -> {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация обезличенного контрольного набора PDF и эталона")
    parser.add_argument("--input-dir", type=str, default="Test_PDFs", help="Каталог исходных PDF")
    parser.add_argument("--output-dir", type=str, default="golden_dataset", help="Выходной каталог golden_dataset")
    args = parser.parse_args()
    build_golden(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
