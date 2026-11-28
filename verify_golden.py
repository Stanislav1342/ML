"""Регрессионная проверка контрольного набора golden_dataset (local или http против LitServe)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

from build_golden_dataset import serialize_page_result
from inference_backend import HttpInferenceBackend, LocalInferenceBackend
from pipeline import Pipeline


def _compare_pages(
    sample_name: str,
    exp_pages: List[Dict[str, Any]],
    act_pages: List[Dict[str, Any]],
    bbox_tol: float,
) -> List[str]:
    diffs: List[str] = []
    if len(exp_pages) != len(act_pages):
        diffs.append(f"{sample_name}: число страниц {len(act_pages)} != {len(exp_pages)}")
        return diffs

    for idx, (exp_p, act_p) in enumerate(zip(exp_pages, act_pages), start=1):
        if exp_p["windows_count"] != act_p["windows_count"]:
            diffs.append(f"{sample_name} p{idx}: windows_count {act_p['windows_count']} != {exp_p['windows_count']}")
        if exp_p["errors"] != act_p["errors"]:
            diffs.append(f"{sample_name} p{idx}: errors {act_p['errors']} != {exp_p['errors']}")
        if exp_p["warnings"] != act_p["warnings"]:
            diffs.append(f"{sample_name} p{idx}: warnings {act_p['warnings']} != {exp_p['warnings']}")

        for w_idx, (ew, aw) in enumerate(zip(exp_p.get("windows", []), act_p.get("windows", []))):
            es, as_ = ew.get("return_split_y"), aw.get("return_split_y")
            if (es is None) != (as_ is None) or (es is not None and abs(es - as_) > bbox_tol):
                diffs.append(f"{sample_name} p{idx} w{w_idx}: return_split_y {as_} != {es}")

            e_els = [(e["label"], e["ocr_value"]) for e in ew.get("elements", [])]
            a_els = [(e["label"], e["ocr_value"]) for e in aw.get("elements", [])]
            if e_els != a_els:
                diffs.append(f"{sample_name} p{idx} w{w_idx}: elements {a_els} != {e_els}")
                continue

            for e_i, (ee, ae) in enumerate(zip(ew.get("elements", []), aw.get("elements", []))):
                for c in range(4):
                    if abs(ee["bbox"][c] - ae["bbox"][c]) > bbox_tol:
                        diffs.append(
                            f"{sample_name} p{idx} w{w_idx} el{e_i}: bbox {ae['bbox']} != {ee['bbox']}"
                        )
                        break
    return diffs


def verify_golden(
    golden_dir: Path,
    mode: str = "local",
    base_url: str = "",
    api_key: str = "",
    bbox_tol: float = 2.0,
) -> int:
    expected_path = golden_dir / "golden_expected.json"
    if not expected_path.is_file():
        print(f"FAIL: не найден {expected_path}", file=sys.stderr)
        return 1

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if mode == "http":
        backend = HttpInferenceBackend(
            base_url=base_url or os.getenv("INFERENCE_BASE_URL", ""),
            api_key=api_key or os.getenv("INFERENCE_API_KEY", ""),
        )
        health = backend.get_health(timeout_sec=5.0, retries=1)
        actual_weights = str(health.get("weights_version") or backend.weights_version)
    else:
        backend = LocalInferenceBackend(require_offline_ocr=True)
        actual_weights = backend.weights_version

    exp_weights = expected.get("weights_version")
    if exp_weights and actual_weights != exp_weights:
        print(f"FAIL: weights_version '{actual_weights}' != '{exp_weights}'", file=sys.stderr)
        return 1

    pipe = Pipeline(backend=backend)
    all_diffs: List[str] = []
    samples = expected.get("samples", {})

    with tempfile.TemporaryDirectory(prefix="golden_run_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for sample_name, exp_data in sorted(samples.items()):
            pdf_path = golden_dir / "samples" / sample_name
            if not pdf_path.is_file():
                all_diffs.append(f"{sample_name}: файл не найден в {pdf_path}")
                print(f"FAIL {sample_name}: файл не найден")
                continue

            segs = pipe.analyze_pdf_segments(
                str(pdf_path),
                str(tmp_path / Path(sample_name).stem),
                max_pages=10,
                image_ext=".png",
                large_pdf_policy="partial",
            )
            act_pages = [serialize_page_result(p) for _, sp in segs for p in sp]
            diffs = _compare_pages(sample_name, exp_data.get("pages", []), act_pages, bbox_tol=bbox_tol)
            if diffs:
                all_diffs.extend(diffs)
                print(f"FAIL {sample_name}: {len(diffs)} расхождений")
                for d in diffs[:5]:
                    print(f"  - {d}")
            else:
                print(f"PASS {sample_name} ({len(act_pages)} стр.)")

    if all_diffs:
        print(f"\nИТОГ: FAIL (всего расхождений: {len(all_diffs)})", file=sys.stderr)
        return 1
    print(f"\nИТОГ: PASS (проверено {len(samples)} PDF, weights_version={actual_weights})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка контрольного набора golden_dataset")
    parser.add_argument("--golden-dir", type=str, default="golden_dataset")
    parser.add_argument("--mode", choices=("local", "http"), default="local")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--bbox-tol", type=float, default=2.0)
    args = parser.parse_args()
    sys.exit(
        verify_golden(
            Path(args.golden_dir),
            mode=args.mode,
            base_url=args.base_url,
            api_key=args.api_key,
            bbox_tol=args.bbox_tol,
        )
    )


if __name__ == "__main__":
    main()
