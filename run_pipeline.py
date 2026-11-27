import os
from pathlib import Path
import shutil

from pipeline import Pipeline


def main() -> None:
    """Пакетно запускает проверку всех PDF из Test_PDFs с выгрузкой в Результаты."""
    print("=== ЭТАП 1: Обработка тестовых МЛ (с формированием папки Результаты) ===")
    results_dir = Path("Результаты")
    results_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("Ошибки", "Архив"):
        sub_path = results_dir / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)
    report_file = results_dir / "report.txt"
    if report_file.exists():
        report_file.unlink()

    p = Pipeline("Новые веса/best_clean.pt", digit_classifier_path="Новые веса/best_digit_classifier.pt")

    test_files = sorted(f for f in os.listdir("Test_PDFs") if f.lower().endswith(".pdf"))
    for f in test_files:
        print(f"Тестовый файл: {f}")
        p.process_pdf(os.path.join("Test_PDFs", f), "Результаты", generate_visuals=True)

    print("\nТестовые МЛ успешно обработаны!\n")


if __name__ == "__main__":
    main()
