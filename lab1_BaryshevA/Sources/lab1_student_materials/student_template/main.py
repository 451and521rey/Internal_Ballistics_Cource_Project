"""Точка входа лабораторной работы № 1.

Запуск:
    uv run python main.py

После выполнения в figures/ должны находиться все рисунки report.md.
Шаблон не задаёт архитектуру расчётной части: её выбирает автор работы.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
RANDOM_SEED = 20260921


def configure_plots() -> None:
    """Единый читаемый стиль авторских графиков."""

    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.6),
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_template_check() -> Path:
    """Создать нейтральный рисунок для проверки конвейера файлов.

    Рисунок не относится к баллистической задаче. После реализации собственной
    программы удалите эту функцию и ссылку на рисунок из отчёта.
    """

    x = np.linspace(0.0, 1.0, 80)
    fig, ax = plt.subplots()
    ax.plot(x, x, color="#2f5d8a", linewidth=2, label="тестовая линия")
    ax.set_xlabel("Безразмерная координата")
    ax.set_ylabel("Безразмерная величина")
    ax.set_title("Проверка воспроизводимого построения рисунка")
    ax.legend()
    fig.tight_layout()
    target = FIGURES_DIR / "template_check.png"
    fig.savefig(target)
    plt.close(fig)
    return target


def build_all_figures() -> list[Path]:
    """Выполнить исследование и пересоздать все рисунки отчёта."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: самостоятельно организуйте прямой расчёт, проверку критерия и
    # ограничений, применение штрафа, выбор исследуемых решений и построение
    # рисунков. Если используется случайность, зафиксируйте seed.

    return [save_template_check()]


def main() -> None:
    configure_plots()
    print(f"pyballistics={version('pyballistics')}")
    print(f"random_seed={RANDOM_SEED}")
    created = build_all_figures()
    for path in created:
        print(f"created={path.relative_to(ROOT)}")
    print("Шаблон запущен. Замените тестовый рисунок результатами исследования.")


if __name__ == "__main__":
    main()

