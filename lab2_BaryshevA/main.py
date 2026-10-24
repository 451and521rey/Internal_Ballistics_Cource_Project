"""Лабораторная работа № 2: ограниченная оптимизация SciPy и Optuna.

Файл остаётся самостоятельным скриптом. Команда ``check`` выполняет быстрые
самопроверки и не запускает вычислительный эксперимент.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


# Контракт задачи из Sources/01_assignment.md.
X_BOUNDS = (-2.5, 2.0)
Y_BOUNDS = (-1.5, 3.0)
REGION_VERTICES = ((2.0, 3.0), (2.0, -1.5), (-2.5, 3.0))
FEASIBILITY_TOLERANCE = 1e-6
DEFAULT_BUDGET = 600
DEFAULT_RHO = 100.0


class BudgetExceeded(RuntimeError):
    """Штатный сигнал остановки до вычисления сверх заданного бюджета."""


@dataclass
class ScalarBudget:
    """Минимальный проверяемый счётчик фактических скалярных вызовов f."""

    limit: int
    calls: int = 0

    def evaluate(self, function: Callable[..., float], *args: object) -> float:
        if self.calls >= self.limit:
            raise BudgetExceeded(
                f"Бюджет {self.limit} исчерпан до следующего вычисления f"
            )
        self.calls += 1
        return float(function(*args))


def _point_xy(point: Sequence[float]) -> tuple[float, float]:
    """Преобразовать точку к паре конечных скаляров."""

    if len(point) != 2:
        raise ValueError("Точка должна содержать ровно две координаты")
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("Координаты точки должны быть конечными")
    return x, y


def raw_function(point: Sequence[float], a: float = 1.0) -> float:
    """Вычислить f_a(x, y) для одной точки; косинусы получают радианы."""

    x, y = _point_xy(point)
    return x * x + y * y + float(a) * (
        2.0 - math.cos(4.0 * x) - math.cos(4.0 * y)
    )


def raw_gradient(point: Sequence[float], a: float = 1.0) -> np.ndarray:
    """Аналитический градиент f_a без дополнительных вычислений f."""

    x, y = _point_xy(point)
    a = float(a)
    return np.array(
        [2.0 * x + 4.0 * a * math.sin(4.0 * x),
         2.0 * y + 4.0 * a * math.sin(4.0 * y)],
        dtype=float,
    )


def normalized_residuals(point: Sequence[float]) -> tuple[float, float, float]:
    """Вернуть g1, g2, g3; допустимая точка имеет gi <= 0 до допуска."""

    x, y = _point_xy(point)
    return (x - 2.0) / 4.5, (y - 3.0) / 4.5, (0.5 - x - y) / 9.0


def violation(point: Sequence[float]) -> float:
    """Максимальное положительное нормированное нарушение."""

    return max(0.0, *normalized_residuals(point))


def is_feasible(
    point: Sequence[float], tolerance: float = FEASIBILITY_TOLERANCE
) -> bool:
    """Проверить исходные ограничения с численным допуском задания."""

    return violation(point) <= tolerance


def penalty(point: Sequence[float]) -> float:
    """Общий квадратичный штраф P для SLSQP-P и TPE-P."""

    return sum(max(0.0, residual) ** 2 for residual in normalized_residuals(point))


def penalty_gradient(point: Sequence[float]) -> np.ndarray:
    """Аналитический градиент общего квадратичного штрафа."""

    g1, g2, g3 = normalized_residuals(point)
    gradient = np.zeros(2, dtype=float)
    if g1 > 0.0:
        gradient += 2.0 * g1 * np.array([1.0 / 4.5, 0.0])
    if g2 > 0.0:
        gradient += 2.0 * g2 * np.array([0.0, 1.0 / 4.5])
    if g3 > 0.0:
        gradient += 2.0 * g3 * np.array([-1.0 / 9.0, -1.0 / 9.0])
    return gradient


def direction_sign(direction: str) -> int:
    """Вернуть +1 для min и -1 для max в минимизируемой цели."""

    if direction == "min":
        return 1
    if direction == "max":
        return -1
    raise ValueError("Направление должно быть 'min' или 'max'")


def objective(
    point: Sequence[float],
    *,
    a: float,
    direction: str,
    rho: float | None = None,
) -> float:
    """Вычислить J_s=s*f либо штрафованную F_s=s*f+rho*P."""

    value = direction_sign(direction) * raw_function(point, a)
    if rho is not None:
        value += float(rho) * penalty(point)
    return value


def objective_gradient(
    point: Sequence[float],
    *,
    a: float,
    direction: str,
    rho: float | None = None,
) -> np.ndarray:
    """Аналитический градиент J_s или F_s без вызова raw_function."""

    gradient = direction_sign(direction) * raw_gradient(point, a)
    if rho is not None:
        gradient = gradient + float(rho) * penalty_gradient(point)
    return gradient


def numerical_gradient(
    function: Callable[[Sequence[float]], float],
    point: Sequence[float],
    step: float = 1e-6,
) -> np.ndarray:
    """Центральная разность только для отдельной диагностической проверки."""

    center = np.asarray(_point_xy(point), dtype=float)
    result = np.empty(2, dtype=float)
    for index in range(2):
        offset = np.zeros(2, dtype=float)
        offset[index] = step
        result[index] = (function(center + offset) - function(center - offset)) / (
            2.0 * step
        )
    return result


def analytical_control() -> dict[str, dict[str, object]]:
    """Вернуть аналитические экстремумы контрольной задачи a=0."""

    minimum = (0.25, 0.25)
    maximum = (-2.5, 3.0)
    return {
        "min": {"point": minimum, "f_raw": raw_function(minimum, a=0.0)},
        "max": {"point": maximum, "f_raw": raw_function(maximum, a=0.0)},
    }


def _assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: ожидалось {expected!r}, получено {actual!r}")


def run_checks() -> None:
    """Быстро проверить контракт без запуска оптимизаторов."""

    checks: list[tuple[str, str]] = []

    control = analytical_control()
    _assert_close(float(control["min"]["f_raw"]), 0.125, 1e-15, "control min")
    _assert_close(float(control["max"]["f_raw"]), 15.25, 1e-15, "control max")
    if not all(is_feasible(data["point"]) for data in control.values()):
        raise AssertionError("Аналитические контрольные точки должны быть допустимыми")
    if is_feasible((0.0, 0.0)):
        raise AssertionError("Точка (0, 0) нарушает x+y>=0.5")
    checks.append(("аналитика a=0", "min=(0.25,0.25), max=(-2.5,3)"))

    for vertex in REGION_VERTICES:
        if not is_feasible(vertex, tolerance=1e-14):
            raise AssertionError(f"Вершина области ошибочно недопустима: {vertex}")
    checks.append(("область", "3 вершины и направления ограничений"))

    gradient_errors = []
    for point in ((0.25, 0.25), (-1.0, 2.0), (1.3, -0.4)):
        analytic = raw_gradient(point, a=1.0)
        numeric = numerical_gradient(lambda z: raw_function(z, a=1.0), point)
        gradient_errors.append(float(np.max(np.abs(analytic - numeric))))
    max_gradient_error = max(gradient_errors)
    if max_gradient_error > 2e-8:
        raise AssertionError(f"Ошибка градиента слишком велика: {max_gradient_error}")
    checks.append(("градиент f", f"max|error|={max_gradient_error:.3e}"))

    penalized_point = (-1.0, 1.0)
    expected_penalty = (0.5 / 9.0) ** 2
    _assert_close(penalty(penalized_point), expected_penalty, 1e-15, "penalty")
    analytic_penalty_gradient = penalty_gradient(penalized_point)
    numeric_penalty_gradient = numerical_gradient(penalty, penalized_point)
    penalty_gradient_error = float(
        np.max(np.abs(analytic_penalty_gradient - numeric_penalty_gradient))
    )
    if penalty_gradient_error > 1e-10:
        raise AssertionError(
            f"Ошибка градиента штрафа слишком велика: {penalty_gradient_error}"
        )
    checks.append(("штраф P", f"P(-1,1)={expected_penalty:.12f}"))

    point = (0.7, 0.2)
    raw = raw_function(point, a=1.0)
    _assert_close(objective(point, a=1.0, direction="min"), raw, 1e-15, "min sign")
    _assert_close(objective(point, a=1.0, direction="max"), -raw, 1e-15, "max sign")
    checks.append(("знак цели", "J_min=f, J_max=-f"))

    budget = ScalarBudget(limit=2)
    budget.evaluate(raw_function, (0.25, 0.25), 0.0)
    budget.evaluate(raw_function, (2.0, 3.0), 0.0)
    try:
        budget.evaluate(raw_function, (-2.5, 3.0), 0.0)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("Счётчик не остановил вычисление сверх бюджета")
    if budget.calls != 2:
        raise AssertionError("Запрещённая попытка не должна увеличивать счётчик")
    checks.append(("бюджет", "остановка до вызова f №3 при B=2"))

    print("ЛР2 — быстрый контроль вычислительного контракта")
    print("=" * 59)
    for label, detail in checks:
        print(f"[OK] {label:<16} {detail}")
    print(f"Итог: {len(checks)} проверок пройдено; оптимизаторы не запускались.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ЛР2: воспроизводимое сравнение методов оптимизации"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("check",),
        default="check",
        help="check — быстрые самопроверки без полного эксперимента",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check":
        run_checks()


if __name__ == "__main__":
    main()
