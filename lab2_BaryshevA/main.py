"""Лабораторная работа № 2: ограниченная оптимизация SciPy и Optuna.

Файл остаётся самостоятельным скриптом. Команда ``check`` выполняет быстрые
самопроверки и не запускает вычислительный эксперимент.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import optuna
import scipy
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult
from scipy.optimize import differential_evolution, minimize


# Контракт задачи из Sources/01_assignment.md.
X_BOUNDS = (-2.5, 2.0)
Y_BOUNDS = (-1.5, 3.0)
REGION_VERTICES = ((2.0, 3.0), (2.0, -1.5), (-2.5, 3.0))
FEASIBILITY_TOLERANCE = 1e-6
DEFAULT_BUDGET = 600
DEFAULT_RHO = 100.0
RUN_SEEDS = (11, 23, 37, 51, 71)
SLSQP_STARTS = (
    (-1.0, 2.0),
    (0.25, 0.25),
    (1.5, 2.5),
    (-2.0, 2.75),
    (1.0, -0.25),
)
SCIPY_METHODS = ("SLSQP-C", "DE-C", "SLSQP-P")
ALL_METHODS = ("SLSQP-C", "DE-C", "SLSQP-P", "TPE-P")
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

EVALUATION_FIELDS = (
    "run_key",
    "phase",
    "method",
    "direction",
    "a",
    "rho",
    "run_id",
    "seed",
    "eval_id",
    "x",
    "y",
    "f_raw",
    "objective",
    "g1",
    "g2",
    "g3",
    "violation",
    "penalty",
    "feasible",
    "elapsed_s",
)
RUN_FIELDS = (
    "run_key",
    "phase",
    "method",
    "direction",
    "a",
    "rho",
    "run_id",
    "seed",
    "start_x",
    "start_y",
    "budget",
    "settings",
    "n_eval",
    "n_grad",
    "n_constraint",
    "n_iteration",
    "elapsed_s",
    "stop_reason",
    "optimizer_success",
    "message",
    "recommendation_x",
    "recommendation_y",
    "recommendation_f_raw",
    "recommendation_objective",
    "recommendation_g1",
    "recommendation_g2",
    "recommendation_g3",
    "recommendation_violation",
    "recommendation_penalty",
    "recommendation_feasible",
    "best_eval_id",
    "best_x",
    "best_y",
    "best_f_raw",
    "best_objective",
    "best_g1",
    "best_g2",
    "best_g3",
    "best_violation",
    "best_penalty",
)
ITERATION_FIELDS = (
    "run_key",
    "phase",
    "method",
    "direction",
    "a",
    "rho",
    "run_id",
    "iteration_id",
    "eval_id_at_callback",
    "x",
    "y",
    "elapsed_s",
)
OPTUNA_TRIAL_FIELDS = (
    "run_key",
    "phase",
    "method",
    "direction",
    "a",
    "rho",
    "run_id",
    "seed",
    "trial_number",
    "eval_id",
    "state",
    "x",
    "y",
    "value",
)


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


@dataclass(frozen=True)
class RunSpec:
    """Полная идентификация одного независимого запуска."""

    phase: str
    method: str
    direction: str
    a: float
    rho: float | None
    run_id: int
    seed: int | None = None
    start: tuple[float, float] | None = None

    @property
    def run_key(self) -> str:
        rho_tag = "none" if self.rho is None else f"{self.rho:g}"
        return (
            f"{self.phase}__{self.direction}__{self.method.lower()}"
            f"__a{self.a:g}__rho{rho_tag}__r{self.run_id}"
        )

    def validate(self) -> None:
        if self.phase not in {"control", "main", "penalty"}:
            raise ValueError(f"Неизвестная фаза: {self.phase}")
        direction_sign(self.direction)
        if self.method.endswith("-P") and self.rho is None:
            raise ValueError(f"Для {self.method} требуется rho")
        if self.method.endswith("-C") and self.rho is not None:
            raise ValueError(f"Для {self.method} rho должен быть пустым")
        if self.method.startswith("SLSQP") and self.start is None:
            raise ValueError(f"Для {self.method} требуется начальная точка")
        if self.method == "DE-C" and self.seed is None:
            raise ValueError("Для DE-C требуется seed")
        if self.method == "TPE-P" and self.seed is None:
            raise ValueError("Для TPE-P требуется seed")


@dataclass
class BudgetedObjective:
    """Цель одного запуска со строгим cap и полной историей реальных вызовов f."""

    spec: RunSpec
    budget: int = DEFAULT_BUDGET
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    n_grad: int = 0
    n_constraint: int = 0
    _started_at: float = field(default_factory=time.perf_counter)

    def evaluate(self, point: Sequence[float]) -> float:
        """Посчитать f ровно один раз и сохранить все производные величины."""

        if len(self.evaluations) >= self.budget:
            raise BudgetExceeded(
                f"{self.spec.run_key}: бюджет B={self.budget} исчерпан"
            )

        # Копируем координаты до передачи в историю: SciPy меняет рабочие массивы.
        x, y = _point_xy(point)
        f_value = raw_function((x, y), self.spec.a)
        residuals = normalized_residuals((x, y))
        point_penalty = sum(max(0.0, value) ** 2 for value in residuals)
        point_violation = max(0.0, *residuals)
        target = direction_sign(self.spec.direction) * f_value
        if self.spec.rho is not None:
            target += self.spec.rho * point_penalty

        self.evaluations.append(
            {
                "run_key": self.spec.run_key,
                "phase": self.spec.phase,
                "method": self.spec.method,
                "direction": self.spec.direction,
                "a": self.spec.a,
                "rho": self.spec.rho,
                "run_id": self.spec.run_id,
                "seed": self.spec.seed,
                "eval_id": len(self.evaluations) + 1,
                "x": x,
                "y": y,
                "f_raw": f_value,
                "objective": target,
                "g1": residuals[0],
                "g2": residuals[1],
                "g3": residuals[2],
                "violation": point_violation,
                "penalty": point_penalty,
                "feasible": point_violation <= FEASIBILITY_TOLERANCE,
                "elapsed_s": time.perf_counter() - self._started_at,
            }
        )
        return target

    def gradient(self, point: Sequence[float]) -> np.ndarray:
        """Учесть аналитический градиент отдельно; f здесь не вызывается."""

        self.n_grad += 1
        return objective_gradient(
            point,
            a=self.spec.a,
            direction=self.spec.direction,
            rho=self.spec.rho,
        )

    def explicit_constraint(self, point: Sequence[float]) -> float:
        """Явное ограничение SLSQP-C в принятой SciPy форме c(x)>=0."""

        self.n_constraint += 1
        x, y = _point_xy(point)
        return x + y - 0.5

    def iteration_callback(self, point: Sequence[float]) -> None:
        """Сохранить принятую итерацию без дополнительного вычисления f."""

        x, y = _point_xy(point)
        self.iterations.append(
            {
                "run_key": self.spec.run_key,
                "phase": self.spec.phase,
                "method": self.spec.method,
                "direction": self.spec.direction,
                "a": self.spec.a,
                "rho": self.spec.rho,
                "run_id": self.spec.run_id,
                "iteration_id": len(self.iterations) + 1,
                "eval_id_at_callback": len(self.evaluations),
                "x": x,
                "y": y,
                "elapsed_s": time.perf_counter() - self._started_at,
            }
        )


def _matching_evaluation(
    evaluations: Sequence[dict[str, Any]], point: Sequence[float]
) -> dict[str, Any] | None:
    """Найти реально вычисленную запись рекомендации без повторного вызова f."""

    x, y = _point_xy(point)
    matches = [
        record
        for record in evaluations
        if max(abs(float(record["x"]) - x), abs(float(record["y"]) - y)) <= 1e-10
    ]
    return matches[-1] if matches else None


def _best_feasible_evaluation(
    evaluations: Sequence[dict[str, Any]], direction: str
) -> dict[str, Any] | None:
    """Выбрать по исходной f лучшую допустимую запись всей истории."""

    feasible = [record for record in evaluations if bool(record["feasible"])]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda record: direction_sign(direction) * float(record["f_raw"]),
    )


def _optimizer_settings(spec: RunSpec) -> dict[str, Any]:
    if spec.method.startswith("SLSQP"):
        return {"method": "SLSQP", "maxiter": 10000, "ftol": 1e-9}
    if spec.method == "DE-C":
        return {
            "method": "differential_evolution",
            "popsize": 15,
            "maxiter": 10000,
            "tol": 0.0,
            "atol": 0.0,
            "polish": False,
            "workers": 1,
            "updating": "immediate",
        }
    if spec.method == "TPE-P":
        return {
            "method": "TPESampler",
            "n_startup_trials": 20,
            "n_jobs": 1,
            "pruning": False,
            "x_distribution": [X_BOUNDS[0], X_BOUNDS[1]],
            "y_distribution": [Y_BOUNDS[0], Y_BOUNDS[1]],
        }
    raise ValueError(f"Нет настроек для метода {spec.method}")


def _result_row(
    evaluator: BudgetedObjective,
    result: OptimizeResult | None,
    *,
    stop_reason: str,
    message: str,
) -> dict[str, Any]:
    """Сформировать сводку, не выполняя диагностических вызовов f."""

    spec = evaluator.spec
    recommendation = None
    recommendation_point: tuple[float, float] | None = None
    if result is not None and getattr(result, "x", None) is not None:
        recommendation_point = _point_xy(result.x)
        recommendation = _matching_evaluation(evaluator.evaluations, result.x)
    best = _best_feasible_evaluation(evaluator.evaluations, spec.direction)

    row: dict[str, Any] = {
        "run_key": spec.run_key,
        "phase": spec.phase,
        "method": spec.method,
        "direction": spec.direction,
        "a": spec.a,
        "rho": spec.rho,
        "run_id": spec.run_id,
        "seed": spec.seed,
        "start_x": None if spec.start is None else spec.start[0],
        "start_y": None if spec.start is None else spec.start[1],
        "budget": evaluator.budget,
        "settings": json.dumps(
            _optimizer_settings(spec), ensure_ascii=False, sort_keys=True
        ),
        "n_eval": len(evaluator.evaluations),
        "n_grad": evaluator.n_grad if spec.method.startswith("SLSQP") else None,
        "n_constraint": evaluator.n_constraint if spec.method == "SLSQP-C" else None,
        "n_iteration": len(evaluator.iterations)
        if spec.method.startswith("SLSQP")
        else (getattr(result, "nit", None) if result is not None else None),
        "elapsed_s": time.perf_counter() - evaluator._started_at,
        "stop_reason": stop_reason,
        "optimizer_success": None if result is None else bool(result.success),
        "message": message,
        "recommendation_x": None
        if recommendation_point is None
        else recommendation_point[0],
        "recommendation_y": None
        if recommendation_point is None
        else recommendation_point[1],
    }

    metric_names = (
        "f_raw",
        "objective",
        "g1",
        "g2",
        "g3",
        "violation",
        "penalty",
        "feasible",
    )
    for metric in metric_names:
        row[f"recommendation_{metric}"] = (
            None if recommendation is None else recommendation[metric]
        )

    if best is None:
        row.update(
            {
                "best_eval_id": None,
                "best_x": None,
                "best_y": None,
                "best_f_raw": None,
                "best_objective": None,
                "best_g1": None,
                "best_g2": None,
                "best_g3": None,
                "best_violation": None,
                "best_penalty": None,
            }
        )
    else:
        row.update(
            {
                "best_eval_id": best["eval_id"],
                "best_x": best["x"],
                "best_y": best["y"],
                "best_f_raw": best["f_raw"],
                "best_objective": best["objective"],
                "best_g1": best["g1"],
                "best_g2": best["g2"],
                "best_g3": best["g3"],
                "best_violation": best["violation"],
                "best_penalty": best["penalty"],
            }
        )
    return row


def run_scipy_method(
    spec: RunSpec, budget: int = DEFAULT_BUDGET
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Выполнить один независимый запуск обязательного метода SciPy."""

    spec.validate()
    if spec.method not in SCIPY_METHODS:
        raise ValueError(f"Метод {spec.method} не является методом SciPy этой работы")
    evaluator = BudgetedObjective(spec=spec, budget=budget)
    result: OptimizeResult | None = None
    stop_reason = "optimizer_stopped"
    message = ""

    try:
        if spec.method == "SLSQP-C":
            constraint = {
                "type": "ineq",
                "fun": evaluator.explicit_constraint,
                "jac": lambda _point: np.array([1.0, 1.0]),
            }
            result = minimize(
                evaluator.evaluate,
                np.asarray(spec.start, dtype=float),
                method="SLSQP",
                jac=evaluator.gradient,
                bounds=Bounds(
                    [X_BOUNDS[0], Y_BOUNDS[0]],
                    [X_BOUNDS[1], Y_BOUNDS[1]],
                ),
                constraints=(constraint,),
                callback=evaluator.iteration_callback,
                options={"maxiter": 10000, "ftol": 1e-9, "disp": False},
            )
        elif spec.method == "SLSQP-P":
            # Здесь намеренно нет явного x+y>=0.5: его учитывает только общий штраф.
            result = minimize(
                evaluator.evaluate,
                np.asarray(spec.start, dtype=float),
                method="SLSQP",
                jac=evaluator.gradient,
                bounds=Bounds(
                    [X_BOUNDS[0], Y_BOUNDS[0]],
                    [X_BOUNDS[1], Y_BOUNDS[1]],
                ),
                callback=evaluator.iteration_callback,
                options={"maxiter": 10000, "ftol": 1e-9, "disp": False},
            )
        else:
            linear_constraint = LinearConstraint([[1.0, 1.0]], [0.5], [np.inf])
            result = differential_evolution(
                evaluator.evaluate,
                bounds=(X_BOUNDS, Y_BOUNDS),
                constraints=(linear_constraint,),
                popsize=15,
                maxiter=10000,
                tol=0.0,
                atol=0.0,
                polish=False,
                workers=1,
                updating="immediate",
                rng=np.random.default_rng(spec.seed),
                disp=False,
            )

        stop_reason = "converged" if bool(result.success) else "optimizer_stopped"
        message = str(result.message)
    except BudgetExceeded as error:
        stop_reason = "budget_reached"
        message = str(error)

    run_row = _result_row(
        evaluator,
        result,
        stop_reason=stop_reason,
        message=message,
    )
    return run_row, evaluator.evaluations, evaluator.iterations


def run_optuna_method(
    spec: RunSpec, budget: int = DEFAULT_BUDGET
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Выполнить TPE-P: один trial равен одному реальному вычислению F."""

    spec.validate()
    if spec.method != "TPE-P":
        raise ValueError("run_optuna_method предназначен только для TPE-P")

    evaluator = BudgetedObjective(spec=spec, budget=budget)
    sampler = optuna.samplers.TPESampler(
        seed=spec.seed,
        n_startup_trials=20,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def trial_objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", X_BOUNDS[0], X_BOUNDS[1])
        y = trial.suggest_float("y", Y_BOUNDS[0], Y_BOUNDS[1])
        return evaluator.evaluate((x, y))

    study.optimize(
        trial_objective,
        n_trials=budget,
        n_jobs=1,
        show_progress_bar=False,
        gc_after_trial=False,
    )

    best_trial = study.best_trial
    result = OptimizeResult(
        x=np.array([best_trial.params["x"], best_trial.params["y"]]),
        fun=best_trial.value,
        success=True,
        message=f"Завершено {budget} последовательных trials без pruning",
    )
    run_row = _result_row(
        evaluator,
        result,
        stop_reason="budget_reached",
        message=str(result.message),
    )

    trial_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        evaluation = evaluator.evaluations[trial.number]
        trial_rows.append(
            {
                "run_key": spec.run_key,
                "phase": spec.phase,
                "method": spec.method,
                "direction": spec.direction,
                "a": spec.a,
                "rho": spec.rho,
                "run_id": spec.run_id,
                "seed": spec.seed,
                "trial_number": trial.number,
                "eval_id": evaluation["eval_id"],
                "state": trial.state.name,
                "x": trial.params["x"],
                "y": trial.params["y"],
                "value": trial.value,
            }
        )
    return run_row, evaluator.evaluations, trial_rows


def build_experiment_plan() -> list[RunSpec]:
    """Построить обязательное ядро из 68 запусков без декартова произведения."""

    plan: list[RunSpec] = []

    # Контроль a=0: 2 направления x 4 метода x один run_id = 8.
    for direction in ("min", "max"):
        for method in ALL_METHODS:
            is_penalty = method.endswith("-P")
            plan.append(
                RunSpec(
                    phase="control",
                    method=method,
                    direction=direction,
                    a=0.0,
                    rho=DEFAULT_RHO if is_penalty else None,
                    run_id=1,
                    seed=RUN_SEEDS[0] if method in {"DE-C", "TPE-P"} else None,
                    start=SLSQP_STARTS[0] if method.startswith("SLSQP") else None,
                )
            )

    # Основной опыт a=1: 2 x 4 x 5 = 40.
    for direction in ("min", "max"):
        for method in ALL_METHODS:
            for index, (seed, start) in enumerate(
                zip(RUN_SEEDS, SLSQP_STARTS, strict=True), start=1
            ):
                is_penalty = method.endswith("-P")
                plan.append(
                    RunSpec(
                        phase="main",
                        method=method,
                        direction=direction,
                        a=1.0,
                        rho=DEFAULT_RHO if is_penalty else None,
                        run_id=index,
                        seed=seed if method in {"DE-C", "TPE-P"} else None,
                        start=start if method.startswith("SLSQP") else None,
                    )
                )

    # Чувствительность штрафа: rho=1,10; rho=100 переиспользуется из main.
    for direction in ("min", "max"):
        for rho in (1.0, 10.0):
            for index, start in enumerate(SLSQP_STARTS, start=1):
                plan.append(
                    RunSpec(
                        phase="penalty",
                        method="SLSQP-P",
                        direction=direction,
                        a=1.0,
                        rho=rho,
                        run_id=index,
                        start=start,
                    )
                )

    if len(plan) != 68:
        raise AssertionError(f"План должен содержать 68 запусков, получено {len(plan)}")
    keys = [spec.run_key for spec in plan]
    if len(keys) != len(set(keys)):
        raise AssertionError("run_key в плане должны быть уникальны")
    for spec in plan:
        spec.validate()
    return plan


def _atomic_write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    """Полностью заменить CSV, не смешивая новый опыт со старым."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in fieldnames})
    temporary.replace(path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def build_config(plan: Sequence[RunSpec]) -> dict[str, Any]:
    """Сформировать проверяемую конфигурацию без времени запуска."""

    return {
        "schema_version": 1,
        "task": {
            "formula": "x^2 + y^2 + a * (2 - cos(4*x) - cos(4*y))",
            "main_a": 1.0,
            "control_a": 0.0,
            "directions": ["min", "max"],
            "constraints": ["x <= 2", "y <= 3", "x + y >= 0.5"],
            "bounds": {"x": list(X_BOUNDS), "y": list(Y_BOUNDS)},
            "residuals": ["(x-2)/4.5", "(y-3)/4.5", "(0.5-x-y)/9"],
            "feasibility_tolerance": FEASIBILITY_TOLERANCE,
            "penalty": "sum(max(0, g_i)^2)",
            "default_rho": DEFAULT_RHO,
        },
        "experiment": {
            "budget_per_run": DEFAULT_BUDGET,
            "cache_enabled": False,
            "parallelism": 1,
            "run_count": len(plan),
            "phase_counts": {
                phase: sum(spec.phase == phase for spec in plan)
                for phase in ("control", "main", "penalty")
            },
            "seeds": list(RUN_SEEDS),
            "slsqp_starts": [list(point) for point in SLSQP_STARTS],
            "method_settings": {
                method: _optimizer_settings(
                    next(spec for spec in plan if spec.method == method)
                )
                for method in ALL_METHODS
            },
            "run_plan": [asdict(spec) | {"run_key": spec.run_key} for spec in plan],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "optuna": optuna.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "environment": {
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "data_contract": {
            "empty_csv_field": "значение отсутствует/неприменимо",
            "diagnostics_in_optimizer_history": False,
            "repeat_run_behavior": "атомарная полная замена файлов основного журнала",
        },
    }


def save_experiment(
    plan: Sequence[RunSpec],
    runs: Sequence[dict[str, Any]],
    evaluations: Sequence[dict[str, Any]],
    iterations: Sequence[dict[str, Any]],
    trials: Sequence[dict[str, Any]],
) -> None:
    """Атомарно перезаписать обязательные журналы завершённого опыта."""

    if len(runs) != len(plan):
        raise AssertionError("Каждому элементу плана должна соответствовать строка runs")
    if any(int(row["n_eval"]) > DEFAULT_BUDGET for row in runs):
        raise AssertionError("Обнаружено превышение бюджета")
    if sum(int(row["n_eval"]) for row in runs) != len(evaluations):
        raise AssertionError("Число evaluations не совпадает с суммой n_eval")

    _atomic_write_json(RESULTS_DIR / "config.json", build_config(plan))
    _atomic_write_csv(RESULTS_DIR / "evaluations.csv", evaluations, EVALUATION_FIELDS)
    _atomic_write_csv(RESULTS_DIR / "runs.csv", runs, RUN_FIELDS)
    _atomic_write_csv(RESULTS_DIR / "iterations.csv", iterations, ITERATION_FIELDS)
    _atomic_write_csv(
        RESULTS_DIR / "optuna_trials.csv", trials, OPTUNA_TRIAL_FIELDS
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_saved_results() -> dict[str, int]:
    """Независимо перечитать журналы и проверить связи и бюджет."""

    runs = _read_csv(RESULTS_DIR / "runs.csv")
    evaluations = _read_csv(RESULTS_DIR / "evaluations.csv")
    iterations = _read_csv(RESULTS_DIR / "iterations.csv")
    trials = _read_csv(RESULTS_DIR / "optuna_trials.csv")
    with (RESULTS_DIR / "config.json").open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    if len(runs) != 68 or int(config["experiment"]["run_count"]) != 68:
        raise AssertionError("В сохранённом опыте должно быть 68 запусков")
    run_keys = [row["run_key"] for row in runs]
    if len(run_keys) != len(set(run_keys)):
        raise AssertionError("В runs.csv обнаружены повторные run_key")

    evaluations_by_run: dict[str, list[dict[str, str]]] = {
        run_key: [] for run_key in run_keys
    }
    for evaluation in evaluations:
        if evaluation["run_key"] not in evaluations_by_run:
            raise AssertionError("evaluation ссылается на неизвестный run_key")
        evaluations_by_run[evaluation["run_key"]].append(evaluation)

    for run in runs:
        records = evaluations_by_run[run["run_key"]]
        n_eval = int(run["n_eval"])
        if n_eval != len(records) or n_eval > DEFAULT_BUDGET:
            raise AssertionError(f"Неверный n_eval для {run['run_key']}")
        if [int(record["eval_id"]) for record in records] != list(
            range(1, n_eval + 1)
        ):
            raise AssertionError(f"Нарушена нумерация eval_id: {run['run_key']}")

        feasible_records = []
        for record in records:
            expected_feasible = (
                float(record["violation"]) <= FEASIBILITY_TOLERANCE
            )
            if (record["feasible"] == "True") != expected_feasible:
                raise AssertionError(f"Неверный feasible: {run['run_key']}")
            if expected_feasible:
                feasible_records.append(record)

        if run["best_eval_id"] == "":
            if feasible_records:
                raise AssertionError(f"Потеряна лучшая допустимая: {run['run_key']}")
        else:
            best_id = int(run["best_eval_id"])
            best_record = records[best_id - 1]
            if best_record["feasible"] != "True":
                raise AssertionError(f"best_eval_id недопустим: {run['run_key']}")
            sign = direction_sign(run["direction"])
            expected = min(
                feasible_records,
                key=lambda record: sign * float(record["f_raw"]),
            )
            if int(expected["eval_id"]) != best_id:
                raise AssertionError(f"Неверно выбрана лучшая точка: {run['run_key']}")
            _assert_close(
                float(run["best_f_raw"]),
                float(best_record["f_raw"]),
                0.0,
                f"best_f_raw {run['run_key']}",
            )

        if run["recommendation_f_raw"] != "":
            recommendation_matches = [
                record
                for record in records
                if max(
                    abs(float(record["x"]) - float(run["recommendation_x"])),
                    abs(float(record["y"]) - float(run["recommendation_y"])),
                )
                <= 1e-10
            ]
            if not recommendation_matches:
                raise AssertionError(
                    f"Рекомендации нет среди реальных evaluations: {run['run_key']}"
                )

    for iteration in iterations:
        run_key = iteration["run_key"]
        if run_key not in evaluations_by_run or not iteration["method"].startswith(
            "SLSQP"
        ):
            raise AssertionError("Некорректная строка iterations.csv")
        if int(iteration["eval_id_at_callback"]) > len(evaluations_by_run[run_key]):
            raise AssertionError(f"Итерация ссылается за пределы истории: {run_key}")

    tpe_evaluation_count = sum(
        len(records)
        for run_key, records in evaluations_by_run.items()
        if "__tpe-p__" in run_key
    )
    if len(trials) != tpe_evaluation_count:
        raise AssertionError("Число Optuna trials не совпадает с evaluations TPE-P")
    if any(trial["state"] != "COMPLETE" for trial in trials):
        raise AssertionError("Все обязательные Optuna trials должны быть COMPLETE")

    return {
        "runs": len(runs),
        "evaluations": len(evaluations),
        "iterations": len(iterations),
        "optuna_trials": len(trials),
    }


def _format_optional(value: object, digits: int = 8) -> str:
    if value is None or value == "":
        return "не найдено"
    return f"{float(value):.{digits}f}"


def print_saved_summary() -> None:
    """Напечатать детерминированную сводку из runs.csv без нового поиска."""

    path = RESULTS_DIR / "runs.csv"
    if not path.exists():
        raise SystemExit("Нет results/runs.csv. Сначала выполните: uv run python main.py run")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    main_rows = [row for row in rows if row["phase"] == "main"]
    print("ЛР2 — сводка основного эксперимента a=1")
    print("=" * 78)
    print("Метод     Напр.  Допустимо  Лучшее f       Медиана f     Медиана n_eval")
    for direction in ("min", "max"):
        for method in ALL_METHODS:
            group = [
                row
                for row in main_rows
                if row["direction"] == direction and row["method"] == method
            ]
            feasible_values = [
                float(row["best_f_raw"]) for row in group if row["best_f_raw"] != ""
            ]
            best_value = (
                min(feasible_values)
                if direction == "min" and feasible_values
                else max(feasible_values) if feasible_values else None
            )
            median_value = (
                statistics.median(feasible_values) if feasible_values else None
            )
            median_cost = statistics.median(int(row["n_eval"]) for row in group)
            print(
                f"{method:<10}{direction:<7}{len(feasible_values):>2}/{len(group):<7}"
                f"{_format_optional(best_value):>15}"
                f"{_format_optional(median_value):>15}{median_cost:>15.1f}"
            )

    control = analytical_control()
    print("\nКонтроль a=0 (абсолютная ошибка лучшего допустимого f):")
    for direction in ("min", "max"):
        expected = float(control[direction]["f_raw"])
        group = [
            row
            for row in rows
            if row["phase"] == "control" and row["direction"] == direction
        ]
        details = []
        for row in group:
            error = (
                None
                if row["best_f_raw"] == ""
                else abs(float(row["best_f_raw"]) - expected)
            )
            details.append(f"{row['method']}={_format_optional(error, 3)}")
        print(f"  {direction}: " + "; ".join(details))


def execute_experiment() -> None:
    """Выполнить 68 запусков и заменить журналы только после полного успеха."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    plan = build_experiment_plan()
    all_runs: list[dict[str, Any]] = []
    all_evaluations: list[dict[str, Any]] = []
    all_iterations: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []

    print(f"ЛР2 — запуск обязательного эксперимента: {len(plan)} независимых запусков")
    for number, spec in enumerate(plan, start=1):
        if spec.method == "TPE-P":
            run, evaluations, trials = run_optuna_method(spec)
            iterations: list[dict[str, Any]] = []
        else:
            run, evaluations, iterations = run_scipy_method(spec)
            trials = []
        all_runs.append(run)
        all_evaluations.extend(evaluations)
        all_iterations.extend(iterations)
        all_trials.extend(trials)
        best = _format_optional(run["best_f_raw"])
        print(
            f"[{number:02d}/{len(plan)}] {spec.run_key:<48} "
            f"n_eval={run['n_eval']:>3}; best f={best}; stop={run['stop_reason']}"
        )

    save_experiment(
        plan,
        all_runs,
        all_evaluations,
        all_iterations,
        all_trials,
    )
    validated = validate_saved_results()
    print(
        "\nЖурналы полностью заменены и проверены: "
        f"runs={validated['runs']}, evaluations={validated['evaluations']}, "
        f"iterations={validated['iterations']}, "
        f"Optuna trials={validated['optuna_trials']}."
    )
    print_saved_summary()


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

    plan = build_experiment_plan()
    phase_counts = {
        phase: sum(spec.phase == phase for spec in plan)
        for phase in ("control", "main", "penalty")
    }
    if phase_counts != {"control": 8, "main": 40, "penalty": 20}:
        raise AssertionError(f"Ошибочный состав плана: {phase_counts}")
    checks.append(("план запусков", "8 control + 40 main + 20 penalty = 68"))

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
        choices=("check", "run", "summary"),
        default="summary",
        help=(
            "check — самопроверки; run — 68 запусков с заменой журналов; "
            "summary — сводка сохранённых результатов"
        ),
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if args.command == "check":
        run_checks()
    elif args.command == "run":
        execute_experiment()
    elif args.command == "summary":
        print_saved_summary()


if __name__ == "__main__":
    main()
