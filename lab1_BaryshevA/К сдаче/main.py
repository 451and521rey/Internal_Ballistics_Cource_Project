"""Безопасный запуск нульмерной модели внутренней баллистики.

Запуск из корня лабораторной работы:

    uv run python main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import pi
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyballistics import ozvb_termo

# Исходные данные варианта № 36 (СИ)
VARIANT_NUMBER = 36
CALIBER_M = 0.085
PROJECTILE_MASS_KG = 9.54
TARGET_MUZZLE_VELOCITY_M_S = 650.0
GUN_TYPE = "нарезное"
BALLISTIC_MODEL = "нульмерная термодинамическая модель"
MAX_PRESSURE_PA = 330.0e6
MAX_BARREL_LENGTH_CALIBERS = 60.0
MAX_BARREL_LENGTH_M = MAX_BARREL_LENGTH_CALIBERS * CALIBER_M
MIN_MUZZLE_VELOCITY_COLD_M_S = 570.0
MAX_MUZZLE_PRESSURE_HOT_PA = 170.0e6
CRITERION_NAME = "W_pm"

# Общие условия и базовое решение прямой задачи (СИ)
P_IGN_0_PA = 5.0e6
FORCING_PRESSURE_PA = 30.0e6
NORMAL_TEMPERATURE_K = 293.15
COLD_TEMPERATURE_K = 223.15
HOT_TEMPERATURE_K = 323.15
PHI_1 = 1.02
BORE_AREA_M2 = 0.26 * pi * CALIBER_M**2
POWDER_NAME = "8/1 тр"
BASE_POWDER_MASS_KG = 1.350
BASE_CHAMBER_VOLUME_M3 = 4.000e-3
BASE_BARREL_LENGTH_M = MAX_BARREL_LENGTH_M

# Численные параметры решателя
INTEGRATION_STEP_S = 5.0e-6
INTEGRATION_METHOD = "rk4"
MAX_INTEGRATION_STEPS = 200_000
NUMERICAL_PRESSURE_CUTOFF_PA = 1.0e9
REQUIRED_TRAJECTORY_FIELDS = ("t", "x_p", "v_p", "p_m")
PENALTY_SCALE = 100.0
ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"

CalculationStatus = Literal["успех", "ошибка"]
EvaluationStatus = Literal["рассчитано", "ошибка"]
ConstraintRelation = Literal["<=", ">="]
FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class BallisticsResult:
    """Результат безопасного запуска, не содержащий проверки ограничений."""

    status: CalculationStatus
    message: str
    stop_reason: str | None = None
    muzzle_velocity_m_s: float | None = None
    max_pressure_pa: float | None = None
    muzzle_pressure_pa: float | None = None
    execution_time_s: float | None = None
    trajectory: dict[str, FloatArray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Результат проверки одного ограничения в исходной размерности."""

    code: str
    description: str
    value: float
    limit: float
    relation: ConstraintRelation
    unit: str
    normalized_violation: float

    @property
    def is_satisfied(self) -> bool:
        """Вернуть признак выполнения ограничения."""

        return self.normalized_violation == 0.0


@dataclass(slots=True)
class SolutionEvaluation:
    """Расчёты при трёх температурах и отдельные показатели допустимости."""

    status: EvaluationStatus
    message: str
    criterion_w_pm_m3: float | None
    calculations: dict[str, BallisticsResult] = field(default_factory=dict)
    constraints: tuple[ConstraintResult, ...] = ()
    is_feasible: bool | None = None
    penalty_m3: float | None = None
    penalized_criterion_m3: float | None = None


@dataclass(frozen=True, slots=True)
class TestScenario:
    """Воспроизводимый сценарий проверки программы."""

    code: str
    title: str
    purpose: str
    powder_mass_kg: float
    chamber_volume_m3: float = BASE_CHAMBER_VOLUME_M3
    barrel_length_m: float = BASE_BARREL_LENGTH_M
    powder_name: str = POWDER_NAME


TEST_SCENARIOS = (
    TestScenario(
        code="valid",
        title="Допустимое решение прямой задачи",
        purpose="проверка нулевого штрафа при выполнении всех ограничений",
        powder_mass_kg=BASE_POWDER_MASS_KG,
    ),
    TestScenario(
        code="error",
        title="Ошибочное решение прямой задачи",
        purpose="проверка безопасной обработки некорректной массы заряда",
        powder_mass_kg=-1.0,
    ),
    TestScenario(
        code="invalid",
        title="Недопустимое решение прямой задачи",
        purpose="проверка штрафа при превышении допустимого давления",
        powder_mass_kg=2.0,
    ),
)


def _error_result(
    message: str,
    *,
    stop_reason: str | None = None,
    execution_time_s: float | None = None,
) -> BallisticsResult:
    """Сформировать единообразный результат ошибочного расчёта."""

    return BallisticsResult(
        status="ошибка",
        message=message,
        stop_reason=stop_reason,
        execution_time_s=execution_time_s,
    )


def run_ballistics_safe(
    powder_mass_kg: float = BASE_POWDER_MASS_KG,
    chamber_volume_m3: float = BASE_CHAMBER_VOLUME_M3,
    barrel_length_m: float = BASE_BARREL_LENGTH_M,
    *,
    powder_name: str = POWDER_NAME,
    initial_temperature_k: float = NORMAL_TEMPERATURE_K,
) -> BallisticsResult:
    """Безопасно решить прямую задачу нульмерной внутренней баллистики.

    Ошибка входных данных, исключение решателя, ошибочный статус, преждевременная
    остановка либо NaN/inf в любом массиве расчётных слоёв возвращаются как
    ``status="ошибка"``. Функция не проверяет ограничения варианта: физически
    рассчитанное, но недопустимое решение остаётся успешным прямым расчётом.
    """

    options = {
        "powders": [{"omega": powder_mass_kg, "dbname": powder_name}],
        "init_conditions": {
            "q": PROJECTILE_MASS_KG,
            "d": CALIBER_M,
            "W_0": chamber_volume_m3,
            "T_0": initial_temperature_k,
            "phi_1": PHI_1,
            "p_0": FORCING_PRESSURE_PA,
            "S": BORE_AREA_M2,
        },
        "igniter": {"p_ign_0": P_IGN_0_PA},
        "meta_termo": {
            "dt": INTEGRATION_STEP_S,
            "method": INTEGRATION_METHOD,
        },
        "stop_conditions": {
            "x_p": barrel_length_m,
            "p_max": NUMERICAL_PRESSURE_CUTOFF_PA,
            "steps_max": MAX_INTEGRATION_STEPS,
        },
    }

    try:
        raw_result = ozvb_termo(options)
        if not isinstance(raw_result, dict):
            return _error_result(
                "Решатель вернул значение неожиданного типа: "
                f"{type(raw_result).__name__}."
            )

        execution_time = raw_result.get("execution_time")
        execution_time_s = (
            float(execution_time) if execution_time is not None else None
        )
        stop_reason_value = raw_result.get("stop_reason")
        stop_reason = (
            str(stop_reason_value) if stop_reason_value is not None else None
        )
        explicit_status = str(raw_result.get("status", "")).lower()

        if stop_reason == "error" or explicit_status in {"error", "ошибка"}:
            diagnostic = raw_result.get("error_message") or "причина не указана"
            return _error_result(
                f"Численный решатель сообщил об ошибке: {diagnostic}",
                stop_reason=stop_reason,
                execution_time_s=execution_time_s,
            )

        if stop_reason is None:
            return _error_result(
                "В ответе решателя отсутствует причина остановки.",
                execution_time_s=execution_time_s,
            )

        trajectory: dict[str, FloatArray] = {}
        for name, values in raw_result.items():
            if not isinstance(values, (np.ndarray, list, tuple)):
                continue

            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or array.size == 0:
                return _error_result(
                    f"Массив расчётных слоёв {name!r} пуст или не одномерен.",
                    stop_reason=stop_reason,
                    execution_time_s=execution_time_s,
                )
            if not np.all(np.isfinite(array)):
                return _error_result(
                    f"Массив расчётных слоёв {name!r} содержит NaN или inf.",
                    stop_reason=stop_reason,
                    execution_time_s=execution_time_s,
                )
            trajectory[name] = array.copy()

        missing_fields = set(REQUIRED_TRAJECTORY_FIELDS) - trajectory.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            return _error_result(
                f"В ответе решателя отсутствуют обязательные массивы: {missing}.",
                stop_reason=stop_reason,
                execution_time_s=execution_time_s,
            )

        layer_count = trajectory["t"].size
        wrong_lengths = sorted(
            name for name, values in trajectory.items() if values.size != layer_count
        )
        if wrong_lengths:
            return _error_result(
                "Массивы расчётных слоёв имеют разную длину: "
                + ", ".join(wrong_lengths)
                + ".",
                stop_reason=stop_reason,
                execution_time_s=execution_time_s,
            )

        if stop_reason != "x_p":
            return _error_result(
                "Расчёт остановлен до достижения дульного среза "
                f"(причина: {stop_reason}).",
                stop_reason=stop_reason,
                execution_time_s=execution_time_s,
            )

        final_position_m = float(trajectory["x_p"][-1])
        position_tolerance_m = max(1.0e-8, barrel_length_m * 1.0e-6)
        if not np.isclose(
            final_position_m,
            barrel_length_m,
            rtol=0.0,
            atol=position_tolerance_m,
        ):
            return _error_result(
                "Расчёт не достиг заданного дульного среза: "
                f"x={final_position_m:.6g} м вместо {barrel_length_m:.6g} м.",
                stop_reason=stop_reason,
                execution_time_s=execution_time_s,
            )

        if np.any(np.diff(trajectory["t"]) < 0.0) or np.any(
            np.diff(trajectory["x_p"]) < -position_tolerance_m
        ):
            return _error_result(
                "В расчётных слоях нарушена монотонность времени или координаты.",
                stop_reason=stop_reason,
                execution_time_s=execution_time_s,
            )

        muzzle_velocity_m_s = float(trajectory["v_p"][-1])
        max_pressure_pa = float(np.max(trajectory["p_m"]))
        muzzle_pressure_pa = float(trajectory["p_m"][-1])
        if muzzle_velocity_m_s < 0.0 or max_pressure_pa < 0.0:
            return _error_result(
                "Решатель вернул физически некорректную скорость или давление.",
                stop_reason=stop_reason,
                execution_time_s=execution_time_s,
            )

        return BallisticsResult(
            status="успех",
            message="Расчёт успешно завершён у дульного среза.",
            stop_reason=stop_reason,
            muzzle_velocity_m_s=muzzle_velocity_m_s,
            max_pressure_pa=max_pressure_pa,
            muzzle_pressure_pa=muzzle_pressure_pa,
            execution_time_s=execution_time_s,
            trajectory=trajectory,
        )
    except Exception as error:  # noqa: BLE001 - граница безопасности решателя
        return _error_result(
            "Перехвачено исключение численного решателя: "
            f"{type(error).__name__}: {error}"
        )


def compute_criterion_w_pm(
    chamber_volume_m3: float,
    barrel_length_m: float,
) -> float:
    """Вычислить чистый критерий ``W_pm = W_0 + S * l`` в м³."""

    if chamber_volume_m3 <= 0.0:
        raise ValueError("Начальный объём каморы должен быть положительным.")
    if barrel_length_m <= 0.0:
        raise ValueError("Длина хода снаряда должна быть положительной.")
    return chamber_volume_m3 + BORE_AREA_M2 * barrel_length_m


def _check_constraint(
    *,
    code: str,
    description: str,
    value: float,
    limit: float,
    relation: ConstraintRelation,
    unit: str,
) -> ConstraintResult:
    """Проверить одно ограничение и нормировать только его нарушение."""

    if not np.isfinite(value) or not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"Некорректные данные ограничения {code!r}.")

    if relation == "<=":
        normalized_violation = max(0.0, (value - limit) / limit)
    elif relation == ">=":
        normalized_violation = max(0.0, (limit - value) / limit)
    else:  # pragma: no cover - ограничено типом ConstraintRelation
        raise ValueError(f"Неизвестный тип ограничения: {relation!r}.")

    return ConstraintResult(
        code=code,
        description=description,
        value=value,
        limit=limit,
        relation=relation,
        unit=unit,
        normalized_violation=normalized_violation,
    )


def check_variant_constraints(
    *,
    normal_max_pressure_pa: float,
    barrel_length_m: float,
    cold_muzzle_velocity_m_s: float,
    hot_muzzle_pressure_pa: float,
) -> tuple[ConstraintResult, ...]:
    """Проверить четыре ограничения варианта № 36 независимо от штрафа."""

    return (
        _check_constraint(
            code="p_m_normal",
            description="максимальное давление при нормальной температуре",
            value=normal_max_pressure_pa,
            limit=MAX_PRESSURE_PA,
            relation="<=",
            unit="Па",
        ),
        _check_constraint(
            code="l_m",
            description="длина ведущей части канала ствола",
            value=barrel_length_m,
            limit=MAX_BARREL_LENGTH_M,
            relation="<=",
            unit="м",
        ),
        _check_constraint(
            code="v_pm_cold",
            description="дульная скорость при температуре −50 °C",
            value=cold_muzzle_velocity_m_s,
            limit=MIN_MUZZLE_VELOCITY_COLD_M_S,
            relation=">=",
            unit="м/с",
        ),
        _check_constraint(
            code="p_mz_hot",
            description="дульное давление при температуре +50 °C",
            value=hot_muzzle_pressure_pa,
            limit=MAX_MUZZLE_PRESSURE_HOT_PA,
            relation="<=",
            unit="Па",
        ),
    )


def compute_penalty(
    criterion_w_pm_m3: float,
    constraints: tuple[ConstraintResult, ...],
    *,
    scale: float = PENALTY_SCALE,
) -> float:
    """Вычислить штраф отдельно от критерия по нормированным нарушениям.

    Используется квадратичная функция
    ``P = scale * W_pm * sum(r_i**2)``, где ``r_i`` — безразмерное
    нормированное нарушение соответствующего ограничения.
    """

    if criterion_w_pm_m3 <= 0.0 or not np.isfinite(criterion_w_pm_m3):
        raise ValueError("Чистый критерий должен быть конечным и положительным.")
    if scale < 0.0 or not np.isfinite(scale):
        raise ValueError("Масштаб штрафа должен быть конечным и неотрицательным.")

    squared_violation_sum = sum(
        constraint.normalized_violation**2 for constraint in constraints
    )
    return scale * criterion_w_pm_m3 * squared_violation_sum


def evaluate_ballistic_solution(
    powder_mass_kg: float,
    chamber_volume_m3: float,
    barrel_length_m: float,
    *,
    powder_name: str = POWDER_NAME,
) -> SolutionEvaluation:
    """Рассчитать решение при трёх температурах и оценить его допустимость.

    Ошибка любого прямого расчёта имеет отдельный статус. Для неё ограничения
    не проверяются, допустимость не назначается, а штраф и штрафованный критерий
    остаются неопределёнными.
    """

    try:
        criterion_w_pm_m3 = compute_criterion_w_pm(
            chamber_volume_m3,
            barrel_length_m,
        )
    except (TypeError, ValueError) as error:
        return SolutionEvaluation(
            status="ошибка",
            message=f"Невозможно вычислить чистый критерий: {error}",
            criterion_w_pm_m3=None,
        )

    temperatures = {
        "normal": NORMAL_TEMPERATURE_K,
        "cold": COLD_TEMPERATURE_K,
        "hot": HOT_TEMPERATURE_K,
    }
    calculations = {
        condition: run_ballistics_safe(
            powder_mass_kg=powder_mass_kg,
            chamber_volume_m3=chamber_volume_m3,
            barrel_length_m=barrel_length_m,
            powder_name=powder_name,
            initial_temperature_k=temperature_k,
        )
        for condition, temperature_k in temperatures.items()
    }

    failed_conditions = [
        condition
        for condition, result in calculations.items()
        if result.status == "ошибка"
    ]
    if failed_conditions:
        diagnostics = "; ".join(
            f"{condition}: {calculations[condition].message}"
            for condition in failed_conditions
        )
        return SolutionEvaluation(
            status="ошибка",
            message="Ошибка прямого расчёта; штраф не применяется. " + diagnostics,
            criterion_w_pm_m3=criterion_w_pm_m3,
            calculations=calculations,
        )

    normal = calculations["normal"]
    cold = calculations["cold"]
    hot = calculations["hot"]
    if (
        normal.max_pressure_pa is None
        or cold.muzzle_velocity_m_s is None
        or hot.muzzle_pressure_pa is None
    ):
        return SolutionEvaluation(
            status="ошибка",
            message="Успешный ответ решателя не содержит обязательных результатов.",
            criterion_w_pm_m3=criterion_w_pm_m3,
            calculations=calculations,
        )

    constraints = check_variant_constraints(
        normal_max_pressure_pa=normal.max_pressure_pa,
        barrel_length_m=barrel_length_m,
        cold_muzzle_velocity_m_s=cold.muzzle_velocity_m_s,
        hot_muzzle_pressure_pa=hot.muzzle_pressure_pa,
    )
    is_feasible = all(constraint.is_satisfied for constraint in constraints)
    penalty_m3 = compute_penalty(criterion_w_pm_m3, constraints)

    return SolutionEvaluation(
        status="рассчитано",
        message=(
            "Все ограничения выполнены."
            if is_feasible
            else "Прямой расчёт выполнен, но имеются нарушения ограничений."
        ),
        criterion_w_pm_m3=criterion_w_pm_m3,
        calculations=calculations,
        constraints=constraints,
        is_feasible=is_feasible,
        penalty_m3=penalty_m3,
        penalized_criterion_m3=criterion_w_pm_m3 + penalty_m3,
    )


def run_test_scenarios() -> list[tuple[TestScenario, SolutionEvaluation]]:
    """Последовательно выполнить все контрольные сценарии без общего падения."""

    results: list[tuple[TestScenario, SolutionEvaluation]] = []
    for scenario in TEST_SCENARIOS:
        try:
            evaluation = evaluate_ballistic_solution(
                powder_mass_kg=scenario.powder_mass_kg,
                chamber_volume_m3=scenario.chamber_volume_m3,
                barrel_length_m=scenario.barrel_length_m,
                powder_name=scenario.powder_name,
            )
        except Exception as error:  # noqa: BLE001 - изоляция сценариев верхнего уровня
            evaluation = SolutionEvaluation(
                status="ошибка",
                message=(
                    "Перехвачена непредвиденная ошибка сценария: "
                    f"{type(error).__name__}: {error}"
                ),
                criterion_w_pm_m3=None,
            )
        results.append((scenario, evaluation))
    return results


def _format_constraint_value(constraint: ConstraintResult) -> tuple[float, float, str]:
    """Подготовить значение и предел ограничения для консольного вывода."""

    if constraint.unit == "Па":
        return constraint.value / 1.0e6, constraint.limit / 1.0e6, "МПа"
    return constraint.value, constraint.limit, constraint.unit


def print_scenario_result(
    scenario: TestScenario,
    evaluation: SolutionEvaluation,
) -> None:
    """Вывести все составляющие оценки одного контрольного сценария."""

    print(f"\n=== {scenario.title} [{scenario.code}] ===")
    print(f"purpose={scenario.purpose}")
    print(
        f"powder={scenario.powder_name}; omega={scenario.powder_mass_kg:.3f} kg; "
        f"W_0={scenario.chamber_volume_m3:.6e} m^3; "
        f"l={scenario.barrel_length_m:.3f} m"
    )
    print(f"calculation_status={evaluation.status}")
    print(f"message={evaluation.message}")
    if evaluation.criterion_w_pm_m3 is None:
        print("criterion_W_pm=—")
    else:
        print(f"criterion_W_pm={evaluation.criterion_w_pm_m3:.9e} m^3")

    if evaluation.status == "ошибка":
        print("feasible=не определено")
        print("constraints=не проверялись")
        print("penalty=—")
        print("penalized_criterion=—")
        return

    print(f"feasible={'да' if evaluation.is_feasible else 'нет'}")
    for constraint in evaluation.constraints:
        value, limit, unit = _format_constraint_value(constraint)
        state = "выполнено" if constraint.is_satisfied else "нарушено"
        print(
            f"constraint[{constraint.code}]: value={value:.6g} {unit}; "
            f"condition={constraint.relation} {limit:.6g} {unit}; "
            f"violation={constraint.normalized_violation:.6f}; {state}"
        )

    print(f"penalty={evaluation.penalty_m3:.9e} m^3")
    print(f"penalized_criterion={evaluation.penalized_criterion_m3:.9e} m^3")


def configure_plots() -> None:
    """Настроить единый читаемый стиль рисунков, пригодный для ЧБ-печати."""

    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.8),
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "0.78",
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.edgecolor": "black",
            "lines.linewidth": 1.7,
        }
    )


def _save_figure(figure: plt.Figure, filename: str) -> Path:
    """Сохранить рисунок в каталоге отчёта и закрыть объект Matplotlib."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    target = FIGURES_DIR / filename
    figure.tight_layout()
    figure.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return target


def _scenario_map(
    scenario_results: list[tuple[TestScenario, SolutionEvaluation]],
) -> dict[str, tuple[TestScenario, SolutionEvaluation]]:
    """Индексировать результаты сценариев по устойчивому коду."""

    return {scenario.code: (scenario, result) for scenario, result in scenario_results}


def _scenario_figure_label(scenario: TestScenario) -> str:
    """Перенести длинное название решения для компактной подписи рисунка."""

    return scenario.title.replace(
        " решение прямой задачи",
        "\nрешение прямой задачи",
    )


def plot_pressure_profiles(
    scenario_results: list[tuple[TestScenario, SolutionEvaluation]],
) -> Path:
    """Построить давление по координате при нормальной температуре."""

    indexed = _scenario_map(scenario_results)
    styles = {
        "valid": {"linestyle": "-", "marker": "o"},
        "invalid": {"linestyle": "--", "marker": "s"},
    }
    figure, axes = plt.subplots()

    for code in ("valid", "invalid"):
        scenario, evaluation = indexed[code]
        calculation = evaluation.calculations.get("normal")
        if calculation is None or calculation.status != "успех":
            continue
        position_m = calculation.trajectory["x_p"]
        pressure_mpa = calculation.trajectory["p_m"] / 1.0e6
        axes.plot(
            position_m,
            pressure_mpa,
            color="black",
            linestyle=styles[code]["linestyle"],
            marker=styles[code]["marker"],
            markevery=max(1, position_m.size // 11),
            markersize=4.0,
            markerfacecolor="white",
            label=_scenario_figure_label(scenario),
        )

    axes.axhline(
        MAX_PRESSURE_PA / 1.0e6,
        color="0.25",
        linestyle="-.",
        linewidth=1.4,
        label=r"Ограничение $p_m=330$ МПа",
    )
    axes.set_xlabel("Путь снаряда, м")
    axes.set_ylabel("Среднебаллистическое давление, МПа")
    axes.set_title("Давление при нормальной начальной температуре")
    axes.set_xlim(left=0.0)
    axes.set_ylim(bottom=0.0)
    axes.legend(loc="best")
    axes.text(
        0.98,
        0.04,
        "Ошибочное решение прямой задачи:\nтраектория отсутствует",
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    return _save_figure(figure, "pressure_profiles.png")


def plot_cold_velocity_profiles(
    scenario_results: list[tuple[TestScenario, SolutionEvaluation]],
) -> Path:
    """Построить скорость по координате для начальной температуры −50 °C."""

    indexed = _scenario_map(scenario_results)
    styles = {
        "valid": {"linestyle": "-", "marker": "o"},
        "invalid": {"linestyle": "--", "marker": "s"},
    }
    figure, axes = plt.subplots()

    for code in ("valid", "invalid"):
        scenario, evaluation = indexed[code]
        calculation = evaluation.calculations.get("cold")
        if calculation is None or calculation.status != "успех":
            continue
        position_m = calculation.trajectory["x_p"]
        velocity_m_s = calculation.trajectory["v_p"]
        axes.plot(
            position_m,
            velocity_m_s,
            color="black",
            linestyle=styles[code]["linestyle"],
            marker=styles[code]["marker"],
            markevery=max(1, position_m.size // 11),
            markersize=4.0,
            markerfacecolor="white",
            label=_scenario_figure_label(scenario),
        )

    axes.axhline(
        MIN_MUZZLE_VELOCITY_COLD_M_S,
        color="0.25",
        linestyle="-.",
        linewidth=1.4,
        label=r"Минимум на дульном срезе $v_{pm,-50}=570$ м/с",
    )
    axes.axvline(
        BASE_BARREL_LENGTH_M,
        color="0.45",
        linestyle=":",
        linewidth=1.2,
        label=r"Дульный срез $l=5{,}10$ м",
    )
    axes.set_xlabel("Путь снаряда, м")
    axes.set_ylabel("Скорость снаряда, м/с")
    axes.set_title(r"Скорость при начальной температуре $-50\,^{\circ}$C")
    axes.set_xlim(left=0.0)
    axes.set_ylim(bottom=0.0)
    axes.legend(loc="best")
    return _save_figure(figure, "cold_velocity_profiles.png")


def _constraint_utilization(constraint: ConstraintResult) -> float:
    """Привести ограничение к виду «не более единицы — допустимо»."""

    if constraint.relation == "<=":
        return constraint.value / constraint.limit
    if constraint.value <= 0.0:
        return np.inf
    return constraint.limit / constraint.value


def plot_constraint_utilization(
    scenario_results: list[tuple[TestScenario, SolutionEvaluation]],
) -> Path:
    """Сопоставить нормированные проверочные величины с границей единица."""

    indexed = _scenario_map(scenario_results)
    labels = (
        r"$p_m$",
        r"$l_m$",
        r"$v_{pm,-50}$",
        r"$p_{mz,+50}$",
    )
    positions = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axes = plt.subplots()

    for offset, code, hatch, gray in (
        (-width / 2, "valid", "...", "0.88"),
        (width / 2, "invalid", "///", "0.62"),
    ):
        scenario, evaluation = indexed[code]
        utilization = [
            _constraint_utilization(constraint)
            for constraint in evaluation.constraints
        ]
        axes.bar(
            positions + offset,
            utilization,
            width,
            color=gray,
            edgecolor="black",
            linewidth=0.9,
            hatch=hatch,
            label=_scenario_figure_label(scenario),
        )

    axes.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Граница допустимости",
    )
    axes.set_xticks(positions, labels)
    axes.set_ylabel("Нормированная проверочная величина, 1")
    axes.set_title("Проверка ограничений варианта № 36")
    axes.set_ylim(bottom=0.0)
    axes.legend(loc="upper right")
    axes.text(
        0.5,
        -0.16,
        "Ошибочное решение прямой задачи:\nограничения не проверяются",
        transform=axes.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
    )
    return _save_figure(figure, "constraint_utilization.png")


def plot_criterion_and_penalty(
    scenario_results: list[tuple[TestScenario, SolutionEvaluation]],
) -> Path:
    """Показать раздельно чистый критерий и добавленный штраф."""

    display_order = ("valid", "invalid", "error")
    indexed = _scenario_map(scenario_results)
    labels = [
        _scenario_figure_label(indexed[code][0]) for code in display_order
    ]
    positions = np.arange(len(display_order), dtype=float)
    criteria = np.zeros(len(display_order))
    penalties = np.zeros(len(display_order))

    for index, code in enumerate(display_order):
        _, evaluation = indexed[code]
        if evaluation.status == "ошибка":
            continue
        criteria[index] = evaluation.criterion_w_pm_m3 or 0.0
        penalties[index] = evaluation.penalty_m3 or 0.0

    figure, axes = plt.subplots()
    axes.bar(
        positions,
        criteria,
        width=0.58,
        color="0.88",
        edgecolor="black",
        linewidth=0.9,
        hatch="...",
        label=r"Чистый критерий $W_{pm}$",
    )
    axes.bar(
        positions,
        penalties,
        width=0.58,
        bottom=criteria,
        color="0.55",
        edgecolor="black",
        linewidth=0.9,
        hatch="///",
        label="Штраф",
    )
    axes.scatter(
        [positions[2]],
        [max(criteria + penalties) * 0.12],
        color="black",
        marker="x",
        s=75,
        linewidths=1.8,
        label="Ошибка: итог не определяется",
        zorder=4,
    )
    axes.set_xticks(positions, labels)
    axes.tick_params(axis="x", labelsize=9)
    axes.set_ylabel(r"Значение показателя, м$^3$")
    axes.set_title("Раздельное представление критерия и штрафа")
    axes.set_ylim(bottom=0.0)
    axes.legend(loc="upper right")
    return _save_figure(figure, "criterion_and_penalty.png")


def build_all_figures(
    scenario_results: list[tuple[TestScenario, SolutionEvaluation]],
) -> list[Path]:
    """Детерминированно пересоздать все рисунки лабораторного отчёта."""

    configure_plots()
    return [
        plot_pressure_profiles(scenario_results),
        plot_cold_velocity_profiles(scenario_results),
        plot_constraint_utilization(scenario_results),
        plot_criterion_and_penalty(scenario_results),
    ]


def main() -> None:
    """Выполнить и вывести три контрольных сценария варианта № 36."""

    try:
        package_version = version("pyballistics")
    except PackageNotFoundError:
        package_version = "не установлена"

    print(f"pyballistics={package_version}")
    print(
        "variant="
        f"{VARIANT_NUMBER}; model={BALLISTIC_MODEL}; criterion={CRITERION_NAME}; "
        f"penalty_scale={PENALTY_SCALE:g}"
    )
    scenario_results = run_test_scenarios()
    for scenario, evaluation in scenario_results:
        print_scenario_result(scenario, evaluation)

    for path in build_all_figures(scenario_results):
        print(f"created={path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
