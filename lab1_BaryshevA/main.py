"""Безопасный запуск нульмерной модели внутренней баллистики.

Запуск из корня лабораторной работы:

    uv run python main.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import pi
from typing import Literal

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

# Общие условия и базовое баллистическое решение (СИ)
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

CalculationStatus = Literal["успех", "ошибка"]
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


def main() -> None:
    """Выполнить один заведомо адекватный тестовый расчёт."""

    try:
        package_version = version("pyballistics")
    except PackageNotFoundError:
        package_version = "не установлена"

    print(f"pyballistics={package_version}")
    print(
        "variant="
        f"{VARIANT_NUMBER}; model={BALLISTIC_MODEL}; powder={POWDER_NAME}; "
        f"omega={BASE_POWDER_MASS_KG:.3f} kg"
    )

    result = run_ballistics_safe(
        powder_mass_kg=BASE_POWDER_MASS_KG,
        chamber_volume_m3=BASE_CHAMBER_VOLUME_M3,
        barrel_length_m=BASE_BARREL_LENGTH_M,
    )
    print(f"status={result.status}; message={result.message}")

    if result.status == "успех":
        print(f"V_k={result.muzzle_velocity_m_s:.3f} m/s")
        print(f"P_max={result.max_pressure_pa:.6e} Pa")
        print(f"P_muzzle={result.muzzle_pressure_pa:.6e} Pa")
        print(f"layers={result.trajectory['t'].size}")


if __name__ == "__main__":
    main()
