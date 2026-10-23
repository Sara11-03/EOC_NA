"""Rydberg pulse parameter helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RydbergSchedule:
    rydberg_radius_um: float
    omega_max_rad_per_us: float | None
    delta_0_rad_per_us: float
    delta_f_rad_per_us: float
    t_rise_ns: int
    t_sweep_ns: int
    t_fall_ns: int
    total_time_ns: int
    delta_f_multiplier: float
    delta_0_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def omega_from_c6_and_radius(c6: float, rydberg_radius_um: float) -> float:
    """Return Omega such that Rb=(C6/Omega)^(1/6)."""
    if c6 <= 0 or rydberg_radius_um <= 0:
        raise ValueError("c6 and rydberg_radius_um must be positive.")
    return float((float(c6) / float(rydberg_radius_um) ** 6))


def omega_max_from_interaction_coeff(c6: float, rydberg_radius_um: float) -> float:
    """Return Omega_max=(C6/Rb^6) for explicit units.

    The expression C6 / Rb**6 is used in units consistent with Pulser.
    """
    return omega_from_c6_and_radius(c6, rydberg_radius_um)


def split_time(total_time_ns: int, *, fractions: tuple[float, float, float] = (1 / 6, 3 / 6, 2 / 6)) -> tuple[int, int, int]:
    total = int(total_time_ns)
    if total <= 0:
        raise ValueError("total_time_ns must be positive.")
    fr, fs, ff = fractions
    if abs((fr + fs + ff) - 1.0) > 1e-9:
        raise ValueError("time fractions must sum to 1.")
    t_rise = int(round(fr * total))
    t_sweep = int(round(fs * total))
    t_fall = total - t_rise - t_sweep
    if min(t_rise, t_sweep, t_fall) <= 0:
        raise ValueError("all pulse segments must be positive.")
    return t_rise, t_sweep, t_fall


def make_schedule(
    *,
    rydberg_radius_um: float,
    omega_max_rad_per_us: float | None,
    total_time_ns: int,
    delta_f_multiplier: float,
    delta_0_multiplier: float = -6.0,
) -> RydbergSchedule:
    if omega_max_rad_per_us is None:
        raise ValueError("Specify omega_max_rad_per_us even for a dry run; dimensionless detunings cannot be labelled rad/us.")
    else:
        U = float(omega_max_rad_per_us) / 2.0
    t_rise, t_sweep, t_fall = split_time(total_time_ns)
    return RydbergSchedule(
        rydberg_radius_um=float(rydberg_radius_um),
        omega_max_rad_per_us=None if omega_max_rad_per_us is None else float(omega_max_rad_per_us),
        delta_0_rad_per_us=float(delta_0_multiplier) * U,
        delta_f_rad_per_us=float(delta_f_multiplier) * U,
        t_rise_ns=t_rise,
        t_sweep_ns=t_sweep,
        t_fall_ns=t_fall,
        total_time_ns=int(total_time_ns),
        delta_f_multiplier=float(delta_f_multiplier),
        delta_0_multiplier=float(delta_0_multiplier),
    )
