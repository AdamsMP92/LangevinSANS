"""Validated data models for the analytical Langevin and SANS backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class InputValidationError(ValueError):
    """Raised when a model input is outside its supported domain."""


@dataclass(frozen=True, slots=True)
class ParticleProperties:
    """Physical properties of identical spherical single-domain particles."""

    radius_nm: float = 5.0
    saturation_magnetization_a_per_m: float = 3.1e5
    temperature_k: float = 300.0

    def __post_init__(self) -> None:
        values = (
            self.radius_nm,
            self.saturation_magnetization_a_per_m,
            self.temperature_k,
        )
        if not all(np.isfinite(value) for value in values):
            raise InputValidationError("Particle properties must be finite.")
        if self.radius_nm <= 0.0:
            raise InputValidationError("Particle radius must be positive.")
        if self.saturation_magnetization_a_per_m <= 0.0:
            raise InputValidationError("Saturation magnetization must be positive.")
        if self.temperature_k <= 0.0:
            raise InputValidationError("Temperature must be positive.")

    @property
    def volume_m3(self) -> float:
        """Particle volume in cubic metres."""

        radius_m = self.radius_nm * 1.0e-9
        return 4.0 * np.pi * radius_m**3 / 3.0

    @property
    def magnetic_moment_a_m2(self) -> float:
        """Fixed magnetic-moment magnitude ``M_s V`` in A m^2."""

        return self.saturation_magnetization_a_per_m * self.volume_m3


@dataclass(frozen=True, slots=True)
class FieldGrid:
    """Symmetric grid of prescribed external magnetic flux densities."""

    field_max_t: float = 1.0
    n_field_points: int = 201

    def __post_init__(self) -> None:
        if not np.isfinite(self.field_max_t) or self.field_max_t <= 0.0:
            raise InputValidationError("The maximum field must be finite and positive.")
        if self.n_field_points < 5 or self.n_field_points % 2 == 0:
            raise InputValidationError("Field points must be an odd integer >= 5.")


@dataclass(frozen=True, slots=True)
class SANSConfig:
    """Sample and numerical inputs for magnetic SANS observables."""

    volume_fraction: float = 0.01
    q_max_per_nm: float = 1.0
    map_points: int = 151
    curve_points: int = 320

    def __post_init__(self) -> None:
        if not np.isfinite(self.volume_fraction) or not (
            0.0 < self.volume_fraction <= 1.0
        ):
            raise InputValidationError("Volume fraction must lie in (0, 1].")
        if not np.isfinite(self.q_max_per_nm) or self.q_max_per_nm <= 0.0:
            raise InputValidationError("The maximum q must be finite and positive.")
        if self.map_points < 31 or self.curve_points < 32:
            raise InputValidationError("Increase the SANS sampling resolution.")


@dataclass(frozen=True, slots=True)
class LangevinResult:
    """Analytical field curve and its first and second orientational moments."""

    particle: ParticleProperties
    field_grid: FieldGrid
    field_t: FloatArray
    xi: FloatArray
    mean_m: FloatArray
    gamma: FloatArray

    @property
    def reduced_magnetization(self) -> FloatArray:
        """Reduced magnetization along the applied-field direction."""

        return self.mean_m[:, 2]

    @property
    def magnetization_a_per_m(self) -> FloatArray:
        """Dimensional ensemble magnetization along ``z``."""

        return (
            self.particle.saturation_magnetization_a_per_m * self.reduced_magnetization
        )

    @property
    def covariance(self) -> FloatArray:
        """Centered covariance, distinct from the raw SANS moments ``gamma``."""

        outer = cast(
            FloatArray,
            np.einsum("...i,...j->...ij", self.mean_m, self.mean_m),
        )
        return self.gamma - outer
