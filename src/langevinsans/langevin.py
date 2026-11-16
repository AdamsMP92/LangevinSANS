"""Closed-form equilibrium moments of the standard Langevin model."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from langevinsans.models import (
    FieldGrid,
    FloatArray,
    InputValidationError,
    LangevinResult,
    ParticleProperties,
)

# The Boltzmann constant is exact in the 2019 SI.
BOLTZMANN_J_PER_K = 1.380649e-23
_SERIES_LIMIT = 0.1


def langevin_function(x: ArrayLike) -> FloatArray:
    """Return ``L(x) = coth(x) - 1/x`` without cancellation near zero."""

    values = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise InputValidationError("The Langevin argument must be finite.")

    small = np.abs(values) < _SERIES_LIMIT
    result = np.empty_like(values)

    x_small = values[small]
    x2 = x_small * x_small
    result[small] = x_small * (
        1.0 / 3.0
        + x2
        * (-1.0 / 45.0 + x2 * (2.0 / 945.0 + x2 * (-1.0 / 4725.0 + x2 * 2.0 / 93555.0)))
    )

    x_regular = values[~small]
    result[~small] = 1.0 / np.tanh(x_regular) - 1.0 / x_regular
    return result


def _langevin_over_x(x: FloatArray) -> FloatArray:
    """Return ``L(x)/x`` with its even analytic continuation at zero."""

    small = np.abs(x) < _SERIES_LIMIT
    result = np.empty_like(x)

    x_small = x[small]
    x2 = x_small * x_small
    result[small] = 1.0 / 3.0 + x2 * (
        -1.0 / 45.0 + x2 * (2.0 / 945.0 + x2 * (-1.0 / 4725.0 + x2 * 2.0 / 93555.0))
    )

    x_regular = x[~small]
    result[~small] = langevin_function(x_regular) / x_regular
    return result


def dimensionless_field(
    particle: ParticleProperties,
    applied_induction_t: ArrayLike,
) -> FloatArray:
    """Return ``xi = mu B_ext / (k_B T)`` for a prescribed external induction."""

    field = np.asarray(applied_induction_t, dtype=np.float64)
    if not np.all(np.isfinite(field)):
        raise InputValidationError("Applied field values must be finite.")
    return (
        particle.magnetic_moment_a_m2
        * field
        / (BOLTZMANN_J_PER_K * particle.temperature_k)
    )


def direction_cosine_density(u: ArrayLike, xi: float) -> FloatArray:
    """Return the exact equilibrium density of ``u = cos(theta)``.

    The density is normalized on ``[-1, 1]`` and is evaluated in a form that
    remains stable for large positive or negative ``xi``. It is the azimuthal
    marginal of the full Langevin orientational distribution.
    """

    direction_cosine = np.asarray(u, dtype=np.float64)
    if not np.all(np.isfinite(direction_cosine)):
        raise InputValidationError("Direction cosines must be finite.")
    if np.any(np.abs(direction_cosine) > 1.0):
        raise InputValidationError("Direction cosines must lie in [-1, 1].")
    if not np.isfinite(xi):
        raise InputValidationError("The dimensionless field must be finite.")

    magnitude = abs(float(xi))
    if magnitude == 0.0:
        return np.full_like(direction_cosine, 0.5)

    field_sign = 1.0 if xi > 0.0 else -1.0
    normalization = magnitude / -np.expm1(-2.0 * magnitude)
    return cast(
        FloatArray,
        normalization * np.exp(magnitude * (field_sign * direction_cosine - 1.0)),
    )


def langevin_moments(xi: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return ``<m>`` and raw second moments ``Gamma = <m m^T>``.

    The applied field defines the laboratory ``z`` axis. The returned shapes
    are ``xi.shape + (3,)`` and ``xi.shape + (3, 3)``. At zero field the
    orientational distribution is isotropic and ``Gamma = I / 3``.
    """

    values = np.asarray(xi, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise InputValidationError("Dimensionless field values must be finite.")

    longitudinal_mean = langevin_function(values)
    transverse_second = _langevin_over_x(values)
    longitudinal_second = 1.0 - 2.0 * transverse_second

    mean_m = np.zeros(values.shape + (3,), dtype=np.float64)
    mean_m[..., 2] = longitudinal_mean

    gamma = np.zeros(values.shape + (3, 3), dtype=np.float64)
    gamma[..., 0, 0] = transverse_second
    gamma[..., 1, 1] = transverse_second
    gamma[..., 2, 2] = longitudinal_second
    return mean_m, gamma


def langevin_curve(
    particle: ParticleProperties,
    field_grid: FieldGrid | None = None,
) -> LangevinResult:
    """Evaluate the reversible analytical equilibrium curve on a symmetric grid."""

    grid = FieldGrid() if field_grid is None else field_grid
    field_t = np.linspace(
        -grid.field_max_t,
        grid.field_max_t,
        grid.n_field_points,
        dtype=np.float64,
    )
    xi = dimensionless_field(particle, field_t)
    mean_m, gamma = langevin_moments(xi)
    return LangevinResult(
        particle=particle,
        field_grid=grid,
        field_t=field_t,
        xi=xi,
        mean_m=mean_m,
        gamma=gamma,
    )
