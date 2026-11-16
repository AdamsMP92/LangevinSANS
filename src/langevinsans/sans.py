"""Magnetic SANS response of dilute spherical Langevin particles."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from langevinsans.models import (
    FloatArray,
    InputValidationError,
    ParticleProperties,
    SANSConfig,
)

type Geometry = Literal["perpendicular", "parallel"]
type CurveSpacing = Literal["log", "linear"]
type CrossSection = Literal["unpolarized", "spin_flip"]
MAGNETIC_SCATTERING_LENGTH = 2.91e8  # A^-1 m^-1


def sphere_form_factor(x: ArrayLike) -> FloatArray:
    """Return ``[j1(x)/x]^2`` with its analytic limit at the origin."""

    values = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise InputValidationError("Form-factor arguments must be finite.")
    small = np.abs(values) < 1.0e-4
    result = np.empty_like(values)
    squared = values[~small] ** 2
    result[~small] = (
        np.sin(values[~small]) / (values[~small] * squared)
        - np.cos(values[~small]) / squared
    ) ** 2
    small_squared = values[small] ** 2
    result[small] = 1.0 / 9.0 - small_squared / 45.0 + small_squared**2 / 525.0
    return result


def _gamma_matrix(gamma: ArrayLike) -> FloatArray:
    values = np.asarray(gamma, dtype=np.float64)
    if values.shape != (3, 3):
        raise InputValidationError("Gamma must have shape (3, 3).")
    if not np.all(np.isfinite(values)):
        raise InputValidationError("Gamma must contain only finite values.")
    if not np.allclose(values, values.T, rtol=1.0e-12, atol=1.0e-14):
        raise InputValidationError("Gamma must be symmetric.")
    return values


def _prefactor(particle: ParticleProperties, config: SANSConfig) -> float:
    radius_m = particle.radius_nm * 1.0e-9
    return (
        config.volume_fraction
        * 16.0
        * np.pi**2
        * radius_m**6
        * MAGNETIC_SCATTERING_LENGTH**2
        * particle.saturation_magnetization_a_per_m**2
        / particle.volume_m3
    )


def _angular_factor(
    gamma: FloatArray,
    sine: FloatArray,
    cosine: FloatArray,
    geometry: Geometry,
    cross_section: CrossSection,
) -> FloatArray:
    if cross_section == "unpolarized":
        if geometry == "perpendicular":
            angular = (
                gamma[0, 0]
                + gamma[1, 1] * cosine**2
                + gamma[2, 2] * sine**2
                - 2.0 * gamma[1, 2] * sine * cosine
            )
        elif geometry == "parallel":
            angular = (
                gamma[0, 0] * sine**2
                + gamma[1, 1] * cosine**2
                + gamma[2, 2]
                - 2.0 * gamma[0, 1] * sine * cosine
            )
        else:
            raise InputValidationError(f"Unknown scattering geometry: {geometry}")
    elif cross_section == "spin_flip":
        if geometry == "perpendicular":
            angular = (
                gamma[0, 0]
                + gamma[1, 1] * cosine**4
                + gamma[2, 2] * sine**2 * cosine**2
                - 2.0 * gamma[1, 2] * sine * cosine**3
            )
        elif geometry == "parallel":
            angular = (
                gamma[0, 0] * sine**2
                + gamma[1, 1] * cosine**2
                - 2.0 * gamma[0, 1] * sine * cosine
            )
        else:
            raise InputValidationError(f"Unknown scattering geometry: {geometry}")
    else:
        raise InputValidationError(f"Unknown SANS cross section: {cross_section}")
    return np.asarray(angular, dtype=np.float64)


def azimuthal_moment(
    gamma: ArrayLike,
    geometry: Geometry,
    cross_section: CrossSection = "unpolarized",
) -> FloatArray:
    """Return the analytical angular factor averaged over detector azimuth."""

    values = np.asarray(gamma, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise InputValidationError("Gamma must end in dimensions (3, 3).")
    if not np.all(np.isfinite(values)):
        raise InputValidationError("Gamma must contain only finite values.")

    if cross_section == "unpolarized":
        if geometry == "perpendicular":
            moment = values[..., 0, 0] + 0.5 * (values[..., 1, 1] + values[..., 2, 2])
        elif geometry == "parallel":
            moment = (
                0.5 * values[..., 0, 0] + 0.5 * values[..., 1, 1] + values[..., 2, 2]
            )
        else:
            raise InputValidationError(f"Unknown scattering geometry: {geometry}")
    elif cross_section == "spin_flip":
        if geometry == "perpendicular":
            moment = (
                values[..., 0, 0]
                + 3.0 / 8.0 * values[..., 1, 1]
                + values[..., 2, 2] / 8.0
            )
        elif geometry == "parallel":
            moment = 0.5 * (values[..., 0, 0] + values[..., 1, 1])
        else:
            raise InputValidationError(f"Unknown scattering geometry: {geometry}")
    else:
        raise InputValidationError(f"Unknown SANS cross section: {cross_section}")
    return np.asarray(moment, dtype=np.float64)


def sans_map(
    gamma: ArrayLike,
    particle: ParticleProperties,
    config: SANSConfig,
    geometry: Geometry,
    cross_section: CrossSection = "unpolarized",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Calculate a square two-dimensional magnetic SANS detector map.

    The intensity is in cm^-1 and both axes are in nm^-1. The horizontal and
    vertical axes are ``(q_z, q_y)`` for perpendicular geometry and
    ``(q_x, q_y)`` for parallel geometry.
    """

    gamma_matrix = _gamma_matrix(gamma)
    axis = np.linspace(
        -config.q_max_per_nm,
        config.q_max_per_nm,
        config.map_points,
    )
    horizontal, vertical = np.meshgrid(axis, axis)
    q = np.hypot(horizontal, vertical)
    sine = np.divide(vertical, q, out=np.zeros_like(q), where=q != 0.0)
    cosine = np.divide(horizontal, q, out=np.ones_like(q), where=q != 0.0)
    angular = _angular_factor(
        gamma_matrix,
        sine,
        cosine,
        geometry,
        cross_section,
    )
    central = azimuthal_moment(gamma_matrix, geometry, cross_section)
    angular = np.where(q == 0.0, central, angular)
    intensity_per_m = (
        _prefactor(particle, config)
        * sphere_form_factor(q * particle.radius_nm)
        * angular
    )
    return axis, axis, np.maximum(intensity_per_m / 100.0, 0.0)


def sans_intensity_curve(
    gamma: ArrayLike,
    particle: ParticleProperties,
    config: SANSConfig,
    geometry: Geometry,
    spacing: CurveSpacing = "log",
    cross_section: CrossSection = "unpolarized",
) -> tuple[FloatArray, FloatArray]:
    """Return the 2-pi azimuthal average ``I(q)`` in cm^-1."""

    gamma_matrix = _gamma_matrix(gamma)
    if spacing == "log":
        q_min = min(config.q_max_per_nm / 10.0, 0.01 / particle.radius_nm)
        q_min = max(q_min, config.q_max_per_nm * 1.0e-5)
        q = np.geomspace(q_min, config.q_max_per_nm, config.curve_points)
    elif spacing == "linear":
        q = np.linspace(0.0, config.q_max_per_nm, config.curve_points)
    else:
        raise InputValidationError(f"Unknown reciprocal-space spacing: {spacing}")
    intensity = (
        _prefactor(particle, config)
        * sphere_form_factor(q * particle.radius_nm)
        * azimuthal_moment(gamma_matrix, geometry, cross_section)
        / 100.0
    )
    return q, np.maximum(intensity, 0.0)


def sans_field_curve(
    field_t: ArrayLike,
    gamma: ArrayLike,
    particle: ParticleProperties,
    config: SANSConfig,
    geometry: Geometry,
    q_per_nm: float,
    cross_section: CrossSection = "unpolarized",
) -> tuple[FloatArray, FloatArray]:
    """Return azimuthally averaged magnetic SANS intensity versus field."""

    field = np.asarray(field_t, dtype=np.float64)
    gamma_values = np.asarray(gamma, dtype=np.float64)
    if field.ndim != 1 or not np.all(np.isfinite(field)):
        raise InputValidationError(
            "Field values must be a finite one-dimensional array."
        )
    if gamma_values.shape != (field.size, 3, 3):
        raise InputValidationError("Gamma must have shape (n_field, 3, 3).")
    if not np.isfinite(q_per_nm) or q_per_nm < 0.0:
        raise InputValidationError("q must be finite and non-negative.")
    intensity = (
        _prefactor(particle, config)
        * sphere_form_factor(q_per_nm * particle.radius_nm)
        * azimuthal_moment(gamma_values, geometry, cross_section)
        / 100.0
    )
    return field.copy(), np.maximum(intensity, 0.0)


def pair_distance_curves(
    gamma: ArrayLike,
    particle: ParticleProperties,
    config: SANSConfig,
    geometry: Geometry,
    cross_section: CrossSection = "unpolarized",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``r``, ``p(r)``, and ``c(r)=p(r)/r^2`` in nm-based units."""

    gamma_matrix = _gamma_matrix(gamma)
    radius_m = particle.radius_nm * 1.0e-9
    r_m = np.linspace(0.0, 2.0 * radius_m, config.curve_points)
    shape = 1.0 - 3.0 * r_m / (4.0 * radius_m) + r_m**3 / (16.0 * radius_m**3)
    amplitude = (
        config.volume_fraction
        * 16.0
        * np.pi**2
        * radius_m**3
        * MAGNETIC_SCATTERING_LENGTH**2
        * particle.saturation_magnetization_a_per_m**2
        / particle.volume_m3
        * np.pi
        / 6.0
        * azimuthal_moment(gamma_matrix, geometry, cross_section)
    )
    correlation_per_m4 = amplitude * shape
    pair_per_m2 = correlation_per_m4 * r_m**2
    return (
        r_m * 1.0e9,
        np.maximum(pair_per_m2 * 1.0e-18, 0.0),
        np.maximum(correlation_per_m4 * 1.0e-36, 0.0),
    )
