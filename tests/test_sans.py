import numpy as np
import pytest

from langevinsans import (
    FieldGrid,
    InputValidationError,
    ParticleProperties,
    SANSConfig,
    langevin_curve,
    pair_distance_curves,
    sans_field_curve,
    sans_intensity_curve,
    sans_map,
    sphere_form_factor,
)


def test_sphere_form_factor_has_correct_origin_limit() -> None:
    values = sphere_form_factor(np.array([0.0, 1.0e-8, 0.1]))
    np.testing.assert_allclose(values[:2], [1.0 / 9.0, 1.0 / 9.0], rtol=1.0e-13)
    assert np.all(values >= 0.0)


def test_isotropic_gamma_produces_isotropic_detector_maps() -> None:
    gamma = np.eye(3) / 3.0
    particle = ParticleProperties()
    config = SANSConfig(map_points=51, curve_points=64)
    for geometry in ("perpendicular", "parallel"):
        _, _, intensity = sans_map(gamma, particle, config, geometry)
        radius = 12
        centre = config.map_points // 2
        ring = np.array(
            [
                intensity[centre + radius, centre],
                intensity[centre - radius, centre],
                intensity[centre, centre + radius],
                intensity[centre, centre - radius],
            ]
        )
        np.testing.assert_allclose(ring, np.full(4, ring[0]), rtol=1.0e-13)


def test_detector_axes_follow_the_documented_coordinate_convention() -> None:
    particle = ParticleProperties()
    config = SANSConfig(map_points=51, curve_points=64)
    centre = config.map_points // 2
    offset = 8

    _, _, perpendicular = sans_map(
        np.diag([0.0, 1.0, 0.0]),
        particle,
        config,
        "perpendicular",
    )
    assert (
        perpendicular[centre, centre + offset] > perpendicular[centre + offset, centre]
    )

    _, _, parallel = sans_map(
        np.diag([1.0, 0.0, 0.0]),
        particle,
        config,
        "parallel",
    )
    assert parallel[centre + offset, centre] > parallel[centre, centre + offset]


def test_sans_curves_are_nonnegative_and_have_compact_pair_support() -> None:
    gamma = np.diag([0.2, 0.3, 0.5])
    particle = ParticleProperties()
    config = SANSConfig(map_points=51, curve_points=64)
    q, intensity = sans_intensity_curve(
        gamma,
        particle,
        config,
        "perpendicular",
    )
    r, pair, correlation = pair_distance_curves(
        gamma,
        particle,
        config,
        "perpendicular",
    )
    assert np.all(np.diff(q) > 0.0)
    assert np.all(intensity >= 0.0)
    assert r[0] == 0.0
    assert r[-1] == 2.0 * particle.radius_nm
    assert pair[0] == 0.0
    assert abs(pair[-1]) < 1.0e-24
    assert abs(correlation[-1]) < 1.0e-24


def test_spin_flip_parallel_channel_excludes_longitudinal_gamma_zz() -> None:
    gamma = np.diag([0.0, 0.0, 1.0])
    particle = ParticleProperties()
    config = SANSConfig(map_points=51, curve_points=64)

    _, _, parallel_map = sans_map(
        gamma,
        particle,
        config,
        "parallel",
        "spin_flip",
    )
    _, parallel_curve = sans_intensity_curve(
        gamma,
        particle,
        config,
        "parallel",
        cross_section="spin_flip",
    )
    np.testing.assert_array_equal(parallel_map, np.zeros_like(parallel_map))
    np.testing.assert_array_equal(parallel_curve, np.zeros_like(parallel_curve))


def test_spin_flip_curves_use_analytic_azimuthal_moments() -> None:
    gamma = np.diag([0.2, 0.3, 0.5])
    particle = ParticleProperties()
    config = SANSConfig(map_points=51, curve_points=64)

    for geometry, expected_ratio in (
        ("perpendicular", 0.375 / 0.6),
        ("parallel", 0.25 / 0.75),
    ):
        _, unpolarized = sans_intensity_curve(
            gamma,
            particle,
            config,
            geometry,
            spacing="linear",
        )
        _, spin_flip = sans_intensity_curve(
            gamma,
            particle,
            config,
            geometry,
            spacing="linear",
            cross_section="spin_flip",
        )
        np.testing.assert_allclose(spin_flip, expected_ratio * unpolarized)


def test_sans_field_curve_is_even_for_the_langevin_model() -> None:
    particle = ParticleProperties()
    result = langevin_curve(particle, FieldGrid(n_field_points=101))
    config = SANSConfig(map_points=51, curve_points=64)

    field, intensity = sans_field_curve(
        result.field_t,
        result.gamma,
        particle,
        config,
        "perpendicular",
        q_per_nm=0.1,
        cross_section="spin_flip",
    )
    np.testing.assert_array_equal(field, result.field_t)
    np.testing.assert_allclose(intensity, intensity[::-1])


def test_intensity_scales_linearly_with_particle_volume_fraction() -> None:
    gamma = np.eye(3) / 3.0
    particle = ParticleProperties()
    dilute = SANSConfig(volume_fraction=0.01, map_points=51, curve_points=64)
    concentrated = SANSConfig(volume_fraction=0.02, map_points=51, curve_points=64)
    _, intensity_dilute = sans_intensity_curve(
        gamma,
        particle,
        dilute,
        "parallel",
    )
    _, intensity_concentrated = sans_intensity_curve(
        gamma,
        particle,
        concentrated,
        "parallel",
    )
    np.testing.assert_allclose(intensity_concentrated, 2.0 * intensity_dilute)


def test_sans_input_shapes_are_validated() -> None:
    particle = ParticleProperties()
    config = SANSConfig(map_points=51, curve_points=64)
    with pytest.raises(InputValidationError):
        sans_map(np.eye(2), particle, config, "parallel")
    with pytest.raises(InputValidationError):
        sans_field_curve(
            [0.0, 1.0],
            np.zeros((3, 3)),
            particle,
            config,
            "parallel",
            0.1,
        )
