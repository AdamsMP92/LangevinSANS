import numpy as np
import pytest

from langevinsans import (
    BOLTZMANN_J_PER_K,
    FieldGrid,
    InputValidationError,
    ParticleProperties,
    dimensionless_field,
    langevin_curve,
    langevin_function,
    langevin_moments,
)


def test_langevin_function_has_stable_origin_and_known_value() -> None:
    values = langevin_function(np.array([0.0, 1.0e-12, 1.0]))
    assert values[0] == 0.0
    np.testing.assert_allclose(values[1], 1.0e-12 / 3.0, rtol=1.0e-14)
    np.testing.assert_allclose(values[2], 0.3130352854993313, rtol=1.0e-14)


def test_langevin_function_is_odd_and_saturates() -> None:
    xi = np.array([0.1, 1.0, 10.0, 1.0e6])
    np.testing.assert_allclose(langevin_function(-xi), -langevin_function(xi))
    np.testing.assert_allclose(langevin_function(xi[-1]), 1.0 - 1.0e-6)


def test_scalar_zero_field_moments_have_documented_shapes() -> None:
    mean_m, gamma = langevin_moments(0.0)
    assert mean_m.shape == (3,)
    assert gamma.shape == (3, 3)
    np.testing.assert_array_equal(mean_m, np.zeros(3))
    np.testing.assert_allclose(gamma, np.eye(3) / 3.0)


def test_closed_form_moments_match_independent_boltzmann_quadrature() -> None:
    xi = 1.7
    direction_cosine, weights = np.polynomial.legendre.leggauss(200)
    boltzmann_weight = np.exp(xi * direction_cosine)
    normalization = np.sum(weights * boltzmann_weight)
    mean_z = np.sum(weights * boltzmann_weight * direction_cosine) / normalization
    second_z = np.sum(weights * boltzmann_weight * direction_cosine**2) / normalization

    mean_m, gamma = langevin_moments(xi)
    np.testing.assert_allclose(mean_m[2], mean_z, rtol=1.0e-13)
    np.testing.assert_allclose(gamma[2, 2], second_z, rtol=1.0e-13)
    np.testing.assert_allclose(gamma[0, 0], (1.0 - second_z) / 2.0, rtol=1.0e-13)


def test_closed_form_moments_are_normalized_and_isotropic_at_zero() -> None:
    xi = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    mean_m, gamma = langevin_moments(xi)

    np.testing.assert_allclose(gamma[2], np.eye(3) / 3.0, atol=1.0e-15)
    np.testing.assert_allclose(np.trace(gamma, axis1=-2, axis2=-1), 1.0)
    np.testing.assert_allclose(gamma[:, 0, 0], gamma[:, 1, 1])
    np.testing.assert_allclose(gamma, gamma[::-1])
    np.testing.assert_allclose(mean_m, -mean_m[::-1])
    assert np.all(np.linalg.eigvalsh(gamma) >= 0.0)


def test_second_moments_match_langevin_identity() -> None:
    xi = np.array([0.01, 0.5, 3.0, 100.0])
    _, gamma = langevin_moments(xi)
    expected_transverse = langevin_function(xi) / xi
    np.testing.assert_allclose(gamma[:, 0, 0], expected_transverse)
    np.testing.assert_allclose(gamma[:, 2, 2], 1.0 - 2.0 * expected_transverse)


def test_physical_scaling_and_symmetric_field_curve() -> None:
    particle = ParticleProperties(
        radius_nm=4.0,
        saturation_magnetization_a_per_m=2.5e5,
        temperature_k=250.0,
    )
    grid = FieldGrid(field_max_t=0.8, n_field_points=101)
    result = langevin_curve(particle, grid)

    expected_xi = (
        particle.magnetic_moment_a_m2
        * result.field_t
        / (BOLTZMANN_J_PER_K * particle.temperature_k)
    )
    np.testing.assert_allclose(result.xi, expected_xi)
    np.testing.assert_allclose(
        result.reduced_magnetization,
        -result.reduced_magnetization[::-1],
    )
    np.testing.assert_allclose(result.gamma, result.gamma[::-1])
    np.testing.assert_allclose(
        result.magnetization_a_per_m,
        particle.saturation_magnetization_a_per_m * result.reduced_magnetization,
    )
    assert result.field_t[grid.n_field_points // 2] == 0.0


def test_covariance_is_distinct_from_raw_second_moment() -> None:
    result = langevin_curve(ParticleProperties(), FieldGrid(n_field_points=5))
    centre = result.field_grid.n_field_points // 2
    np.testing.assert_allclose(result.covariance[centre], result.gamma[centre])
    assert result.covariance[-1, 2, 2] < result.gamma[-1, 2, 2]
    assert np.all(np.linalg.eigvalsh(result.covariance) >= -1.0e-14)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ParticleProperties(radius_nm=0.0),
        lambda: ParticleProperties(temperature_k=-1.0),
        lambda: FieldGrid(field_max_t=0.0),
        lambda: FieldGrid(n_field_points=100),
    ],
)
def test_invalid_model_inputs_are_rejected(factory: object) -> None:
    with pytest.raises(InputValidationError):
        factory()  # type: ignore[operator]


def test_custom_field_validation() -> None:
    particle = ParticleProperties()
    with pytest.raises(InputValidationError):
        dimensionless_field(particle, [0.0, np.nan])
    with pytest.raises(InputValidationError):
        langevin_moments([0.0, np.inf])
