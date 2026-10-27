import math

import pytest

from openmc_plasma_source.stellarator_source import (
    ActionGrid,
    CoefficientLayout,
    FourierZernikeSurface,
    boundary_error,
    jax_value_and_gradient,
    make_action,
    optimize_surface,
    rotating_ellipse_boundary_coefficients,
    rotating_ellipse_coordinates,
    rotating_ellipse_initial_guess,
    validate_surface,
    zernike_radial,
    zernike_radial_derivative,
    scalar_action_reference,
)


def test_low_order_zernike_polynomials():
    assert zernike_radial(0.37, 0, 0) == pytest.approx(1.0)
    assert zernike_radial(0.37, 1, 1) == pytest.approx(0.37)
    assert zernike_radial(0.37, 2, 0) == pytest.approx(2 * 0.37**2 - 1)
    assert zernike_radial_derivative(0.37, 1, 1) == pytest.approx(1.0)
    assert zernike_radial_derivative(0.37, 2, 0) == pytest.approx(4 * 0.37)


def test_rotating_ellipse_coefficients_reconstruct_boundary():
    major_radius = 906.0
    minor_radius = 292.258
    nfp = 4
    elongation = 1.6
    rotation = 1
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(
        major_radius, minor_radius, nfp, elongation, rotation
    )
    r_coefficients = {(n, m, 1): value for (n, m), value in r_boundary.items()}
    z_coefficients = {(n, m, 1): value for (n, m), value in z_boundary.items()}
    r_coefficients[(0, 0, 0)] = r_coefficients.pop((0, 0, 1))
    surface = FourierZernikeSurface(
        r_coefficients,
        z_coefficients,
        nfp,
        field_period_poloidal_shift=rotation * math.pi,
    )

    for theta, zeta in ((0.2, 0.1), (1.3, 0.7), (4.1, 2.4)):
        expected_r, expected_z = rotating_ellipse_coordinates(
            1.0,
            theta,
            zeta,
            major_radius,
            minor_radius,
            nfp,
            elongation,
            rotation,
        )
        point = surface.evaluate(1.0, theta, zeta)
        assert point.R == pytest.approx(expected_r)
        assert point.Z == pytest.approx(expected_z)


def test_initial_guess_matches_direct_rotating_ellipse():
    parameters = (906.0, 292.258, 4, 1.6, 1)
    r_coefficients, z_coefficients = rotating_ellipse_initial_guess(*parameters)
    surface = FourierZernikeSurface(
        r_coefficients,
        z_coefficients,
        parameters[2],
        field_period_poloidal_shift=parameters[4] * math.pi,
    )

    for s, theta, zeta in ((0.2, 0.2, 0.1), (0.7, 1.3, 0.7), (1.0, 4.1, 2.4)):
        expected_r, expected_z = rotating_ellipse_coordinates(
            s, theta, zeta, *parameters
        )
        point = surface.evaluate(s, theta, zeta)
        assert point.R == pytest.approx(expected_r)
        assert point.Z == pytest.approx(expected_z)


def test_cylindrical_and_cartesian_jacobians_agree():
    r_coefficients, z_coefficients = rotating_ellipse_initial_guess(
        906.0, 292.258, 4, 1.6, 1
    )
    surface = FourierZernikeSurface(
        r_coefficients,
        z_coefficients,
        4,
        field_period_poloidal_shift=math.pi,
    )
    point = surface.evaluate(0.6, 0.9, 0.4)
    assert point.jacobian == pytest.approx(point.cartesian_jacobian(0.4))


def test_rotating_ellipse_requires_integer_rotation():
    with pytest.raises(ValueError, match="integer"):
        rotating_ellipse_boundary_coefficients(906.0, 292.258, 4, 1.6, 0.5)


def test_asymmetric_boundary_breaks_pi_rotation_but_preserves_validation():
    parameters = (906.0, 292.258, 4, 1.6, 1, 0.05)
    r_coefficients, z_coefficients = rotating_ellipse_initial_guess(*parameters)
    surface = FourierZernikeSurface(
        r_coefficients,
        z_coefficients,
        parameters[2],
        field_period_poloidal_shift=parameters[4] * math.pi,
    )
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(*parameters)
    diagnostics = validate_surface(
        surface,
        s_values=(0.1, 0.5, 1.0),
        theta_values=(0.0, 0.7, 2.1),
        zeta_values=(0.0, 0.3, 1.2),
        r_boundary=r_boundary,
        z_boundary=z_boundary,
    )
    first = surface.evaluate(0.8, 0.4, 0.2)
    opposite = surface.evaluate(0.8, 0.4 + 3.141592653589793, 0.2)
    central_symmetry_error = abs(
        (first.R - 906.0) + (opposite.R - 906.0)
    ) + abs(first.Z + opposite.Z)
    assert central_symmetry_error > 1.0e-6
    assert diagnostics["boundary_error"] == pytest.approx(0.0)
    assert diagnostics["periodicity_error"] == pytest.approx(0.0)
    assert diagnostics["stellarator_symmetry_error"] == pytest.approx(0.0)
    assert diagnostics["minimum_nonaxis_jacobian"] > 0.0


def test_asymmetry_is_a_rigidly_rotated_single_cross_section():
    major_radius = 906.0
    minor_radius = 292.258
    nfp = 5
    elongation = 1.6
    rotation = 1
    asymmetry = 0.05
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(
        major_radius, minor_radius, nfp, elongation, rotation, asymmetry
    )
    amplitude = asymmetry * minor_radius
    assert r_boundary[(rotation, 2)] == pytest.approx(amplitude)
    assert z_boundary[(rotation, 2)] == pytest.approx(amplitude)

    theta = 0.73
    zeta = 0.41
    normalized_radius = 0.8
    delta = 0.5 * rotation * nfp * zeta
    radial = minor_radius * normalized_radius
    expected_r = major_radius + (
        radial * math.cos(theta) * math.cos(delta)
        - elongation * radial * math.sin(theta) * math.sin(delta)
        + asymmetry
        * minor_radius
        * normalized_radius**2
        * math.cos(2.0 * theta - rotation * nfp * zeta)
    )
    expected_z = (
        radial * math.cos(theta) * math.sin(delta)
        + elongation * radial * math.sin(theta) * math.cos(delta)
        + asymmetry
        * minor_radius
        * normalized_radius**2
        * math.sin(2.0 * theta - rotation * nfp * zeta)
    )
    actual_r, actual_z = rotating_ellipse_coordinates(
        normalized_radius,
        theta,
        zeta,
        major_radius,
        minor_radius,
        nfp,
        elongation,
        rotation,
        asymmetry,
    )
    assert actual_r == pytest.approx(expected_r)
    assert actual_z == pytest.approx(expected_z)


def test_stage_two_validation_for_rotating_ellipse():
    parameters = (906.0, 292.258, 4, 1.6, 1)
    r_coefficients, z_coefficients = rotating_ellipse_initial_guess(*parameters)
    surface = FourierZernikeSurface(
        r_coefficients,
        z_coefficients,
        parameters[2],
        field_period_poloidal_shift=parameters[4] * math.pi,
    )
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(*parameters)
    diagnostics = validate_surface(
        surface,
        s_values=(0.1, 0.5, 1.0),
        theta_values=(0.0, 0.7, 2.1),
        zeta_values=(0.0, 0.3, 1.2),
        r_boundary=r_boundary,
        z_boundary=z_boundary,
    )
    assert diagnostics["boundary_error"] == pytest.approx(0.0)
    assert diagnostics["periodicity_error"] == pytest.approx(0.0)
    assert diagnostics["stellarator_symmetry_error"] == pytest.approx(0.0)
    assert diagnostics["minimum_jacobian"] > 0.0
    assert diagnostics["minimum_radial_spacing"] > 0.0


def test_coefficient_layout_enforces_boundary_constraints():
    parameters = (906.0, 292.258, 4, 1.6, 1)
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(*parameters)
    layout = CoefficientLayout.from_orders(
        n_max=1,
        m_max=1,
        l_max=3,
        r_boundary=r_boundary,
        z_boundary=z_boundary,
        twisted_rotation=parameters[4],
    )
    r_initial, z_initial = rotating_ellipse_initial_guess(*parameters)
    vector = layout.initial_vector(r_initial, z_initial)
    r_full, z_full = layout.coefficients_from_vector(vector)
    surface = FourierZernikeSurface(
        r_full,
        z_full,
        parameters[2],
        field_period_poloidal_shift=parameters[4] * math.pi,
    )
    assert boundary_error(surface, r_boundary, z_boundary) == pytest.approx(0.0)


def test_jax_action_matches_scalar_reference_and_has_gradient():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    parameters = (906.0, 292.258, 4, 1.6, 1)
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(*parameters)
    layout = CoefficientLayout.from_orders(
        n_max=1,
        m_max=1,
        l_max=3,
        r_boundary=r_boundary,
        z_boundary=z_boundary,
        twisted_rotation=parameters[4],
    )
    r_initial, z_initial = rotating_ellipse_initial_guess(*parameters)
    vector = list(layout.initial_vector(r_initial, z_initial))
    # Move away from the symmetric zero higher-order modes so that the scalar
    # finite-difference diagnostic is not dominated by cancellation.
    vector[0] = 10.0
    grid = ActionGrid.uniform(n_s=3, n_theta=4, n_zeta=2)
    omega = 0.25
    action = make_action(layout, parameters[2], grid, omega)
    value_and_gradient = jax_value_and_gradient(action, jit=False)
    value, gradient = value_and_gradient(jax.numpy.asarray(vector, dtype=jax.numpy.float64))
    r_full, z_full = layout.coefficients_from_vector(vector)
    reference = scalar_action_reference(r_full, z_full, parameters[2], grid, omega)
    assert float(value) == pytest.approx(reference, rel=1.0e-12)
    assert gradient.shape == (layout.n_free,)
    assert all(math.isfinite(float(component)) for component in gradient)

    # The geometry is expressed in centimetres and the action is large in
    # absolute units, so use a resolvable perturbation for the scalar check.
    step = 1.0e-3
    plus = list(vector)
    minus = list(vector)
    plus[0] += step
    minus[0] -= step
    plus_r, plus_z = layout.coefficients_from_vector(plus)
    minus_r, minus_z = layout.coefficients_from_vector(minus)
    finite_difference = (
        scalar_action_reference(plus_r, plus_z, parameters[2], grid, omega)
        - scalar_action_reference(minus_r, minus_z, parameters[2], grid, omega)
    ) / (2.0 * step)
    assert float(gradient[0]) == pytest.approx(finite_difference, rel=1.0e-5)


def test_scipy_optimizer_returns_valid_surface():
    pytest.importorskip("scipy")
    parameters = (10.0, 2.0, 2, 1.4, 1)
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(*parameters)
    layout = CoefficientLayout.from_orders(
        n_max=1,
        m_max=1,
        l_max=3,
        r_boundary=r_boundary,
        z_boundary=z_boundary,
        twisted_rotation=parameters[4],
    )
    r_initial, z_initial = rotating_ellipse_initial_guess(*parameters)
    initial = list(layout.initial_vector(r_initial, z_initial))
    initial[0] = 0.05
    result = optimize_surface(
        layout,
        parameters[2],
        ActionGrid.uniform(n_s=3, n_theta=6, n_zeta=2),
        omega=0.25,
        initial_vector=initial,
        r_boundary=r_boundary,
        z_boundary=z_boundary,
        field_period_poloidal_shift=parameters[4] * math.pi,
        options={"maxiter": 50, "gtol": 1.0e-8},
        jit=True,
    )
    assert result.success
    assert result.diagnostics["valid_surface"] == pytest.approx(1.0)
    assert result.diagnostics["boundary_error"] == pytest.approx(0.0)
    assert result.diagnostics["minimum_nonaxis_jacobian"] > 0.0
