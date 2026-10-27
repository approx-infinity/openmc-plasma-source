"""SciPy optimization wrapper for the JAX stellarator action."""

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .geometry import FourierZernikeSurface
from .optimization import (
    ActionGrid,
    CoefficientLayout,
    _evaluate_point,
    _zernike_radial_derivative_jax,
    _zernike_radial_jax,
    jax_value_and_gradient,
    make_action,
)
from .validation import validate_surface

ModeKey = Tuple[float, int, int]


@dataclass
class StellaratorOptimizationResult:
    """Surface, optimizer state, and diagnostics from one minimization.

    Attributes
    ----------
    surface : FourierZernikeSurface
        Mapping assembled from the final coefficient vector.
    coefficient_vector : tuple of float
        Final free coefficients in the exact order defined by the layout.
    scipy_result : scipy.optimize.OptimizeResult
        Full SciPy result, including status, message, gradients, and evaluation
        counts.
    diagnostics : dict of str to float
        Initial/final objective, gradient norm, iteration counts, and any
        geometry or boundary diagnostics that could be computed. The key
        ``valid_surface`` is 1.0 when validation passed and 0.0 when it failed
        but ``require_valid_surface=False`` allowed the result to be returned.
    """

    surface: FourierZernikeSurface
    coefficient_vector: Tuple[float, ...]
    scipy_result: object
    diagnostics: Dict[str, float]

    @property
    def success(self) -> bool:
        """Whether SciPy reports that its termination criteria were met."""

        return bool(self.scipy_result.success)

    @property
    def message(self) -> str:
        """Message from SciPy describing why the optimization stopped."""

        return str(self.scipy_result.message)


def _evaluate_geometry_point(r_terms, z_terms, nfp: int, s, theta, zeta):
    """Evaluate R, Z, and signed cylindrical Jacobian with JAX scalars.

    The Jacobian is ``R*(R_s*Z_theta - R_theta*Z_s)``. This helper computes
    only the derivatives needed by the geometry barrier and avoids building a
    :class:`SurfaceEvaluation` object during JAX tracing.
    """

    import jax.numpy as jnp

    R = jnp.asarray(0.0)
    R_s = jnp.asarray(0.0)
    R_theta = jnp.asarray(0.0)
    Z = jnp.asarray(0.0)
    Z_s = jnp.asarray(0.0)
    Z_theta = jnp.asarray(0.0)

    for (n, m, l), coefficient in r_terms:
        phase = m * theta - n * nfp * zeta
        radial = _zernike_radial_jax(s, l, m)
        radial_s = _zernike_radial_derivative_jax(s, l, m)
        R = R + coefficient * radial * jnp.cos(phase)
        R_s = R_s + coefficient * radial_s * jnp.cos(phase)
        R_theta = R_theta - coefficient * radial * m * jnp.sin(phase)

    for (n, m, l), coefficient in z_terms:
        phase = m * theta - n * nfp * zeta
        radial = _zernike_radial_jax(s, l, m)
        radial_s = _zernike_radial_derivative_jax(s, l, m)
        Z = Z + coefficient * radial * jnp.sin(phase)
        Z_s = Z_s + coefficient * radial_s * jnp.sin(phase)
        Z_theta = Z_theta + coefficient * radial * m * jnp.cos(phase)

    jacobian = R * (R_s * Z_theta - R_theta * Z_s)
    return R, Z, jacobian


def make_geometry_penalty(
    layout: CoefficientLayout,
    nfp: int,
    grid: ActionGrid,
    *,
    jacobian_floor: float = 0.0,
    jacobian_scale: float = 1.0,
    radius_floor: float = 0.0,
    radius_scale: float = 1.0,
    penalty_weight: float = 1.0e12,
):
    """Create a soft barrier for radius and Jacobian violations on the grid.

    Softplus barriers penalize non-positive Jacobians and cylindrical radii
    without using non-differentiable ``abs`` or hard clipping. The final
    surface is still checked with strict validation after SciPy terminates.

    Parameters
    ----------
    layout : CoefficientLayout
        Basis and boundary constraints used to assemble the candidate mapping.
    nfp : int
        Number of field periods.
    grid : ActionGrid
        Sample locations used by the barrier. Jacobians are checked for
        non-axis radial nodes; radius is checked at all radial nodes.
    jacobian_floor, radius_floor : float, default=0
        Minimum preferred cylindrical Jacobian and radius. Values below each
        floor contribute to the penalty.
    jacobian_scale, radius_scale : float, default=1
        Positive softplus transition scales in the corresponding physical
        units. Smaller scales make the barrier more sharply concentrated near
        a floor.
    penalty_weight : float, default=1e12
        Non-negative multiplier applied to the sum of squared softplus
        violations. Its magnitude must be chosen relative to the action scale.

    Returns
    -------
    callable
        Differentiable function ``penalty(x)`` of the packed free-coefficient
        vector. It returns a scalar JAX value.

    Raises
    ------
    ValueError
        If either scale is non-positive or the penalty weight is negative.
    """

    import jax.numpy as jnp

    if jacobian_scale <= 0.0 or radius_scale <= 0.0:
        raise ValueError("barrier scales must be positive")
    if penalty_weight < 0.0:
        raise ValueError("penalty_weight must be non-negative")

    def penalty(x):
        """Evaluate the summed radius/Jacobian barrier for one vector."""

        r_terms, z_terms = layout.coefficients(x)
        total = jnp.asarray(0.0)
        dtheta = 2.0 * pi / grid.n_theta
        dzeta = 2.0 * pi / grid.n_zeta
        for s in grid.s_nodes:
            for j in range(grid.n_theta):
                theta = j * dtheta
                for k in range(grid.n_zeta):
                    zeta = k * dzeta
                    radius, _z, jacobian = _evaluate_geometry_point(
                        r_terms, z_terms, nfp, s, theta, zeta
                    )
                    bad_radius = jnp.logaddexp(
                        0.0, (radius_floor - radius) / radius_scale
                    )
                    if s > 0.0:
                        bad_jacobian = jnp.logaddexp(
                            0.0, (jacobian_floor - jacobian) / jacobian_scale
                        )
                        total = total + bad_jacobian**2
                    total = total + bad_radius**2
        return penalty_weight * total

    return penalty


def make_optimization_objective(
    layout: CoefficientLayout,
    nfp: int,
    grid: ActionGrid,
    omega: float,
    *,
    use_geometry_penalty: bool = True,
    jacobian_floor: float = 0.0,
    jacobian_scale: float = 1.0,
    radius_floor: float = 0.0,
    radius_scale: float = 1.0,
    penalty_weight: float = 1.0e12,
):
    """Combine the source action with an optional geometry soft barrier.

    Parameters match :func:`make_action` and :func:`make_geometry_penalty`.
    When ``use_geometry_penalty`` is false, the returned function is exactly
    the action. When true, it is the sum of the action and the sampled soft
    barrier; this changes the optimized objective and is not part of the bare
    theory action.

    Returns
    -------
    callable
        Scalar JAX objective function of the packed free coefficients.
    """

    action = make_action(layout, nfp, grid, omega)
    if not use_geometry_penalty:
        return action
    penalty = make_geometry_penalty(
        layout,
        nfp,
        grid,
        jacobian_floor=jacobian_floor,
        jacobian_scale=jacobian_scale,
        radius_floor=radius_floor,
        radius_scale=radius_scale,
        penalty_weight=penalty_weight,
    )

    def objective(x):
        """Return the action and geometry penalty for one coefficient vector."""

        return action(x) + penalty(x)

    return objective


def optimize_surface(
    layout: CoefficientLayout,
    nfp: int,
    grid: ActionGrid,
    omega: float,
    initial_vector: Sequence[float],
    *,
    validation_s_values: Iterable[float] | None = None,
    validation_theta_values: Iterable[float] | None = None,
    validation_zeta_values: Iterable[float] | None = None,
    r_boundary: Mapping[Tuple[int, int], float] | None = None,
    z_boundary: Mapping[Tuple[int, int], float] | None = None,
    method: str = "L-BFGS-B",
    options: Mapping[str, float | int] | None = None,
    use_geometry_penalty: bool = True,
    jacobian_floor: float = 0.0,
    jacobian_scale: float = 1.0,
    radius_floor: float = 0.0,
    radius_scale: float = 1.0,
    penalty_weight: float = 1.0e12,
    jit: bool = True,
    require_valid_surface: bool = True,
    field_period_poloidal_shift: float = 0.0,
    geometry_description: str | None = None,
) -> StellaratorOptimizationResult:
    """Minimize the action over free coefficients and construct the surface.

    SciPy performs the selected optimization method using the value and
    gradient supplied by JAX. The optimizer can add a differentiable sampled
    geometry penalty, then validates the returned surface on explicit grids or
    on grids derived from ``grid``. Boundary checks are performed only when
    both boundary mappings are supplied.

    Parameters
    ----------
    layout : CoefficientLayout
        Basis modes, boundary coefficients, and free-variable ordering.
    nfp : int
        Number of toroidal field periods.
    grid : ActionGrid
        Action quadrature grid and default validation grid.
    omega : float
        Weight of radial length in the action.
    initial_vector : sequence of float
        Starting free-coefficient vector; its length must equal ``layout.n_free``.
    validation_s_values, validation_theta_values, validation_zeta_values : iterable, optional
        Explicit validation coordinates. Omitted values default to the radial
        nodes and angular cell origins from ``grid``.
    r_boundary, z_boundary : mapping, optional
        Expected external-boundary Fourier coefficients. Supply both or
        neither.
    method : str, default="L-BFGS-B"
        SciPy minimization method name.
    options : mapping, optional
        Method-specific SciPy options passed to ``scipy.optimize.minimize``.
    use_geometry_penalty : bool, default=True
        If true, add the sampled soft barrier from
        :func:`make_geometry_penalty` to the action.
    jacobian_floor, radius_floor : float, default=0
        Preferred lower bounds used by the geometry barrier.
    jacobian_scale, radius_scale : float, default=1
        Positive softplus transition scales for that barrier.
    penalty_weight : float, default=1e12
        Non-negative geometry-barrier multiplier.
    jit : bool, default=True
        JIT-compile the JAX value-and-gradient function.
    require_valid_surface : bool, default=True
        If true, raise when validation fails or SciPy reports unsuccessful
        termination. If false, return the best result with
        ``diagnostics['valid_surface']`` set to 0.0 when geometry validation
        fails; inspect ``scipy_result.success`` separately.
    field_period_poloidal_shift : float, default=0
        Poloidal seam shift in radians applied by the resulting surface's
        periodicity diagnostics.
    geometry_description : str, optional
        Short label included in the startup message.

    Returns
    -------
    StellaratorOptimizationResult
        Final surface, coefficient vector, SciPy result, and diagnostics.

    Raises
    ------
    ValueError
        If vector lengths or boundary arguments are inconsistent, or strict
        surface validation fails.
    RuntimeError
        If SciPy does not report convergence while ``require_valid_surface``
        is true.
    """

    import numpy as np
    from scipy.optimize import minimize

    if len(initial_vector) != layout.n_free:
        raise ValueError(
            f"expected {layout.n_free} initial coefficients, got {len(initial_vector)}"
        )
    if (r_boundary is None) != (z_boundary is None):
        raise ValueError("both R and Z boundary coefficients are required")

    objective = make_optimization_objective(
        layout,
        nfp,
        grid,
        omega,
        use_geometry_penalty=use_geometry_penalty,
        jacobian_floor=jacobian_floor,
        jacobian_scale=jacobian_scale,
        radius_floor=radius_floor,
        radius_scale=radius_scale,
        penalty_weight=penalty_weight,
    )
    value_and_gradient = jax_value_and_gradient(objective, jit=jit)

    def scipy_objective(vector):
        """Adapt JAX's value-gradient pair to SciPy's ``jac=True`` protocol."""

        value, gradient = value_and_gradient(np.asarray(vector, dtype=float))
        return float(value), np.asarray(gradient, dtype=float)

    description = f", {geometry_description}" if geometry_description else ""
    print(
        f"Starting action minimization for {nfp}-period stellarator"
        f"{description}, omega={omega:g}..."
    )
    scipy_result = minimize(
        scipy_objective,
        np.asarray(initial_vector, dtype=float),
        jac=True,
        method=method,
        options=dict(options or {}),
    )
    print("Optimization finished!")
    print(f"Current function value: {scipy_result.fun:.6f}")
    print(f"Iterations: {getattr(scipy_result, 'nit', 'n/a')}")
    print(f"Function evaluations: {getattr(scipy_result, 'nfev', 'n/a')}")
    print(f"Gradient evaluations: {getattr(scipy_result, 'njev', 'n/a')}")
    print(f"Final Action Value: {scipy_result.fun}")

    coefficient_vector = tuple(float(value) for value in scipy_result.x)
    r_coefficients, z_coefficients = layout.coefficients_from_vector(
        coefficient_vector
    )
    surface = FourierZernikeSurface(
        r_coefficients,
        z_coefficients,
        nfp,
        field_period_poloidal_shift=field_period_poloidal_shift,
    )

    s_values = tuple(
        validation_s_values if validation_s_values is not None else grid.s_nodes
    )
    theta_values = tuple(
        validation_theta_values
        if validation_theta_values is not None
        else (index * 2.0 * pi / grid.n_theta for index in range(grid.n_theta))
    )
    zeta_values = tuple(
        validation_zeta_values
        if validation_zeta_values is not None
        else (index * 2.0 * pi / grid.n_zeta for index in range(grid.n_zeta))
    )

    diagnostics: Dict[str, float] = {
        "initial_objective": float(scipy_objective(initial_vector)[0]),
        "final_objective": float(scipy_result.fun),
        "final_gradient_norm": float(np.linalg.norm(scipy_result.jac)),
        "iterations": float(getattr(scipy_result, "nit", 0)),
        "function_evaluations": float(getattr(scipy_result, "nfev", 0)),
    }
    try:
        diagnostics.update(
            validate_surface(
                surface,
                s_values,
                theta_values,
                zeta_values,
                r_boundary=r_boundary,
                z_boundary=z_boundary,
            )
        )
        diagnostics["valid_surface"] = 1.0
    except ValueError:
        diagnostics["valid_surface"] = 0.0
        if require_valid_surface:
            raise

    if not scipy_result.success and require_valid_surface:
        raise RuntimeError(
            "stellarator surface optimization did not converge: "
            f"{scipy_result.message}"
        )

    return StellaratorOptimizationResult(
        surface=surface,
        coefficient_vector=coefficient_vector,
        scipy_result=scipy_result,
        diagnostics=diagnostics,
    )
