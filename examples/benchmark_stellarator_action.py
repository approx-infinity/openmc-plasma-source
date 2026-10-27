"""Benchmark the source action for the paper's one-parameter bean example."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from openmc_plasma_source.stellarator_source import ActionGrid, CoefficientLayout
from openmc_plasma_source.stellarator_source.geometry import FourierZernikeSurface
from openmc_plasma_source.stellarator_source.optimization import (
    jax_value_and_gradient,
    make_action,
    scalar_action_reference,
)


jax.config.update("jax_enable_x64", True)

R_BOUNDARY = {(0.0, 0): 1.0, (0.0, 1): 0.6, (0.0, 2): 0.8}
Z_BOUNDARY = {(0.0, 1): 0.8, (0.0, 2): 0.1}
GRID = ActionGrid.uniform(7, 32, 1)
PAPER_R20_AT_ZERO_OMEGA = -0.42
OMEGA_VALUES = (0.0, 0.01, 0.5, 1.0)


def make_layout() -> CoefficientLayout:
    """Build the source layout, dropping the identically zero sin(0) mode."""

    layout = CoefficientLayout.from_orders(
        n_max=0,
        m_max=2,
        l_max=2,
        r_boundary=R_BOUNDARY,
        z_boundary=Z_BOUNDARY,
    )
    return CoefficientLayout(
        r_modes=layout.r_modes,
        z_modes=tuple(
            mode
            for mode in layout.z_modes
            if not (mode[0] == 0.0 and mode[1] == 0)
        ),
        r_boundary=layout.r_boundary,
        z_boundary=layout.z_boundary,
    )


def main() -> None:
    layout = make_layout()
    if layout.n_free != 1:
        raise RuntimeError(f"expected one active coefficient, found {layout.n_free}")

    def source_action(r20: float, omega: float) -> float:
        r_coefficients, z_coefficients = layout.coefficients_from_vector([r20])
        return scalar_action_reference(
            r_coefficients,
            z_coefficients,
            nfp=1,
            grid=GRID,
            omega=omega,
        )

    print(f"Action grid: {len(GRID.s_nodes)} radial nodes x {GRID.n_theta} theta cells")
    print(f"Paper omega=0 reference: r20={PAPER_R20_AT_ZERO_OMEGA:.10f}")
    minima = {}
    for omega in OMEGA_VALUES:
        result = minimize_scalar(
            lambda r20: source_action(float(r20), omega),
            bounds=(-0.8, 0.2),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        minima[omega] = float(result.x)
        r_coefficients, z_coefficients = layout.coefficients_from_vector([result.x])
        surface = FourierZernikeSurface(r_coefficients, z_coefficients, nfp=1)
        try:
            geometry = surface.validate(
                np.linspace(0.0, 1.0, 41),
                np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False),
                (0.0,),
                require_positive_jacobian=False,
            )
            minimum_jacobian = geometry["minimum_nonaxis_jacobian"]
            validity = (
                f"minJ={minimum_jacobian:.3e}, "
                f"maxJ={geometry['maximum_jacobian']:.3e}, "
                f"{'valid' if minimum_jacobian > 0.0 else 'invalid'}"
            )
        except ValueError as error:
            validity = f"invalid: {error}"
        print(
            f"omega={omega:g}: r20={result.x:.10f}, "
            f"S={result.fun:.10f}, geometry={validity}"
        )

    print("Stationary points of the unpenalized source action:")
    parameter_samples = np.linspace(-0.8, 0.2, 101)
    derivative_step = 1.0e-5
    for omega in OMEGA_VALUES[1:]:
        def derivative(r20: float) -> float:
            return (
                source_action(r20 + derivative_step, omega)
                - source_action(r20 - derivative_step, omega)
            ) / (2.0 * derivative_step)

        sampled_derivatives = [derivative(float(value)) for value in parameter_samples]
        roots = []
        for left, right, dleft, dright in zip(
            parameter_samples[:-1],
            parameter_samples[1:],
            sampled_derivatives[:-1],
            sampled_derivatives[1:],
        ):
            if dleft * dright < 0.0:
                roots.append(brentq(derivative, float(left), float(right)))
        print(f"  omega={omega:g}: " + (", ".join(f"r20={root:.8f}" for root in roots) or "no interior stationary point"))

    print(
        "Source action at paper r20=-0.42: "
        + ", ".join(
            f"omega={omega:g}: {source_action(PAPER_R20_AT_ZERO_OMEGA, omega):.10f}"
            for omega in OMEGA_VALUES
        )
    )

    action = make_action(layout, nfp=1, grid=GRID, omega=0.0)
    value_and_gradient = jax_value_and_gradient(action, jit=False)
    value, gradient = value_and_gradient(jnp.asarray([PAPER_R20_AT_ZERO_OMEGA]))
    print(f"JAX source action at paper r20: S={float(value):.10f}")
    print(f"JAX dS/dr20 at paper r20: {float(gradient[0]):.10f}")
    step = 1.0e-5
    finite_difference = (
        source_action(PAPER_R20_AT_ZERO_OMEGA + step, 0.0)
        - source_action(PAPER_R20_AT_ZERO_OMEGA - step, 0.0)
    ) / (2.0 * step)
    print(f"Scalar-reference centered derivative: {finite_difference:.10f}")

    for label, r20 in (
        ("omega=0 source minimum", minima[0.0]),
        ("paper reference", PAPER_R20_AT_ZERO_OMEGA),
    ):
        r_coefficients, z_coefficients = layout.coefficients_from_vector([r20])
        surface = FourierZernikeSurface(r_coefficients, z_coefficients, nfp=1)
        try:
            geometry = surface.validate(
                np.linspace(0.0, 1.0, 41),
                np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False),
                (0.0,),
            )
            print(
                f"{label} geometry: valid, "
                f"minimum non-axis Jacobian="
                f"{geometry['minimum_nonaxis_jacobian']:.10e}"
            )
        except ValueError as error:
            print(f"{label} geometry: invalid ({error})")

    print("Sampled source action curve:")
    for r20 in np.linspace(-0.6, 0.0, 7):
        print(
            f"  r20={r20:+.3f}: "
            f"S(omega=0)={source_action(float(r20), 0.0):.10f}, "
            f"S(omega=0.01)={source_action(float(r20), 0.01):.10f}"
        )


if __name__ == "__main__":
    main()
