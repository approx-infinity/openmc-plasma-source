"""Shared reaction-rate and fusion-power calculations."""

from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from NeSST.spectral_model import reac_DD, reac_DT, reac_TT

from ..fuel_types import get_reactions_from_fuel

_EV_TO_J = 1.602176634e-19
_FUSION_Q_MEV = {
    "DD": 7.3,
    "DT": 17.6,
    "TT": 11.3,
}
_REACTION_REACTIVITY = {
    "DD": reac_DD,
    "DT": reac_DT,
    "TT": reac_TT,
}


def reaction_rate_density(
    ion_density: Union[float, NDArray],
    ion_temperature: Union[float, NDArray],
    reaction: str,
) -> NDArray:
    """Calculate one reaction's rate density from reactivity and density.

    The reactivity is selected from NeSST for DD, DT, or TT and evaluated at
    the supplied ion temperature. ``ion_density`` is the effective reacting
    density product: ``n_a*n_b`` for unlike reactants or ``0.5*n**2`` for
    identical reactants. In this package it is normally supplied by
    :func:`reaction_fuel_density`.

    Parameters
    ----------
    ion_density : float or array-like
        Effective reacting-density product in m^-6.
    ion_temperature : float or array-like
        Ion temperature in eV, as expected by the NeSST reactivity functions.
    reaction : str
        Reaction key: ``"DD"``, ``"DT"``, or ``"TT"``.

    Returns
    -------
    numpy.ndarray
        Reaction rate density in reactions/s/m^3, broadcast over the input
        shapes.

    Raises
    ------
    ValueError
        If ``reaction`` is not one of the supported reaction keys.
    """

    try:
        reactivity = _REACTION_REACTIVITY[reaction]
    except KeyError as exc:
        raise ValueError(
            f'Reaction {reaction} not in available options ["DD", "DT", "TT"]'
        ) from exc
    return np.asarray(ion_density) * reactivity(np.asarray(ion_temperature))


def reaction_fuel_density(
    ion_density: Union[float, NDArray],
    fuel: Dict[str, float],
    reaction: str,
) -> NDArray:
    """Form the reacting-density product for DD, DT, or TT fuel.

    Fuel fractions are multiplied by the total ion density to obtain species
    densities. Identical-particle reactions include the factor of one half to
    avoid counting each pair twice.

    Parameters
    ----------
    ion_density : float or array-like
        Total ion density in m^-3.
    fuel : mapping of str to float
        Species fractions, such as ``{"D": 0.5, "T": 0.5}``.
    reaction : str
        Reaction key ``"DD"``, ``"DT"``, or ``"TT"``.

    Returns
    -------
    numpy.ndarray
        Effective density product in m^-6: ``n_D**2/2``, ``n_D*n_T``, or
        ``n_T**2/2``.

    Raises
    ------
    KeyError
        If the fuel mapping does not include a species required by the chosen
        reaction.
    ValueError
        If ``reaction`` is unsupported.
    """

    fuel_densities = {
        key: np.asarray(ion_density) * value for key, value in fuel.items()
    }
    if reaction == "DD":
        return 0.5 * fuel_densities["D"] ** 2
    if reaction == "TT":
        return 0.5 * fuel_densities["T"] ** 2
    if reaction == "DT":
        return fuel_densities["T"] * fuel_densities["D"]
    raise ValueError(
        f'Reaction {reaction} not in available options ["DD", "DT", "TT"]'
    )


def fusion_emission_densities(
    ion_density: Union[float, NDArray],
    ion_temperature: Union[float, NDArray],
    fuel: Dict[str, float],
) -> Tuple[NDArray, NDArray]:
    """Return neutron-source and fusion-power densities.

    Returns
    -------
    neutron_source_density:
        Neutrons/s/m3, including the two-neutron yield of T-T reactions.
    fusion_power_density:
        W/m3 using the existing reaction Q values.

    Parameters
    ----------
    ion_density : float or array-like
        Total ion density in m^-3.
    ion_temperature : float or array-like
        Ion temperature in eV. Density and temperature arrays are broadcast
        together before reaction rates are evaluated.
    fuel : mapping of str to float
        Fuel species fractions used to form DD, DT, and TT reacting densities.
        Only reaction channels returned by ``get_reactions_from_fuel`` are
        included.

    Returns
    -------
    tuple of numpy.ndarray
        ``(neutron_source_density, fusion_power_density)``. Units are
        neutrons/s/m^3 and W/m^3, respectively. The T-T channel contributes
        two neutrons per reaction; the other supported channels contribute one.
    """

    density = np.asarray(ion_density)
    temperature = np.asarray(ion_temperature)
    density, temperature = np.broadcast_arrays(density, temperature)
    total_source_density = np.zeros_like(density, dtype=float)
    total_fusion_power_density = np.zeros_like(density, dtype=float)

    for reaction in get_reactions_from_fuel(fuel):
        fuel_density = reaction_fuel_density(density, fuel, reaction)
        reaction_rate = reaction_rate_density(fuel_density, temperature, reaction)
        total_source_density += reaction_rate * (2 if reaction == "TT" else 1)
        total_fusion_power_density += (
            reaction_rate * _FUSION_Q_MEV[reaction] * 1e6 * _EV_TO_J
        )

    return total_source_density, total_fusion_power_density
