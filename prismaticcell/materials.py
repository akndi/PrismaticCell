"""Material access and thermal homogenization of the electrode sandwich.

Implements PHYSICS.md §1. All effective properties are computed from the per-layer
material data — nothing is hard-coded. A sandwich (one repeat unit) is homogenized into
an anisotropic block: through-plane (z) conductivity is a series/harmonic average,
in-plane (x,y) is a thickness-weighted parallel average, and heat capacity/density are
mass(thickness)-weighted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import Layer, Material


@dataclass
class HomogenizedProps:
    """Effective anisotropic thermal properties of a homogenized region."""
    density: float      # kg/m^3, effective density
    cp: float           # J/kg/K, effective specific heat
    k_x: float          # W/m/K, in-plane
    k_y: float          # W/m/K, in-plane
    k_z: float          # W/m/K, through-plane
    thickness: float    # m, total thickness of the homogenized set

    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity [J/m^3/K]."""
        return self.density * self.cp


def sandwich_thickness(layers: List[Layer]) -> float:
    """Total through-plane thickness of one sandwich [m]."""
    return sum(layer.thickness for layer in layers)


def homogenize(layers: List[Layer], materials: Dict[str, Material]) -> HomogenizedProps:
    """Homogenize a set of layers into anisotropic effective properties (PHYSICS §1).

    - k_z  = (Σ t_i) / Σ (t_i / k_through_i)          (series / harmonic)
    - k_xy = Σ (t_i k_in_i) / Σ t_i                    (parallel / thickness-weighted)
    - ρcp  = Σ (t_i ρ_i cp_i) / Σ t_i                  (volumetric heat capacity)
    - ρ    = Σ (t_i ρ_i) / Σ t_i
    """
    if not layers:
        raise ValueError("homogenize() requires at least one layer")

    t_total = 0.0
    inv_kz_sum = 0.0        # Σ t_i / k_through_i
    kx_sum = 0.0            # Σ t_i k_in_i
    ky_sum = 0.0
    rho_t_sum = 0.0        # Σ t_i ρ_i
    rhocp_t_sum = 0.0      # Σ t_i ρ_i cp_i

    for layer in layers:
        mat = materials[layer.material]
        t = layer.thickness
        if t <= 0:
            raise ValueError(f"layer '{layer.role}' has non-positive thickness {t}")
        if mat.k_through <= 0 or mat.k_in <= 0:
            raise ValueError(f"material '{mat.name}' has non-positive conductivity")
        t_total += t
        inv_kz_sum += t / mat.k_through
        kx_sum += t * mat.k_in
        ky_sum += t * mat.k_in
        rho_t_sum += t * mat.density
        rhocp_t_sum += t * mat.density * mat.cp

    density = rho_t_sum / t_total
    cp = (rhocp_t_sum / t_total) / density if density > 0 else 0.0
    k_z = t_total / inv_kz_sum
    k_x = kx_sum / t_total
    k_y = ky_sum / t_total
    return HomogenizedProps(density=density, cp=cp, k_x=k_x, k_y=k_y, k_z=k_z, thickness=t_total)


def isotropic_props(material: Material, thickness: float) -> HomogenizedProps:
    """Wrap a single (isotropic or anisotropic) material as HomogenizedProps.

    Used for non-composite regions such as the can wall, tabs, and gas gaps.
    """
    return HomogenizedProps(
        density=material.density,
        cp=material.cp,
        k_x=material.k_in,
        k_y=material.k_in,
        k_z=material.k_through,
        thickness=thickness,
    )
