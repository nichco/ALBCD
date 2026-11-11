"""Spanwise distributions of the initial and the monolithic optimal design, for the
problem-description figures (fig_problem_*.py).

Evaluates the aerodynamic and structural models of models.py at the initial design
and at the solution in monolithic_solution.npz (regenerate it with monolithic.py),
and saves the lift, twist, wall thickness, von Mises stress and deflection along
the span, plus the twisted VLM mesh and its pressure coefficients, to
cessna_problem.npz.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # the models work in float64
import jax.numpy as jnp

from models import (aero_model, structures_model, twist_cp0, thickness_cp0,
                    bspline_twist, bspline_thickness, y, b, q, v_inf, rho_atm, mesh0_jnp, vlm_geom, vlm_aero,
                    solve_aero, r, load_factor, safety_factor, sigma_yield_mpa, min_gauge, chords0, m0)
from jvlm.analysis import panel_pressure

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(twist_cp, thickness_cp):
    """Spanwise distributions of one design, with the structure loaded by its own aero loads."""
    twist = bspline_twist @ twist_cp
    thickness = bspline_thickness @ thickness_cp

    def_mesh = vlm_geom.apply_twist_rotation(mesh0_jnp, jnp.rad2deg(twist),
                                             ref_axis_pos=0.25, symmetry=False, rotate_x=True)
    out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
    strip_forces = vlm_aero.sectional_forces(out["sec_forces"])
    lift_per_span = strip_forces[:, 2] / out["widths"]   # N/m, 1 g cruise
    cp = panel_pressure(out["mesh"], out["sec_forces"], out["normals"], orient="up") / q

    # the structure, with its stress and deflection fields
    _, aero_loads, lift = aero_model(twist_cp)
    max_sigma_ks, min_thickness_ks, weight, beam, u = structures_model(aero_loads, thickness_cp, full_output=True)
    sigma_mpa = beam.stress(u) / 1e6

    return dict(twist_deg=np.degrees(np.array(twist)), thickness_mm=np.array(thickness) * 1e3,
                lift_per_span=np.array(lift_per_span), sigma_mpa=np.array(sigma_mpa),
                w=np.array(u[:, 2]), mesh=np.array(out["mesh"]), cp=np.array(cp),
                CD=float(out["CD"]), CL=float(out["CL"]), L=float(lift), W=float(weight),
                spar_mass=float(beam.mass), max_sigma_ks=float(max_sigma_ks),
                min_thickness_ks=float(min_thickness_ks))


solution = np.load(os.path.join(HERE, "monolithic_solution.npz"))
designs = {"init": evaluate(twist_cp0, thickness_cp0),
           "opt": evaluate(solution["twist_cp"], solution["thickness_cp"])}

data = dict(y=y, y_elem=0.5 * (y[:-1] + y[1:]), b=b, radius=np.array(r), chords=np.array(chords0),
            sigma_yield=sigma_yield_mpa, min_gauge_mm=min_gauge * 1e3, m0=m0, n_load=load_factor * safety_factor,
            twist_cp_opt=solution["twist_cp"], thickness_cp_opt=solution["thickness_cp"])
for tag, d in designs.items():
    data.update({f"{k}_{tag}": v for k, v in d.items()})
    print(f"{tag}: CD = {d['CD']:.5f}, CL = {d['CL']:.4f}, L = {d['L']:.0f} N, W = {d['W']:.0f} N, "
          f"spar mass = {d['spar_mass']:.1f} kg, max sigma = {d['sigma_mpa'].max():.1f} MPa, "
          f"tip deflection = {d['w'][-1]:.3f} m")

np.savez(os.path.join(HERE, "cessna_problem.npz"), **data)
