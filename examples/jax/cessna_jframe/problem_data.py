"""Spanwise distributions of the initial and the monolithic optimal design, for the
problem figure (fig_problem_render.py).

Evaluates the aerodynamic and structural models of models.py at the initial design
and at the solution in monolithic_solution.npz (regenerate it with monolithic.py),
and saves the lift, twist, spar and strut wall thickness, von Mises stress and
spar deflection, plus the twisted VLM mesh and its pressure coefficients and the
spar and strut geometry, to cessna_problem.npz.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # the models work in float64
import jax.numpy as jnp

from models import (aero_model, structures_model, build_frame, twist_cp0, thickness_cp0, strut_thickness0,
                    bspline_twist, bspline_thickness, y, b, q, v_inf, rho_atm, mesh0_jnp, vlm_geom, vlm_aero,
                    solve_aero, spar_nodes, spar_mesh, y_spar_elem, r_spar, r_strut, strut_meshes, chords0,
                    load_factor, safety_factor, sigma_yield_mpa, min_gauge, m0)
from vlm_jax.analysis import panel_pressure

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(twist_cp, thickness_cp, strut_thickness):
    """Spanwise distributions of one design, with the structure loaded by its own aero loads."""
    twist = bspline_twist @ twist_cp
    thickness = bspline_thickness @ thickness_cp

    def_mesh = vlm_geom.apply_twist_rotation(mesh0_jnp, jnp.rad2deg(twist),
                                             ref_axis_pos=0.25, symmetry=False, rotate_x=True)
    out = solve_aero(def_mesh, v_inf, rho_atm, symmetry=False, with_viscous=False)
    strip_forces = vlm_aero.sectional_forces(out["sec_forces"])
    lift_per_span = strip_forces[:, 2] / out["widths"]   # N/m, 1 g cruise
    cp = panel_pressure(out["mesh"], out["sec_forces"], out["normals"], orient="up") / q

    # the same frame as structures_model, solved here to also return the stress and deflection fields
    _, aero_loads, lift = aero_model(twist_cp)
    frame = build_frame(aero_loads, thickness_cp, strut_thickness)
    u = frame.solve()
    sigma_mpa = [np.array(beam.stress(u_beam)) / 1e6 for beam, u_beam in zip(frame.beams, u)]
    max_sigma_ks, min_thickness_ks, weight = structures_model(aero_loads, thickness_cp, strut_thickness)

    return dict(twist_deg=np.degrees(np.array(twist)), thickness_mm=np.array(thickness) * 1e3,
                strut_thickness_mm=np.array(strut_thickness) * 1e3, lift_per_span=np.array(lift_per_span),
                sigma_spar_mpa=np.concatenate(sigma_mpa[:2]), sigma_strut_mpa=np.stack(sigma_mpa[2:]),
                w=np.concatenate([np.array(u_beam[:, 2]) for u_beam in u[:2]]),
                mesh=np.array(out["mesh"]), cp=np.array(cp),
                CD=float(out["CD"]), CL=float(out["CL"]), L=float(lift), W=float(weight),
                spar_mass=float(sum(beam.mass for beam in frame.beams[:2])),
                strut_mass=float(sum(beam.mass for beam in frame.beams[2:])),
                max_sigma_ks=float(max_sigma_ks), min_thickness_ks=float(min_thickness_ks))


solution = np.load(os.path.join(HERE, "monolithic_solution.npz"))
designs = {"init": evaluate(twist_cp0, thickness_cp0, strut_thickness0),
           "opt": evaluate(solution["twist_cp"], solution["thickness_cp"], solution["strut_thickness"])}

# spar and strut geometry: the spar elements' midpoints and radii, the spar nodes, and the
# strut nodes, all ordered [left, right] as in models.py
data = dict(y=y, b=b, y_elem=y_spar_elem, radius=np.array(r_spar), chords=np.array(chords0),
            y_spar_node=y[np.concatenate(spar_nodes)], spar_mesh=spar_mesh[np.concatenate(spar_nodes)],
            strut_mesh=np.stack(strut_meshes), strut_radius=r_strut,
            sigma_yield=sigma_yield_mpa, min_gauge_mm=min_gauge * 1e3, m0=m0, n_load=load_factor * safety_factor,
            twist_cp_opt=solution["twist_cp"], thickness_cp_opt=solution["thickness_cp"],
            strut_thickness_opt=solution["strut_thickness"])
for tag, d in designs.items():
    data.update({f"{k}_{tag}": v for k, v in d.items()})
    print(f"{tag}: CD = {d['CD']:.5f}, CL = {d['CL']:.4f}, L = {d['L']:.0f} N, W = {d['W']:.0f} N, "
          f"spar mass = {d['spar_mass']:.1f} kg, strut mass = {d['strut_mass']:.2f} kg, "
          f"max spar sigma = {d['sigma_spar_mpa'].max():.1f} MPa, max strut sigma = {d['sigma_strut_mpa'].max():.1f} MPa, "
          f"tip deflection = {d['w'][-1]:.3f} m")

np.savez(os.path.join(HERE, "cessna_problem.npz"), **data)
