"""Wing VLM camber mesh and beam mesh of the Vashon Ranger R7, generated from its STEP geometry with jeometry, and
an FFD parameterization of the wing's jig twist that updates both meshes. ranger.stp holds the entire aircraft
(fuselage, wing, tails, propeller and landing gear); the meshes and the FFD use only its wing, and
``twist_ffd.twisted_aircraft`` puts the twisted wing back on the aircraft for plotting. models.py imports
``twist_ffd``; run this file to plot the meshes on the aircraft at an example twist."""
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jeometry as jg

ranger = jg.import_geometry(Path(__file__).parent / "ranger.stp")
wing = ranger.declare_component(function_search_names=["WingGeom"], name="wing")
num_spanwise, num_chordwise = 41, 12

# region wing vlm
# The wing is a straight, untwisted rectangle, set on the fuselage at a constant incidence of about 2 deg (nose up),
# so its leading (trailing) edge is the frontmost (rearmost) point of its surface, repeated along the span. The
# camber surface lies midway between the upper and lower surfaces, found by projecting the chord surface
# vertically onto the wing from above and below; it carries the incidence, which the twist adds to.
points = np.asarray(wing.evaluate(wing.generate_parametric_grid(100)))
leading_edge = np.tile(points[points[:, 0].argmin()], (num_spanwise, 1))
trailing_edge = np.tile(points[points[:, 0].argmax()], (num_spanwise, 1))
leading_edge[:, 1] = trailing_edge[:, 1] = np.linspace(points[:, 1].min(), points[:, 1].max(), num_spanwise)
chord_surface = np.linspace(leading_edge, trailing_edge, num_chordwise)  # (chordwise, spanwise, 3)

up = np.array([0., 0., 1.])
camber_upper = wing.project(chord_surface + up, direction=-up)  # parametric coordinates on the wing
camber_lower = wing.project(chord_surface - up, direction=up)
wing_camber_mesh = ((wing.evaluate(camber_upper) + wing.evaluate(camber_lower)) / 2).reshape(chord_surface.shape)
# endregion wing vlm

# region wing beam
# The beam runs along a fixed fraction of the chord, with a node at each VLM spanwise station, midway between the
# upper and lower surfaces there; their distance is the depth available to the wing box.
beam_chord_fraction = 0.25
beam_chord_points = (1 - beam_chord_fraction) * leading_edge + beam_chord_fraction * trailing_edge
beam_upper = wing.project(beam_chord_points + up, direction=-up)
beam_lower = wing.project(beam_chord_points - up, direction=up)
wing_beam_mesh = (wing.evaluate(beam_upper) + wing.evaluate(beam_lower)) / 2  # (spanwise, 3)
wing_beam_height = jnp.linalg.norm(wing.evaluate(beam_upper) - wing.evaluate(beam_lower), axis=-1)  # (spanwise,)
# endregion wing beam

# region htail vlm
# The horizontal tail is built like the wing: a straight, untwisted rectangle (at an incidence of about -1 deg), so
# its camber surface comes from the same vertical projections of its chord surface. So far it is only plotted.
htail = ranger.declare_component(function_search_names=["htail"], name="htail")
htail_num_spanwise, htail_num_chordwise = 21, 12
htail_points = np.asarray(htail.evaluate(htail.generate_parametric_grid(100)))
htail_leading_edge = np.tile(htail_points[htail_points[:, 0].argmin()], (htail_num_spanwise, 1))
htail_trailing_edge = np.tile(htail_points[htail_points[:, 0].argmax()], (htail_num_spanwise, 1))
htail_leading_edge[:, 1] = htail_trailing_edge[:, 1] = np.linspace(htail_points[:, 1].min(), htail_points[:, 1].max(),
                                                                   htail_num_spanwise)
htail_chord_surface = np.linspace(htail_leading_edge, htail_trailing_edge, htail_num_chordwise)

htail_camber_upper = htail.project(htail_chord_surface + up, direction=-up)  # parametric coordinates on the htail
htail_camber_lower = htail.project(htail_chord_surface - up, direction=up)
htail_camber_mesh = ((htail.evaluate(htail_camber_upper) + htail.evaluate(htail_camber_lower)) / 2
                     ).reshape(htail_chord_surface.shape)  # (chordwise, spanwise, 3)
# endregion htail vlm

# The wing's faces are linear across each semispan, with control points only at its root and tip, so an FFD could
# only twist them linearly. Insert knots spanwise, exactly (by linear interpolation), so that every face has a row
# of control points at each VLM station of its semispan. The surface and its parametrization are unchanged, so the
# projections above still hold; they are done first because projecting onto the refined faces is slower.
num_semispan = num_spanwise // 2 + 1
for i, f in wing.functions.items():
    assert f.space.degree[0] == 1 and f.space.coefficients_shape[0] == 2
    knots = (np.concatenate([[0.], np.linspace(0., 1., num_semispan), [1.]]), np.asarray(f.space.knots[1]))
    space = jg.BSplineSpace(degree=f.space.degree, knots=knots)
    wing.functions[i] = jg.BSpline(space, np.linspace(f.coefficients[0], f.coefficients[1], num_semispan), f.name)

# region twist parameterization
class WingTwistFFD:
    """The wing's jig twist, parameterized by an FFD block: twist control points in, VLM and beam meshes out.

    A box FFD block around the wing (linear in all three directions) is viewed as ``num_sections`` spanwise
    sections. Each section rotates about the spanwise direction through the beam axis by the twist there, a cubic
    jeometry B-spline of the control points across the span (left tip to right tip), positive nose up. The twist is
    applied explicitly, with no ``ParameterizationSolver``.

    The meshes are evaluated at fixed parametric coordinates on the wing, whose control points move linearly with
    the FFD block's, so the map from the block's control points to the mesh points is a constant matrix, computed
    once. A call is then a rotation of the sections and one matrix product, compiled with ``jax.jit`` and
    differentiable in the twist.
    """

    def __init__(self, wing, camber_upper, camber_lower, beam_upper, beam_lower, camber_shape, num_twist_cp=7,
                 num_sections=None, aircraft=None):
        """The ``camber_*`` and ``beam_*`` arguments are the parametric coordinates (from ``wing.project``) of the
        camber surface's upper and lower points, ``camber_shape`` (chordwise, spanwise, 3) points in all, and of the
        beam's. ``num_sections`` defaults to one section per beam node. ``aircraft`` is the geometry the wing is a
        component of (its functions keep their indices there), for ``twisted_aircraft``; by default, the wing."""
        mesh_coordinates = [camber_upper, camber_lower, beam_upper, beam_lower]
        num_sections = len(beam_upper) if num_sections is None else num_sections
        self.camber_shape = camber_shape
        self.wing = wing
        self.aircraft = wing if aircraft is None else aircraft
        self.ffd_block = jg.construct_ffd_block_around_entities(wing, num_coefficients=(2, num_sections, 2),
                                                                degree=(1, 1, 1))
        self.ffd_points = np.asarray(self.ffd_block.coefficients)  # (2, num_sections, 2, 3): x, span, z

        # Linear map from the FFD block's control points to the mesh points (the same for x, y and z).
        def mesh_points(ffd_points):
            deformed = wing.copy()
            deformed.set_coefficients(self.ffd_block.evaluate_ffd(ffd_points))
            return jnp.concatenate([deformed.evaluate(c).reshape(-1, 3) for c in mesh_coordinates])

        jacobian = jax.jacfwd(mesh_points)(jnp.asarray(self.ffd_points.reshape(-1, 3)))
        self.mesh_matrix = jacobian[:, 0, :, 0]  # (num_mesh_points, num_ffd_points)
        self.split = np.cumsum([len(c) for c in mesh_coordinates])[:-1]

        # Twist at each section: a cubic B-spline of the control points over the span.
        twist_space = jg.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(num_twist_cp,))
        self.twist_basis = jnp.asarray(twist_space.compute_basis_matrix(np.linspace(0., 1., num_sections)[:, None]))

        # Twist axis: the beam axis, at the in-section (x, z) parametric coordinates of the untwisted beam's root
        # node in the box.
        box_min, box_max = self.ffd_points.reshape(-1, 3).min(0), self.ffd_points.reshape(-1, 3).max(0)
        beam_node = np.asarray(self._meshes(jnp.zeros(num_twist_cp), twist=False)[1][num_sections // 2])
        self.twist_axis_coordinate = ((beam_node - box_min) / (box_max - box_min))[[0, 2]]
        self._evaluate = jax.jit(self._meshes)

    def __call__(self, twist_cp):
        """The camber mesh (chordwise, spanwise, 3) and beam mesh (spanwise, 3) twisted by ``twist_cp`` (radians)."""
        return self._evaluate(jnp.asarray(twist_cp, dtype=float))

    def twisted_wing(self, twist_cp):
        """The wing geometry twisted by ``twist_cp`` (radians)."""
        twisted = self.wing.copy()
        twisted.set_coefficients(self.ffd_block.evaluate_ffd(self._twist(jnp.asarray(twist_cp, dtype=float))))
        return twisted

    def twisted_aircraft(self, twist_cp):
        """The entire aircraft with its wing twisted by ``twist_cp`` (radians); the other components are shared
        with ``aircraft``, unchanged."""
        functions = {**self.aircraft.functions, **self.twisted_wing(twist_cp).functions}
        return jg.Geometry(functions, self.aircraft.function_names, name=self.aircraft.name)

    def _twist(self, twist_cp):
        """The FFD block's control points with its sections rotated by the twist."""
        sections = jg.SectionalParameterization(self.ffd_points, principal_parametric_dimension=1)
        parameters = jg.SectionalParameters()
        parameters.add_rotation(axis=np.array([0., 1., 0.]), rotation=self.twist_basis @ twist_cp,
                                parametric_coordinate=self.twist_axis_coordinate)
        return sections.evaluate(parameters)

    def _meshes(self, twist_cp, twist=True):
        ffd_points = self._twist(twist_cp) if twist else self.ffd_points
        camber_upper, camber_lower, beam_upper, beam_lower = jnp.split(self.mesh_matrix @ ffd_points.reshape(-1, 3),
                                                                       self.split)
        return ((camber_upper + camber_lower) / 2).reshape(self.camber_shape), (beam_upper + beam_lower) / 2
# endregion twist parameterization


twist_ffd = WingTwistFFD(wing, camber_upper, camber_lower, beam_upper, beam_lower, chord_surface.shape,
                         aircraft=ranger)

if __name__ == "__main__":
    example_twist_cp = np.deg2rad([0., 2., 4., 6., 4., 2., 0.])
    twisted_camber_mesh, twisted_beam_mesh = twist_ffd(example_twist_cp)
    twist_ffd.twisted_aircraft(example_twist_cp).plot_meshes(
        [twisted_camber_mesh, twisted_beam_mesh, htail_camber_mesh], mesh_color="silver", mesh_opacity=0.4)
