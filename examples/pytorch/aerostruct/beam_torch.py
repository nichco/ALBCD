"""3D Euler-Bernoulli beam finite-element model with a thin-walled tube cross-section, in PyTorch.

Run this file directly to check the tip displacement and root stress of a
cantilever against the analytic solutions.
"""

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt


class CSTube:
    def __init__(self, radius, thickness):
        r_i = radius - thickness

        self.radius = radius
        self.thickness = thickness
        self.r_i = r_i    # inner radius
        self.r_o = radius # outer radius

        r_o4_minus_r_i4 = radius**4 - r_i**4
        self.area = torch.pi * (radius**2 - r_i**2) # cross-sectional area
        self.J  = torch.pi * r_o4_minus_r_i4 / 2    # polar moment of inertia
        self.Iy = torch.pi * r_o4_minus_r_i4 / 4    # 2nd moment of inertia about y
        self.Iz = torch.pi * r_o4_minus_r_i4 / 4    # 2nd moment of inertia about z

    def max_von_mises(
        self,
        axial_strain,
        kappa_y,
        kappa_z,
        torsion_rate,
        E,
        G,
    ):
        """
        Maximum von-Mises stress anywhere on the tube cross-section.

        Parameters
        ----------
        axial_strain : float
            du/dx

        kappa_y : float
            d(theta_y)/dx

        kappa_z : float
            d(theta_z)/dx

        torsion_rate : float
            d(theta_x)/dx

        Returns
        -------
        sigma_vm_max : float
        """

        r = self.r_o

        # resultant bending curvature magnitude
        kappa = torch.sqrt(kappa_y**2 + kappa_z**2)

        sigma_axial = E * axial_strain
        sigma_bending = E * r * kappa

        sigma_max = sigma_axial + sigma_bending
        sigma_min = sigma_axial - sigma_bending

        tau_torsion = G * r * torsion_rate

        vm_pos = torch.sqrt(sigma_max**2 + 3.0 * tau_torsion**2)
        vm_neg = torch.sqrt(sigma_min**2 + 3.0 * tau_torsion**2)

        return torch.maximum(vm_pos, vm_neg)


class Beam:

    def __init__(self,
                 mesh,
                 E:           float,
                 G:           float,
                 rho:         float,
                 cs:         CSTube,
                 F,
                 fixed_nodes: list[int] | None = None,
                 fixed_dofs:  list[int] | None = None):

        self.num_nodes    = len(mesh)
        self.num_elements = self.num_nodes - 1
        n = self.num_elements
        self.cs = cs

        self.connectivity = np.array([[i, i + 1] for i in range(n)])

        # Precompute the 12 global DOF indices for every element
        self.elem_dofs = np.array([
            np.r_[6*n1 : 6*n1+6, 6*n2 : 6*n2+6]
            for n1, n2 in self.connectivity
        ])  # (num_elements, 12)

        if fixed_nodes is None:
            fixed_nodes = [0]
        if fixed_dofs is None:
            fixed_dofs = list(range(6))
        self.fixed_dofs = np.array([6 * nd + d for nd in fixed_nodes for d in fixed_dofs])
        self.free_dofs  = np.setdiff1d(np.arange(self.num_nodes * 6), self.fixed_dofs)

        # Index tensors for the gather/scatter steps below. Torch indexing wants
        # tensors (not np.ix_ grids), and these are plain integer constants, so
        # they never enter the autograd graph.
        self.elem_dofs_t = torch.as_tensor(self.elem_dofs, dtype=torch.long)
        self.free_dofs_t = torch.as_tensor(self.free_dofs, dtype=torch.long)

        self.mesh = torch.as_tensor(mesh, dtype=torch.float64)
        self.E    = torch.as_tensor(E,   dtype=torch.float64)
        self.G    = torch.as_tensor(G,   dtype=torch.float64)
        self.rho  = torch.as_tensor(rho, dtype=torch.float64)
        self.F    = torch.as_tensor(F,   dtype=torch.float64)

        self.A  = cs.area
        self.J  = cs.J
        self.Iy = cs.Iy
        self.Iz = cs.Iz

        # Element lengths and unit axes (torch, so mesh is differentiable too)
        p1   = self.mesh[self.connectivity[:, 0]]
        p2   = self.mesh[self.connectivity[:, 1]]
        diff = p2 - p1
        self.L   = torch.linalg.norm(diff, dim=1) # (n,)
        self.e_x = diff / self.L[:, None] # (n, 3)

        # calculate the beam's mass
        self.mass = torch.sum(self.rho * self.A * self.L)


    def _local_stiffness(self) -> torch.Tensor:
        L, A, E, G = self.L, self.A, self.E, self.G
        z        = torch.zeros_like(L)
        AEL      = A * E / L
        GJL      = G * self.J  / L
        c12z, c6z, c4z, c2z = (12*E*self.Iz/L**3, 6*E*self.Iz/L**2,
                                4*E*self.Iz/L, 2*E*self.Iz/L)
        c12y, c6y, c4y, c2y = (12*E*self.Iy/L**3, 6*E*self.Iy/L**2,
                                4*E*self.Iy/L, 2*E*self.Iy/L)

        # Each row_i is (n_elem, 12); stack into (n_elem, 12, 12)
        rows = [
            torch.stack([ AEL,  z,   z,   z,   z,   z,  -AEL,  z,   z,   z,   z,   z  ], 1),  # 0
            torch.stack([ z, c12z,  z,   z,   z,  c6z,  z, -c12z, z,   z,   z,  c6z  ], 1),  # 1
            torch.stack([ z,  z, c12y,  z, -c6y,  z,   z,   z, -c12y, z, -c6y,  z    ], 1),  # 2
            torch.stack([ z,  z,  z,  GJL,  z,   z,   z,   z,   z, -GJL, z,   z      ], 1),  # 3
            torch.stack([ z,  z, -c6y, z,  c4y,  z,   z,   z,  c6y,  z,  c2y,  z     ], 1),  # 4
            torch.stack([ z, c6z, z,   z,   z,  c4z,  z, -c6z,  z,   z,   z,  c2z   ], 1),  # 5
            torch.stack([-AEL, z,  z,   z,   z,   z,  AEL,  z,   z,   z,   z,   z   ], 1),  # 6
            torch.stack([ z,-c12z, z,   z,   z, -c6z,  z,  c12z, z,   z,   z, -c6z  ], 1),  # 7
            torch.stack([ z,  z,-c12y, z,  c6y,  z,   z,   z,  c12y, z,  c6y,  z    ], 1),  # 8
            torch.stack([ z,  z,  z, -GJL,  z,   z,   z,   z,   z,  GJL, z,   z     ], 1),  # 9
            torch.stack([ z,  z, -c6y, z,  c2y,  z,   z,   z,  c6y,  z,  c4y,  z    ], 1),  # 10
            torch.stack([ z, c6z, z,   z,   z,  c2z,  z, -c6z,  z,   z,   z,  c4z   ], 1),  # 11
        ]
        return torch.stack(rows, dim=1)   # (n_elem, 12, 12)


    @staticmethod
    def _rotation_matrix(ex: torch.Tensor) -> torch.Tensor:
        """
        Build the 3x3 rotation matrix for one element.

        We use torch.where (not a Python if) so this function is safe to
        call under torch.func transforms -- torch evaluates both branches
        and blends, rather than branching on a traced value.
        """
        z_axis = torch.tensor([0., 0., 1.], dtype=ex.dtype)
        x_axis = torch.tensor([1., 0., 0.], dtype=ex.dtype)

        # Pick whichever reference axis is least parallel to the element
        ref = torch.where(torch.abs(ex @ z_axis) > 0.9, x_axis, z_axis)

        ez = torch.linalg.cross(ex, ref)
        ez = ez / torch.linalg.norm(ez)
        ey = torch.linalg.cross(ez, ex)           # right-hand rule

        return torch.stack([ex, ey, ez])   # rows = local axes in global frame


    def _transform_stiffness(self, K_local: torch.Tensor) -> torch.Tensor:
        """
        Rotate every element's 12x12 stiffness from local -> global frame.

        The 12x12 transformation matrix T is block-diagonal: four copies of
        the 3x3 rotation matrix R (one per node/DOF-triplet).  We build T
        efficiently as a Kronecker product:

            T = I_4 (x) R   ->   torch.kron(torch.eye(4), R)

        then apply  K_global = T^T K_local T  via vmap over all elements.
        """
        def transform_one(K_e, ex):
            R = Beam._rotation_matrix(ex)
            T = torch.kron(torch.eye(4, dtype=R.dtype), R)    # (12, 12) block-diagonal
            return T.T @ K_e @ T

        return torch.vmap(transform_one)(K_local, self.e_x)


    def _assemble(self, K_elem: torch.Tensor) -> torch.Tensor:
        n_dofs = self.num_nodes * 6
        # elem_dofs: (n_elem, 12) — broadcast to (n_elem, 12, 12) row/col grids
        dofs = self.elem_dofs_t                       # (n_elem, 12)
        rows = dofs[:, :, None]                      # (n_elem, 12,  1)
        cols = dofs[:, None, :]                      # (n_elem,  1, 12)
        flat_idx = (rows * n_dofs + cols).reshape(-1)  # (n_elem*144,)
        flat_val = K_elem.reshape(-1)                   # (n_elem*144,)
        # index_add is the out-of-place equivalent of jax's .at[idx].add(): it
        # accumulates (rather than overwrites) on repeated indices, which is
        # exactly what an FE scatter of shared DOFs needs.
        return torch.zeros(n_dofs * n_dofs, dtype=flat_val.dtype).index_add(
            0, flat_idx, flat_val).reshape(n_dofs, n_dofs)


    def solve(self) -> torch.Tensor:
        """
        Solve K u = f for the free DOFs.

        Returns
        -------
        u : (num_nodes, 6)  nodal displacements / rotations in global frame
        """
        K_local = self._local_stiffness()
        K_elem  = self._transform_stiffness(K_local)
        K       = self._assemble(K_elem)

        f = self.F.flatten()

        # Static integer index tensors -> safe inside torch.func transforms
        K_ff = K[self.free_dofs_t[:, None], self.free_dofs_t[None, :]]
        f_f  = f[self.free_dofs_t]
        u_f  = torch.linalg.solve(K_ff, f_f)

        # Scatter free-DOF solution back into the full vector
        u = torch.zeros(self.num_nodes * 6, dtype=u_f.dtype).index_copy(
            0, self.free_dofs_t, u_f)
        return u.reshape(self.num_nodes, 6)


    def recover_strain(self, u):
        """
        Elemental strain recovery.

        Parameters
        ----------
        u : (num_nodes, 6)

        Returns
        -------
        axial_strain : (n_elem,)
        kappa_y      : (n_elem,)
        kappa_z      : (n_elem,)
        torsion_rate : (n_elem,)
        """

        # gather global element DOFs
        elem_u = u.reshape(-1)[self.elem_dofs_t]  # (n_elem, 12)

        # transform to local coordinates
        def to_local(ue, ex):
            R = Beam._rotation_matrix(ex)
            T = torch.kron(torch.eye(4, dtype=R.dtype), R)
            return T @ ue

        ul = torch.vmap(to_local)(elem_u, self.e_x)

        L = self.L

        # axial strain
        axial_strain = (ul[:, 6] - ul[:, 0]) / L

        # twist rate
        torsion_rate = (ul[:, 9] - ul[:, 3]) / L

        # bending curvatures
        kappa_y = (ul[:, 10] - ul[:, 4]) / L
        kappa_z = (ul[:, 11] - ul[:, 5]) / L

        return axial_strain, kappa_y, kappa_z, torsion_rate


    def recover_stress(self, u):

        axial_strain, kappa_y, kappa_z, torsion_rate = self.recover_strain(u)

        sigma_vm = self.cs.max_von_mises(
            axial_strain,
            kappa_y,
            kappa_z,
            torsion_rate,
            self.E,
            self.G,
        )

        return sigma_vm


if __name__ == "__main__":

    num_nodes = 31
    length    = 10.0
    mesh      = np.zeros((num_nodes, 3))
    mesh[:, 1] = np.linspace(0, length, num_nodes)

    E   = 69e9
    G   = 26e9
    rho = 2700
    P = 10_000.0 # tip load in N

    F = np.zeros((num_nodes, 6))
    F[-1, 2] = P # load in the global Z direction

    radius = 0.5
    thickness = 0.001
    cs   = CSTube(radius=radius, thickness=thickness)
    beam = Beam(mesh=mesh, E=E, G=G, rho=rho, cs=cs, F=F, fixed_nodes=[0])

    u = beam.solve()
    print('shape of u:', u.shape)

    plt.plot(mesh[:, 1], u[:, 2], marker='o')
    plt.title("Vertical displacement along the beam")
    plt.xlabel("Spanwise position (m)")
    plt.ylabel("Vertical displacement (m)")
    plt.show()

    delta = P * length**3 / (3 * E * cs.Iz)

    print(f"  Tip displacement (FEA):      {u[-1, 2]:.6e} m")
    print(f"  Tip displacement (analytic): {delta:.6e} m")
    print(f"  Relative error:              {abs(u[-1, 2] - delta) / delta * 100:.4f} %")

    sigma = beam.recover_stress(u)   # (num_elements,)
    print('shape of sigma:', sigma.shape)

    # plot the stress distribution along the beam
    plt.plot(sigma)
    plt.xlabel("Spanwise position (m)")
    plt.ylabel("Bending stress (Pa)")
    plt.grid()
    plt.show()

    # Root moment = P * L  and  sigma_root = P * L * c / I
    sigma_root_analytic = P * length * radius / float(cs.Iz)
    sigma_root_fea      = float(sigma[0])   # element 0, node-1 end (fixed root)

    print(f"  Root stress (FEA):      {sigma_root_fea:.6e} Pa")
    print(f"  Root stress (analytic): {sigma_root_analytic:.6e} Pa")
    print(f"  Error:                  {abs(sigma_root_fea - sigma_root_analytic) / sigma_root_analytic * 100:.4f} %")
