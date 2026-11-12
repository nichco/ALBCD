import time
from typing import Optional, Sequence, Union
import numpy as np
from scipy.optimize import BFGS

class ALBCD2():


    def __init__(self,
                 subproblems: Sequence,
                 x0: np.ndarray,
                 mu0: Optional[np.ndarray] = None,
                 data0: Optional[dict] = None,
                 max_mu: float = 1e3,
                 rho: float = 1.2,
                 tau: float = 0.5,
                 feas_tol: float = 1e-3,
                 opt_tol: float = 1e-4,
                 max_outer_iter: int = 100,
                 max_inner_iter: int = 10,
                 max_y: float = 1e6,
                 verbose: bool = True,
                 ):

        self.subproblems = subproblems
        self.x = np.array(x0, dtype=float) # copy, so the caller's x0 is never modified
        self.data = {} if data0 is None else dict(data0) # other quantities shared between subproblems; never holds x, y, mu or phi
        self.tf = None

        # force float dtype; an integer-dtype mu would otherwise truncate penalty growth.
        self.mu = np.asarray(mu0, dtype=float) # penalty parameter(s)
        self.max_mu = max_mu
        self.rho = rho
        self.tau = tau
        self.feas_tol = feas_tol
        self.opt_tol = opt_tol

        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.max_y = max_y
        self.verbose = verbose
        self.x_history = [self.x.copy()]
        self.feas_history = [] # feasibility (max constraint violation) after each sweep
        self.opt_history = []  # optimality (max-norm KKT residual) after each sweep
        self.y = np.zeros_like(self.mu) # Lagrange multipliers, one per coupling constraint
        self.phi = None # coupling constraints at the current x, from the latest subproblem solve



        # damped-BFGS approximation of the Hessian of f + y^T phi w.r.t. all variables
        # self.B = BFGS(exception_strategy="damp_update", init_scale="auto")
        # self.B.initialize(self.x.size, "hess")

    def _log(self, msg: str) -> None:
        if self.verbose: print(msg)

    def _update_mu(self, mu, phi_new, phi_old) -> None:
        # increase the penalty parameter of each scalar constraint if
        # it is infeasible and its feasibility is not decreasing fast enough.
        
        grow = np.abs(phi_new) > np.maximum(self.tau * np.abs(phi_old), self.feas_tol)
        mu_new = np.where(grow, self.rho * mu, mu)

        return np.minimum(mu_new, self.max_mu)

    def _update_y(self, y, mu, phi) -> None:
        
        y_new = np.clip(y + mu * phi, -self.max_y, self.max_y)

        return y_new

    def _block_coordinate_descent(self, opt_tol):

        for i in range(self.max_inner_iter):

            for subproblem in self.subproblems:

                outputs = {}
                subproblem.solve(self.x, self.y, self.mu, self.data, outputs)
                self.x = outputs.pop("x")
                # phi at the new x; none when unconstrained. A missing or wrong-sized phi fails the reshape
                self.phi = np.array(outputs.pop("phi", []), dtype=float).reshape(self.mu.shape)
                self.data.update(outputs)  # any other output is shared through data

                self.x_history.append(self.x.copy())

            opt = max(np.linalg.norm(sp.gradient(self.x, self.y, self.mu, self.data), ord=np.inf)
                                for sp in self.subproblems)
            self.opt_history.append(opt)

            if opt <= opt_tol:
                self._log(f"inner loop converged with optimality {opt:.3e} in {i + 1} iterations")
                break

        else: self._log(f"inner loop failed with optimality {opt:.3e} in {i + 1} iterations")

        return opt

    def _augmented_lagrangian(self, feas_tol, opt_tol):

        phi_old = np.full_like(self.mu, np.inf)
        success = False
        
        k = 0
        while k < self.max_outer_iter:
            k += 1

            optimality = self._block_coordinate_descent(opt_tol)

            feas = np.max(np.abs(self.phi))

            if feas <= feas_tol:
                success = optimality <= opt_tol
                break

            self.y = self._update_y(self.y, self.mu, self.phi)
            self.mu = self._update_mu(self.mu, self.phi, phi_old)
            phi_old = self.phi
            
        return success


    def solve(self) -> "ALBCD2":

        t0 = time.perf_counter()
        feas_tol = self.feas_tol
        opt_tol = self.opt_tol

        success = self._augmented_lagrangian(feas_tol, opt_tol)

        self.tf = time.perf_counter() - t0
        self._log(f"{'ALBCD converged' if success else 'ALBCD failed'} ({self.tf:.2f} s)")

        return self















if __name__ == "__main__":


    import numpy as np
    import modopt as mo
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from subproblem import Subproblem
    import warnings
    warnings.filterwarnings("ignore")


    class Subproblem1(Subproblem):

        def objective(self, v, other, y, mu):
            x1 = v[0]
            s = v[1]
            x2 = other[0]
            obj = jnp.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

            g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s

            return obj + jnp.sum(y * g0) + 0.5 * jnp.sum(mu * g0**2)

        def solve(self, x, y, mu, data, outputs) -> None:
            v0 = np.asarray(self.decompose(x), dtype=float)
            other = jnp.asarray(self.other(x))

            xl = np.array([-np.inf, 0])
            xu = np.array([np.inf, np.inf])

            jax_obj = lambda v: self.objective(v, other, y, mu)

            jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=xl, xu=xu)

            optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
            optimizer.solve()
            # optimizer.print_results()
            v_new = optimizer.results['x']

            # No general (in)equality constraints at the block level -- only box
            # bounds -- so cache the bounds for the projected-gradient stationarity
            # check in residual() instead of constraint multipliers/Jacobian.
            self._xl = xl
            self._xu = xu

            x_new = self.recompose(x, v_new)

            # coupling constraint at the new x
            x1, s, x2 = x_new
            g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s

            outputs["x"] = x_new
            outputs["phi"] = np.array([g0])

        def gradient(self, x, y, mu, data) -> np.ndarray:

            v = jnp.asarray(self.decompose(x))
            other = jnp.asarray(self.other(x))
            grad_f = jax.grad(self.objective)(v, other, y, mu)
            
            v = np.asarray(v)
            grad_f = np.asarray(grad_f)
            
            # projected-gradient stationarity residual for box-constrained v:
            # zero iff v satisfies the KKT conditions for xl <= v <= xu
            proj = np.clip(v - grad_f, self._xl, self._xu)
            grad = v - proj
            return grad


    class Subproblem2(Subproblem):

        def objective(self, v, other, y, mu):
            x2 = v[0]
            x1, s = other[0], other[1]
            obj = jnp.squeeze(x1**2 + x2**2 - 1.5 * x1 * x2)

            g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s
            
            return obj + jnp.sum(y * g0) + 0.5 * jnp.sum(mu * g0**2)

        def solve(self, x, y, mu, data, outputs) -> None:
            v0 = np.asarray(self.decompose(x), dtype=float)
            other = jnp.asarray(self.other(x))

            jax_obj = lambda v: self.objective(v, other, y, mu)

            jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj)

            optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-8}, turn_off_outputs=True)
            optimizer.solve()
            # optimizer.print_results()
            v_new = optimizer.results['x']

            # No box bounds on this block either -- record unbounded limits so
            # residual() can use the same projected-gradient check.
            self._xl = np.full_like(v_new, -np.inf)
            self._xu = np.full_like(v_new, np.inf)

            x_new = self.recompose(x, v_new)

            # coupling constraint at the new x
            x1, s, x2 = x_new
            g0 = -1 * (x1**2 + x2**2) + 0.5**2 + s

            outputs["x"] = x_new
            outputs["phi"] = np.array([g0])

        def gradient(self, x, y, mu, data) -> np.ndarray:

            v = jnp.asarray(self.decompose(x))
            other = jnp.asarray(self.other(x))
            grad_f = jax.grad(self.objective)(v, other, y, mu)
            
            v = np.asarray(v)
            grad_f = np.asarray(grad_f)
            
            # projected-gradient stationarity residual for box-constrained v:
            # zero iff v satisfies the KKT conditions for xl <= v <= xu
            proj = np.clip(v - grad_f, self._xl, self._xu)
            grad = v - proj
            return grad


    v_init = np.array([-0.5, 0.0, 1.0])

    opt = ALBCD2(subproblems=[Subproblem1(index=slice(0, 2)), # owns x1 and s
                            Subproblem2(index=slice(2, 3))], # owns x2
                x0=np.array(v_init),
                mu0=np.array([1.0]),
                max_mu=1e3,
                rho=1.2,
                tau=0.5,
                feas_tol=1e-3,
                opt_tol=1e-4,
                max_y=1e6,
                max_outer_iter=100,
                max_inner_iter=100)

    opt.solve()


    history = np.array(opt.x_history)
    x1_history = history[:, 0]
    s_history = history[:, 1]
    x2_history = history[:, 2]


    plt.rcParams.update({'font.size': 14})

    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 - 1.5 * X * Y
    levels = np.linspace(0, max(Z.flatten()), 30)
    plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
    plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

    plt.plot(x1_history, x2_history, 's-', color='tab:orange', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x, y)$')
    plt.plot(x1_history[-1], x2_history[-1], 's', color='#D462AD', markersize=8, zorder=11, mec='k')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('y')

    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 * np.cos(theta)
    circle_y = 0.5 * np.sin(theta)
    plt.plot(circle_x, circle_y, '--', color='black', linewidth=2, alpha=0.5)
    plt.fill(circle_x, circle_y, color='black', alpha=0.3)

    ticks = [-1, 0, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.gca().set_aspect('equal')
    plt.show()