import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_default_dtype(torch.float64) # modopt and SLSQP work in float64
import modopt as mo
import warnings
warnings.filterwarnings("ignore")

N = 4 # number of subproblems
n0 = 10 # number of global variables
ni = 20 # number of local variables


def objective(v):

    x0 = v[:n0]
    xs = v[n0:].reshape(N, ni)
    w = torch.hstack([x0.tile(N, 1), xs])

    terms = 100 * (w[:, 1:] - w[:, :-1] ** 2) ** 2 + (1.0 - w[:, :-1]) ** 2

    return torch.mean(torch.sum(terms, dim=1))


def constraints(v):

    x0 = v[:n0]
    xs = v[n0:].reshape(N, ni)
    w = torch.hstack([x0.tile(N, 1), xs])

    spheres = torch.sum(w ** 2, dim=1) / (n0 + ni)       # N local constraints
    plane = torch.mean(xs[:, 0])      # 1 global constraint
    return torch.cat([spheres, plane[None]])


x0 = np.ones(n0 + N * ni) * 0
xl, xu = -1.5, 1.5
cu = np.concatenate([np.full(N, 0.8 ** 2), [0.93]])
cl = np.full(N + 1, -np.inf)

# modopt drives the solve with numpy callbacks; torch.func supplies the exact derivatives
prob = mo.ProblemLite(x0=x0,
                      obj=lambda x: np.float64(objective(torch.as_tensor(x))),
                      grad=lambda x: np.array(torch.func.grad(objective)(torch.as_tensor(x))),
                      con=lambda x: np.array(constraints(torch.as_tensor(x))),
                      jac=lambda x: np.array(torch.func.jacrev(constraints)(torch.as_tensor(x))),
                      xl=xl, xu=xu, cl=cl, cu=cu, o_scaler=1e-2)

optimizer = mo.SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()

x = optimizer.results['x']
print(x)
