from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper

import irksome

ButcherTableau = irksome.ButcherTableaux.ButcherTableau
import time


if __name__ == "__main__":

    N_cells = 100000
    dt = 1E-7
    t_end = 1E-5
    visulisation = False

    mesh = PeriodicIntervalMesh(N_cells, 1.0)

    V = FunctionSpace(mesh, "DG", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)

    (x,) = SpatialCoordinate(mesh)

    velocity = as_vector((1.0,))
    u = Function(W).interpolate(velocity)

    q = Function(V).interpolate(exp(-20.0 * (x - 0.5) ** 2))
    q_init = Function(V).assign(q)

    phi = TestFunction(V)

    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))
    


    dt = Constant(dt)
    dq = TrialFunction(V)

    L = inner(phi, q) * dx + dt * (q * div(phi * u) * dx - (phi("+") - phi("-")) * (un("+") * q("+") - un("-") * q("-")) * dS)
    A = inner(phi, dq) * dx


    #solver_parameters = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}
    solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}
    
    qn = Function(V)
    qn.assign(q)

    if visulisation:
        solution_output = File("output/solution.pvd")
    
    t0 = time.time()
    for stepx in range(int(float(t_end)/float(dt))):

        q.assign(qn)
        solve(A == L, qn, bcs=[], solver_parameters=solver_parameters)
        
        if stepx % 10 == 0:
            print(stepx, (time.time() - t0)/ (stepx + 1))
            if visulisation:
                solution_output.write(q)

    if visulisation:
        solution_output.write(q)

    print(norm(q - q_init) / norm(q_init))

