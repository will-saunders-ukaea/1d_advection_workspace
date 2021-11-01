from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper

import irksome

ButcherTableau = irksome.ButcherTableaux.ButcherTableau
import time


if __name__ == "__main__":

    N_cells = 1000
    dt = 0.0001
    t_end = 1.0

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

    L = q * div(phi * u) * dx - (phi("+") - phi("-")) * (un("+") * q("+") - un("-") * q("-")) * dS
    A = inner(phi, Dt(q)) * dx
    F = A - L

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    # butcher_tableau = GaussLegendre(2)
    
    # "Classic RK4"
    third = 1.0 / 3.0
    sixth = 1.0 / 6.0

    butcher_tableau = ButcherTableau(
        A=np.array(
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0, 0.0),
                (0.0, 0.5, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            )
        ),
        b=np.array((sixth, third, third, sixth)),
        btilde=None,
        c=np.array((0.0, 0.5, 0.5, 1.0)),
        order=4,
    )
    print(butcher_tableau.A)
    print(butcher_tableau.b)
    print(butcher_tableau.btilde)
    print(butcher_tableau.c)
    print(butcher_tableau.order)

    dt = Constant(dt)
    t = Constant(0.0)

    stepper = TimeStepper(F, butcher_tableau, t, dt, q, bcs=[], solver_parameters=luparams)

    solution_output = File("output/solution.pvd")

    step = 0
    solution_output.write(q)
    t0 = time.time()
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        step += 1
        if step % 100 == 0:
            print(
                "Step {}. Simulation time (s): {}. Time per step (s) {}".format(
                    step, float(t), (time.time() - t0) / step
                )
            )
            solution_output.write(q)

    solution_output.write(q)

    print(norm(q - q_init) / norm(q_init))
