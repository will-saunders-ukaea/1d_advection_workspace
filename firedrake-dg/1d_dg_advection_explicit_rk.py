from firedrake import *
import time


if __name__ == "__main__":

    N_cells = 100000
    dt = 5e-7
    t_end = 5e-3
    visulisation = False

    # create the mesh and function space
    mesh = PeriodicIntervalMesh(N_cells, 1.0)

    # function space for q
    V = FunctionSpace(mesh, "DG", 1)

    # function space for the constant velocity field
    W = VectorFunctionSpace(mesh, "CG", 1)

    (x,) = SpatialCoordinate(mesh)

    velocity = as_vector((1.0,))
    u = Function(W).interpolate(velocity)

    # initial condition to be advected
    q = Function(V).interpolate(exp(-20.0 * (x - 0.5) ** 2))
    q_init = Function(V).assign(q)

    # create the weak forms for "classic rk4"
    phi = TestFunction(V)

    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))

    third = 1.0 / 3.0
    sixth = 1.0 / 6.0

    dt = Constant(dt)
    dq = TrialFunction(V)

    qn = Function(V)

    f = dt * (q * div(phi * u) * dx - (phi("+") - phi("-")) * (un("+") * q("+") - un("-") * q("-")) * dS)
    A = inner(phi, dq) * dx

    k1 = Function(V)
    k2 = Function(V)
    k3 = Function(V)
    k4 = Function(V)

    # solver_parameters = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}
    solver_parameters = {"ksp_type": "cg", "pc_type": "none"}

    prob1 = LinearVariationalProblem(A, f, k1)
    solv1 = LinearVariationalSolver(prob1, solver_parameters=solver_parameters)
    prob2 = LinearVariationalProblem(A, f, k2)
    solv2 = LinearVariationalSolver(prob2, solver_parameters=solver_parameters)
    prob3 = LinearVariationalProblem(A, f, k3)
    solv3 = LinearVariationalSolver(prob3, solver_parameters=solver_parameters)
    prob4 = LinearVariationalProblem(A, f, k4)
    solv4 = LinearVariationalSolver(prob4, solver_parameters=solver_parameters)

    qn.assign(q)

    if visulisation:
        solution_output = File("output/solution.pvd")

    t0 = time.time()
    for stepx in range(int(float(t_end) / float(dt))):

        q.assign(qn)
        solv1.solve()

        q.assign(qn + 0.5 * k1)
        solv2.solve()

        q.assign(qn + 0.5 * k2)
        solv3.solve()

        q.assign(qn + k3)
        solv4.solve()

        qn.assign(qn + sixth * k1 + third * k2 + third * k3 + sixth * k4)

        if stepx % 100 == 0:
            print(stepx, (time.time() - t0) / (stepx + 1))
            if visulisation:
                solution_output.write(qn)

    if visulisation:
        solution_output.write(qn)

    print(norm(qn - q_init) / norm(q_init))
