# ruff: noqa: F821
linear = np.array([1.0, 1.2, 0.9, 1.1], dtype=float)
constraints = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
rhs = np.array([1.0, 1.0], dtype=float)
variable_names = ("特征_0", "特征_1", "特征_2", "特征_3")
problem = LinearConstraintBinaryProblem(
    linear=linear,
    constraints=constraints,
    rhs=rhs,
    sense="max",
    variable_names=variable_names,
)
transition_basis = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]], dtype=int)
feasible_state = np.array([1, 0, 1, 0], dtype=int)
circuit = build_rasengan_circuit(
    problem,
    layers=1,
    transition_basis=transition_basis,
    feasible_state=feasible_state,
)
best = problem.best_feasible_state()
