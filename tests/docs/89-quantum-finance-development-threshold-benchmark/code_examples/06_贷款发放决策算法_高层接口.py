# ruff: noqa: F821
program = DecisionProgram(name="贷款特征选择", kind="decision", domain="贷款发放")
x = program.add_variables("特征", 4, vtype="binary")
program.add_constraints([x[0] + x[1] == 1, x[2] + x[3] == 1])
program.set_objective(
    1.0 * x[0] + 1.2 * x[1] + 0.9 * x[2] + 1.1 * x[3],
    sense="maximize",
)
problem = program.to_rasengan_problem()
circuit = build_rasengan_circuit(problem, layers=1)
best = problem.best_feasible_state()
