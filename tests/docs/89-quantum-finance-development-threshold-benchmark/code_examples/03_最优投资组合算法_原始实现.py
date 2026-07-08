# ruff: noqa: F821
expected_returns = np.asarray(expected_returns, dtype=float)
covariance = np.asarray(covariance, dtype=float)
matrix = np.zeros((4, 4), dtype=float)
matrix[0, 2:] = expected_returns
matrix[1, 2:] = 1.0
matrix[2:, 0] = expected_returns
matrix[2:, 1] = 1.0
matrix[2:, 2:] = covariance
vector = np.zeros(4, dtype=float)
vector[0] = target_return
vector[1] = 1.0
solution = np.linalg.solve(matrix, vector)
weights = {"资产甲": float(solution[2]), "资产乙": float(solution[3])}
