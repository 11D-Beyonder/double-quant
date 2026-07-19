# ruff: noqa: F821
library = default_operator_library()
result = library.execute(
    "func_1",
    {
        "expected_returns": expected_returns,
        "covariance": covariance,
        "target_return": target_return,
        "assets": ["资产甲", "资产乙"],
    },
    max_qpe_qubits=4,
)
weights = result.financial_result["weights"]
