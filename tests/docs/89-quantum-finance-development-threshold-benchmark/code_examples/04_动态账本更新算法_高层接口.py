# ruff: noqa: F821
algorithm = DynamicLedgerUpdateAlgorithm(modulus=15, base=2, phase_qubits=6)
circuit = algorithm.build_circuit()
