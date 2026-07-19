# ruff: noqa: F821
logical_variables = 8
num_qubits = math.ceil(logical_variables / 2)
circuit = QuantumCircuit(num_qubits, num_qubits)
circuit.h(range(num_qubits))
for _ in range(2):
    target = num_qubits - 1
    circuit.h(target)
    circuit.mcx(list(range(target)), target)
    circuit.h(target)
    circuit.h(range(num_qubits))
    circuit.x(range(num_qubits))
    circuit.h(target)
    circuit.mcx(list(range(target)), target)
    circuit.h(target)
    circuit.x(range(num_qubits))
    circuit.h(range(num_qubits))
circuit.measure(range(num_qubits), range(num_qubits)[::-1])
