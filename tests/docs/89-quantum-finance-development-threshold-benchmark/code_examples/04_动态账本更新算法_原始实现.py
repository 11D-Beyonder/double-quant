# ruff: noqa: F821
phase_qubits = 6
work_qubits = 4
circuit = QuantumCircuit(phase_qubits + work_qubits, phase_qubits)
phase_register = list(range(phase_qubits))
work_register = list(range(phase_qubits, phase_qubits + work_qubits))
circuit.h(phase_register)
circuit.x(work_register[0])
for target in work_register[1:]:
    circuit.cswap(phase_register[0], work_register[0], target)
circuit.cswap(phase_register[1], work_register[0], work_register[2])
circuit.cswap(phase_register[1], work_register[1], work_register[3])
for index in range(phase_qubits // 2):
    circuit.swap(phase_register[index], phase_register[phase_qubits - index - 1])
for target_index in range(phase_qubits):
    for control_index in range(target_index):
        angle = -math.pi / float(2 ** (target_index - control_index))
        circuit.cp(angle, phase_register[control_index], phase_register[target_index])
    circuit.h(phase_register[target_index])
circuit.measure(phase_register, phase_register)
