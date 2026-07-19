from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from double_quant.algorithm.grover import (
    build_grover_circuit,
    build_sfs_grover_circuit,
    grover_success_probability,
)
from double_quant.algorithm.hhl import HHLSolver
from double_quant.algorithm.qubo import (
    NumPyMinimumEigensolverSolver,
    QAOASolver,
    SamplingVQESolver,
)
from double_quant.algorithm.rasengan import (
    LinearConstraintBinaryProblem,
    build_penalty_qaoa_circuit,
    build_rasengan_circuit,
)
from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QuantumShapleyCalculator,
)
from double_quant.algorithm.shor import build_shor_period_finding_circuit
from double_quant.algorithm.shor.baseline import (
    classical_trial_division_operations,
)
from double_quant.application.antifraud_monitoring import AntifraudMonitoringAlgorithm
from double_quant.application.branch_location import BranchLocationAlgorithm
from double_quant.application.defi_management import DefiManagementAlgorithm
from double_quant.application.dynamic_ledger_update import DynamicLedgerUpdateAlgorithm
from double_quant.application.index_tracking import IndexTrackingAlgorithm
from double_quant.application.loan_decision import LoanDecisionAlgorithm
from double_quant.application.payment_settlement import PaymentSettlementAlgorithm
from double_quant.application.portfolio import PortfolioOptimizer
from double_quant.application.risk import RiskAttributor
from double_quant.common import IsingProblem, LinearSystem, QUBOProblem
from double_quant.common.metric import expected_shortfall
from double_quant.programming.measures import EuropeanCallPriceMeasure


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Stable definition for a quantum-finance operator."""

    id: str
    name: str
    domain: str
    task: str
    problem_form: str
    quantum_primitive: str
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    supported_backends: tuple[str, ...]
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Execution resources reported with a normalized operator result."""

    num_qubits: int | None = None
    circuit_depth: int | None = None
    two_qubit_gates: int | None = None
    shots: int | None = None
    oracle_calls: int | None = None
    optimizer_evals: int | None = None
    runtime_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class OperatorResult:
    """Normalized result returned by every operator in the library."""

    operator_id: str
    backend: str
    financial_result: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    resources: ResourceProfile = field(default_factory=ResourceProfile)
    raw_result: Any | None = None


def _circuit_resource_profile(circuit: QuantumCircuit) -> ResourceProfile:
    two_qubit_gates = sum(
        1 for instruction in circuit.data if instruction.operation.num_qubits == 2
    )
    shots = None
    if circuit.metadata is not None and isinstance(circuit.metadata.get("shots"), int):
        shots = int(circuit.metadata["shots"])
    return ResourceProfile(
        num_qubits=int(circuit.num_qubits),
        circuit_depth=int(circuit.depth()),
        two_qubit_gates=int(two_qubit_gates),
        shots=shots,
    )


class QuantumFinancialOperator(Protocol):
    """Runtime protocol shared by all quantum-finance operators."""

    spec: OperatorSpec

    def validate(self, inputs: dict[str, Any]) -> None: ...

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult: ...


class QuantumFinancialOperatorLibrary:
    """Registry of reusable quantum-finance operators."""

    def __init__(self, operators: list[QuantumFinancialOperator] | None = None) -> None:
        self._operators: dict[str, QuantumFinancialOperator] = {}
        for operator in operators or []:
            self.register(operator)

    def register(self, operator: QuantumFinancialOperator) -> None:
        if operator.spec.id in self._operators:
            raise ValueError(f"Operator already registered: {operator.spec.id}")
        self._operators[operator.spec.id] = operator

    def get(self, operator_id: str) -> QuantumFinancialOperator:
        try:
            return self._operators[operator_id]
        except KeyError as exc:
            raise KeyError(f"Unknown quantum financial operator: {operator_id}") from exc

    def list_specs(self) -> list[OperatorSpec]:
        return [self._operators[key].spec for key in sorted(self._operators)]

    def execute(
        self,
        operator_id: str,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        operator = self.get(operator_id)
        operator.validate(inputs)
        return operator.execute(inputs, backend=backend, **options)


class ExpectedShortfallOperator:
    spec = OperatorSpec(
        id="func_2",
        name="风险价值计量算法（Func-2）",
        domain="风险计量",
        task="valuation",
        problem_form="historical_tail_loss",
        quantum_primitive="classical_measure",
        input_fields=("portfolio_returns", "alpha"),
        output_fields=("expected_shortfall",),
        supported_backends=("classical",),
        assumptions=("historical simulation",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        if "portfolio_returns" not in inputs:
            raise ValueError("portfolio_returns is required")
        alpha = float(inputs.get("alpha", 0.95))
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        returns = np.asarray(inputs["portfolio_returns"], dtype=float)
        if returns.ndim != 1 or returns.size == 0:
            raise ValueError("portfolio_returns must be a non-empty 1D array")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "classical"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        alpha = float(inputs.get("alpha", 0.95))
        value = expected_shortfall(
            np.asarray(inputs["portfolio_returns"], dtype=float),
            alpha,
        )
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={"expected_shortfall": value},
            diagnostics={"alpha": alpha},
        )


class LinearSystemSolveOperator:
    spec = OperatorSpec(
        id="linear_system.hhl_sapo",
        name="HHL/SAPO Linear System Solver",
        domain="portfolio",
        task="linear_solve",
        problem_form="linear_system",
        quantum_primitive="hhl",
        input_fields=("matrix", "vector"),
        output_fields=("solution",),
        supported_backends=("statevector",),
        assumptions=("matrix is symmetric or Hermitian", "dimension is a power of two"),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        matrix, vector = self._coerce_system(inputs)
        if matrix.shape[0] & (matrix.shape[0] - 1):
            raise ValueError("matrix dimension must be a positive power of 2")
        if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
            raise ValueError("matrix and vector must contain only finite values")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "statevector"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        matrix, vector = self._coerce_system(inputs)
        solver_options = dict(options)
        solution = HHLSolver.solve(matrix, vector, "sapo", **solver_options)
        resources = self._resource_profile(matrix, vector, solver_options)
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={"solution": solution},
            diagnostics={
                "dimension": int(matrix.shape[0]),
                "solver_variant": "sapo",
            },
            resources=resources,
        )

    @staticmethod
    def _coerce_system(inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if "system" in inputs:
            system = inputs["system"]
            if not isinstance(system, LinearSystem):
                raise TypeError("system must be a LinearSystem")
            return system.matrix, system.vector
        if "matrix" not in inputs or "vector" not in inputs:
            raise ValueError("matrix and vector are required")
        matrix = np.asarray(inputs["matrix"], dtype=float)
        vector = np.asarray(inputs["vector"], dtype=float)
        LinearSystem(matrix, vector)
        return matrix, vector

    @staticmethod
    def _resource_profile(
        matrix: np.ndarray,
        vector: np.ndarray,
        solver_options: dict[str, Any],
    ) -> ResourceProfile:
        try:
            circuit = HHLSolver.build_circuit(matrix, vector, "sapo", **solver_options)
        except Exception:
            return ResourceProfile()
        two_qubit_gates = sum(1 for instruction in circuit.data if instruction.operation.num_qubits == 2)
        return ResourceProfile(
            num_qubits=int(circuit.num_qubits),
            circuit_depth=int(circuit.depth()),
            two_qubit_gates=int(two_qubit_gates),
        )


class BinaryOptimizationOperator:
    spec = OperatorSpec(
        id="optimization.qubo",
        name="QUBO/Ising Binary Optimization",
        domain="optimization",
        task="binary_optimization",
        problem_form="qubo_or_ising",
        quantum_primitive="qaoa_or_sampling_vqe",
        input_fields=("problem",),
        output_fields=("best_bitstring", "best_objective"),
        supported_backends=("classical_exact", "qaoa", "sampling_vqe"),
        assumptions=("problem is already modeled as QUBO or Ising",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        problem = self._coerce_problem(inputs)
        if problem.num_variables <= 0:
            raise ValueError("problem must contain at least one variable")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "classical_exact"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        problem = self._coerce_problem(inputs)

        if selected_backend == "qaoa":
            solver = QAOASolver(**options)
        elif selected_backend == "sampling_vqe":
            solver = SamplingVQESolver(**options)
        else:
            solver = NumPyMinimumEigensolverSolver(**options)

        raw_result = solver.solve(problem)
        optimizer_evals = None
        if raw_result.metadata is not None:
            optimizer_evals = raw_result.metadata.get("optimizer_evals")
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={
                "best_bitstring": raw_result.best_bitstring,
                "best_objective": raw_result.best_objective,
                "best_energy": raw_result.best_energy,
                "best_probability": raw_result.best_probability,
                "variable_names": list(problem.variable_names or []),
            },
            diagnostics={"num_variables": problem.num_variables},
            resources=ResourceProfile(
                num_qubits=problem.num_variables,
                optimizer_evals=optimizer_evals if isinstance(optimizer_evals, int) else None,
            ),
            raw_result=raw_result,
        )

    @staticmethod
    def _coerce_problem(inputs: dict[str, Any]) -> QUBOProblem | IsingProblem:
        if "problem" in inputs:
            problem = inputs["problem"]
            if not isinstance(problem, (QUBOProblem, IsingProblem)):
                raise TypeError("problem must be QUBOProblem or IsingProblem")
            return problem
        if "quadratic_matrix" not in inputs:
            raise ValueError("problem or quadratic_matrix is required")
        return QUBOProblem(
            np.asarray(inputs["quadratic_matrix"], dtype=float),
            constant=float(inputs.get("constant", 0.0)),
            variable_names=inputs.get("variable_names"),
        )


class RiskAttributionOperator:
    spec = OperatorSpec(
        id="risk.quantum_shapley",
        name="Quantum Shapley Risk Attribution",
        domain="risk",
        task="attribution",
        problem_form="shapley_game",
        quantum_primitive="quantum_shapley",
        input_fields=("returns", "alpha"),
        output_fields=("src",),
        supported_backends=("classical_exact", "classical_mc", "quantum_statevector"),
        assumptions=("mode='rs' is required for quantum Shapley backend",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        if "returns" not in inputs:
            raise ValueError("returns is required")
        returns = inputs["returns"]
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if returns.empty or len(returns.columns) == 0:
            raise ValueError("returns must contain at least one asset column")
        alpha = float(inputs.get("alpha", 0.95))
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "quantum_statevector"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        returns = inputs["returns"]
        alpha = float(inputs.get("alpha", 0.95))

        if selected_backend == "classical_exact":
            solver_class = BinaryEnumerationCalculator
            mode = "es"
            solver_options: dict[str, Any] = {}
        elif selected_backend == "classical_mc":
            solver_class = PermutationMCCalculator
            mode = "es"
            solver_options = dict(options)
        else:
            solver_class = QuantumShapleyCalculator
            mode = "rs"
            solver_options = {"extraction_mode": "statevector", **options}

        attributor = RiskAttributor(
            returns,
            solver_class,
            alpha=alpha,
            mode=mode,
            **solver_options,
        )
        src = attributor.attribute()
        oracle_calls = self._oracle_calls(attributor.solver, len(returns.columns))
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={"src": src},
            diagnostics={"mode": mode, "assets": list(returns.columns), "alpha": alpha},
            resources=ResourceProfile(
                oracle_calls=sum(oracle_calls.values()) if oracle_calls else None,
            ),
            raw_result={"oracle_calls_by_asset": oracle_calls},
        )

    @staticmethod
    def _oracle_calls(solver: Any, num_assets: int) -> dict[str, int] | None:
        if not hasattr(solver, "get_oracle_count"):
            return None
        counts: dict[str, int] = {}
        for index in range(num_assets):
            count = solver.get_oracle_count(index)
            if count is None:
                return None
            counts[str(index)] = int(count)
        return counts


class EuropeanCallOptionOperator:
    spec = OperatorSpec(
        id="func_3",
        name="金融衍生品定价算法（Func-3）",
        domain="衍生品定价",
        task="valuation",
        problem_form="discounted_payoff_expectation",
        quantum_primitive="amplitude_estimation_ready_measure",
        input_fields=("terminal_price_scenarios", "strike"),
        output_fields=("option_price",),
        supported_backends=("classical_scenarios",),
        assumptions=("terminal scenarios are already generated",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        if "terminal_price_scenarios" not in inputs:
            raise ValueError("terminal_price_scenarios is required")
        if "strike" not in inputs:
            raise ValueError("strike is required")
        scenarios = np.asarray(inputs["terminal_price_scenarios"], dtype=float)
        if scenarios.ndim != 1 or scenarios.size == 0:
            raise ValueError("terminal_price_scenarios must be a non-empty 1D array")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "classical_scenarios"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        parameters = {
            "strike": inputs["strike"],
            "risk_free_rate": inputs.get("risk_free_rate", 0.0),
            "maturity": inputs.get("maturity", 1.0),
            **options,
        }
        value = EuropeanCallPriceMeasure.evaluate(inputs, parameters)
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={"option_price": value},
            diagnostics={
                "strike": float(parameters["strike"]),
                "scenario_count": int(
                    np.asarray(inputs["terminal_price_scenarios"]).size
                ),
            },
        )


class GroverSearchOperator:
    spec = OperatorSpec(
        id="search.grover_sfs",
        name="Grover/SFS Search Circuit",
        domain="search",
        task="amplitude_amplified_search",
        problem_form="binary_search_space",
        quantum_primitive="grover_sfs",
        input_fields=("logical_variables", "iterations"),
        output_fields=("circuit", "success_probability"),
        supported_backends=("sfs_circuit", "plain_circuit"),
        assumptions=("marked_state defaults to all ones",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        logical_variables = int(inputs.get("logical_variables", 0))
        iterations = int(inputs.get("iterations", 1))
        if logical_variables <= 0:
            raise ValueError("logical_variables must be positive")
        if iterations < 0:
            raise ValueError("iterations must be non-negative")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "sfs_circuit"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        merged = {**inputs, **options}
        self.validate(merged)
        logical_variables = int(merged["logical_variables"])
        iterations = int(merged.get("iterations", 1))
        marked_state = merged.get("marked_state")
        if selected_backend == "plain_circuit":
            circuit = build_grover_circuit(
                num_qubits=logical_variables,
                iterations=iterations,
                marked_state=marked_state,
            )
        else:
            circuit = build_sfs_grover_circuit(
                logical_variables=logical_variables,
                iterations=iterations,
                compressed_qubits=merged.get("compressed_qubits"),
                marked_state=marked_state,
            )
        search_space_size = int(circuit.metadata["search_space_size"])
        success_probability = grover_success_probability(
            num_items=search_space_size,
            iterations=iterations,
        )
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={
                "circuit": circuit,
                "success_probability": success_probability,
            },
            diagnostics=dict(circuit.metadata or {}),
            resources=_circuit_resource_profile(circuit),
            raw_result=circuit,
        )


class ShorPeriodFindingOperator:
    spec = OperatorSpec(
        id="number_theory.shor_period_finding",
        name="Shor Period Finding Circuit",
        domain="ledger",
        task="period_finding",
        problem_form="modular_order_finding",
        quantum_primitive="shor",
        input_fields=("modulus", "base"),
        output_fields=("circuit",),
        supported_backends=("optimized_circuit",),
        assumptions=("modulus and base are coprime",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        if "modulus" not in inputs:
            raise ValueError("modulus is required")
        modulus = int(inputs["modulus"])
        base = int(inputs.get("base", 2))
        if modulus <= 2:
            raise ValueError("modulus must be greater than 2")
        if not 1 < base < modulus:
            raise ValueError("base must satisfy 1 < base < modulus")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "optimized_circuit"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        merged = {**inputs, **options}
        self.validate(merged)
        modulus = int(merged["modulus"])
        base = int(merged.get("base", 2))
        circuit = build_shor_period_finding_circuit(
            modulus,
            base=base,
            phase_qubits=merged.get("phase_qubits"),
            work_qubits=merged.get("work_qubits"),
        )
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={"circuit": circuit},
            diagnostics={
                **dict(circuit.metadata or {}),
                "classical_trial_divisions": classical_trial_division_operations(
                    modulus
                ),
            },
            resources=_circuit_resource_profile(circuit),
            raw_result=circuit,
        )


class RasenganOptimizationOperator:
    spec = OperatorSpec(
        id="optimization.rasengan",
        name="Rasengan Constrained Binary Optimization",
        domain="optimization",
        task="constrained_binary_optimization",
        problem_form="linear_constraint_binary_problem",
        quantum_primitive="rasengan",
        input_fields=("problem",),
        output_fields=("best_bitstring", "best_objective", "circuit"),
        supported_backends=("rasengan_circuit", "penalty_qaoa"),
        assumptions=("constraints are linear equalities over binary variables",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        if not isinstance(inputs.get("problem"), LinearConstraintBinaryProblem):
            raise TypeError("problem must be a LinearConstraintBinaryProblem")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "rasengan_circuit"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        problem = inputs["problem"]
        layers = int(options.get("layers", inputs.get("layers", 1)))
        if selected_backend == "penalty_qaoa":
            circuit = build_penalty_qaoa_circuit(problem, layers=layers)
        else:
            circuit = build_rasengan_circuit(
                problem,
                layers=layers,
                transition_basis=options.get("transition_basis"),
                feasible_state=options.get("feasible_state"),
            )
        best = problem.best_feasible_state()
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={
                "best_bitstring": best,
                "best_objective": problem.objective_value(best),
                "circuit": circuit,
            },
            diagnostics={
                **dict(circuit.metadata or {}),
                "num_feasible_states": sum(
                    1 for state in problem.iter_binary_states() if problem.is_feasible(state)
                ),
            },
            resources=_circuit_resource_profile(circuit),
            raw_result=circuit,
        )


class PortfolioOptimizationOperator:
    spec = OperatorSpec(
        id="func_1",
        name="最优投资组合算法（Func-1）",
        domain="投资组合",
        task="portfolio_optimization",
        problem_form="constrained_linear_system",
        quantum_primitive="hhl",
        input_fields=("expected_returns", "covariance", "target_return"),
        output_fields=("weights",),
        supported_backends=("hhl_sapo",),
        assumptions=("portfolio system is expanded to a power-of-two dimension",),
    )

    def validate(self, inputs: dict[str, Any]) -> None:
        for field_name in ("expected_returns", "covariance", "target_return"):
            if field_name not in inputs:
                raise ValueError(f"{field_name} is required")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "hhl_sapo"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        optimizer = PortfolioOptimizer(
            inputs["expected_returns"],
            inputs["covariance"],
            float(inputs["target_return"]),
            assets=inputs.get("assets"),
            solver_class=HHLSolver,
            **options,
        )
        weights = optimizer.optimize()
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result={"weights": weights},
            diagnostics={"assets": list(weights)},
        )


@dataclass(frozen=True, slots=True)
class ApplicationOperatorConfig:
    operator_id: str
    name: str
    domain: str
    algorithm_class: type
    quantum_primitive: str
    default_parameters: dict[str, Any] = field(default_factory=dict)


class ApplicationCircuitOperator:
    """Operator wrapper around concrete algorithms in double_quant.application."""

    def __init__(self, config: ApplicationOperatorConfig) -> None:
        self.config = config
        self.spec = OperatorSpec(
            id=config.operator_id,
            name=config.name,
            domain=config.domain,
            task="application_circuit_build",
            problem_form="finance_application",
            quantum_primitive=config.quantum_primitive,
            input_fields=tuple(config.default_parameters),
            output_fields=("circuit",),
            supported_backends=("application_circuit", "baseline_circuit"),
            assumptions=("parameters default to the application class defaults",),
        )

    def validate(self, inputs: dict[str, Any]) -> None:
        if not isinstance(inputs, dict):
            raise TypeError("inputs must be a dictionary")

    def execute(
        self,
        inputs: dict[str, Any],
        *,
        backend: str | None = None,
        **options: Any,
    ) -> OperatorResult:
        selected_backend = backend or "application_circuit"
        if selected_backend not in self.spec.supported_backends:
            raise ValueError(f"Unsupported backend for {self.spec.id}: {selected_backend}")
        self.validate(inputs)
        parameters = {**self.config.default_parameters, **inputs, **options}
        algorithm = self.config.algorithm_class(**parameters)
        if selected_backend == "baseline_circuit":
            circuit = algorithm.build_baseline_circuit()
        else:
            circuit = algorithm.build_circuit()
        problem = algorithm.build_problem() if hasattr(algorithm, "build_problem") else None
        financial_result: dict[str, Any] = {"circuit": circuit}
        if problem is not None:
            best = problem.best_feasible_state()
            financial_result["best_bitstring"] = best
            financial_result["best_objective"] = problem.objective_value(best)
        return OperatorResult(
            operator_id=self.spec.id,
            backend=selected_backend,
            financial_result=financial_result,
            diagnostics={
                "application_class": self.config.algorithm_class.__name__,
                **dict(circuit.metadata or {}),
            },
            resources=_circuit_resource_profile(circuit),
            raw_result=circuit,
        )


@dataclass(frozen=True, slots=True)
class TemplateStep:
    name: str
    operator_id: str
    purpose: str


@dataclass(frozen=True, slots=True)
class TemplateResult:
    template_id: str
    algorithm: str
    operator_ids: tuple[str, ...]
    financial_result: dict[str, Any]
    step_results: tuple[OperatorResult, ...]


class RiskAttributionSoftwareTemplate:
    """Software template assembled from the operator library for risk attribution."""

    id = "template.risk_attribution"
    algorithm = "quantum_shapley_risk_attribution"
    steps = (
        TemplateStep(
            "portfolio_tail_risk",
            "risk.expected_shortfall",
            "Compute portfolio-level tail risk used as the valuation baseline.",
        ),
        TemplateStep(
            "asset_risk_contribution",
            "risk.quantum_shapley",
            "Compute asset-level Shapley risk contribution with the selected backend.",
        ),
    )

    def run(
        self,
        library: QuantumFinancialOperatorLibrary,
        inputs: dict[str, Any],
        *,
        backend: str = "quantum_statevector",
        **options: Any,
    ) -> TemplateResult:
        returns = inputs["returns"]
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        alpha = float(inputs.get("alpha", 0.95))

        portfolio_returns = returns.mean(axis=1).to_numpy(dtype=float)
        es_result = library.execute(
            "risk.expected_shortfall",
            {"portfolio_returns": portfolio_returns, "alpha": alpha},
        )
        src_result = library.execute(
            "risk.quantum_shapley",
            {"returns": returns, "alpha": alpha},
            backend=backend,
            **options,
        )
        return TemplateResult(
            template_id=self.id,
            algorithm=self.algorithm,
            operator_ids=tuple(step.operator_id for step in self.steps),
            financial_result={
                "portfolio_expected_shortfall": es_result.financial_result[
                    "expected_shortfall"
                ],
                "src": src_result.financial_result["src"],
            },
            step_results=(es_result, src_result),
        )


class ApplicationCircuitCatalogTemplate:
    """Software template that builds the configured application algorithm circuits."""

    id = "template.func_application_catalog"
    algorithm = "应用算法电路目录"
    default_operator_ids = (
        "func_4",
        "func_5",
        "func_6",
        "func_7",
        "func_8",
        "func_9",
        "func_10",
    )

    def run(
        self,
        library: QuantumFinancialOperatorLibrary,
        inputs: dict[str, Any] | None = None,
        *,
        backend: str = "application_circuit",
        **options: Any,
    ) -> TemplateResult:
        inputs = inputs or {}
        operator_ids = tuple(inputs.get("operator_ids", self.default_operator_ids))
        parameters_by_operator = inputs.get("parameters", {})
        step_results = []
        resource_table: dict[str, dict[str, int | None]] = {}
        for operator_id in operator_ids:
            operator_inputs = dict(parameters_by_operator.get(operator_id, {}))
            result = library.execute(
                operator_id,
                operator_inputs,
                backend=backend,
                **options,
            )
            step_results.append(result)
            resource_table[operator_id] = {
                "num_qubits": result.resources.num_qubits,
                "circuit_depth": result.resources.circuit_depth,
                "two_qubit_gates": result.resources.two_qubit_gates,
            }
        return TemplateResult(
            template_id=self.id,
            algorithm=self.algorithm,
            operator_ids=operator_ids,
            financial_result={"resource_table": resource_table},
            step_results=tuple(step_results),
        )


def _application_operators() -> list[ApplicationCircuitOperator]:
    return [
        ApplicationCircuitOperator(
            ApplicationOperatorConfig(
                operator_id="func_4",
                name="动态账本更新算法（Func-4）",
                domain="动态账本",
                algorithm_class=DynamicLedgerUpdateAlgorithm,
                quantum_primitive="shor",
                default_parameters={"modulus": 15, "base": 2},
            )
        ),
        ApplicationCircuitOperator(
            ApplicationOperatorConfig(
                operator_id="func_5",
                name="去中心化金融管理算法（Func-5）",
                domain="去中心化金融管理",
                algorithm_class=DefiManagementAlgorithm,
                quantum_primitive="grover_sfs",
                default_parameters={"logical_variables": 8, "grover_iterations": 2},
            )
        ),
        ApplicationCircuitOperator(
            ApplicationOperatorConfig(
                operator_id="func_6",
                name="反欺诈监测算法（Func-6）",
                domain="反欺诈监测",
                algorithm_class=AntifraudMonitoringAlgorithm,
                quantum_primitive="rasengan",
                default_parameters={"groups": 3, "layers": 1},
            )
        ),
        ApplicationCircuitOperator(
            ApplicationOperatorConfig(
                operator_id="func_7",
                name="支付与结算系统算法（Func-7）",
                domain="支付与结算",
                algorithm_class=PaymentSettlementAlgorithm,
                quantum_primitive="rasengan",
                default_parameters={"accounts": 3, "layers": 1},
            )
        ),
        ApplicationCircuitOperator(
            ApplicationOperatorConfig(
                operator_id="func_8",
                name="贷款发放决策算法（Func-8）",
                domain="贷款发放决策",
                algorithm_class=LoanDecisionAlgorithm,
                quantum_primitive="rasengan",
                default_parameters={"feature_groups": 3, "layers": 1},
            )
        ),
        ApplicationCircuitOperator(
            ApplicationOperatorConfig(
                operator_id="func_9",
                name="银行网点布局优化算法（Func-9）",
                domain="银行网点布局",
                algorithm_class=BranchLocationAlgorithm,
                quantum_primitive="grover_sfs",
                default_parameters={"candidate_sites": 8, "grover_iterations": 2},
            )
        ),
        ApplicationCircuitOperator(
            ApplicationOperatorConfig(
                operator_id="func_10",
                name="指数追踪算法（Func-10）",
                domain="指数追踪",
                algorithm_class=IndexTrackingAlgorithm,
                quantum_primitive="rasengan",
                default_parameters={"sectors": 3, "layers": 1},
            )
        ),
    ]


def default_operator_library() -> QuantumFinancialOperatorLibrary:
    return QuantumFinancialOperatorLibrary(
        [
            PortfolioOptimizationOperator(),
            ExpectedShortfallOperator(),
            EuropeanCallOptionOperator(),
            *_application_operators(),
        ]
    )


def default_software_templates() -> dict[str, Any]:
    application_template = ApplicationCircuitCatalogTemplate()
    return {
        application_template.id: application_template,
    }
