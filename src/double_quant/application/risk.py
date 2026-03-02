from typing import Literal

import pandas as pd

from double_quant.common.metric import expected_shortfall
from double_quant.algorithm.shapley import ShapleyCalculator


class ExpectedShortfallValueFunction:
    """
    Direct ES characteristic function for Shapley value calculation.
    V(S) = ES(S)

    Note: ES is subadditive, so this is NOT compatible with QuantumCalculator,
    which requires a superadditive value function. Use RiskSavingValueFunction
    for quantum-compatible computation.
    """

    def __init__(self, returns_df: pd.DataFrame, alpha: float = 0.95):
        self.returns_df = returns_df
        self.alpha = alpha
        self.assets = returns_df.columns.tolist()
        self.num_assets = len(self.assets)

    def __getitem__(self, bitmask: int) -> float:
        """
        Calculate V(S) = ES(S) for a subset of assets defined by a bitmask.
        """
        if bitmask == 0:
            return 0.0

        selected_assets = [
            self.assets[i] for i in range(self.num_assets) if (bitmask >> i) & 1
        ]
        portfolio_returns = self.returns_df[selected_assets].mean(axis=1).to_numpy()
        return expected_shortfall(portfolio_returns, self.alpha)


class RiskSavingValueFunction:
    """
    Implements the 'Risk Saving' (RS) characteristic function for Shapley value calculation.
    RS(S) = sum_{i in S} ES({i}) - ES(S)
    """

    def __init__(self, returns_df: pd.DataFrame, alpha: float = 0.95):
        self.returns_df = returns_df
        self.alpha = alpha
        self.assets = returns_df.columns.tolist()
        self.num_assets = len(self.assets)

        # Pre-calculate individual ES for each asset
        self.individual_es = {
            asset: expected_shortfall(returns_df[asset].values, alpha)
            for asset in self.assets
        }

    def __getitem__(self, bitmask: int) -> float:
        """
        Calculate RS(S) for a subset of assets defined by a bitmask.
        """
        if bitmask == 0:
            return 0.0

        selected_assets = []
        for i in range(self.num_assets):
            if (bitmask >> i) & 1:
                selected_assets.append(self.assets[i])

        if not selected_assets:
            return 0.0

        # sum_{i in S} ES({i})
        sum_individual_es = sum(self.individual_es[asset] for asset in selected_assets)

        # ES(S)
        # HACK: Assuming equal weight for assets in the subset (as is common in risk attribution experiments)
        # or weighted based on returns_df if it represents weighted returns.
        # Here we calculate the portfolio return as the average of selected assets' returns.
        portfolio_returns = self.returns_df[selected_assets].mean(axis=1).values
        portfolio_es = expected_shortfall(portfolio_returns, self.alpha)

        return sum_individual_es - portfolio_es


class RiskAttributor:
    """
    Coordinates the risk attribution process using Shapley solvers.

    Two modes are supported:
    - "rs" (default): Uses the Risk Saving characteristic function RS(S) = Σ ES({i}) − ES(S).
      The Shapley value Φ_i^RS is computed, then SRC_i = ES({i}) − Φ_i^RS.
      RS is superadditive, so this mode is compatible with QuantumCalculator.
    - "es": Uses ES directly as the characteristic function V(S) = ES(S).
      The Shapley value Φ_i^ES is computed, then SRC_i = Φ_i^ES.
      ES is subadditive, so this mode is NOT compatible with QuantumCalculator.

    Both modes produce mathematically equivalent results.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        solver_class: type[ShapleyCalculator],
        alpha: float = 0.95,
        mode: Literal["es", "rs"] = "rs",
        **solver_kwargs,
    ):
        self.mode = mode
        self.assets = returns_df.columns.tolist()
        self.num_assets = len(self.assets)

        if mode == "rs":
            self.vfunc = RiskSavingValueFunction(returns_df, alpha)
        else:
            self.vfunc = ExpectedShortfallValueFunction(returns_df, alpha)

        self.solver = solver_class(self.num_assets, self.vfunc, **solver_kwargs)

    def attribute(self) -> dict[str, float]:
        """
        Calculate the Shapley Risk Contribution (SRC) for each asset.

        mode="rs": SRC_i = ES({i}) − Φ_i^RS  (indirect, quantum-compatible)
        mode="es": SRC_i = Φ_i^ES            (direct, classical only)
        """
        phi_list = self.solver.get_all()

        if self.mode == "rs":
            return {
                asset: self.vfunc.individual_es[asset] - phi_list[i]  # type: ignore[union-attr]
                for i, asset in enumerate(self.assets)
            }
        else:
            return {asset: phi_list[i] for i, asset in enumerate(self.assets)}
