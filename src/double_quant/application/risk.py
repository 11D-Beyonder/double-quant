import pandas as pd
from double_quant.common.metric import expected_shortfall


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
        # Assuming equal weight for assets in the subset (as is common in risk attribution experiments)
        # or weighted based on returns_df if it represents weighted returns.
        # Here we calculate the portfolio return as the average of selected assets' returns.
        portfolio_returns = self.returns_df[selected_assets].mean(axis=1).values
        portfolio_es = expected_shortfall(portfolio_returns, self.alpha)

        return sum_individual_es - portfolio_es


class RiskAttributor:
    """
    Coordinates the risk attribution process using the Risk Saving model and Quantum/Classical Shapley solvers.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        solver_class,
        alpha: float = 0.95,
        **solver_kwargs,
    ):
        self.returns_df = returns_df
        self.alpha = alpha
        self.assets = returns_df.columns.tolist()
        self.num_assets = len(self.assets)

        self.rs_vfunc = RiskSavingValueFunction(returns_df, alpha)
        self.solver = solver_class(self.num_assets, self.rs_vfunc, **solver_kwargs)

    def attribute(self) -> dict[str, float]:
        """
        Calculate the Shapley Risk Contribution (SRC) for each asset.
        SRC_i = ES({i}) - Phi_i^RS
        """
        phi_rs_list = self.solver.get_all()
        src_results = {}

        for i, asset in enumerate(self.assets):
            phi_rs = phi_rs_list[i]
            ind_es = self.rs_vfunc.individual_es[asset]
            src = ind_es - phi_rs
            src_results[asset] = src

        return src_results
