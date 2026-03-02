import numpy as np
from itertools import permutations
from scipy import special

from double_quant.algorithm.shapley.protocol import ValueFunction


class ShapleyCalculator:
    """计算 Shapley 值的基类 只需要实现 _calculate_one 方法，该方法计算给定 player 的 Shapley 值。"""

    def __init__(self, num_players: int, value_dict: ValueFunction | None = None):
        """初始化 Shapley 计算器。

        Args:
            num_players (int): 玩家数。
            value_dict (ValueFunction, optional): 子集收益值字典，为 None 时随机生成。
        """
        self.num_players = num_players

        self.value_dict = value_dict

        self._shapley_cache: list[float | None] = [None] * num_players

    def _calculate_one(self, target_player: int) -> float:
        raise NotImplementedError

    def get_one(self, target_player: int) -> float:
        if self._shapley_cache[target_player] is None:
            self._shapley_cache[target_player] = self._calculate_one(target_player)

        val = self._shapley_cache[target_player]
        if val is None:
            raise RuntimeError(
                f"Failed to calculate Shapley value for player {target_player}"
            )
        return val

    def get_all(self) -> list[float]:
        return [self.get_one(i) for i in range(self.num_players)]


class BinaryEnumerationCalculator(ShapleyCalculator):
    def __init__(self, num_players: int, value_dict: ValueFunction | None = None):
        super().__init__(num_players, value_dict)
        self.__factorial = [1] * (num_players + 1)
        for i in range(1, num_players + 1):
            self.__factorial[i] = self.__factorial[i - 1] * i

    def _calculate_one(self, target_player: int):
        if self.value_dict is None:
            raise ValueError("value_dict is required")

        contribution = 0
        for subset in range(2**self.num_players):
            # NOTE: 枚举不包含 player_index 的子集。
            if (1 << target_player) & subset:
                continue
            subset_size = bin(subset).count("1")
            weight = (
                self.__factorial[self.num_players - subset_size - 1]
                / self.__factorial[self.num_players]
                * self.__factorial[subset_size]
            )
            contribution += weight * (
                self.value_dict[subset | (1 << target_player)] - self.value_dict[subset]
            )
        return contribution


class PermutationEnumerationCalculator(ShapleyCalculator):
    def __init__(self, num_players: int, value_dict: ValueFunction | None = None):
        super().__init__(num_players, value_dict)
        self.__weight = special.factorial(self.num_players, exact=True)

    def _calculate_one(self, target_player):
        if self.value_dict is None:
            raise ValueError("value_dict is required")

        contribution = 0
        for subset in permutations(range(self.num_players), self.num_players - 1):
            bin_pre = 0
            for idx in subset:
                if idx == target_player:
                    break
                bin_pre |= 1 << idx
            contribution += (
                self.value_dict[bin_pre | (1 << target_player)]
                - self.value_dict[bin_pre]
            ) / self.__weight
        return contribution


class PermutationMCCalculator(ShapleyCalculator):
    """Classic Monte Carlo Shapley estimator using permutation sampling.

    Implements the algorithm from Castro et al. (2009):
    φ̂_i = (1/T) Σ_{t=1}^T [v(P_i^t ∪ {i}) - v(P_i^t)]

    where P_i^t is the set of players preceding i in the t-th random permutation.
    """

    def __init__(
        self,
        num_players: int,
        value_dict: ValueFunction | None = None,
        num_samples: int = 100,
        seed: int | None = None,
    ):
        super().__init__(num_players, value_dict)
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def _calculate_one(self, target_player: int) -> float:
        if self.value_dict is None:
            raise ValueError("value_dict is required")

        contribution = 0.0
        for _ in range(self.num_samples):
            # Generate random permutation
            perm = list(range(self.num_players))
            self.rng.shuffle(perm)

            # Find position of target player and compute marginal contribution
            precedent_mask = 0
            for p in perm:
                if p == target_player:
                    break
                precedent_mask |= 1 << p

            # v(S ∪ {i}) - v(S)
            with_player = precedent_mask | (1 << target_player)
            contribution += (
                self.value_dict[with_player] - self.value_dict[precedent_mask]
            )

        return contribution / self.num_samples

    def get_oracle_count(self, player_index: int) -> int:
        """Each sample requires 2 value function lookups."""
        return self.num_samples * 2
