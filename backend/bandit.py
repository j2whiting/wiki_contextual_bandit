from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class LinearThompsonSamplingBandit:
    """Simple diagonal-approximation linear Thompson Sampling bandit.

    We maintain:
      A_diag = diag(lambda * I + sum x_i^2)
      b = sum r_i * x_i

    Posterior mean (approx):
      theta_mean = b / A_diag

    And sample:
      theta ~ N(theta_mean, nu^2 * diag(1 / A_diag))
    """
    def __init__(self, dim: int, lambda_: float = 1.0, nu: float = 1.0) -> None:
        self.dim = dim
        self.lambda_ = float(lambda_)
        self.nu = float(nu)

        # A = lambda * I  (diagonal stored)
        self.A_diag = np.full(self.dim, self.lambda_, dtype=np.float64)
        self.b = np.zeros(self.dim, dtype=np.float64)

        self.interactions: int = 0
        self.reward_history: List[float] = []

    @property
    def A_inv_diag(self) -> np.ndarray:
        return 1.0 / self.A_diag

    def theta_mean(self) -> np.ndarray:
        return self.b * self.A_inv_diag

    def sample_theta(self) -> np.ndarray:
        mean = self.theta_mean()
        std = self.nu * np.sqrt(self.A_inv_diag)
        return np.random.normal(loc=mean, scale=std)

    def predict_mean_reward(self, x: np.ndarray) -> float:
        theta = self.theta_mean()
        return float(np.dot(theta, x))

    def update(self, x: np.ndarray, reward: float) -> None:
        x = x.astype(np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"Feature dimension mismatch: {x.shape[0]} != {self.dim}")

        reward = float(reward)
        self.A_diag += x * x
        self.b += reward * x

        self.interactions += 1
        self.reward_history.append(reward)

    def avg_reward(self) -> float:
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history))

    def to_state_dict(self, max_reward_history: Optional[int] = 500) -> Dict[str, Any]:
        history: List[float] = list(self.reward_history)
        if max_reward_history is not None and max_reward_history > 0:
            history = history[-max_reward_history:]
        return {
            "dim": int(self.dim),
            "lambda_": float(self.lambda_),
            "nu": float(self.nu),
            "A_diag": self.A_diag.astype(float).tolist(),
            "b": self.b.astype(float).tolist(),
            "interactions": int(self.interactions),
            "reward_history": history,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return

        dim = int(state.get("dim", self.dim))
        if dim != self.dim:
            raise ValueError(f"Bandit dim mismatch: state dim={dim} != bandit dim={self.dim}")

        self.lambda_ = float(state.get("lambda_", self.lambda_))
        self.nu = float(state.get("nu", self.nu))

        A_diag = np.asarray(state.get("A_diag", self.A_diag), dtype=np.float64).reshape(-1)
        b = np.asarray(state.get("b", self.b), dtype=np.float64).reshape(-1)
        if A_diag.shape[0] != self.dim or b.shape[0] != self.dim:
            raise ValueError("Bandit state has wrong vector sizes.")

        self.A_diag = A_diag
        self.b = b

        self.interactions = int(state.get("interactions", self.interactions))
        history = state.get("reward_history", [])
        if isinstance(history, list):
            self.reward_history = [float(x) for x in history]
