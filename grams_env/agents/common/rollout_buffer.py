"""Rollout Buffer com suporte a GAE para PPO.

Armazena trajetórias coletadas durante o rollout e calcula
Generalized Advantage Estimation (GAE-λ) para atualização da política.
Suporta observações Dict (node_features + adjacency_matrix).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Buffer de trajetórias para PPO com GAE.

    Armazena transições (obs, action, reward, done, log_prob, value)
    e calcula advantages via GAE-λ após o rollout completo.

    Parameters
    ----------
    rollout_steps : int
        Número de steps por rollout.
    num_rbs : int
        Número de Resource Blocks (K) — dimensão da ação.
    gamma : float
        Fator de desconto.
    gae_lambda : float
        Parâmetro λ do GAE.
    device : torch.device
        Dispositivo para os tensores.
    """

    rollout_steps: int
    num_rbs: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self._ptr: int = 0
        self._full: bool = False

        # Observações: armazenadas como listas de dicts numpy (tamanho V varia)
        self._obs_node_features: list[np.ndarray] = []
        self._obs_adj_matrix: list[np.ndarray] = []

        # Ações, log_probs, values, rewards, dones
        self._actions: list[np.ndarray] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

        # GAE computado
        self._advantages: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    def reset(self) -> None:
        """Limpa o buffer para novo rollout."""
        self._ptr = 0
        self._full = False
        self._obs_node_features.clear()
        self._obs_adj_matrix.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        self._advantages = None
        self._returns = None

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Adiciona uma transição ao buffer.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação com 'node_features' e 'adjacency_matrix'.
        action : np.ndarray
            Ação tomada (K,).
        reward : float
            Recompensa recebida.
        done : bool
            Se o episódio terminou (terminated ou truncated).
        log_prob : float
            Log-probabilidade da ação sob a política atual.
        value : float
            Estimativa de valor do estado atual.
        """
        self._obs_node_features.append(obs["node_features"].copy())
        self._obs_adj_matrix.append(obs["adjacency_matrix"].copy())
        self._actions.append(action.copy())
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)
        self._ptr += 1

    @property
    def size(self) -> int:
        """Número de transições armazenadas."""
        return self._ptr

    def compute_gae(self, last_value: float) -> None:
        """Calcula GAE-λ advantages e returns.

        Parameters
        ----------
        last_value : float
            V(s_{T+1}) — valor do último estado (para bootstrap).
        """
        n = self._ptr
        rewards = np.array(self._rewards[:n], dtype=np.float64)
        values = np.array(self._values[:n], dtype=np.float64)
        dones = np.array(self._dones[:n], dtype=np.float64)

        advantages = np.zeros(n, dtype=np.float64)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t]
                + self.gamma * next_value * next_non_terminal
                - values[t]
            )
            last_gae = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
            advantages[t] = last_gae

        self._advantages = advantages
        self._returns = advantages + values

    def get_batches(
        self,
        batch_size: int,
    ) -> list[dict[str, torch.Tensor | list]]:
        """Gera mini-batches aleatórios para atualização PPO.

        Parameters
        ----------
        batch_size : int
            Tamanho de cada mini-batch.

        Returns
        -------
        list[dict]
            Lista de mini-batches, cada um contendo:
            - 'obs_node_features': lista de tensores (V_i, 3)
            - 'obs_adj_matrix': lista de tensores (V_i, V_i)
            - 'actions': tensor (B, K)
            - 'old_log_probs': tensor (B,)
            - 'advantages': tensor (B,)
            - 'returns': tensor (B,)
        """
        assert self._advantages is not None, (
            "Chame compute_gae() antes de get_batches()."
        )

        n = self._ptr
        indices = np.random.permutation(n)
        batches = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            batch = {
                "obs_node_features": [
                    torch.as_tensor(
                        self._obs_node_features[i],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for i in batch_idx
                ],
                "obs_adj_matrix": [
                    torch.as_tensor(
                        self._obs_adj_matrix[i],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for i in batch_idx
                ],
                "actions": torch.as_tensor(
                    np.stack([self._actions[i] for i in batch_idx]),
                    dtype=torch.long,
                    device=self.device,
                ),
                "old_log_probs": torch.tensor(
                    [self._log_probs[i] for i in batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "advantages": torch.as_tensor(
                    self._advantages[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "returns": torch.as_tensor(
                    self._returns[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                ),
            }
            batches.append(batch)

        return batches
