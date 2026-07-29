"""Actor-Critic MLP Baseline — Fully-Connected para comparação.

Baseline que demonstra a limitação de redes fully-connected para
alocação de Resource Blocks: o MLP concatena e "achata" todas as
features em um vetor fixo, tornando-o dependente do número exato
de UEs (V) usado no treinamento.

Contraste com o GNNActorCritic:
    - MLP: input_dim = V×3 + V×(V-1)/2 → FIXO para V de treino.
    - GNN: message passing por nó → funciona com QUALQUER V.

Isso prova a tese do artigo: sem GNN, a rede não generaliza para
densidades diferentes de UEs (zero-shot impossível).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from grams_env.agents.common.utils import obs_to_tensors


class MLPActorCritic(nn.Module):
    """Baseline Actor-Critic com MLP (sem GNN).

    Parameters
    ----------
    num_ues : int
        Número fixo de UEs (V) — define o tamanho do input.
    num_rbs : int
        Número de Resource Blocks (K=50).
    hidden_dim : int
        Dimensão das camadas ocultas.
    device : torch.device | str
        Dispositivo de computação.
    """

    def __init__(
        self,
        num_ues: int = 20,
        num_rbs: int = 50,
        hidden_dim: int = 256,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.num_ues = num_ues
        self.num_rbs = num_rbs

        # Dimensão do input flat:
        # node_features: V × 3
        # adjacency upper triangle: V × (V-1) / 2
        self._node_feat_dim = num_ues * 3
        self._adj_tri_dim = num_ues * (num_ues - 1) // 2
        self._flat_dim = self._node_feat_dim + self._adj_tri_dim

        # Actor: flat → logits (K × V)
        self.actor = nn.Sequential(
            nn.Linear(self._flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_rbs * num_ues),
        )

        # Critic: flat → valor escalar
        self.critic = nn.Sequential(
            nn.Linear(self._flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _flatten_obs(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Concatena observação em vetor flat.

        Parameters
        ----------
        node_features : torch.Tensor (V, 3)
        adj_matrix : torch.Tensor (V, V)

        Returns
        -------
        torch.Tensor (flat_dim,)
        """
        # Flatten node features
        nf_flat = node_features.reshape(-1)  # (V×3,)

        # Upper triangle da adjacency (sem diagonal)
        idx = torch.triu_indices(
            self.num_ues, self.num_ues, offset=1,
            device=adj_matrix.device,
        )
        adj_flat = adj_matrix[idx[0], idx[1]]  # (V×(V-1)/2,)

        return torch.cat([nf_flat, adj_flat])  # (flat_dim,)

    def _get_policy_and_value(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Calcula distribuição de política e valor.

        Returns
        -------
        dist : Categorical
            Distribuição (K, V) sobre UEs por RB.
        value : torch.Tensor
            Valor escalar.
        """
        flat = self._flatten_obs(node_features, adj_matrix)  # (flat_dim,)

        # Actor
        logits = self.actor(flat)  # (K×V,)
        logits = logits.reshape(self.num_rbs, self.num_ues)  # (K, V)
        dist = Categorical(logits=logits)

        # Critic
        value = self.critic(flat).squeeze(-1)  # escalar

        return dist, value

    def act(
        self,
        obs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, float]:
        """Seleciona ação para interação com o ambiente.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação com 'node_features' (V, 3) e 'adjacency_matrix' (V, V).

        Returns
        -------
        action : np.ndarray (K,)
        log_prob : float
        value : float
        """
        obs_t = obs_to_tensors(obs, self._device)
        dist, value = self._get_policy_and_value(
            obs_t["node_features"], obs_t["adjacency_matrix"],
        )

        # Cada RB tem sua própria distribuição Categorical
        actions = dist.sample()              # (K,)
        log_probs = dist.log_prob(actions)    # (K,)
        total_log_prob = log_probs.sum()      # escalar

        return (
            actions.cpu().numpy(),
            total_log_prob.item(),
            value.item(),
        )

    def evaluate(
        self,
        obs_node_features: list[torch.Tensor],
        obs_adj_matrix: list[torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Avalia ações para um batch (usado no update PPO).

        Parameters
        ----------
        obs_node_features : list[torch.Tensor]
            Lista de (V, 3) — uma por amostra.
        obs_adj_matrix : list[torch.Tensor]
            Lista de (V, V) — uma por amostra.
        actions : torch.Tensor (B, K)

        Returns
        -------
        log_probs : torch.Tensor (B,)
        values : torch.Tensor (B,)
        entropy : torch.Tensor (B,)
        """
        batch_log_probs = []
        batch_values = []
        batch_entropy = []

        for i, (nf, adj) in enumerate(
            zip(obs_node_features, obs_adj_matrix)
        ):
            dist, value = self._get_policy_and_value(nf, adj)

            action_i = actions[i]  # (K,)
            log_prob_i = dist.log_prob(action_i).sum()  # escalar
            entropy_i = dist.entropy().sum()             # soma sobre K RBs

            batch_log_probs.append(log_prob_i)
            batch_values.append(value)
            batch_entropy.append(entropy_i)

        return (
            torch.stack(batch_log_probs),   # (B,)
            torch.stack(batch_values),       # (B,)
            torch.stack(batch_entropy),      # (B,)
        )
