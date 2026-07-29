"""Actor-Critic com GNN backbone para alocação de Resource Blocks.

Acopla o GraphEncoder (GATConv) ao algoritmo PPO:
    - Actor: GNN embeddings → MLP head → Categorical sobre V UEs × K RBs.
    - Critic: GNN embeddings → Global Mean Pool → MLP head → valor escalar.

A rede é invariante ao número de UEs (V), permitindo zero-shot:
treinar com V=20 e inferir com V=50, 100, 200.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from grams_env.agents.common.utils import adj_to_edge_index, obs_to_tensors
from grams_env.agents.gnn.graph_encoder import GraphEncoder


class GNNActorCritic(nn.Module):
    """Modelo Actor-Critic com GNN para PPO.

    Parameters
    ----------
    in_features : int
        Dimensão das node features (3).
    hidden_dim : int
        Dimensão dos embeddings GNN.
    num_layers : int
        Número de camadas GATConv.
    num_heads : int
        Número de heads de atenção.
    num_rbs : int
        Número de Resource Blocks (K=50).
    device : torch.device | str
        Dispositivo de computação.
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_rbs: int = 50,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.num_rbs = num_rbs

        # GNN Backbone
        self.encoder = GraphEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Actor Head: por nó → logit escalar → softmax sobre V UEs
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Critic Head: global pooling → valor escalar
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _encode(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Passa observação pela GNN e retorna embeddings.

        Parameters
        ----------
        node_features : torch.Tensor (V, 3)
        adj_matrix : torch.Tensor (V, V)

        Returns
        -------
        torch.Tensor (V, hidden_dim)
        """
        edge_index, edge_attr = adj_to_edge_index(adj_matrix)
        edge_index = edge_index.to(self._device)
        edge_attr = edge_attr.to(self._device)
        return self.encoder(node_features, edge_index, edge_attr)

    def _get_policy_and_value(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Calcula distribuição de política e valor para uma observação.

        Returns
        -------
        dist : Categorical
            Distribuição sobre V UEs.
        value : torch.Tensor
            Valor escalar do estado.
        """
        embeddings = self._encode(node_features, adj_matrix)  # (V, hidden)

        # Actor: logits por nó → distribuição sobre UEs
        logits = self.actor_head(embeddings).squeeze(-1)  # (V,)
        dist = Categorical(logits=logits)

        # Critic: mean pooling → valor
        pooled = embeddings.mean(dim=0)  # (hidden,)
        value = self.critic_head(pooled).squeeze(-1)  # escalar

        return dist, value

    def act(
        self,
        obs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, float]:
        """Seleciona ação para interação com o ambiente (inference).

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação com 'node_features' (V, 3) e 'adjacency_matrix' (V, V).

        Returns
        -------
        action : np.ndarray (K,)
            IDs dos UEs alocados a cada RB.
        log_prob : float
            Log-probabilidade total da ação.
        value : float
            Estimativa de valor V(s).
        """
        obs_t = obs_to_tensors(obs, self._device)
        dist, value = self._get_policy_and_value(
            obs_t["node_features"], obs_t["adjacency_matrix"],
        )

        # Amostra K RBs da mesma distribuição Categorical
        actions = dist.sample((self.num_rbs,))  # (K,)
        log_probs = dist.log_prob(actions)       # (K,)
        total_log_prob = log_probs.sum()          # escalar

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

        Processa cada amostra individualmente (V pode variar entre amostras),
        depois empilha os resultados.

        Parameters
        ----------
        obs_node_features : list[torch.Tensor]
            Lista de (V_i, 3) — uma por amostra.
        obs_adj_matrix : list[torch.Tensor]
            Lista de (V_i, V_i) — uma por amostra.
        actions : torch.Tensor (B, K)
            Ações do batch.

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
            entropy_i = dist.entropy()                   # escalar

            batch_log_probs.append(log_prob_i)
            batch_values.append(value)
            batch_entropy.append(entropy_i)

        return (
            torch.stack(batch_log_probs),   # (B,)
            torch.stack(batch_values),       # (B,)
            torch.stack(batch_entropy),      # (B,)
        )
