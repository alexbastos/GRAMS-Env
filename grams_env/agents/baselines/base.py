"""Interface base para agentes clássicos (baselines).

Define o contrato que RoundRobin e ProportionalFair implementam.
A interface é compatível com o ActorCriticProtocol do PPOTrainer,
mas retorna valores dummy para log_prob e value (não se aplicam
a algoritmos heurísticos).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaselineAgent(ABC):
    """Interface comum para agentes clássicos de alocação de RBs.

    Compatível com o loop de avaliação do GRAMS-Env:
        action, _, _ = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    Subclasses implementam ``_select_action`` com a lógica de alocação.

    Parameters
    ----------
    num_rbs : int
        Número de Resource Blocks disponíveis (K=50).
    num_ues : int
        Número de User Equipments (V).
    """

    def __init__(self, num_rbs: int = 50, num_ues: int = 10) -> None:
        self.num_rbs = num_rbs
        self.num_ues = num_ues

    @abstractmethod
    def _select_action(
        self,
        obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Retorna a alocação de RBs dado o estado observado.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação com 'node_features' (V, 3) e 'adjacency_matrix' (V, V).

        Returns
        -------
        np.ndarray
            Array (K,) com o ID do UE atribuído a cada RB.
        """
        ...

    def act(
        self,
        obs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, float]:
        """Seleciona ação — interface compatível com ActorCriticProtocol.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação Gymnasium.

        Returns
        -------
        action : np.ndarray (K,)
            IDs dos UEs alocados a cada RB.
        log_prob : float
            Sempre 0.0 (heurísticas não têm log-probabilidade).
        value : float
            Sempre 0.0 (heurísticas não estimam valor).
        """
        action = self._select_action(obs)
        return action, 0.0, 0.0

    def reset(self) -> None:
        """Reseta estado interno entre episódios (se necessário).

        Subclasses podem sobrescrever para limpar estado acumulado.
        """
