"""Round Robin — alocação cíclica determinística de Resource Blocks.

Distribui os K RBs de forma cíclica entre os V UEs, avançando o
ponteiro de início a cada TTI. É o baseline mais simples possível
e garante fairness perfeita em número de RBs, independente das
condições de canal.

Exemplo com K=6 RBs e V=3 UEs:
    TTI 0: [0, 1, 2, 0, 1, 2]   (começa no UE 0)
    TTI 1: [1, 2, 0, 1, 2, 0]   (começa no UE 1)
    TTI 2: [2, 0, 1, 2, 0, 1]   (começa no UE 2)
"""

from __future__ import annotations

import numpy as np

from grams_env.agents.baselines.base import BaselineAgent


class RoundRobinAgent(BaselineAgent):
    """Alocação Round Robin de Resource Blocks.

    A cada TTI, o ponteiro cíclico avança em 1, de modo que o UE
    que inicia a alocação rotaciona uniformemente.

    Parameters
    ----------
    num_rbs : int
        Número de Resource Blocks (K=50).
    num_ues : int
        Número de User Equipments (V).
    """

    def __init__(self, num_rbs: int = 50, num_ues: int = 10) -> None:
        super().__init__(num_rbs=num_rbs, num_ues=num_ues)
        self._pointer: int = 0

    def _select_action(
        self,
        obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Aloca RBs ciclicamente a partir do ponteiro atual.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação (não utilizada — RR é agnóstico ao canal).

        Returns
        -------
        np.ndarray
            Array (K,) com alocação cíclica de UEs.
        """
        # Gera sequência cíclica: [ptr, ptr+1, ..., V-1, 0, 1, ..., ptr-1, ...]
        action = np.array(
            [(self._pointer + k) % self.num_ues for k in range(self.num_rbs)],
            dtype=np.int64,
        )
        # Avança ponteiro para o próximo TTI
        self._pointer = (self._pointer + 1) % self.num_ues
        return action

    def reset(self) -> None:
        """Reseta o ponteiro cíclico para o início."""
        self._pointer = 0
