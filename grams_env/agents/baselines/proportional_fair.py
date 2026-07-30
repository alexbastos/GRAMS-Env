"""Proportional Fair (PF) — alocação de RBs por prioridade PF.

Implementa o escalonador Proportional Fair clássico de telecom:

    prioridade_v(t) = taxa_instantânea_v(t) / throughput_médio_v(t)

O UE com maior prioridade recebe o próximo RB. O throughput médio
é atualizado com uma média móvel exponencial (EWMA):

    T̄_v(t) = (1 − 1/τ) · T̄_v(t−1) + (1/τ) · throughput_real_v(t−1)

onde τ (``window``) controla a "memória" do escalonador.

A taxa instantânea é derivada da CQI (coluna 0 de node_features),
que é proporcional a log₂(1 + SINR). Dessa forma, o PF balanceia
eficiência espectral (favorecendo UEs com bom canal) com justiça
temporal (favorecendo UEs historicamente subatendidos).

Ref: 3GPP TS 36.213, Viswanath et al. "Opportunistic beamforming
using dumb antennas" (IEEE Trans. IT, 2002).
"""

from __future__ import annotations

import numpy as np

from grams_env.agents.baselines.base import BaselineAgent


class ProportionalFairAgent(BaselineAgent):
    """Escalonador Proportional Fair para alocação de Resource Blocks.

    Parameters
    ----------
    num_rbs : int
        Número de Resource Blocks (K=50).
    num_ues : int
        Número de User Equipments (V).
    window : float
        Janela da média móvel exponencial (τ). Valores típicos: 10–100 TTIs.
        Quanto maior, mais lenta a adaptação do throughput médio histórico.
    initial_avg_throughput : float
        Valor inicial do throughput médio (evita divisão por zero no
        primeiro TTI). Deve ser um valor positivo pequeno.
    """

    def __init__(
        self,
        num_rbs: int = 50,
        num_ues: int = 10,
        window: float = 50.0,
        initial_avg_throughput: float = 1.0,
    ) -> None:
        super().__init__(num_rbs=num_rbs, num_ues=num_ues)
        self._window = window
        self._initial_avg = initial_avg_throughput
        # Throughput médio histórico por UE — EWMA
        self._avg_throughput = np.full(
            num_ues, initial_avg_throughput, dtype=np.float64,
        )

    def _select_action(
        self,
        obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Aloca RBs usando a métrica Proportional Fair.

        Procedimento por RB (greedy):
            1. Calcula taxa instantânea por UE a partir da CQI.
            2. Computa prioridade PF = taxa_inst / throughput_médio.
            3. Atribui o RB ao UE com maior prioridade.

        Após alocar todos os K RBs, atualiza o throughput médio
        histórico com EWMA baseado na quantidade de RBs obtidos.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação com 'node_features' (V, 3):
                col 0 — CQI (proxy de qualidade do canal em dB normalizada).
                col 1 — Tamanho da fila (bits).
                col 2 — Carga CBR (bytes).

        Returns
        -------
        np.ndarray
            Array (K,) com o ID do UE atribuído a cada RB.
        """
        node_features = obs["node_features"]  # (V, 3)
        num_ues = node_features.shape[0]

        # ---- Taxa instantânea a partir da CQI ----
        # CQI = 10·log₁₀(gain) + 200, portanto gain_db = CQI − 200
        # Taxa instantânea ∝ log₂(1 + 10^(gain_db/10))
        cqi = node_features[:, 0].astype(np.float64)
        gain_db = cqi - 200.0
        sinr_linear = np.power(10.0, gain_db / 10.0)
        instant_rate = np.log2(1.0 + sinr_linear)  # (V,)

        # ---- Prioridade PF: taxa instantânea / throughput médio ----
        priority = instant_rate / (self._avg_throughput + 1e-20)  # (V,)

        # ---- Alocação gulosa: atribui cada RB ao UE de maior prioridade ----
        # Acumula RBs alocados para ponderar a prioridade
        rbs_allocated = np.zeros(num_ues, dtype=np.int64)
        action = np.empty(self.num_rbs, dtype=np.int64)

        for k in range(self.num_rbs):
            ue = int(np.argmax(priority))
            action[k] = ue
            rbs_allocated[ue] += 1

            # Reduz levemente a prioridade do UE já alocado para
            # distribuir RBs de forma mais uniforme dentro do TTI.
            # Fator de escala: 1 / (1 + n_allocated) impede que um
            # único UE monopolize todos os RBs quando tem CQI muito alto.
            priority[ue] = instant_rate[ue] / (
                self._avg_throughput[ue] * (1 + rbs_allocated[ue]) + 1e-20
            )

        # ---- Atualiza throughput médio histórico (EWMA) ----
        # Proxy: o throughput neste TTI é proporcional à taxa × n_RBs
        throughput_this_tti = instant_rate * rbs_allocated.astype(np.float64)
        alpha = 1.0 / self._window
        self._avg_throughput = (
            (1.0 - alpha) * self._avg_throughput
            + alpha * throughput_this_tti
        )

        return action

    def reset(self) -> None:
        """Reseta o throughput médio histórico para o valor inicial."""
        self._avg_throughput = np.full(
            self.num_ues, self._initial_avg, dtype=np.float64,
        )
