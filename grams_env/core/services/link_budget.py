"""Cálculo de SINR e capacidade de Shannon — domínio puro."""

from __future__ import annotations

import numpy as np

from grams_env.core.domain.cell import CellConfig


class LinkBudget:
    """Serviço de cálculo de enlace: SINR e capacidade Shannon.

    Recebe CellConfig via DI. Sem estado mutável, sem gymnasium.
    """

    def __init__(self, config: CellConfig) -> None:
        self._cfg = config

    def compute_sinr(
        self,
        direct_gains: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """SINR = P_tx × g_direct / N  para UEs ativos.

        Parameters
        ----------
        direct_gains : np.ndarray
            Ganho direto linear por UE (V,).
        active_mask : np.ndarray
            Máscara booleana de UEs ativos (V,).

        Returns
        -------
        np.ndarray
            Vetor (V,) de SINR linear por UE.
        """
        return np.where(
            active_mask,
            self._cfg.gnb_tx_power_w * direct_gains / self._cfg.noise_w,
            0.0,
        )

    def shannon_capacity_bits(
        self,
        sinr: np.ndarray,
        num_rbs_per_ue: np.ndarray,
    ) -> np.ndarray:
        """C = B · log₂(1 + SINR) · TTI × num_rbs   [bits/TTI].

        Aplica o limiar de SINR: se SINR < limiar, C = 0.

        Parameters
        ----------
        sinr : np.ndarray
            SINR linear por UE (V,).
        num_rbs_per_ue : np.ndarray
            Quantidade de RBs alocados a cada UE (V,).

        Returns
        -------
        np.ndarray
            Capacidade em bits por TTI (V,).
        """
        above = sinr >= self._cfg.sinr_threshold_linear
        per_rb = np.where(
            above,
            self._cfg.rb_bandwidth_hz
            * np.log2(1.0 + sinr)
            * self._cfg.tti_s,
            0.0,
        )
        return per_rb * num_rbs_per_ue
