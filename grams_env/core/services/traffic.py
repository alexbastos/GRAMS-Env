"""Gerador de tráfego CBR — domínio puro."""

from __future__ import annotations

import numpy as np

from grams_env.core.domain.cell import CellConfig


class CBRTrafficGenerator:
    """Serviço de geração de tráfego CBR (Constant Bit Rate).

    Cada UE gera uma quantidade fixa de dados (conforme perfil CBR)
    a cada TTI, convertida para bits e adicionada à fila.
    """

    def __init__(self, config: CellConfig) -> None:
        self._cfg = config

    def init_cbr_profiles(
        self,
        num_ues: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Atribui perfis de tráfego CBR aleatórios aos UEs.

        Parameters
        ----------
        num_ues : int
            Número de UEs.
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        np.ndarray
            Array (V,) com bytes/TTI por UE.
        """
        cbr_arr = np.array(self._cfg.cbr_profiles_bytes, dtype=np.float64)
        cbr_indices = rng.integers(0, len(cbr_arr), size=num_ues)
        return cbr_arr[cbr_indices]

    @staticmethod
    def generate(
        queues: np.ndarray,
        cbr_bytes: np.ndarray,
    ) -> np.ndarray:
        """Insere pacotes CBR nas filas de todos os UEs.

        Parameters
        ----------
        queues : np.ndarray
            Filas atuais (V,) em bits (será modificado in-place).
        cbr_bytes : np.ndarray
            Carga CBR por UE (V,) em bytes/TTI.

        Returns
        -------
        np.ndarray
            Filas atualizadas (V,) em bits.
        """
        arrival_bits = cbr_bytes * 8.0  # bytes → bits
        queues += arrival_bits
        return queues
