"""Snapshot do estado da rede em um TTI — Value Object imutável."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NetworkState:
    """Estado completo da rede em um TTI, para consumo pelos serviços.

    Nota: numpy é usado aqui como tipo de dado matricial (Value Object),
    NÃO como lógica computacional. É equivalente a uma "list of floats"
    performática. As operações matemáticas ficam nos Services.
    """

    positions: np.ndarray             # (V, 2)
    speeds: np.ndarray                # (V,)
    directions: np.ndarray            # (V,)
    queues: np.ndarray                # (V,)
    cbr_bytes: np.ndarray             # (V,)
    direct_gains: np.ndarray          # (V,)
    interference_gains: np.ndarray    # (V, V)
    is_los: np.ndarray                # (V,) bool

    @property
    def num_ues(self) -> int:
        return len(self.queues)
