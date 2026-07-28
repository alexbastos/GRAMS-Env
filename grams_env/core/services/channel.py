"""Calculador de ganhos de canal — path loss + shadowing + fading."""

from __future__ import annotations

import numpy as np

from grams_env.core.domain.cell import CellConfig
from grams_env.core.ports.propagation_port import PropagationModel


class ChannelGainCalculator:
    """Serviço de cálculo de ganhos de canal direto e de interferência.

    Componentes:
        • Path Loss determinístico (via PropagationModel injetado)
        • Log-normal Shadowing (lento, fixo por episódio)
        • Rayleigh Fading (rápido, renovado a cada TTI)

    Recebe CellConfig e PropagationModel via DI.
    """

    def __init__(
        self,
        config: CellConfig,
        propagation: PropagationModel,
    ) -> None:
        self._cfg = config
        self._prop = propagation

    def sample_los(
        self,
        d_2d: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Amostra o estado LOS/NLOS de cada UE conforme a probabilidade.

        Parameters
        ----------
        d_2d : np.ndarray
            Distância 2D de cada UE ao gNB (V,).
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        np.ndarray
            Array booleano (V,); True = LOS.
        """
        p_los = self._prop.los_probability(d_2d)
        return rng.random(size=d_2d.shape) < p_los

    def generate_shadowing(
        self,
        is_los: np.ndarray,
        num_ues: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera shadowing lento (fixo por episódio).

        Parameters
        ----------
        is_los : np.ndarray
            Estado LOS/NLOS de cada UE (V,).
        num_ues : int
            Número de UEs.
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (shadowing_direct (V,), shadowing_inter (V, V)) em dB.
        """
        c = self._cfg
        shadowing_direct = np.where(
            is_los,
            rng.normal(0, c.shadowing_std_los_db, num_ues),
            rng.normal(0, c.shadowing_std_nlos_db, num_ues),
        )
        shadowing_inter = rng.normal(
            0, c.shadowing_std_nlos_db, (num_ues, num_ues)
        )
        np.fill_diagonal(shadowing_inter, 0.0)
        return shadowing_direct, shadowing_inter

    def compute_gains(
        self,
        positions: np.ndarray,
        is_los: np.ndarray,
        shadowing_direct: np.ndarray,
        shadowing_inter: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recalcula os ganhos de canal direto e de interferência.

        Parameters
        ----------
        positions : np.ndarray
            Posições dos UEs (V, 2) em metros.
        is_los : np.ndarray
            Estado LOS/NLOS de cada UE (V,).
        shadowing_direct : np.ndarray
            Shadowing direto (V,) em dB (fixo por episódio).
        shadowing_inter : np.ndarray
            Shadowing inter-UE (V, V) em dB (fixo por episódio).
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (direct_gain (V,), interference_gain (V, V)) em escala linear.
        """
        num_ues = len(positions)

        # --- Ganho direto (gNB → UE) ---
        dist_direct = np.linalg.norm(positions, axis=1)  # (V,)
        pl_direct_db = self._prop.path_loss_direct_db(dist_direct, is_los)

        # Ganho total = −PL − Shadowing (em dB) → linear
        total_loss_db = pl_direct_db + shadowing_direct
        pl_linear = 10 ** (-total_loss_db / 10)

        # Rayleigh fading: |h|² ∼ Exp(1), renovado a cada TTI
        fading_direct = rng.exponential(1.0, size=num_ues)
        direct_gain = (pl_linear * fading_direct).astype(np.float64)

        # --- Ganho de interferência (UE ↔ UE) ---
        diff = (
            positions[:, np.newaxis, :]
            - positions[np.newaxis, :, :]
        )  # (V, V, 2)
        dist_inter = np.linalg.norm(diff, axis=2)  # (V, V)

        pl_inter_db = self._prop.path_loss_inter_ue_db(dist_inter)
        total_inter_db = pl_inter_db + shadowing_inter
        pl_inter_linear = 10 ** (-total_inter_db / 10)

        fading_inter = rng.exponential(1.0, size=(num_ues, num_ues))
        interference_gain = (
            pl_inter_linear * fading_inter
        ).astype(np.float64)
        np.fill_diagonal(interference_gain, 0.0)

        return direct_gain, interference_gain
