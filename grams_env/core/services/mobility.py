"""Modelo de mobilidade — random walk com reflexão."""

from __future__ import annotations

import numpy as np

from grams_env.core.domain.cell import CellConfig


class MobilityModel:
    """Serviço de mobilidade: random walk suave com reflexão.

    Atualiza posições 2D dos UEs com base na velocidade e direção.
    UEs que ultrapassam o raio da célula são refletidos para dentro.

    Recebe CellConfig via DI.
    """

    def __init__(self, config: CellConfig) -> None:
        self._cfg = config

    def init_positions(
        self,
        num_ues: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posiciona UEs uniformemente em disco.

        Parameters
        ----------
        num_ues : int
            Número de UEs.
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        np.ndarray
            Posições (V, 2) em metros.
        """
        c = self._cfg
        angles = rng.uniform(0, 2 * np.pi, size=num_ues)
        radii = c.cell_radius_m * np.sqrt(
            rng.uniform(0, 1, size=num_ues)
        )
        radii = np.clip(radii, c.min_distance_m, c.cell_radius_m)
        return np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )

    def init_speeds(
        self,
        num_ues: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Atribui velocidades aleatórias dos perfis de mobilidade.

        Parameters
        ----------
        num_ues : int
            Número de UEs.
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        np.ndarray
            Velocidades (V,) em m/s.
        """
        speed_ms = np.array(
            [s / 3.6 for s in self._cfg.mobility_speeds_kmh],
            dtype=np.float64,
        )
        speed_indices = rng.integers(0, len(speed_ms), size=num_ues)
        return speed_ms[speed_indices]

    def init_directions(
        self,
        num_ues: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Inicializa direções aleatórias.

        Returns
        -------
        np.ndarray
            Direções (V,) em radianos.
        """
        return rng.uniform(0, 2 * np.pi, size=num_ues)

    def update(
        self,
        positions: np.ndarray,
        speeds: np.ndarray,
        directions: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Atualiza a posição 2D dos UEs com random walk.

        Parameters
        ----------
        positions : np.ndarray
            Posições atuais (V, 2) em metros (será modificado in-place).
        speeds : np.ndarray
            Velocidades (V,) em m/s.
        directions : np.ndarray
            Direções atuais (V,) em radianos (será modificado in-place).
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (positions, directions) atualizados.
        """
        c = self._cfg
        num_ues = len(speeds)

        # Perturbação angular (desvio padrão de ±15°)
        angle_perturbation = rng.normal(0, np.radians(15), size=num_ues)
        directions += angle_perturbation

        # Deslocamento: Δx = v · cos(θ) · TTI,  Δy = v · sin(θ) · TTI
        dx = speeds * np.cos(directions) * c.tti_s
        dy = speeds * np.sin(directions) * c.tti_s
        positions[:, 0] += dx
        positions[:, 1] += dy

        # Reflexão: se o UE saiu do raio da célula, refletir a direção
        distances = np.linalg.norm(positions, axis=1)
        out_of_cell = distances > c.cell_radius_m

        if np.any(out_of_cell):
            scale = c.cell_radius_m / distances[out_of_cell]
            positions[out_of_cell] *= scale[:, np.newaxis]
            directions[out_of_cell] += np.pi  # inversão de 180°

        # Garantir distância mínima do gNB
        distances = np.linalg.norm(positions, axis=1)
        too_close = distances < c.min_distance_m
        if np.any(too_close):
            scale = c.min_distance_m / (distances[too_close] + 1e-10)
            positions[too_close] *= scale[:, np.newaxis]

        return positions, directions
