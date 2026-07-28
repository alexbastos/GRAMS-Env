"""Adapter de RNG — encapsula numpy.random.Generator."""

from __future__ import annotations

import numpy as np


class NumpyRNG:
    """Adapter para numpy.random.Generator.

    Encapsula o RNG do numpy para que serviços de domínio
    não dependam diretamente do numpy. Útil para substituir
    o RNG em testes ou para usar outro backend.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    @property
    def generator(self) -> np.random.Generator:
        """Retorna o gerador numpy subjacente."""
        return self._rng

    def uniform(self, low: float, high: float, size: int) -> np.ndarray:
        return self._rng.uniform(low, high, size=size)

    def normal(self, loc: float, scale: float, size: int | tuple[int, ...]) -> np.ndarray:
        return self._rng.normal(loc, scale, size=size)

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return self._rng.integers(low, high, size=size)

    def random(self, size: int | tuple[int, ...]) -> np.ndarray:
        return self._rng.random(size=size)

    def exponential(self, scale: float, size: int | tuple[int, ...]) -> np.ndarray:
        return self._rng.exponential(scale, size=size)
