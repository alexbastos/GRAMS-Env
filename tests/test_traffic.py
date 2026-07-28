"""Testes unitários para o gerador de tráfego — CBR e Poisson."""

import numpy as np
import pytest

from grams_env.core.domain.cell import CellConfig
from grams_env.core.services.traffic import CBRTrafficGenerator


class TestCBRProfiles:
    """Testa a inicialização de perfis CBR."""

    def test_init_profiles_shapes(self):
        """Perfis devem ter shape (V,)."""
        config = CellConfig()
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)
        profiles = gen.init_cbr_profiles(10, rng)
        assert profiles.shape == (10,)

    def test_init_profiles_values(self):
        """Perfis devem conter apenas valores dos cbr_profiles_bytes."""
        config = CellConfig(cbr_profiles_bytes=(1000, 4000))
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)
        profiles = gen.init_cbr_profiles(100, rng)
        unique = set(profiles.tolist())
        assert unique.issubset({1000.0, 4000.0})


class TestDeterministicTraffic:
    """Testa o modo de tráfego determinístico (CBR puro)."""

    def test_deterministic_exact_arrival(self):
        """Filas devem crescer exatamente cbr_bytes * 8 bits por TTI."""
        config = CellConfig(traffic_mode="deterministic")
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)

        queues = np.zeros(3, dtype=np.float64)
        cbr_bytes = np.array([1000.0, 4000.0, 1000.0])

        gen.generate(queues, cbr_bytes, rng)

        expected = np.array([8000.0, 32000.0, 8000.0])
        np.testing.assert_array_equal(queues, expected)

    def test_deterministic_is_constant(self):
        """Cada chamada deve adicionar exatamente o mesmo valor."""
        config = CellConfig(traffic_mode="deterministic")
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)

        cbr = np.array([1000.0])
        q1 = np.zeros(1)
        q2 = np.zeros(1)

        gen.generate(q1, cbr, rng)
        gen.generate(q2, cbr, rng)
        assert q1[0] == q2[0] == 8000.0


class TestPoissonTraffic:
    """Testa o modo de tráfego Poisson estocástico."""

    def test_poisson_mode_is_default(self):
        """O modo padrão deve ser 'poisson'."""
        config = CellConfig()
        gen = CBRTrafficGenerator(config)
        assert gen.mode == "poisson"

    def test_poisson_requires_rng(self):
        """Modo Poisson sem rng deve levantar ValueError."""
        config = CellConfig(traffic_mode="poisson")
        gen = CBRTrafficGenerator(config)
        queues = np.zeros(3)
        cbr = np.array([1000.0, 1000.0, 1000.0])
        with pytest.raises(ValueError, match="rng"):
            gen.generate(queues, cbr, rng=None)

    def test_poisson_mean_converges(self):
        """Média de muitas amostras Poisson deve convergir para λ.

        Para λ=1000, com 10000 amostras, a média deve estar em
        [990, 1010] com altíssima probabilidade.
        """
        config = CellConfig(traffic_mode="poisson")
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)

        num_samples = 10000
        cbr = np.array([1000.0])
        total_bits = 0.0

        for _ in range(num_samples):
            q = np.zeros(1)
            gen.generate(q, cbr, rng)
            total_bits += q[0]

        mean_bytes = total_bits / num_samples / 8.0
        assert 990.0 < mean_bytes < 1010.0, (
            f"Média Poisson: {mean_bytes:.1f}, esperado ~1000"
        )

    def test_poisson_variance_positive(self):
        """Tráfego Poisson deve ter variância > 0 (não é constante)."""
        config = CellConfig(traffic_mode="poisson")
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)

        arrivals = []
        cbr = np.array([4000.0])
        for _ in range(100):
            q = np.zeros(1)
            gen.generate(q, cbr, rng)
            arrivals.append(q[0])

        variance = np.var(arrivals)
        assert variance > 0, "Tráfego Poisson deve ter variância > 0"

    def test_poisson_different_profiles(self):
        """λ=4000 deve gerar mais tráfego médio que λ=1000."""
        config = CellConfig(traffic_mode="poisson")
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)

        total_1000 = 0.0
        total_4000 = 0.0
        n = 5000

        for _ in range(n):
            q1 = np.zeros(1)
            q4 = np.zeros(1)
            gen.generate(q1, np.array([1000.0]), rng)
            gen.generate(q4, np.array([4000.0]), rng)
            total_1000 += q1[0]
            total_4000 += q4[0]

        mean_1000 = total_1000 / n / 8.0
        mean_4000 = total_4000 / n / 8.0

        assert mean_4000 > mean_1000 * 3.5, (
            f"λ=4000 ({mean_4000:.0f}) deveria ser ~4x λ=1000 ({mean_1000:.0f})"
        )

    def test_poisson_vectorized(self):
        """Deve funcionar com vetores de UEs com perfis diferentes."""
        config = CellConfig(traffic_mode="poisson")
        gen = CBRTrafficGenerator(config)
        rng = np.random.default_rng(42)

        queues = np.zeros(5, dtype=np.float64)
        cbr = np.array([1000.0, 4000.0, 1000.0, 4000.0, 1000.0])

        gen.generate(queues, cbr, rng)

        assert queues.shape == (5,)
        assert np.all(queues >= 0), "Filas não podem ser negativas"
        assert np.any(queues > 0), "Pelo menos um UE deve ter tráfego"


class TestTrafficModeConfig:
    """Testa a configuração do modo de tráfego."""

    def test_invalid_traffic_mode(self):
        """Modo de tráfego inválido deve levantar ValueError."""
        with pytest.raises(ValueError, match="traffic_mode"):
            CellConfig(traffic_mode="invalid")

    def test_deterministic_mode(self):
        """traffic_mode='deterministic' deve ser aceito."""
        config = CellConfig(traffic_mode="deterministic")
        assert config.traffic_mode == "deterministic"

    def test_poisson_mode(self):
        """traffic_mode='poisson' deve ser aceito."""
        config = CellConfig(traffic_mode="poisson")
        assert config.traffic_mode == "poisson"
