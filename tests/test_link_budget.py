"""Testes unitários para o serviço de SINR e Shannon — SEM gymnasium."""

import numpy as np

from grams_env.core.domain.cell import CellConfig
from grams_env.core.services.link_budget import LinkBudget


class TestComputeSINR:
    """Testa o cálculo de SINR."""

    def test_sinr_active_ues(self):
        """SINR deve ser > 0 para UEs ativos com ganho > 0."""
        config = CellConfig()
        lb = LinkBudget(config)
        gains = np.array([1e-10, 1e-8, 1e-12])
        active = np.array([True, True, False])
        sinr = lb.compute_sinr(gains, active)
        assert sinr[0] > 0 and sinr[1] > 0
        assert sinr[2] == 0.0, "UE inativo deve ter SINR = 0"

    def test_sinr_inactive_ue_zero(self):
        """SINR deve ser exatamente 0 para UEs inativos."""
        config = CellConfig()
        lb = LinkBudget(config)
        gains = np.array([1e-8])
        active = np.array([False])
        sinr = lb.compute_sinr(gains, active)
        assert sinr[0] == 0.0

    def test_sinr_proportional_to_gain(self):
        """SINR deve ser proporcional ao ganho direto."""
        config = CellConfig()
        lb = LinkBudget(config)
        gains = np.array([1e-10, 2e-10])
        active = np.array([True, True])
        sinr = lb.compute_sinr(gains, active)
        np.testing.assert_allclose(sinr[1] / sinr[0], 2.0, rtol=1e-10)


class TestShannonCapacity:
    """Testa a capacidade de Shannon."""

    def test_shannon_zero_below_threshold(self):
        """Shannon deve retornar 0 quando SINR < limiar."""
        config = CellConfig()
        lb = LinkBudget(config)
        sinr = np.array([1.0])  # ~0 dB, abaixo do limiar 14.8 dB
        capacity = lb.shannon_capacity_bits(sinr, np.array([5.0]))
        assert capacity[0] == 0.0

    def test_shannon_positive_above_threshold(self):
        """Shannon deve retornar > 0 quando SINR > limiar."""
        config = CellConfig()
        lb = LinkBudget(config)
        # SINR = 10^(14.8/10) ≈ 30.2 (exatamente no limiar)
        sinr = np.array([config.sinr_threshold_linear + 1.0])
        capacity = lb.shannon_capacity_bits(sinr, np.array([5.0]))
        assert capacity[0] > 0, f"Capacidade deveria ser > 0, obteve {capacity[0]}"

    def test_shannon_zero_rbs(self):
        """Capacidade deve ser 0 quando num_rbs = 0."""
        config = CellConfig()
        lb = LinkBudget(config)
        sinr = np.array([100.0])  # SINR alta
        capacity = lb.shannon_capacity_bits(sinr, np.array([0.0]))
        assert capacity[0] == 0.0

    def test_shannon_scales_with_rbs(self):
        """Capacidade deve escalar linearmente com número de RBs."""
        config = CellConfig()
        lb = LinkBudget(config)
        sinr = np.array([100.0])
        cap_5 = lb.shannon_capacity_bits(sinr, np.array([5.0]))
        cap_10 = lb.shannon_capacity_bits(sinr, np.array([10.0]))
        np.testing.assert_allclose(cap_10[0] / cap_5[0], 2.0, rtol=1e-10)

    def test_shannon_vectorized(self):
        """Deve funcionar com vetores."""
        config = CellConfig()
        lb = LinkBudget(config)
        sinr = np.array([1.0, 50.0, 100.0])
        rbs = np.array([3.0, 5.0, 10.0])
        cap = lb.shannon_capacity_bits(sinr, rbs)
        assert cap.shape == (3,)
        assert cap[0] == 0.0  # abaixo do limiar
        assert cap[1] > 0.0   # acima do limiar
        assert cap[2] > 0.0   # acima do limiar
