"""Testes unitários para o modelo de propagação 3GPP TR 36.873 — SEM gymnasium."""

import numpy as np

from grams_env.core.domain.cell import CellConfig
from grams_env.core.services.propagation import TR36873_UMa


class TestLOSProbability:
    """Testa a probabilidade de Line-of-Sight."""

    def test_los_probability_close_distance(self):
        """P_LOS deve ser alta para distâncias curtas."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d = np.array([10.0])
        p = model.los_probability(d)
        assert p[0] > 0.9, f"P_LOS a 10m deveria ser > 0.9, obteve {p[0]}"

    def test_los_probability_far_distance(self):
        """P_LOS deve diminuir com a distância."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d_close = np.array([50.0])
        d_far = np.array([400.0])
        p_close = model.los_probability(d_close)
        p_far = model.los_probability(d_far)
        assert p_close[0] > p_far[0], (
            f"P_LOS a 50m ({p_close[0]}) deveria ser > P_LOS a 400m ({p_far[0]})"
        )

    def test_los_probability_bounded_0_1(self):
        """P_LOS deve estar entre 0 e 1."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d = np.array([1.0, 10.0, 100.0, 500.0, 1000.0])
        p = model.los_probability(d)
        assert np.all(p >= 0.0) and np.all(p <= 1.0), (
            f"P_LOS fora do range [0, 1]: {p}"
        )

    def test_los_probability_vectorized(self):
        """Deve funcionar com vetores de distâncias."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d = np.linspace(10, 500, 50)
        p = model.los_probability(d)
        assert p.shape == (50,)


class TestPathLossDirect:
    """Testa o path loss direto gNB→UE."""

    def test_path_loss_los_at_100m(self):
        """Path loss LOS a 100m deve estar entre 60-80 dB para 2 GHz."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d = np.array([100.0])
        is_los = np.array([True])
        pl = model.path_loss_direct_db(d, is_los)
        assert 60.0 < pl[0] < 80.0, (
            f"Path loss LOS a 100m: {pl[0]:.1f} dB, esperado [60, 80]"
        )

    def test_path_loss_nlos_greater_than_los(self):
        """Path loss NLOS deve ser >= path loss LOS."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d = np.array([200.0])
        pl_los = model.path_loss_direct_db(d, np.array([True]))
        pl_nlos = model.path_loss_direct_db(d, np.array([False]))
        assert pl_nlos[0] >= pl_los[0], (
            f"NLOS ({pl_nlos[0]:.1f}) deveria ser >= LOS ({pl_los[0]:.1f})"
        )

    def test_path_loss_increases_with_distance(self):
        """Path loss deve aumentar com a distância."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d_near = np.array([50.0])
        d_far = np.array([300.0])
        is_los = np.array([True])
        pl_near = model.path_loss_direct_db(d_near, is_los)
        pl_far = model.path_loss_direct_db(d_far, is_los)
        assert pl_far[0] > pl_near[0], (
            f"PL a 300m ({pl_far[0]:.1f}) deveria ser > PL a 50m ({pl_near[0]:.1f})"
        )

    def test_path_loss_vectorized(self):
        """Deve funcionar com vetores."""
        config = CellConfig()
        model = TR36873_UMa(config)
        d = np.array([50.0, 100.0, 200.0, 300.0, 500.0])
        is_los = np.array([True, False, True, False, True])
        pl = model.path_loss_direct_db(d, is_los)
        assert pl.shape == (5,)
        assert np.all(pl > 0), "Path loss deve ser positivo"


class TestPathLossInterUE:
    """Testa o path loss inter-UE."""

    def test_path_loss_inter_ue_matrix(self):
        """Deve retornar uma matriz (V, V) de path loss."""
        config = CellConfig()
        model = TR36873_UMa(config)
        dist_matrix = np.array([
            [0.0, 50.0, 100.0],
            [50.0, 0.0, 70.0],
            [100.0, 70.0, 0.0],
        ])
        pl = model.path_loss_inter_ue_db(dist_matrix)
        assert pl.shape == (3, 3)
        assert np.all(pl > 0), "Path loss inter-UE deve ser positivo"

    def test_path_loss_inter_ue_symmetric_input(self):
        """Para distâncias simétricas, path loss deve ser simétrico."""
        config = CellConfig()
        model = TR36873_UMa(config)
        dist_matrix = np.array([
            [0.0, 100.0],
            [100.0, 0.0],
        ])
        pl = model.path_loss_inter_ue_db(dist_matrix)
        np.testing.assert_allclose(pl[0, 1], pl[1, 0], rtol=1e-10)
