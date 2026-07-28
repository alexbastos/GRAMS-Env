"""Testes de integração — OpenRAN_RBA_Env com gymnasium."""

import numpy as np
import pytest

from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env
from grams_env.core.domain.cell import CellConfig


class TestEnvInit:
    """Testa a inicialização do ambiente."""

    def test_default_init(self):
        """Deve inicializar com defaults sem erros."""
        env = OpenRAN_RBA_Env(num_ues=10)
        assert env.num_ues == 10
        assert env.num_rbs == 50

    def test_custom_config(self):
        """Deve aceitar CellConfig customizado."""
        config = CellConfig(num_rbs=25, cell_radius_m=300.0)
        env = OpenRAN_RBA_Env(num_ues=5, config=config)
        assert env.num_rbs == 25
        assert env.config.cell_radius_m == 300.0


class TestEnvReset:
    """Testa o reset do ambiente."""

    def test_reset_returns_correct_shapes(self):
        """Shapes das observações devem corresponder ao observation_space."""
        env = OpenRAN_RBA_Env(num_ues=10)
        obs, info = env.reset(seed=42)
        assert obs["node_features"].shape == (10, 3)
        assert obs["adjacency_matrix"].shape == (10, 10)

    def test_reset_returns_info(self):
        """Info deve conter metadados do episódio."""
        env = OpenRAN_RBA_Env(num_ues=5)
        obs, info = env.reset(seed=42)
        assert "ue_speeds_kmh" in info
        assert "ue_cbr_bytes" in info
        assert "ue_is_los" in info
        assert len(info["ue_speeds_kmh"]) == 5

    def test_reset_observation_in_space(self):
        """Observação deve estar dentro do observation_space."""
        env = OpenRAN_RBA_Env(num_ues=10)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_reset_reproducibility(self):
        """Dois resets com mesma seed devem produzir mesma observação."""
        env = OpenRAN_RBA_Env(num_ues=10)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(
            obs1["node_features"], obs2["node_features"]
        )


class TestEnvStep:
    """Testa o step do ambiente."""

    def test_step_returns_correct_types(self):
        """Step deve retornar (obs, reward, terminated, truncated, info)."""
        env = OpenRAN_RBA_Env(num_ues=5)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shapes(self):
        """Shapes devem ser consistentes após step."""
        env = OpenRAN_RBA_Env(num_ues=10)
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs["node_features"].shape == (10, 3)
        assert obs["adjacency_matrix"].shape == (10, 10)

    def test_step_observation_in_space(self):
        """Observação pós-step deve estar no observation_space."""
        env = OpenRAN_RBA_Env(num_ues=10)
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)

    def test_step_info_keys(self):
        """Info deve conter todas as métricas esperadas."""
        env = OpenRAN_RBA_Env(num_ues=5)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        expected_keys = {
            "total_throughput_bits",
            "mean_sinr_db",
            "total_queue_bits",
            "num_active_ues",
            "num_failed_sinr",
            "step",
        }
        assert expected_keys.issubset(info.keys())

    def test_multiple_steps_no_crash(self):
        """Deve executar múltiplos steps sem erros."""
        env = OpenRAN_RBA_Env(num_ues=10)
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    def test_step_without_reset_raises(self):
        """Step sem reset deve levantar AssertionError."""
        env = OpenRAN_RBA_Env(num_ues=5)
        with pytest.raises(AssertionError, match="reset"):
            env.step(env.action_space.sample())

    def test_queues_grow_without_allocation(self):
        """Filas devem crescer quando não há alocação efetiva."""
        env = OpenRAN_RBA_Env(num_ues=5)
        env.reset(seed=42)
        # Executa vários steps
        for _ in range(10):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
        # Filas devem ter crescido (nem toda capacidade atende toda demanda)
        assert info["total_queue_bits"] > 0


class TestEnvRender:
    """Testa o render do ambiente."""

    def test_render_before_reset(self, capsys):
        """Render antes de reset deve imprimir mensagem."""
        env = OpenRAN_RBA_Env(num_ues=5)
        env.render()
        captured = capsys.readouterr()
        assert "reset" in captured.out.lower()

    def test_render_after_reset(self, capsys):
        """Render após reset deve imprimir estado."""
        env = OpenRAN_RBA_Env(num_ues=5)
        env.reset(seed=42)
        env.render()
        captured = capsys.readouterr()
        assert "TTI" in captured.out
