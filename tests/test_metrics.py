"""Testes unitários e de integração — Pipeline de Métricas.

Valida:
    1. MetricsCollector: fórmulas de SE, delay P95, latência.
    2. MetricsExporter: exportação CSV (resumo + série temporal) e HDF5.
    3. EpisodeRunner: integração com ambiente e baselines.
    4. Pipeline completo: run → export → read back.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from grams_env.metrics.collector import EpisodeResult, MetricsCollector
from grams_env.metrics.exporter import MetricsExporter
from grams_env.metrics.runner import EpisodeRunner


# ====================================================================== #
#  Fixtures & Helpers                                                      #
# ====================================================================== #

BANDWIDTH_HZ = 10e6   # 10 MHz
TTI_S = 1e-3           # 1 ms
NUM_UES = 5


def _make_obs(queue_bits: np.ndarray, cbr_bytes: np.ndarray) -> dict:
    """Cria observação sintética para testes do collector."""
    v = len(queue_bits)
    cqi = np.full(v, 150.0, dtype=np.float32)
    node_features = np.column_stack([
        cqi, queue_bits, cbr_bytes,
    ]).astype(np.float32)
    adjacency_matrix = np.zeros((v, v), dtype=np.float32)
    return {
        "node_features": node_features,
        "adjacency_matrix": adjacency_matrix,
    }


def _make_info(throughput: float, queue_total: float) -> dict:
    """Cria info dict sintético para testes do collector."""
    return {
        "total_throughput_bits": throughput,
        "mean_sinr_db": 20.0,
        "total_queue_bits": queue_total,
        "num_active_ues": NUM_UES,
        "num_failed_sinr": 0,
        "step": 1,
    }


def _make_result(
    num_steps: int = 10,
    num_ues: int = NUM_UES,
) -> EpisodeResult:
    """Cria EpisodeResult sintético para testes do exporter."""
    return EpisodeResult(
        spectral_efficiency_mean_bps_hz=5.123456,
        queue_delay_p95_ms=2.5678,
        ric_latency_mean_ms=0.012345,
        total_reward=1234.56,
        mean_throughput_bits_per_tti=51234.0,
        mean_queue_bits=8000.0,
        num_steps=num_steps,
        spectral_efficiency_ts=np.random.rand(num_steps),
        queue_delay_per_ue_ts=np.random.rand(num_steps, num_ues),
        inference_latency_ts=np.random.rand(num_steps) * 0.1,
        throughput_ts=np.random.rand(num_steps) * 100000,
        reward_ts=np.random.rand(num_steps) * 100,
        queue_total_ts=np.random.rand(num_steps) * 10000,
        config={
            "agent_name": "TestAgent",
            "num_ues": num_ues,
            "carrier_freq_ghz": 2.0,
            "bandwidth_mhz": 10.0,
            "seed": 42,
        },
    )


# ====================================================================== #
#  Testes — MetricsCollector                                               #
# ====================================================================== #

class TestMetricsCollector:
    """Testa acumulação de dados e cálculo de métricas."""

    def test_empty_compute_raises(self):
        """compute() sem record_step deve levantar ValueError."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)
        with pytest.raises(ValueError, match="Nenhum step"):
            collector.compute()

    def test_num_steps_property(self):
        """num_steps deve refletir o número de records."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)
        assert collector.num_steps == 0

        obs = _make_obs(
            np.zeros(NUM_UES), np.full(NUM_UES, 1000.0),
        )
        info = _make_info(50000.0, 0.0)
        collector.record_step(obs, info, 100.0, 0.001)
        assert collector.num_steps == 1

    def test_spectral_efficiency_formula(self):
        """SE = throughput / (BW × TTI)."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        throughput = 100_000.0  # bits
        expected_se = throughput / (BANDWIDTH_HZ * TTI_S)  # 10.0 bps/Hz

        obs = _make_obs(
            np.zeros(NUM_UES), np.full(NUM_UES, 1000.0),
        )
        info = _make_info(throughput, 0.0)
        collector.record_step(obs, info, 0.0, 0.001)

        result = collector.compute()
        assert result.spectral_efficiency_mean_bps_hz == pytest.approx(
            expected_se, rel=1e-10,
        )
        assert result.spectral_efficiency_ts[0] == pytest.approx(
            expected_se, rel=1e-10,
        )

    def test_queue_delay_formula(self):
        """W = queue_bits × tti_s / (cbr_bytes × 8) [s] → ms."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        queue_bits = np.array([8000.0, 16000.0, 0.0], dtype=np.float32)
        cbr_bytes = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)

        # delay_v = queue_v × tti_s / (cbr_v × 8) × 1000 [ms]
        # UE 0: 8000 / (8e6) * 1000 = 1.0 ms
        # UE 1: 16000 / (8e6) * 1000 = 2.0 ms
        # UE 2: 0 / (8e6) * 1000 = 0.0 ms
        expected_delays = np.array([1.0, 2.0, 0.0])

        obs = _make_obs(queue_bits, cbr_bytes)
        info = _make_info(50000.0, float(np.sum(queue_bits)))
        collector.record_step(obs, info, 0.0, 0.001)

        result = collector.compute()
        np.testing.assert_allclose(
            result.queue_delay_per_ue_ts[0],
            expected_delays,
            rtol=1e-6,
        )

    def test_queue_delay_zero_cbr_no_division_error(self):
        """cbr_bytes = 0 não deve causar divisão por zero."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        queue_bits = np.array([8000.0, 0.0], dtype=np.float32)
        cbr_bytes = np.array([0.0, 0.0], dtype=np.float32)

        obs = _make_obs(queue_bits, cbr_bytes)
        info = _make_info(0.0, 8000.0)
        collector.record_step(obs, info, 0.0, 0.001)

        result = collector.compute()
        # Com cbr=0, delay deve ser 0 (sem chegada, sem "espera")
        np.testing.assert_array_equal(
            result.queue_delay_per_ue_ts[0], [0.0, 0.0],
        )

    def test_p95_percentile_calculation(self):
        """P95 deve ser calculado sobre todos os UEs × TTIs."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        # 100 TTIs com 2 UEs: delays controlados
        cbr_bytes = np.array([1000.0, 1000.0], dtype=np.float32)
        for t in range(100):
            # UE 0: delay cresce linearmente de 0 a ~12.5 ms
            # UE 1: delay fixo em 0.5 ms
            q0 = float(t * 1000)  # bits
            q1 = 4000.0           # bits → 4000/(8000) = 0.5 ms
            obs = _make_obs(
                np.array([q0, q1], dtype=np.float32), cbr_bytes,
            )
            info = _make_info(50000.0, q0 + q1)
            collector.record_step(obs, info, 0.0, 0.001)

        result = collector.compute()
        # P95 deve existir e ser > 0
        assert result.queue_delay_p95_ms > 0.0
        assert result.num_steps == 100

    def test_inference_latency_recording(self):
        """Latência de inferência deve ser registrada em ms."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        obs = _make_obs(
            np.zeros(NUM_UES), np.full(NUM_UES, 1000.0),
        )
        info = _make_info(50000.0, 0.0)

        # Simula 3 TTIs com latências diferentes
        collector.record_step(obs, info, 0.0, 0.001)   # 1.0 ms
        collector.record_step(obs, info, 0.0, 0.002)   # 2.0 ms
        collector.record_step(obs, info, 0.0, 0.003)   # 3.0 ms

        result = collector.compute()
        expected_mean = (1.0 + 2.0 + 3.0) / 3.0
        assert result.ric_latency_mean_ms == pytest.approx(
            expected_mean, rel=1e-6,
        )
        np.testing.assert_allclose(
            result.inference_latency_ts, [1.0, 2.0, 3.0], rtol=1e-6,
        )

    def test_reset_clears_data(self):
        """reset() deve limpar todos os dados acumulados."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        obs = _make_obs(
            np.zeros(NUM_UES), np.full(NUM_UES, 1000.0),
        )
        info = _make_info(50000.0, 0.0)
        collector.record_step(obs, info, 0.0, 0.001)
        assert collector.num_steps == 1

        collector.reset()
        assert collector.num_steps == 0
        with pytest.raises(ValueError):
            collector.compute()

    def test_timeseries_shapes(self):
        """Séries temporais devem ter shapes corretas."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        n_steps = 20
        obs = _make_obs(
            np.zeros(NUM_UES), np.full(NUM_UES, 1000.0),
        )
        info = _make_info(50000.0, 0.0)
        for _ in range(n_steps):
            collector.record_step(obs, info, 100.0, 0.001)

        result = collector.compute()
        assert result.spectral_efficiency_ts.shape == (n_steps,)
        assert result.queue_delay_per_ue_ts.shape == (n_steps, NUM_UES)
        assert result.inference_latency_ts.shape == (n_steps,)
        assert result.throughput_ts.shape == (n_steps,)
        assert result.reward_ts.shape == (n_steps,)
        assert result.queue_total_ts.shape == (n_steps,)

    def test_total_reward_is_sum(self):
        """total_reward deve ser a soma de todas as recompensas."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        obs = _make_obs(
            np.zeros(NUM_UES), np.full(NUM_UES, 1000.0),
        )
        info = _make_info(50000.0, 0.0)

        rewards = [10.0, 20.0, 30.0]
        for r in rewards:
            collector.record_step(obs, info, r, 0.001)

        result = collector.compute()
        assert result.total_reward == pytest.approx(sum(rewards), rel=1e-10)

    def test_config_propagation(self):
        """Metadados de config devem ser propagados ao resultado."""
        collector = MetricsCollector(BANDWIDTH_HZ, TTI_S)

        obs = _make_obs(
            np.zeros(NUM_UES), np.full(NUM_UES, 1000.0),
        )
        info = _make_info(50000.0, 0.0)
        collector.record_step(obs, info, 0.0, 0.001)

        config = {"agent_name": "TestAgent", "seed": 42}
        result = collector.compute(config)
        assert result.config["agent_name"] == "TestAgent"
        assert result.config["seed"] == 42


# ====================================================================== #
#  Testes — MetricsExporter (CSV)                                          #
# ====================================================================== #

class TestExporterCSV:
    """Testa exportação para CSV."""

    def test_summary_csv_created(self, tmp_path):
        """Deve criar arquivo CSV com header e uma linha de dados."""
        result = _make_result()
        path = tmp_path / "summary.csv"

        MetricsExporter.to_summary_csv(result, path)
        assert path.exists()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert "spectral_efficiency_bps_hz" in rows[0]
        assert "queue_delay_p95_ms" in rows[0]
        assert "ric_latency_mean_ms" in rows[0]
        assert rows[0]["agent"] == "TestAgent"

    def test_summary_csv_append(self, tmp_path):
        """Modo append deve adicionar linhas sem duplicar header."""
        result = _make_result()
        path = tmp_path / "summary.csv"

        MetricsExporter.to_summary_csv(result, path, append=False)
        MetricsExporter.to_summary_csv(result, path, append=True)
        MetricsExporter.to_summary_csv(result, path, append=True)

        with open(path) as f:
            lines = f.readlines()

        # 1 header + 3 linhas de dados
        assert len(lines) == 4

    def test_summary_csv_columns_match(self, tmp_path):
        """Colunas do CSV devem corresponder a SUMMARY_COLUMNS."""
        result = _make_result()
        path = tmp_path / "summary.csv"

        MetricsExporter.to_summary_csv(result, path)

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == MetricsExporter.SUMMARY_COLUMNS

    def test_summary_csv_creates_parent_dirs(self, tmp_path):
        """Deve criar diretórios-pai inexistentes."""
        result = _make_result()
        path = tmp_path / "deep" / "nested" / "summary.csv"

        MetricsExporter.to_summary_csv(result, path)
        assert path.exists()

    def test_timeseries_csv_created(self, tmp_path):
        """Deve criar CSV de série temporal com linhas = num_steps."""
        num_steps = 15
        result = _make_result(num_steps=num_steps)
        path = tmp_path / "timeseries.csv"

        MetricsExporter.to_timeseries_csv(result, path)
        assert path.exists()

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert "step" in header
        assert "spectral_efficiency_bps_hz" in header
        assert len(rows) == num_steps

    def test_timeseries_csv_step_numbers(self, tmp_path):
        """Steps devem ser numerados de 1 a T."""
        num_steps = 10
        result = _make_result(num_steps=num_steps)
        path = tmp_path / "timeseries.csv"

        MetricsExporter.to_timeseries_csv(result, path)

        with open(path) as f:
            reader = csv.DictReader(f)
            steps = [int(row["step"]) for row in reader]

        assert steps == list(range(1, num_steps + 1))


# ====================================================================== #
#  Testes — MetricsExporter (HDF5)                                         #
# ====================================================================== #

class TestExporterHDF5:
    """Testa exportação para HDF5 (requer h5py)."""

    @pytest.fixture(autouse=True)
    def _check_h5py(self):
        pytest.importorskip("h5py")

    def test_hdf5_created(self, tmp_path):
        """Deve criar arquivo HDF5 com 3 grupos."""
        import h5py

        result = _make_result()
        path = tmp_path / "result.h5"

        MetricsExporter.to_hdf5(result, path)
        assert path.exists()

        with h5py.File(path, "r") as f:
            assert "summary" in f
            assert "timeseries" in f
            assert "config" in f

    def test_hdf5_summary_attrs(self, tmp_path):
        """Atributos de resumo devem conter as 3 métricas."""
        import h5py

        result = _make_result()
        path = tmp_path / "result.h5"

        MetricsExporter.to_hdf5(result, path)

        with h5py.File(path, "r") as f:
            s = f["summary"]
            assert s.attrs["spectral_efficiency_mean_bps_hz"] == pytest.approx(
                result.spectral_efficiency_mean_bps_hz,
            )
            assert s.attrs["queue_delay_p95_ms"] == pytest.approx(
                result.queue_delay_p95_ms,
            )
            assert s.attrs["ric_latency_mean_ms"] == pytest.approx(
                result.ric_latency_mean_ms,
            )

    def test_hdf5_timeseries_shapes(self, tmp_path):
        """Datasets de séries temporais devem ter shapes corretas."""
        import h5py

        num_steps = 20
        num_ues = 5
        result = _make_result(num_steps=num_steps, num_ues=num_ues)
        path = tmp_path / "result.h5"

        MetricsExporter.to_hdf5(result, path)

        with h5py.File(path, "r") as f:
            ts = f["timeseries"]
            assert ts["spectral_efficiency"].shape == (num_steps,)
            assert ts["queue_delay_per_ue"].shape == (num_steps, num_ues)
            assert ts["inference_latency"].shape == (num_steps,)
            assert ts["throughput"].shape == (num_steps,)
            assert ts["reward"].shape == (num_steps,)
            assert ts["queue_total"].shape == (num_steps,)

    def test_hdf5_config_attrs(self, tmp_path):
        """Atributos de config devem ser salvos."""
        import h5py

        result = _make_result()
        path = tmp_path / "result.h5"

        MetricsExporter.to_hdf5(result, path)

        with h5py.File(path, "r") as f:
            cfg = f["config"]
            assert cfg.attrs["agent_name"] == "TestAgent"
            assert cfg.attrs["seed"] == 42


# ====================================================================== #
#  Testes — EpisodeRunner                                                  #
# ====================================================================== #

class TestEpisodeRunner:
    """Testa a integração do runner com ambiente e baselines."""

    def test_run_with_round_robin(self):
        """Deve completar um episódio com RR e retornar métricas."""
        from grams_env.agents.baselines import RoundRobinAgent
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        env = OpenRAN_RBA_Env(num_ues=5, max_steps=50)
        agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=5)
        runner = EpisodeRunner(env, agent, agent_name="RoundRobin")

        result = runner.run(seed=42, max_steps=50)

        assert result.num_steps == 50
        assert result.spectral_efficiency_mean_bps_hz > 0
        assert result.queue_delay_p95_ms >= 0
        assert result.ric_latency_mean_ms > 0
        assert result.total_reward != 0.0
        assert result.config["agent_name"] == "RoundRobin"
        assert result.config["num_ues"] == 5
        assert result.config["seed"] == 42

    def test_run_with_proportional_fair(self):
        """Deve completar um episódio com PF e retornar métricas."""
        from grams_env.agents.baselines import ProportionalFairAgent
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        env = OpenRAN_RBA_Env(num_ues=5, max_steps=50)
        agent = ProportionalFairAgent(num_rbs=env.num_rbs, num_ues=5)
        runner = EpisodeRunner(env, agent, agent_name="PF")

        result = runner.run(seed=42, max_steps=50)

        assert result.num_steps == 50
        assert result.spectral_efficiency_mean_bps_hz > 0
        assert result.config["agent_name"] == "PF"

    def test_timeseries_shapes_match(self):
        """Shapes das séries temporais devem bater com num_steps e num_ues."""
        from grams_env.agents.baselines import RoundRobinAgent
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        num_ues = 10
        max_steps = 30
        env = OpenRAN_RBA_Env(num_ues=num_ues, max_steps=max_steps)
        agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=num_ues)
        runner = EpisodeRunner(env, agent)

        result = runner.run(seed=42, max_steps=max_steps)

        assert result.spectral_efficiency_ts.shape == (max_steps,)
        assert result.queue_delay_per_ue_ts.shape == (max_steps, num_ues)
        assert result.inference_latency_ts.shape == (max_steps,)

    def test_metrics_physically_reasonable(self):
        """Métricas devem estar em faixas fisicamente razoáveis."""
        from grams_env.agents.baselines import RoundRobinAgent
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        env = OpenRAN_RBA_Env(num_ues=10, max_steps=100)
        agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=10)
        runner = EpisodeRunner(env, agent)

        result = runner.run(seed=42, max_steps=100)

        # SE: 0 < SE < 100 bps/Hz (faixa razoável para 10 MHz @ 50 RBs)
        assert 0 < result.spectral_efficiency_mean_bps_hz < 100

        # Delay P95: >= 0 ms
        assert result.queue_delay_p95_ms >= 0

        # Latência RIC: 0 < latency < 100 ms (heurísticas são rápidas)
        assert 0 < result.ric_latency_mean_ms < 100

    def test_random_agent_works(self):
        """Deve funcionar com um agente que usa action_space.sample()."""
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        env = OpenRAN_RBA_Env(num_ues=5, max_steps=20)

        class RandomAgent:
            def __init__(self, env):
                self._env = env

            def act(self, obs):
                return self._env.action_space.sample(), 0.0, 0.0

        agent = RandomAgent(env)
        runner = EpisodeRunner(env, agent, agent_name="Random")
        result = runner.run(seed=42, max_steps=20)

        assert result.num_steps == 20
        assert result.config["agent_name"] == "Random"


# ====================================================================== #
#  Testes — Pipeline Completo                                              #
# ====================================================================== #

class TestFullPipeline:
    """Testa o pipeline completo: run → export → read back."""

    def test_run_and_export_csv(self, tmp_path):
        """Pipeline RR → CSV resumo → leitura."""
        from grams_env.agents.baselines import RoundRobinAgent
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        env = OpenRAN_RBA_Env(num_ues=5, max_steps=50)
        agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=5)
        runner = EpisodeRunner(env, agent, agent_name="RR")

        result = runner.run(seed=42, max_steps=50)

        # Exporta
        summary_path = tmp_path / "summary.csv"
        ts_path = tmp_path / "timeseries.csv"
        MetricsExporter.to_summary_csv(result, summary_path)
        MetricsExporter.to_timeseries_csv(result, ts_path)

        # Lê de volta e verifica
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert float(row["spectral_efficiency_bps_hz"]) == pytest.approx(
                result.spectral_efficiency_mean_bps_hz, rel=1e-4,
            )
            assert float(row["queue_delay_p95_ms"]) == pytest.approx(
                result.queue_delay_p95_ms, rel=1e-3,
            )
            assert row["agent"] == "RR"

        with open(ts_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 50

    def test_multiple_agents_csv_append(self, tmp_path):
        """Deve acumular resultados de RR + PF em um único CSV."""
        from grams_env.agents.baselines import (
            ProportionalFairAgent,
            RoundRobinAgent,
        )
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        env = OpenRAN_RBA_Env(num_ues=5, max_steps=30)
        path = tmp_path / "combined.csv"

        for name, AgentClass in [
            ("RR", RoundRobinAgent),
            ("PF", ProportionalFairAgent),
        ]:
            agent = AgentClass(num_rbs=env.num_rbs, num_ues=5)
            runner = EpisodeRunner(env, agent, agent_name=name)
            result = runner.run(seed=42, max_steps=30)
            MetricsExporter.to_summary_csv(result, path, append=True)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["agent"] == "RR"
        assert rows[1]["agent"] == "PF"

    def test_reproducibility_same_seed(self):
        """Duas execuções com mesma seed devem dar métricas idênticas."""
        from grams_env.agents.baselines import RoundRobinAgent
        from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

        env = OpenRAN_RBA_Env(num_ues=5, max_steps=50)
        agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=5)

        runner = EpisodeRunner(env, agent, agent_name="RR")
        r1 = runner.run(seed=123, max_steps=50)
        r2 = runner.run(seed=123, max_steps=50)

        assert r1.spectral_efficiency_mean_bps_hz == pytest.approx(
            r2.spectral_efficiency_mean_bps_hz, rel=1e-10,
        )
        assert r1.queue_delay_p95_ms == pytest.approx(
            r2.queue_delay_p95_ms, rel=1e-10,
        )
        assert r1.total_reward == pytest.approx(
            r2.total_reward, rel=1e-10,
        )
