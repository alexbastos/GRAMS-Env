"""Runner de episódio — orquestra agent ↔ env com coleta de métricas.

Executa um episódio completo, cronometra cada chamada ``agent.act()``
e delega a acumulação de dados ao ``MetricsCollector``.

Uso::

    from grams_env.agents.baselines import RoundRobinAgent
    from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env
    from grams_env.metrics import EpisodeRunner, MetricsExporter

    env = OpenRAN_RBA_Env(num_ues=10)
    agent = RoundRobinAgent(num_rbs=50, num_ues=10)
    runner = EpisodeRunner(env, agent, agent_name="RoundRobin")

    result = runner.run(seed=42)
    print(f"SE:      {result.spectral_efficiency_mean_bps_hz:.4f} bps/Hz")
    print(f"Delay:   {result.queue_delay_p95_ms:.4f} ms")
    print(f"Latency: {result.ric_latency_mean_ms:.4f} ms")

    MetricsExporter.to_summary_csv(result, "results/summary.csv")
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from grams_env.metrics.collector import EpisodeResult, MetricsCollector


class EpisodeRunner:
    """Executa um episódio completo no ambiente com coleta de métricas.

    Orquestra a interação agent ↔ env, cronometra a inferência
    e delega a acumulação de dados ao MetricsCollector.

    Compatível com qualquer agente que implemente
    ``act(obs) -> (action, log_prob, value)``:
    baselines (RR, PF), agentes DRL (GNN+PPO, MLP+PPO) e
    agentes aleatórios.

    Parameters
    ----------
    env : gymnasium.Env
        Ambiente OpenRAN_RBA_Env.
    agent : object
        Agente com método ``act(obs) -> (action, log_prob, value)``.
        Se possuir método ``reset()``, será chamado no início.
    agent_name : str
        Nome do agente para metadados (ex: "RoundRobin", "PF", "GNN+PPO").
    """

    def __init__(
        self,
        env: Any,
        agent: Any,
        agent_name: str = "unknown",
    ) -> None:
        self._env = env
        self._agent = agent
        self._agent_name = agent_name

    def run(
        self,
        seed: int = 42,
        max_steps: int | None = None,
    ) -> EpisodeResult:
        """Executa um episódio e retorna as métricas calculadas.

        Parameters
        ----------
        seed : int
            Semente aleatória para reprodutibilidade.
        max_steps : int | None
            Máximo de TTIs a executar. Se None, usa ``env.max_steps``.

        Returns
        -------
        EpisodeResult
            Resultado com as 3 métricas do artigo, séries temporais
            e metadados do experimento.
        """
        env = self._env
        agent = self._agent
        max_steps = max_steps or getattr(env, "max_steps", 3600)

        # Metadados do experimento
        config = self._build_config(seed)

        # Inicializa coletor
        bandwidth_hz = env.config.bandwidth_mhz * 1e6
        collector = MetricsCollector(
            bandwidth_hz=bandwidth_hz,
            tti_s=env.config.tti_s,
        )

        # Reset ambiente e agente
        obs, info = env.reset(seed=seed)
        if hasattr(agent, "reset") and callable(agent.reset):
            agent.reset()

        # Loop do episódio
        for _ in range(max_steps):
            # Cronometra inferência do agente (Métrica 3: RIC Latency)
            t0 = time.perf_counter()
            action, _, _ = agent.act(obs)
            inference_time_s = time.perf_counter() - t0

            # Executa TTI no ambiente
            obs, reward, terminated, truncated, info = env.step(action)

            # Registra dados do TTI
            collector.record_step(obs, info, reward, inference_time_s)

            if terminated or truncated:
                break

        return collector.compute(config)

    def _build_config(self, seed: int) -> dict[str, Any]:
        """Constrói dicionário de metadados do experimento.

        Extrai parâmetros relevantes do CellConfig do ambiente
        para inclusão no EpisodeResult.

        Parameters
        ----------
        seed : int
            Semente aleatória utilizada.

        Returns
        -------
        dict[str, Any]
            Metadados do experimento.
        """
        cfg = self._env.config
        return {
            "agent_name": self._agent_name,
            "num_ues": self._env.num_ues,
            "num_rbs": cfg.num_rbs,
            "carrier_freq_ghz": cfg.carrier_freq_ghz,
            "bandwidth_mhz": cfg.bandwidth_mhz,
            "cbr_profile": str(cfg.cbr_profiles_bytes),
            "speed_profile": str(cfg.mobility_speeds_kmh),
            "traffic_mode": cfg.traffic_mode,
            "seed": seed,
        }


# ====================================================================== #
#  Validação Rápida — Demo com Baselines                                   #
# ====================================================================== #
if __name__ == "__main__":
    from grams_env.agents.baselines import (
        ProportionalFairAgent,
        RoundRobinAgent,
    )
    from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env
    from grams_env.metrics.exporter import MetricsExporter

    NUM_UES = 10
    MAX_STEPS = 500

    env = OpenRAN_RBA_Env(num_ues=NUM_UES, max_steps=MAX_STEPS)

    print("=" * 78)
    print("  GRAMS-Env — Pipeline de Métricas (Demo)")
    print("=" * 78)
    print(f"  UEs           : {NUM_UES}")
    print(f"  RBs           : {env.num_rbs}")
    print(f"  TTIs          : {MAX_STEPS}")
    print(f"  Bandwidth     : {env.config.bandwidth_mhz} MHz")
    print(f"  Frequência    : {env.config.carrier_freq_ghz} GHz")
    print("=" * 78)

    agents = {
        "RoundRobin": RoundRobinAgent(
            num_rbs=env.num_rbs, num_ues=NUM_UES,
        ),
        "ProportionalFair": ProportionalFairAgent(
            num_rbs=env.num_rbs, num_ues=NUM_UES,
        ),
    }

    for name, agent in agents.items():
        runner = EpisodeRunner(env, agent, agent_name=name)
        result = runner.run(seed=42, max_steps=MAX_STEPS)

        print(f"\n  📊 {name}:")
        print(f"     Eficiência Espectral : "
              f"{result.spectral_efficiency_mean_bps_hz:.4f} bps/Hz")
        print(f"     Atraso Fila P95      : "
              f"{result.queue_delay_p95_ms:.4f} ms")
        print(f"     Latência RIC (média) : "
              f"{result.ric_latency_mean_ms:.6f} ms")
        print(f"     Recompensa total     : "
              f"{result.total_reward:.2f}")
        print(f"     Throughput médio/TTI : "
              f"{result.mean_throughput_bits_per_tti:.0f} bits")
        print(f"     Fila média           : "
              f"{result.mean_queue_bits:.0f} bits")

        # Exporta CSV de resumo (append para acumular baselines)
        MetricsExporter.to_summary_csv(
            result, "results/demo_summary.csv", append=True,
        )

    # Exporta série temporal do último agente (PF)
    MetricsExporter.to_timeseries_csv(
        result, "results/demo_timeseries_pf.csv",
    )

    print("\n" + "=" * 78)
    print("  ✅ Resultados exportados para results/")
    print("     • results/demo_summary.csv      (resumo RR + PF)")
    print("     • results/demo_timeseries_pf.csv (série temporal PF)")
    print("=" * 78)
