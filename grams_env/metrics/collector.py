"""Coletor de métricas por TTI — acumula dados durante um episódio.

Calcula as 3 métricas definidas no artigo (Tabela II):

1. **Eficiência Espectral do Sistema** (bps/Hz):
    SE(t) = throughput_total(t) / (bandwidth_hz × tti_s)
    Resultado: média temporal sobre todos os TTIs.

2. **Atraso de Fila no Percentil 95** (ms):
    Estimado via Lei de Little:
        W_v(t) = queue_bits_v(t) / λ_v
    onde λ_v = cbr_bytes_v × 8 / tti_s [bits/s].
    Resultado: percentil 95 sobre todos os UEs e TTIs.

3. **Latência de Inferência do RIC** (ms):
    Tempo de parede (wall-clock) de agent.act(obs).
    Resultado: média temporal sobre todos os TTIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ====================================================================== #
#  Resultado do Episódio                                                   #
# ====================================================================== #

@dataclass
class EpisodeResult:
    """Resultado completo de um episódio com as 3 métricas do artigo.

    Contém os valores agregados para a Tabela II, séries temporais
    para geração de gráficos e metadados do experimento.

    Attributes
    ----------
    spectral_efficiency_mean_bps_hz : float
        Eficiência espectral média do sistema (bps/Hz).
    queue_delay_p95_ms : float
        Atraso de fila no percentil 95 (ms), sobre todos os UEs e TTIs.
    ric_latency_mean_ms : float
        Latência média de inferência do RIC (ms).
    total_reward : float
        Recompensa acumulada no episódio.
    mean_throughput_bits_per_tti : float
        Throughput médio por TTI (bits).
    mean_queue_bits : float
        Tamanho médio total das filas (bits).
    num_steps : int
        Número de TTIs executados.
    spectral_efficiency_ts : np.ndarray
        Série temporal de SE (T,) em bps/Hz.
    queue_delay_per_ue_ts : np.ndarray
        Atraso de fila por UE por TTI (T, V) em ms.
    inference_latency_ts : np.ndarray
        Latência de inferência por TTI (T,) em ms.
    throughput_ts : np.ndarray
        Throughput total por TTI (T,) em bits.
    reward_ts : np.ndarray
        Recompensa por TTI (T,).
    queue_total_ts : np.ndarray
        Fila total por TTI (T,) em bits.
    config : dict[str, Any]
        Metadados do experimento (agent_name, num_ues, seed, etc.).
    """

    # --- Métricas do artigo (Tabela II) ---
    spectral_efficiency_mean_bps_hz: float
    queue_delay_p95_ms: float
    ric_latency_mean_ms: float

    # --- Agregados adicionais ---
    total_reward: float
    mean_throughput_bits_per_tti: float
    mean_queue_bits: float
    num_steps: int

    # --- Séries temporais ---
    spectral_efficiency_ts: np.ndarray     # (T,) bps/Hz
    queue_delay_per_ue_ts: np.ndarray      # (T, V) ms
    inference_latency_ts: np.ndarray       # (T,) ms
    throughput_ts: np.ndarray              # (T,) bits
    reward_ts: np.ndarray                  # (T,)
    queue_total_ts: np.ndarray             # (T,) bits

    # --- Metadados ---
    config: dict[str, Any] = field(default_factory=dict)


# ====================================================================== #
#  Coletor de Métricas                                                     #
# ====================================================================== #

class MetricsCollector:
    """Acumula dados por TTI e calcula as métricas agregadas.

    Uso típico::

        collector = MetricsCollector(bandwidth_hz=10e6, tti_s=1e-3)
        for step in range(T):
            t0 = time.perf_counter()
            action, _, _ = agent.act(obs)
            latency = time.perf_counter() - t0
            obs, reward, _, _, info = env.step(action)
            collector.record_step(obs, info, reward, latency)
        result = collector.compute(config_metadata)

    Parameters
    ----------
    bandwidth_hz : float
        Largura de banda total do sistema em Hz (10 MHz → 10e6).
    tti_s : float
        Duração do TTI em segundos (1 ms → 1e-3).
    """

    def __init__(self, bandwidth_hz: float, tti_s: float) -> None:
        self._bw_hz = bandwidth_hz
        self._tti_s = tti_s
        self.reset()

    def reset(self) -> None:
        """Limpa todos os dados acumulados para reutilização."""
        self._se_list: list[float] = []
        self._delay_list: list[np.ndarray] = []
        self._latency_list: list[float] = []
        self._throughput_list: list[float] = []
        self._reward_list: list[float] = []
        self._queue_total_list: list[float] = []

    @property
    def num_steps(self) -> int:
        """Número de TTIs registrados até o momento."""
        return len(self._se_list)

    def record_step(
        self,
        obs: dict[str, np.ndarray],
        info: dict[str, Any],
        reward: float,
        inference_time_s: float,
    ) -> None:
        """Registra os dados de um TTI.

        Extrai informações do observation (per-UE) e do info (agregados)
        para calcular métricas instantâneas.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação pós-step com 'node_features' (V, 3):
                col 0 — CQI (dB normalizada).
                col 1 — queue_bits por UE.
                col 2 — cbr_bytes por UE.
        info : dict[str, Any]
            Dicionário info retornado pelo env.step().
        reward : float
            Recompensa escalar do TTI.
        inference_time_s : float
            Tempo de inferência do agente em segundos (wall-clock).
        """
        node_features = obs["node_features"]
        queue_bits = node_features[:, 1].astype(np.float64)
        cbr_bytes = node_features[:, 2].astype(np.float64)

        # 1. Eficiência Espectral: SE = throughput / (BW × TTI)
        throughput = info["total_throughput_bits"]
        se = throughput / (self._bw_hz * self._tti_s)
        self._se_list.append(se)

        # 2. Atraso de fila por UE — Lei de Little:
        #    W = L / λ = queue_bits / (cbr_bytes × 8 / tti_s) [s]
        #    W_ms = queue_bits × tti_s × 1000 / (cbr_bytes × 8)
        arrival_rate_bps = cbr_bytes * 8.0 / self._tti_s  # (V,) bits/s
        safe_mask = arrival_rate_bps > 0
        delay_s = np.zeros_like(queue_bits)
        np.divide(
            queue_bits, arrival_rate_bps,
            out=delay_s, where=safe_mask,
        )
        delay_ms = delay_s * 1000.0  # (V,) ms
        self._delay_list.append(delay_ms)

        # 3. Latência de inferência do RIC
        self._latency_list.append(inference_time_s * 1000.0)

        # Métricas auxiliares
        self._throughput_list.append(throughput)
        self._reward_list.append(reward)
        self._queue_total_list.append(info["total_queue_bits"])

    def compute(
        self,
        config: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        """Calcula as métricas agregadas e retorna o resultado completo.

        Parameters
        ----------
        config : dict[str, Any] | None
            Metadados do experimento (agent_name, num_ues, seed, etc.).
            Incluídos no EpisodeResult para rastreabilidade.

        Returns
        -------
        EpisodeResult
            Resultado com as 3 métricas do artigo, séries temporais
            e metadados.

        Raises
        ------
        ValueError
            Se nenhum step foi registrado via record_step().
        """
        if not self._se_list:
            raise ValueError(
                "Nenhum step registrado. Chame record_step() primeiro."
            )

        # Séries temporais
        se_ts = np.array(self._se_list, dtype=np.float64)
        delay_ts = np.array(self._delay_list)  # (T, V) float64
        latency_ts = np.array(self._latency_list, dtype=np.float64)
        throughput_ts = np.array(self._throughput_list, dtype=np.float64)
        reward_ts = np.array(self._reward_list, dtype=np.float64)
        queue_total_ts = np.array(self._queue_total_list, dtype=np.float64)

        # P95 do atraso sobre TODOS os UEs × TODOS os TTIs
        all_delays = delay_ts.ravel()
        p95 = float(np.percentile(all_delays, 95))

        return EpisodeResult(
            spectral_efficiency_mean_bps_hz=float(np.mean(se_ts)),
            queue_delay_p95_ms=p95,
            ric_latency_mean_ms=float(np.mean(latency_ts)),
            total_reward=float(np.sum(reward_ts)),
            mean_throughput_bits_per_tti=float(np.mean(throughput_ts)),
            mean_queue_bits=float(np.mean(queue_total_ts)),
            num_steps=len(self._se_list),
            spectral_efficiency_ts=se_ts,
            queue_delay_per_ue_ts=delay_ts,
            inference_latency_ts=latency_ts,
            throughput_ts=throughput_ts,
            reward_ts=reward_ts,
            queue_total_ts=queue_total_ts,
            config=config or {},
        )
