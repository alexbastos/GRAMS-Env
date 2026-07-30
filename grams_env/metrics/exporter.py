"""Exportador de métricas — CSV (resumo + séries temporais) e HDF5.

Formatos de saída:

CSV Resumo (``to_summary_csv``):
    Uma linha por episódio contendo as 3 métricas do artigo +
    metadados do experimento. Modo append permite acumular
    resultados de múltiplas seeds/configurações em um único arquivo.

CSV Séries Temporais (``to_timeseries_csv``):
    Uma linha por TTI com métricas instantâneas para geração de
    gráficos de convergência e análise detalhada.

HDF5 (``to_hdf5``):
    Formato hierárquico com 3 grupos:
        /summary    — atributos com as 3 métricas + agregados.
        /timeseries — datasets com séries temporais (T,) e (T, V).
        /config     — atributos com parâmetros do experimento.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from grams_env.metrics.collector import EpisodeResult


class MetricsExporter:
    """Exporta EpisodeResult para CSV e HDF5.

    Todos os métodos são estáticos — sem estado interno.
    """

    # Colunas do CSV de resumo, na ordem de escrita
    SUMMARY_COLUMNS: list[str] = [
        "agent",
        "num_ues",
        "carrier_freq_ghz",
        "bandwidth_mhz",
        "cbr_profile",
        "speed_profile",
        "seed",
        "num_steps",
        "spectral_efficiency_bps_hz",
        "queue_delay_p95_ms",
        "ric_latency_mean_ms",
        "total_reward",
        "mean_throughput_bits",
        "mean_queue_bits",
    ]

    # ------------------------------------------------------------------ #
    #  CSV Resumo                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_summary_csv(
        result: EpisodeResult,
        path: str | Path,
        append: bool = False,
    ) -> None:
        """Exporta resumo do episódio como uma linha CSV.

        Se ``append=True``, adiciona ao arquivo existente sem
        reescrever o header. Isso permite acumular resultados de
        múltiplas seeds/configurações para a matriz fatorial.

        Parameters
        ----------
        result : EpisodeResult
            Resultado do episódio.
        path : str | Path
            Caminho do arquivo CSV de saída.
        append : bool
            Se True, faz append no arquivo existente.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cfg = result.config
        row = {
            "agent": cfg.get("agent_name", "unknown"),
            "num_ues": cfg.get("num_ues", ""),
            "carrier_freq_ghz": cfg.get("carrier_freq_ghz", ""),
            "bandwidth_mhz": cfg.get("bandwidth_mhz", ""),
            "cbr_profile": cfg.get("cbr_profile", ""),
            "speed_profile": cfg.get("speed_profile", ""),
            "seed": cfg.get("seed", ""),
            "num_steps": result.num_steps,
            "spectral_efficiency_bps_hz": (
                f"{result.spectral_efficiency_mean_bps_hz:.6f}"
            ),
            "queue_delay_p95_ms": (
                f"{result.queue_delay_p95_ms:.4f}"
            ),
            "ric_latency_mean_ms": (
                f"{result.ric_latency_mean_ms:.6f}"
            ),
            "total_reward": f"{result.total_reward:.2f}",
            "mean_throughput_bits": (
                f"{result.mean_throughput_bits_per_tti:.2f}"
            ),
            "mean_queue_bits": f"{result.mean_queue_bits:.2f}",
        }

        write_header = not (append and path.exists())
        mode = "a" if append else "w"

        with open(path, mode, newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=MetricsExporter.SUMMARY_COLUMNS,
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------ #
    #  CSV Séries Temporais                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_timeseries_csv(
        result: EpisodeResult,
        path: str | Path,
    ) -> None:
        """Exporta séries temporais por TTI para CSV.

        Colunas:
            step, spectral_efficiency_bps_hz, throughput_bits,
            total_queue_bits, ric_latency_ms, reward

        Parameters
        ----------
        result : EpisodeResult
            Resultado do episódio.
        path : str | Path
            Caminho do arquivo CSV de saída.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "spectral_efficiency_bps_hz",
                "throughput_bits",
                "total_queue_bits",
                "ric_latency_ms",
                "reward",
            ])
            for t in range(result.num_steps):
                writer.writerow([
                    t + 1,
                    f"{result.spectral_efficiency_ts[t]:.6f}",
                    f"{result.throughput_ts[t]:.2f}",
                    f"{result.queue_total_ts[t]:.2f}",
                    f"{result.inference_latency_ts[t]:.6f}",
                    f"{result.reward_ts[t]:.2f}",
                ])

    # ------------------------------------------------------------------ #
    #  HDF5                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_hdf5(
        result: EpisodeResult,
        path: str | Path,
    ) -> None:
        """Exporta resultado completo para HDF5.

        Estrutura hierárquica do arquivo::

            /summary     — attrs: SE, P95_delay, RIC_latency, ...
            /timeseries/ — datasets:
                spectral_efficiency  (T,)
                queue_delay_per_ue   (T, V)
                inference_latency    (T,)
                throughput           (T,)
                reward               (T,)
                queue_total          (T,)
            /config      — attrs: agent_name, num_ues, seed, ...

        Parameters
        ----------
        result : EpisodeResult
            Resultado do episódio.
        path : str | Path
            Caminho do arquivo HDF5 de saída.

        Raises
        ------
        ImportError
            Se h5py não estiver instalado.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py é necessário para exportar HDF5. "
                "Instale com: pip install h5py"
            ) from None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            # --- Summary ---
            summary = f.create_group("summary")
            summary.attrs["spectral_efficiency_mean_bps_hz"] = (
                result.spectral_efficiency_mean_bps_hz
            )
            summary.attrs["queue_delay_p95_ms"] = (
                result.queue_delay_p95_ms
            )
            summary.attrs["ric_latency_mean_ms"] = (
                result.ric_latency_mean_ms
            )
            summary.attrs["total_reward"] = result.total_reward
            summary.attrs["mean_throughput_bits_per_tti"] = (
                result.mean_throughput_bits_per_tti
            )
            summary.attrs["mean_queue_bits"] = result.mean_queue_bits
            summary.attrs["num_steps"] = result.num_steps

            # --- Timeseries ---
            ts = f.create_group("timeseries")
            ts.create_dataset(
                "spectral_efficiency",
                data=result.spectral_efficiency_ts,
            )
            ts.create_dataset(
                "queue_delay_per_ue",
                data=result.queue_delay_per_ue_ts,
            )
            ts.create_dataset(
                "inference_latency",
                data=result.inference_latency_ts,
            )
            ts.create_dataset(
                "throughput",
                data=result.throughput_ts,
            )
            ts.create_dataset(
                "reward",
                data=result.reward_ts,
            )
            ts.create_dataset(
                "queue_total",
                data=result.queue_total_ts,
            )

            # --- Config ---
            cfg_group = f.create_group("config")
            for key, value in result.config.items():
                try:
                    cfg_group.attrs[key] = value
                except TypeError:
                    cfg_group.attrs[key] = str(value)
