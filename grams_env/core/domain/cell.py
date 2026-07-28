"""Configuração da célula — parâmetros físicos 3GPP puros."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellConfig:
    """Parâmetros físicos da célula, imutáveis por episódio.

    Todos os valores seguem 3GPP TR 36.873, Table 7.2-1.
    Sem dependências de numpy, gymnasium ou qualquer framework.
    """

    carrier_freq_ghz: float = 2.0
    bandwidth_mhz: float = 10.0
    num_rbs: int = 50
    rb_bandwidth_hz: float = 180_000.0
    gnb_tx_power_dbm: float = 46.0
    ue_tx_power_dbm: float = 23.0
    noise_floor_dbm_per_hz: float = -174.0
    cell_radius_m: float = 500.0
    min_distance_m: float = 35.0
    gnb_height_m: float = 25.0
    ue_height_m: float = 1.5
    street_width_m: float = 20.0
    building_height_m: float = 20.0
    tti_s: float = 1e-3
    sinr_threshold_db: float = 14.8

    # Shadowing (desvio padrão em dB)
    shadowing_std_los_db: float = 4.0
    shadowing_std_nlos_db: float = 6.0

    # Perfis de mobilidade em km/h
    mobility_speeds_kmh: tuple[float, ...] = (0.0, 3.0, 20.0)

    # Perfis de tráfego CBR em bytes por TTI
    cbr_profiles_bytes: tuple[int, ...] = (1000, 4000)

    # Penalidade de fila na recompensa
    queue_penalty_weight: float = 1e-4

    @property
    def gnb_tx_power_w(self) -> float:
        """Potência de transmissão do gNB em Watts."""
        return 10 ** ((self.gnb_tx_power_dbm - 30) / 10)

    @property
    def noise_w(self) -> float:
        """Potência de ruído por RB em Watts."""
        n0 = 10 ** ((self.noise_floor_dbm_per_hz - 30) / 10)
        return n0 * self.rb_bandwidth_hz

    @property
    def sinr_threshold_linear(self) -> float:
        """Limiar SINR em escala linear."""
        return 10 ** (self.sinr_threshold_db / 10)

    @property
    def delta_h(self) -> float:
        """Diferença de alturas gNB − UE para cálculo de d_3D."""
        return self.gnb_height_m - self.ue_height_m
