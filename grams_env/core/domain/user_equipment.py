"""Entidade User Equipment — estado e perfil de cada UE."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserEquipment:
    """Representa um UE com estado mutável a cada TTI.

    Atributos puramente descritivos — sem lógica de framework.
    """

    ue_id: int
    x: float = 0.0                       # posição X (metros)
    y: float = 0.0                       # posição Y (metros)
    speed_m_s: float = 0.0               # velocidade (m/s)
    direction_rad: float = 0.0           # direção (radianos)
    cbr_bytes: float = 1000.0            # carga CBR por TTI
    queue_bits: float = 0.0              # fila pendente (bits)
    direct_gain: float = 0.0             # ganho direto linear
    is_los: bool = False                 # estado LOS/NLOS
    shadowing_db: float = 0.0            # shadowing (fixo por episódio)

    @property
    def distance_to_gnb(self) -> float:
        """Distância 2D do UE ao gNB (gNB na origem)."""
        return (self.x**2 + self.y**2) ** 0.5
