"""Wrapper Gymnasium — camada fina de infraestrutura.

OpenRAN_RBA_Env — Ambiente Gymnasium para Resource Block Allocation em redes 6G/Open RAN.

Simula a camada MAC de um gNodeB que aloca K=50 Resource Blocks a V User
Equipments (UEs) em uma célula Urban Macro (UMa). O modelo de propagação
segue o 3GPP TR 36.873 (Path Loss UMa + Log-normal Shadowing + Rayleigh
Fading). As observações são estruturadas como grafos (node_features +
adjacency_matrix) para consumo direto por Graph Neural Networks.

Autor  : GRAMS Lab
Licença: MIT
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from grams_env.adapters.graph_builder import GraphBuilder
from grams_env.adapters.reward import ThroughputQueueReward
from grams_env.core.domain.cell import CellConfig
from grams_env.core.domain.network_state import NetworkState
from grams_env.core.services.channel import ChannelGainCalculator
from grams_env.core.services.link_budget import LinkBudget
from grams_env.core.services.mobility import MobilityModel
from grams_env.core.services.propagation import TR36873_UMa
from grams_env.core.services.traffic import CBRTrafficGenerator


class OpenRAN_RBA_Env(gym.Env):
    """Wrapper Gymnasium fino — toda lógica delegada aos serviços.

    Esta classe NÃO contém equações de Shannon, path loss,
    ou qualquer regra de telecom. Apenas:
      1. Define action_space / observation_space (Gymnasium).
      2. Orquestra os serviços no step().
      3. Converte resultados para o formato Gymnasium.

    Observation (grafo):
        node_features    — (V, 3): [CQI, queue_bits, cbr_load_bytes] por UE.
        adjacency_matrix — (V, V): ganho do canal de interferência entre UEs.

    Action:
        Array de tamanho K=50 (num_rbs). Cada posição indica o ID do UE
        (0..V-1) que receberá aquele RB.

    Reward:
        Soma do throughput real de todos os UEs menos penalidade proporcional
        ao tamanho agregado das filas (incentiva baixa latência).
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        num_ues: int = 10,
        config: CellConfig | None = None,
    ) -> None:
        """Inicializa o ambiente OpenRAN_RBA_Env.

        Parameters
        ----------
        num_ues : int
            Número de User Equipments (V).
        config : CellConfig | None
            Configuração da célula. Se None, usa os defaults 3GPP.
        """
        super().__init__()

        self.num_ues = num_ues
        self.config = config or CellConfig()

        # Injeção de dependência dos serviços de domínio
        propagation = TR36873_UMa(self.config)
        self._channel = ChannelGainCalculator(self.config, propagation)
        self._link = LinkBudget(self.config)
        self._mobility = MobilityModel(self.config)
        self._traffic = CBRTrafficGenerator(self.config)
        self._graph = GraphBuilder()
        self._reward_fn = ThroughputQueueReward(
            weight=self.config.queue_penalty_weight,
        )

        # Espaços Gymnasium (ÚNICO local com dependência de gymnasium)
        self.num_rbs = self.config.num_rbs
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_rbs, num_ues, dtype=np.int64)
        )
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                0.0, np.inf, (num_ues, 3), np.float32
            ),
            "adjacency_matrix": spaces.Box(
                0.0, np.inf, (num_ues, num_ues), np.float32
            ),
        })

        # Estado interno (inicializado em reset)
        self._positions: np.ndarray | None = None         # (V, 2) metros
        self._speeds: np.ndarray | None = None             # (V,) m/s
        self._directions: np.ndarray | None = None         # (V,) radianos
        self._cbr_bytes: np.ndarray | None = None          # (V,) bytes/TTI
        self._queues: np.ndarray | None = None             # (V,) bits
        self._direct_gain: np.ndarray | None = None        # (V,) linear
        self._interference_gain: np.ndarray | None = None  # (V,V) linear
        self._shadowing_direct: np.ndarray | None = None   # (V,) dB
        self._shadowing_inter: np.ndarray | None = None    # (V,V) dB
        self._is_los: np.ndarray | None = None             # (V,) bool
        self._step_count: int = 0

    # ================================================================== #
    #  RESET                                                               #
    # ================================================================== #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reinicia o ambiente para um novo episódio.

        1. Posiciona V UEs uniformemente no disco da célula.
        2. Atribui mobilidade (0 / 3 / 20 km/h) e perfil CBR (1000 / 4000 B).
        3. Gera shadowing e calcula ganhos de canal iniciais (TR 36.873).
        4. Inicializa todas as filas em zero.

        Returns
        -------
        observation : dict[str, np.ndarray]
            Observação inicial com node_features e adjacency_matrix.
        info : dict[str, Any]
            Metadados auxiliares do episódio.
        """
        super().reset(seed=seed)
        rng = self.np_random
        self._step_count = 0

        # 1. Posicionar UEs uniformemente em disco
        self._positions = self._mobility.init_positions(self.num_ues, rng)

        # 2. Mobilidade
        self._speeds = self._mobility.init_speeds(self.num_ues, rng)
        self._directions = self._mobility.init_directions(self.num_ues, rng)

        # 3. Perfil de tráfego CBR
        self._cbr_bytes = self._traffic.init_cbr_profiles(self.num_ues, rng)

        # 4. Filas inicializadas em zero
        self._queues = np.zeros(self.num_ues, dtype=np.float64)

        # 5. Determinar LOS / NLOS por UE
        dist_2d = np.linalg.norm(self._positions, axis=1)
        self._is_los = self._channel.sample_los(dist_2d, rng)

        # 6. Shadowing (lento, fixo por episódio)
        self._shadowing_direct, self._shadowing_inter = (
            self._channel.generate_shadowing(
                self._is_los, self.num_ues, rng
            )
        )

        # 7. Ganhos de canal iniciais
        self._direct_gain, self._interference_gain = (
            self._channel.compute_gains(
                self._positions,
                self._is_los,
                self._shadowing_direct,
                self._shadowing_inter,
                rng,
            )
        )

        state = self._build_network_state()
        obs = self._graph.build(state)
        observation = self._graph.to_dict(obs)

        info: dict[str, Any] = {
            "ue_speeds_kmh": (self._speeds * 3.6).tolist(),
            "ue_cbr_bytes": self._cbr_bytes.tolist(),
            "ue_is_los": self._is_los.tolist(),
        }
        return observation, info

    # ================================================================== #
    #  STEP                                                                #
    # ================================================================== #
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Executa um TTI (1 ms) do ambiente.

        Sequência cronológica por TTI:
            1. Mobilidade — atualizar posições dos UEs.
            2. Canal — recalcular ganhos (path loss + fading).
            3. Tráfego — inserir pacotes CBR nas filas.
            4. SINR — calcular com limiar de 14.8 dB.
            5. Capacidade — Shannon para SINR ≥ limiar; 0 caso contrário.
            6. Filas — subtrair throughput real.
            7. Recompensa — throughput total − penalidade de fila.

        Parameters
        ----------
        action : np.ndarray
            Array (K,) indicando qual UE recebe cada RB.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        assert self._queues is not None, "Chame reset() antes de step()."
        action = np.asarray(action, dtype=np.int64)
        rng = self.np_random
        self._step_count += 1

        # ---- 1. Mobilidade ----
        self._positions, self._directions = self._mobility.update(
            self._positions, self._speeds, self._directions, rng
        )

        # ---- 2. Recalcular canal de rádio (TR 36.873) ----
        self._direct_gain, self._interference_gain = (
            self._channel.compute_gains(
                self._positions,
                self._is_los,
                self._shadowing_direct,
                self._shadowing_inter,
                rng,
            )
        )

        # ---- 3. Chegada de tráfego (CBR determinístico ou Poisson) ----
        self._queues = self._traffic.generate(
            self._queues, self._cbr_bytes, rng
        )

        # ---- 4. Cálculo de SINR com interferência co-canal ----
        rbs_per_ue = np.bincount(
            action, minlength=self.num_ues
        ).astype(np.float64)  # (V,)
        active_mask = rbs_per_ue > 0

        sinr_per_ue = self._link.compute_sinr(
            self._direct_gain, active_mask
        )  # (V,)

        # ---- 5. Capacidade com limiar de SINR ----
        total_capacity = self._link.shannon_capacity_bits(
            sinr_per_ue, rbs_per_ue
        )  # (V,)

        # ---- 6. Atualização de filas ----
        real_throughput = np.minimum(total_capacity, self._queues)
        self._queues = np.maximum(0.0, self._queues - real_throughput)

        # ---- 7. Recompensa ----
        reward = self._reward_fn.compute(real_throughput, self._queues)

        # ---- Observação e info ----
        state = self._build_network_state()
        obs = self._graph.build(state)
        observation = self._graph.to_dict(obs)

        sinr_above = sinr_per_ue >= self.config.sinr_threshold_linear
        num_active = int(np.sum(rbs_per_ue > 0))
        num_failed = int(np.sum((~sinr_above) & (rbs_per_ue > 0)))

        info: dict[str, Any] = {
            "total_throughput_bits": float(np.sum(real_throughput)),
            "mean_sinr_db": float(
                10 * np.log10(np.mean(sinr_per_ue) + 1e-20)
            ),
            "total_queue_bits": float(np.sum(self._queues)),
            "num_active_ues": num_active,
            "num_failed_sinr": num_failed,
            "step": self._step_count,
        }
        return observation, reward, False, False, info

    # ================================================================== #
    #  MÉTODOS INTERNOS                                                    #
    # ================================================================== #

    def _build_network_state(self) -> NetworkState:
        """Constrói o snapshot NetworkState a partir do estado interno."""
        return NetworkState(
            positions=self._positions,
            speeds=self._speeds,
            directions=self._directions,
            queues=self._queues,
            cbr_bytes=self._cbr_bytes,
            direct_gains=self._direct_gain,
            interference_gains=self._interference_gain,
            is_los=self._is_los,
        )

    # ================================================================== #
    #  RENDER                                                              #
    # ================================================================== #

    def render(self) -> None:
        """Renderização textual simples do estado atual."""
        if self._queues is None:
            print("[OpenRAN_RBA_Env] Ambiente não inicializado. Chame reset().")
            return
        print(
            f"[TTI {self._step_count:>5d}]  "
            f"Queue total: {np.sum(self._queues):>12.0f} bits  |  "
            f"Queue média: {np.mean(self._queues):>10.0f} bits  |  "
            f"Queue máx: {np.max(self._queues):>10.0f} bits"
        )


# ====================================================================== #
#  VALIDAÇÃO RÁPIDA — Random Agent                                        #
# ====================================================================== #
if __name__ == "__main__":
    NUM_STEPS = 10
    NUM_UES = 10

    env = OpenRAN_RBA_Env(num_ues=NUM_UES)
    obs, info = env.reset(seed=42)

    print("=" * 78)
    print("  OpenRAN_RBA_Env — Validação com Random Agent (3GPP TR 36.873)")
    print("=" * 78)
    print(f"  UEs             : {NUM_UES}")
    print(f"  RBs             : {env.num_rbs}")
    print(f"  Bandwidth       : {env.config.bandwidth_mhz} MHz")
    print(f"  Frequency       : {env.config.carrier_freq_ghz} GHz")
    print(f"  gNB Tx Power    : {env.config.gnb_tx_power_dbm} dBm")
    print(f"  SINR Threshold  : {env.config.sinr_threshold_db} dB")
    print(f"  Action Space    : {env.action_space}")
    print(f"  Observation     : {env.observation_space}")
    print(f"  node_features   : {obs['node_features'].shape}")
    print(f"  adjacency_matrix: {obs['adjacency_matrix'].shape}")
    print("-" * 78)
    print(f"  Speeds (km/h)   : {info['ue_speeds_kmh']}")
    print(f"  CBR (bytes)     : {info['ue_cbr_bytes']}")
    print(f"  LOS             : {info['ue_is_los']}")
    print("=" * 78)

    cumulative_reward = 0.0

    for step in range(1, NUM_STEPS + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        print(
            f"  Step {step:>3d}  |  "
            f"Reward: {reward:>12.2f}  |  "
            f"Throughput: {info['total_throughput_bits']:>12.0f} bits  |  "
            f"SINR médio: {info['mean_sinr_db']:>7.2f} dB  |  "
            f"Fila: {info['total_queue_bits']:>12.0f} bits  |  "
            f"SINR falhas: {info['num_failed_sinr']}"
        )

        if terminated or truncated:
            break

    print("=" * 78)
    print(f"  Recompensa acumulada ({NUM_STEPS} steps): {cumulative_reward:.2f}")
    print("  ✅ Nenhum erro de dimensionalidade detectado.")
    print("=" * 78)
