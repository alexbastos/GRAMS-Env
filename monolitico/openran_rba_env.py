"""
OpenRAN_RBA_Env — Ambiente Gymnasium para Resource Block Allocation em redes 6G/Open RAN.

Simula a camada MAC de um gNodeB que aloca K=50 Resource Blocks a V User
Equipments (UEs) em uma célula Urban Macro (UMa). O modelo de propagação
segue o 3GPP TR 36.873 (Path Loss UMa + Log-normal Shadowing + Rayleigh
Fading). As observações são estruturadas como grafos (node_features +
adjacency_matrix) para consumo direto por Graph Neural Networks.

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OpenRAN_RBA_Env(gym.Env):
    """Ambiente de alocação de Resource Blocks para redes 6G/Open RAN.

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

    # ================================================================== #
    #  Constantes físicas — 3GPP TR 36.873 / UMa                          #
    # ================================================================== #
    BANDWIDTH_MHZ: float = 10.0             # largura de banda do sistema
    NUM_RBS: int = 50                       # 10 MHz → 50 RBs
    CARRIER_FREQ_GHZ: float = 2.0           # frequência portadora
    GNB_TX_POWER_DBM: float = 46.0          # potência Tx do gNodeB
    UE_TX_POWER_DBM: float = 23.0           # potência Tx do UE (referência)
    RB_BANDWIDTH_HZ: float = 180_000.0      # largura de banda de 1 RB
    TTI_S: float = 1e-3                     # duração de 1 TTI (1 ms)

    # Geometria da célula e alturas (3GPP TR 36.873 Table 7.2-1)
    CELL_RADIUS_M: float = 500.0            # raio da célula UMa
    MIN_DISTANCE_M: float = 35.0            # distância mínima UE–gNB
    GNB_HEIGHT_M: float = 25.0              # altura do gNB (h_BS)
    UE_HEIGHT_M: float = 1.5                # altura do UE (h_UT)
    STREET_WIDTH_M: float = 20.0            # largura da rua (W)
    BUILDING_HEIGHT_M: float = 20.0         # altura média de edifícios (h)

    # Limiar de SINR para decodificação
    SINR_THRESHOLD_DB: float = 14.8
    SINR_THRESHOLD_LINEAR: float = 10 ** (14.8 / 10)  # ≈ 30.20

    # Ruído
    NOISE_FLOOR_DBM_PER_HZ: float = -174.0  # densidade de ruído térmico

    # Shadowing (desvio padrão em dB)
    SHADOWING_STD_LOS_DB: float = 4.0       # σ_SF LOS (TR 36.873)
    SHADOWING_STD_NLOS_DB: float = 6.0      # σ_SF NLOS (TR 36.873)

    # Penalidade de fila na recompensa
    QUEUE_PENALTY_WEIGHT: float = 1e-4

    # Perfis de mobilidade em km/h
    MOBILITY_SPEEDS_KMH: list[float] = [0.0, 3.0, 20.0]

    # Perfis de tráfego CBR em bytes por TTI
    CBR_PROFILES_BYTES: list[int] = [1000, 4000]

    def __init__(
        self,
        num_ues: int = 10,
        seed: int | None = None,
    ) -> None:
        """Inicializa o ambiente OpenRAN_RBA_Env.

        Parameters
        ----------
        num_ues : int
            Número de User Equipments (V).
        seed : int | None
            Semente para reprodutibilidade do gerador aleatório.
        """
        super().__init__()

        self.num_ues = num_ues
        self.num_rbs = self.NUM_RBS

        # Potência linear do gNB (watts)
        self.gnb_tx_power_w: float = 10 ** (
            (self.GNB_TX_POWER_DBM - 30) / 10
        )

        # Ruído total linear (watts) por RB: N = N0 (W/Hz) × BW_RB (Hz)
        noise_density_w: float = 10 ** (
            (self.NOISE_FLOOR_DBM_PER_HZ - 30) / 10
        )
        self.noise_w: float = noise_density_w * self.RB_BANDWIDTH_HZ

        # Diferença de alturas para d_3D (constante)
        self._delta_h: float = self.GNB_HEIGHT_M - self.UE_HEIGHT_M

        # ----- Espaços Gymnasium ----------------------------------------
        self.action_space = spaces.MultiDiscrete(
            nvec=np.full(self.num_rbs, self.num_ues, dtype=np.int64),
        )

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.num_ues, 3),
                    dtype=np.float32,
                ),
                "adjacency_matrix": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.num_ues, self.num_ues),
                    dtype=np.float32,
                ),
            }
        )

        # ----- Estado interno (inicializado em reset) --------------------
        self._ue_positions: np.ndarray | None = None       # (V, 2) metros
        self._ue_speeds: np.ndarray | None = None           # (V,) m/s
        self._ue_directions: np.ndarray | None = None       # (V,) radianos
        self._ue_cbr_bytes: np.ndarray | None = None        # (V,) bytes/TTI
        self._queues: np.ndarray | None = None              # (V,) bits
        self._direct_gain: np.ndarray | None = None         # (V,) linear
        self._interference_gain: np.ndarray | None = None   # (V,V) linear
        self._shadowing_direct: np.ndarray | None = None    # (V,) dB
        self._shadowing_inter: np.ndarray | None = None     # (V,V) dB
        self._is_los: np.ndarray | None = None              # (V,) bool
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

        # 1. Posicionar UEs uniformemente em disco -------------------
        angles = rng.uniform(0, 2 * np.pi, size=self.num_ues)
        radii = self.CELL_RADIUS_M * np.sqrt(
            rng.uniform(0, 1, size=self.num_ues)
        )
        radii = np.clip(radii, self.MIN_DISTANCE_M, self.CELL_RADIUS_M)
        self._ue_positions = np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )

        # 2. Mobilidade -----------------------------------------------
        speed_ms = np.array(
            [s / 3.6 for s in self.MOBILITY_SPEEDS_KMH], dtype=np.float64
        )
        speed_indices = rng.integers(0, len(speed_ms), size=self.num_ues)
        self._ue_speeds = speed_ms[speed_indices]
        self._ue_directions = rng.uniform(0, 2 * np.pi, size=self.num_ues)

        # 3. Perfil de tráfego CBR ------------------------------------
        cbr_arr = np.array(self.CBR_PROFILES_BYTES, dtype=np.float64)
        cbr_indices = rng.integers(0, len(cbr_arr), size=self.num_ues)
        self._ue_cbr_bytes = cbr_arr[cbr_indices]

        # 4. Filas inicializadas em zero ------------------------------
        self._queues = np.zeros(self.num_ues, dtype=np.float64)

        # 5. Determinar LOS / NLOS por UE -----------------------------
        dist_2d = np.linalg.norm(self._ue_positions, axis=1)
        self._is_los = self._sample_los(dist_2d, rng)

        # 6. Shadowing (lento, fixo por episódio) ---------------------
        self._shadowing_direct = np.where(
            self._is_los,
            rng.normal(0, self.SHADOWING_STD_LOS_DB, self.num_ues),
            rng.normal(0, self.SHADOWING_STD_NLOS_DB, self.num_ues),
        )
        self._shadowing_inter = rng.normal(
            0, self.SHADOWING_STD_NLOS_DB, (self.num_ues, self.num_ues)
        )
        np.fill_diagonal(self._shadowing_inter, 0.0)

        # 7. Ganhos de canal iniciais ----------------------------------
        self._update_channel_gains(rng)

        observation = self._build_observation()
        info: dict[str, Any] = {
            "ue_speeds_kmh": (self._ue_speeds * 3.6).tolist(),
            "ue_cbr_bytes": self._ue_cbr_bytes.tolist(),
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

        # ---- 1. Mobilidade ---------------------------------------------
        self._update_mobility(rng)

        # ---- 2. Recalcular canal de rádio (TR 36.873) ------------------
        self._update_channel_gains(rng)

        # ---- 3. Chegada de tráfego CBR ---------------------------------
        self._generate_traffic()

        # ---- 4. Cálculo de SINR com interferência co-canal -------------
        sinr_per_ue = self._compute_sinr(action)               # (V,)

        # ---- 5. Capacidade com limiar de SINR --------------------------
        sinr_above = sinr_per_ue >= self.SINR_THRESHOLD_LINEAR  # (V,) bool

        # Shannon: C = B · log₂(1 + SINR) · TTI  [bits por RB por TTI]
        capacity_per_rb = np.where(
            sinr_above,
            self.RB_BANDWIDTH_HZ * np.log2(1.0 + sinr_per_ue) * self.TTI_S,
            0.0,
        )                                                       # (V,)

        # Multiplicar pela quantidade de RBs atribuídos a cada UE
        rbs_per_ue = np.bincount(
            action, minlength=self.num_ues
        ).astype(np.float64)                                    # (V,)
        total_capacity = capacity_per_rb * rbs_per_ue           # (V,)

        # ---- 6. Atualização de filas -----------------------------------
        real_throughput = np.minimum(total_capacity, self._queues)
        self._queues = np.maximum(0.0, self._queues - real_throughput)

        # ---- 7. Recompensa ---------------------------------------------
        total_throughput = float(np.sum(real_throughput))
        queue_penalty = float(np.sum(self._queues)) * self.QUEUE_PENALTY_WEIGHT
        reward = total_throughput - queue_penalty

        # ---- Observação e info -----------------------------------------
        observation = self._build_observation()
        num_active = int(np.sum(rbs_per_ue > 0))
        num_failed = int(np.sum((~sinr_above) & (rbs_per_ue > 0)))

        info: dict[str, Any] = {
            "total_throughput_bits": total_throughput,
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
    #  MÉTODOS AUXILIARES — MOBILIDADE                                     #
    # ================================================================== #

    def _update_mobility(self, rng: np.random.Generator) -> None:
        """Atualiza a posição 2D dos UEs com base na velocidade e direção.

        UEs que ultrapassam o raio da célula são refletidos para dentro.
        A direção sofre uma leve perturbação angular a cada TTI para
        simular um random walk suave.

        Parameters
        ----------
        rng : np.random.Generator
            Gerador de números aleatórios.
        """
        # Perturbação angular (desvio padrão de ±15°)
        angle_perturbation = rng.normal(0, np.radians(15), size=self.num_ues)
        self._ue_directions += angle_perturbation

        # Deslocamento: Δx = v · cos(θ) · TTI,  Δy = v · sin(θ) · TTI
        dx = self._ue_speeds * np.cos(self._ue_directions) * self.TTI_S
        dy = self._ue_speeds * np.sin(self._ue_directions) * self.TTI_S
        self._ue_positions[:, 0] += dx
        self._ue_positions[:, 1] += dy

        # Reflexão: se o UE saiu do raio da célula, refletir a direção
        distances = np.linalg.norm(self._ue_positions, axis=1)
        out_of_cell = distances > self.CELL_RADIUS_M

        if np.any(out_of_cell):
            # Reposicionar no limite e inverter a direção
            scale = self.CELL_RADIUS_M / distances[out_of_cell]
            self._ue_positions[out_of_cell] *= scale[:, np.newaxis]
            self._ue_directions[out_of_cell] += np.pi  # inversão de 180°

        # Garantir distância mínima do gNB
        distances = np.linalg.norm(self._ue_positions, axis=1)
        too_close = distances < self.MIN_DISTANCE_M
        if np.any(too_close):
            scale = self.MIN_DISTANCE_M / (distances[too_close] + 1e-10)
            self._ue_positions[too_close] *= scale[:, np.newaxis]

    # ================================================================== #
    #  MÉTODOS AUXILIARES — MODELO DE PROPAGAÇÃO (3GPP TR 36.873)          #
    # ================================================================== #

    @staticmethod
    def _los_probability(d_2d: np.ndarray) -> np.ndarray:
        """Calcula a probabilidade de Line-of-Sight (3GPP TR 36.873, Tab 7.2-2).

        P_LOS(d) = min(18/d, 1) · (1 − exp(−d/63)) + exp(−d/63)

        Parameters
        ----------
        d_2d : np.ndarray
            Distância 2D em metros.

        Returns
        -------
        np.ndarray
            Probabilidade LOS (mesmo shape).
        """
        d = np.clip(d_2d, 1.0, None)
        return np.minimum(18.0 / d, 1.0) * (1 - np.exp(-d / 63)) + np.exp(
            -d / 63
        )

    def _sample_los(
        self, d_2d: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Amostra o estado LOS/NLOS de cada UE conforme a probabilidade.

        Parameters
        ----------
        d_2d : np.ndarray
            Distância 2D de cada UE ao gNB.
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        np.ndarray
            Array booleano (V,); True = LOS.
        """
        p_los = self._los_probability(d_2d)
        return rng.random(size=d_2d.shape) < p_los

    def _calculate_pathloss_direct(
        self, d_2d: np.ndarray
    ) -> np.ndarray:
        """Calcula o path loss UE–gNB (3GPP TR 36.873, Tab 7.2-1, 3D-UMa).

        LOS:
            PL = 22.0·log₁₀(d_3D) + 28.0 + 20·log₁₀(f_c)

        NLOS:
            PL = 161.04 − 7.1·log₁₀(W) + 7.5·log₁₀(h)
                 − (24.37 − 3.7·(h/h_BS)²)·log₁₀(h_BS)
                 + (43.42 − 3.1·log₁₀(h_BS))·(log₁₀(d_3D) − 3)
                 + 20·log₁₀(f_c)
                 − (3.2·(log₁₀(17.625))² − 4.97)
                 − 0.6·(h_UT − 1.5)

        Parameters
        ----------
        d_2d : np.ndarray
            Distância 2D (V,) de cada UE ao gNB em metros.

        Returns
        -------
        np.ndarray
            Path loss em dB (V,).
        """
        d_2d_safe = np.clip(d_2d, 10.0, None)
        d_3d = np.sqrt(d_2d_safe**2 + self._delta_h**2)

        f_c = self.CARRIER_FREQ_GHZ
        h_bs = self.GNB_HEIGHT_M
        h_ut = self.UE_HEIGHT_M
        w = self.STREET_WIDTH_M
        h = self.BUILDING_HEIGHT_M

        # LOS path loss
        pl_los = (
            22.0 * np.log10(d_3d)
            + 28.0
            + 20.0 * np.log10(f_c)
        )

        # NLOS path loss
        pl_nlos = (
            161.04
            - 7.1 * np.log10(w)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * (h / h_bs) ** 2) * np.log10(h_bs)
            + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d_3d) - 3)
            + 20.0 * np.log10(f_c)
            - (3.2 * (np.log10(17.625)) ** 2 - 4.97)
            - 0.6 * (h_ut - 1.5)
        )

        # Selecionar com base no estado LOS/NLOS
        pl_db = np.where(self._is_los, pl_los, np.maximum(pl_los, pl_nlos))
        return pl_db

    def _calculate_pathloss_inter_ue(
        self, dist_matrix: np.ndarray
    ) -> np.ndarray:
        """Calcula o path loss entre pares de UEs (3GPP UMa NLOS simplificado).

        Como UEs estão à mesma altura, usa-se a fórmula NLOS com d_3D ≈ d_2D
        (Δh ≈ 0). Isto modela a atenuação do sinal de interferência
        percebida entre UEs.

        Parameters
        ----------
        dist_matrix : np.ndarray
            Matriz (V, V) de distâncias 2D entre UEs.

        Returns
        -------
        np.ndarray
            Path loss em dB (V, V).
        """
        d = np.clip(dist_matrix, 1.0, None)
        f_c = self.CARRIER_FREQ_GHZ
        h_bs = self.GNB_HEIGHT_M
        w = self.STREET_WIDTH_M
        h = self.BUILDING_HEIGHT_M
        h_ut = self.UE_HEIGHT_M

        # NLOS path loss (d_3D ≈ d_2D para UEs na mesma altura)
        pl_db = (
            161.04
            - 7.1 * np.log10(w)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * (h / h_bs) ** 2) * np.log10(h_bs)
            + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d) - 3)
            + 20.0 * np.log10(f_c)
            - (3.2 * (np.log10(17.625)) ** 2 - 4.97)
            - 0.6 * (h_ut - 1.5)
        )
        return pl_db

    def _update_channel_gains(self, rng: np.random.Generator) -> None:
        """Recalcula os ganhos de canal direto e de interferência.

        Componentes:
            • Path Loss determinístico (3GPP TR 36.873 UMa)
            • Log-normal Shadowing (lento, fixo por episódio)
            • Rayleigh Fading (rápido, renovado a cada TTI)

        Parameters
        ----------
        rng : np.random.Generator
            Gerador de números aleatórios.
        """
        # --- Ganho direto (gNB → UE) -----------------------------------
        dist_direct = np.linalg.norm(self._ue_positions, axis=1)  # (V,)
        pl_direct_db = self._calculate_pathloss_direct(dist_direct)

        # Ganho total = −PL − Shadowing  (em dB)  →  linear
        total_loss_db = pl_direct_db + self._shadowing_direct
        pl_linear = 10 ** (-total_loss_db / 10)

        # Rayleigh fading: |h|² ∼ Exp(1), renovado a cada TTI
        fading_direct = rng.exponential(1.0, size=self.num_ues)
        self._direct_gain = (pl_linear * fading_direct).astype(np.float64)

        # --- Ganho de interferência (UE ↔ UE) ---------------------------
        diff = (
            self._ue_positions[:, np.newaxis, :]
            - self._ue_positions[np.newaxis, :, :]
        )                                                       # (V, V, 2)
        dist_inter = np.linalg.norm(diff, axis=2)              # (V, V)

        pl_inter_db = self._calculate_pathloss_inter_ue(dist_inter)
        total_inter_db = pl_inter_db + self._shadowing_inter
        pl_inter_linear = 10 ** (-total_inter_db / 10)

        fading_inter = rng.exponential(
            1.0, size=(self.num_ues, self.num_ues)
        )
        self._interference_gain = (
            pl_inter_linear * fading_inter
        ).astype(np.float64)
        np.fill_diagonal(self._interference_gain, 0.0)

    # ================================================================== #
    #  MÉTODOS AUXILIARES — TRÁFEGO                                        #
    # ================================================================== #

    def _generate_traffic(self) -> None:
        """Insere pacotes CBR nas filas de todos os UEs.

        Cada UE gera uma quantidade fixa de dados (1000 ou 4000 bytes)
        a cada TTI, convertida para bits.
        """
        arrival_bits = self._ue_cbr_bytes * 8.0  # bytes → bits
        self._queues += arrival_bits

    # ================================================================== #
    #  MÉTODOS AUXILIARES — SINR                                           #
    # ================================================================== #

    def _compute_sinr(self, action: np.ndarray) -> np.ndarray:
        """Calcula a SINR por UE no enlace descendente OFDMA.

        Em OFDMA de célula única, os Resource Blocks são ortogonais.
        Portanto, a SINR de cada UE depende exclusivamente do seu ganho
        direto (gNB → UE) e do ruído térmico:

            SINR_u = P_tx × g_direct[u]  /  N

        A matriz de adjacência (interferência inter-UE) é mantida como
        feature do grafo para a GNN — codifica a correlação espacial
        entre UEs que compete pela alocação de RBs.

        Se o cenário for estendido para multi-célula, a interferência
        inter-celular pode ser incorporada diretamente aqui.

        Parameters
        ----------
        action : np.ndarray
            Array (K,) indicando qual UE recebe cada RB.

        Returns
        -------
        np.ndarray
            Vetor (V,) de SINR linear por UE.
        """
        rbs_per_ue = np.bincount(action, minlength=self.num_ues)
        active_mask = rbs_per_ue > 0

        # SINR = P_tx × g_direct / N  (para UEs ativos; 0 para inativos)
        sinr = np.where(
            active_mask,
            self.gnb_tx_power_w * self._direct_gain / self.noise_w,
            0.0,
        )

        return sinr

    # ================================================================== #
    #  MÉTODOS AUXILIARES — OBSERVAÇÃO                                     #
    # ================================================================== #

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Constrói a observação no formato de grafo para a GNN.

        node_features (V, 3):
            col 0 — CQI: proxy do ganho direto em escala dB normalizada.
            col 1 — Tamanho da fila atual (bits).
            col 2 — Carga de tráfego alvo CBR (bytes: 1000 ou 4000).

        adjacency_matrix (V, V):
            Ganho de canal de interferência entre UEs (escala linear).

        Returns
        -------
        dict[str, np.ndarray]
        """
        # CQI: ganho direto → dB + offset para garantir valores positivos
        cqi = 10 * np.log10(self._direct_gain + 1e-20) + 200

        node_features = np.column_stack(
            [
                cqi,                    # canal direto (proxy CQI)
                self._queues,           # fila atual (bits)
                self._ue_cbr_bytes,     # perfil CBR (1000 ou 4000 bytes)
            ]
        ).astype(np.float32)

        adjacency_matrix = self._interference_gain.astype(np.float32)

        return {
            "node_features": node_features,
            "adjacency_matrix": adjacency_matrix,
        }

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

    env = OpenRAN_RBA_Env(num_ues=NUM_UES, seed=42)
    obs, info = env.reset(seed=42)

    print("=" * 78)
    print("  OpenRAN_RBA_Env — Validação com Random Agent (3GPP TR 36.873)")
    print("=" * 78)
    print(f"  UEs             : {NUM_UES}")
    print(f"  RBs             : {env.num_rbs}")
    print(f"  Bandwidth       : {env.BANDWIDTH_MHZ} MHz")
    print(f"  Frequency       : {env.CARRIER_FREQ_GHZ} GHz")
    print(f"  gNB Tx Power    : {env.GNB_TX_POWER_DBM} dBm")
    print(f"  SINR Threshold  : {env.SINR_THRESHOLD_DB} dB")
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
