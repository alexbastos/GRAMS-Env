"""Gerador de tráfego — CBR determinístico e Poisson estocástico."""

from __future__ import annotations

import numpy as np

from grams_env.core.domain.cell import CellConfig


class CBRTrafficGenerator:
    """Serviço de geração de tráfego com dois modos de operação.

    Modo 'deterministic' (CBR puro):
        Cada UE gera exatamente cbr_bytes bytes por TTI.
        Adequado para análise baseline e comparação controlada.

    Modo 'poisson' (estocástico):
        O número de bytes gerados por UE por TTI segue uma distribuição
        de Poisson com taxa média λ_v = cbr_bytes. Isso modela rajadas
        de tráfego realistas conforme descrito na especificação do artigo.

        arrival_bytes_v ~ Poisson(λ_v),  onde λ_v ∈ {1000, 4000}

    O modo é definido pelo campo traffic_mode do CellConfig.
    """

    def __init__(self, config: CellConfig) -> None:
        self._cfg = config

    @property
    def mode(self) -> str:
        """Retorna o modo de tráfego configurado."""
        return self._cfg.traffic_mode

    def init_cbr_profiles(
        self,
        num_ues: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Atribui perfis de tráfego CBR aleatórios aos UEs.

        Cada UE recebe um dos perfis definidos em CellConfig.cbr_profiles_bytes.
        No modo Poisson, o valor atribuído serve como a taxa média λ_v
        do processo estocástico.

        Parameters
        ----------
        num_ues : int
            Número de UEs.
        rng : np.random.Generator
            Gerador de números aleatórios.

        Returns
        -------
        np.ndarray
            Array (V,) com bytes/TTI (ou λ_v) por UE.
        """
        cbr_arr = np.array(self._cfg.cbr_profiles_bytes, dtype=np.float64)
        cbr_indices = rng.integers(0, len(cbr_arr), size=num_ues)
        return cbr_arr[cbr_indices]

    def generate(
        self,
        queues: np.ndarray,
        cbr_bytes: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Insere tráfego nas filas de todos os UEs.

        Despacha automaticamente para o modo configurado (determinístico
        ou Poisson).

        Parameters
        ----------
        queues : np.ndarray
            Filas atuais (V,) em bits (será modificado in-place).
        cbr_bytes : np.ndarray
            Carga CBR por UE (V,) em bytes/TTI. No modo Poisson,
            este valor é a taxa média λ_v.
        rng : np.random.Generator | None
            Gerador de números aleatórios (obrigatório no modo Poisson).

        Returns
        -------
        np.ndarray
            Filas atualizadas (V,) em bits.
        """
        if self._cfg.traffic_mode == "poisson":
            return self._generate_poisson(queues, cbr_bytes, rng)
        return self._generate_deterministic(queues, cbr_bytes)

    @staticmethod
    def _generate_deterministic(
        queues: np.ndarray,
        cbr_bytes: np.ndarray,
    ) -> np.ndarray:
        """Tráfego CBR determinístico: exatamente cbr_bytes por TTI.

        Parameters
        ----------
        queues : np.ndarray
            Filas atuais (V,) em bits.
        cbr_bytes : np.ndarray
            Carga fixa por UE (V,) em bytes/TTI.

        Returns
        -------
        np.ndarray
            Filas atualizadas (V,) em bits.
        """
        arrival_bits = cbr_bytes * 8.0  # bytes → bits
        queues += arrival_bits
        return queues

    @staticmethod
    def _generate_poisson(
        queues: np.ndarray,
        cbr_bytes: np.ndarray,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        """Tráfego Poisson estocástico: arrival ~ Poisson(λ_v).

        O número de bytes que chega a cada UE por TTI é amostrado de
        uma distribuição de Poisson com taxa média λ_v = cbr_bytes[v].
        Isso gera rajadas variáveis ao redor da carga nominal,
        modelando realismo no tráfego de rede.

        Para perfis de alto volume (cbr_bytes = 4000), a distribuição
        produz rajadas mais intensas, enquanto perfis de baixo volume
        (cbr_bytes = 1000) geram tráfego mais suave.

        Parameters
        ----------
        queues : np.ndarray
            Filas atuais (V,) em bits.
        cbr_bytes : np.ndarray
            Taxa média λ_v por UE (V,) em bytes/TTI.
        rng : np.random.Generator | None
            Gerador de números aleatórios.

        Returns
        -------
        np.ndarray
            Filas atualizadas (V,) em bits.

        Raises
        ------
        ValueError
            Se rng não for fornecido no modo Poisson.
        """
        if rng is None:
            raise ValueError(
                "rng é obrigatório no modo 'poisson'. "
                "Passe o np_random do gymnasium."
            )
        # Amostrar bytes chegados via Poisson(λ_v)
        arrival_bytes = rng.poisson(lam=cbr_bytes).astype(np.float64)
        arrival_bits = arrival_bytes * 8.0  # bytes → bits
        queues += arrival_bits
        return queues
