"""Gerenciador de Sementes e Execução Fatorial.

Executa as 72 configurações base do artigo:
    - 3 variações de UEs (10, 30, 50)
    - 3 perfis de mobilidade (0, 3, 20 km/h) homogeneamente distribuídos
    - 2 níveis de tráfego CBR (1000, 4000 Bytes)
    - 2 frequências portadoras (0.7 GHz, 2.0 GHz)
    = 36 ambientes distintos.

Para os algoritmos clássicos (Round Robin, Proportional Fair), isso gera
72 configurações de teste. Cada configuração é executada com 10 sementes
aleatórias para validação estatística (totalizando 720 simulações).

Uso:
    python run_experiments.py --agents baselines --workers 4
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from grams_env.agents.baselines import ProportionalFairAgent, RoundRobinAgent
from grams_env.core.domain.cell import CellConfig
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env
from grams_env.metrics.exporter import MetricsExporter
from grams_env.metrics.runner import EpisodeRunner

# Configuração Padrão do Experimento
UES_LIST = [10, 30, 50]
SPEEDS_LIST = [(0.0,), (3.0,), (20.0,)]
CBR_LIST = [(1000,), (4000,)]
FREQ_LIST = [0.7, 2.0]
SEEDS = list(range(42, 52))  # 10 sementes (42 a 51)
MAX_STEPS = 3600  # 3600 TTIs

RESULTS_DIR = Path("results/factorial")
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


def run_single_episode(
    agent_name: str,
    num_ues: int,
    speed_profile: tuple[float, ...],
    cbr_profile: tuple[int, ...],
    freq_ghz: float,
    seed: int,
    max_steps: int,
):
    """Executa e retorna o resultado de uma única rodada."""
    config = CellConfig(
        carrier_freq_ghz=freq_ghz,
        mobility_speeds_kmh=speed_profile,
        cbr_profiles_bytes=cbr_profile,
    )
    env = OpenRAN_RBA_Env(num_ues=num_ues, config=config, max_steps=max_steps)

    if agent_name == "RoundRobin":
        agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=num_ues)
    elif agent_name == "ProportionalFair":
        agent = ProportionalFairAgent(num_rbs=env.num_rbs, num_ues=num_ues)
    else:
        raise ValueError(f"Agente {agent_name} desconhecido.")

    runner = EpisodeRunner(env, agent, agent_name=agent_name)
    return runner.run(seed=seed, max_steps=max_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execução Fatorial GRAMS-Env")
    parser.add_argument(
        "--agents",
        choices=["baselines", "all"],
        default="baselines",
        help="Agentes a serem avaliados.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Número de workers no multiprocessing.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=MAX_STEPS,
        help="Número de TTIs por episódio.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.agents == "baselines":
        agents = ["RoundRobin", "ProportionalFair"]
    else:
        agents = ["RoundRobin", "ProportionalFair"]

    combinations = list(
        itertools.product(
            agents, UES_LIST, SPEEDS_LIST, CBR_LIST, FREQ_LIST, SEEDS
        )
    )
    total_runs = len(combinations)
    print("=" * 80)
    print(" 🚀 INICIANDO PIPELINE DE EXPERIMENTOS FATORIAIS")
    print("=" * 80)
    print(f" Agentes        : {agents}")
    print(f" Configurações  : {total_runs // (10 * len(agents))} por agente")
    print(f" Sementes       : 10 ({SEEDS[0]} a {SEEDS[-1]})")
    print(f" Total Runs     : {total_runs}")
    print(f" Threads (CPUs) : {args.workers}")
    print(f" Saída CSV      : {SUMMARY_CSV}")
    print("=" * 80)

    # Garante arquivo limpo
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()

    # Submete ao pool e coleta iterativamente
    first_write = True
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_episode,
                agent, ues, speeds, cbr, freq, seed, args.steps,
            ): (agent, ues, speeds, cbr, freq, seed)
            for (agent, ues, speeds, cbr, freq, seed) in combinations
        }

        with tqdm(total=total_runs, desc="Simulando Episódios") as pbar:
            for future in as_completed(futures):
                result = future.result()
                # A primeira escrita cria o arquivo com header, as próximas fazem append
                MetricsExporter.to_summary_csv(
                    result, SUMMARY_CSV, append=not first_write
                )
                first_write = False
                pbar.update(1)

    print("\n✅ Experimentos finalizados com sucesso!")
    print(f"📊 Resultados salvos em: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
