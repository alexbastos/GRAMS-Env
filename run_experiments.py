"""Pipeline Fatorial Completo — Baselines + GNN Zero-Shot + MLP.

Executa a matriz fatorial definida no artigo:
    - 6 densidades de UEs: 1, 10, 50, 100, 150, 200
    - 3 perfis de mobilidade: 0, 3, 20 km/h
    - 2 níveis de tráfego CBR: 1000, 4000 Bytes
    - 2 frequências portadoras: 0.7 GHz, 2.0 GHz
    = 72 ambientes distintos.

Agentes avaliados:
    - RoundRobin (baseline clássica)
    - ProportionalFair (baseline clássica)
    - GNN+PPO (zero-shot: treinado com V=20, avaliado em qualquer V)
    - MLP+PPO (treinado com V=20, avalia APENAS com V=20 — falha nos demais)

Cada configuração é executada com 10 sementes aleatórias para
validação estatística.

Uso:
    # Apenas baselines clássicas (sem PyTorch)
    python run_experiments.py --agents baselines --workers 4

    # Todos os agentes (requer PyTorch + modelo treinado)
    python run_experiments.py --agents all --workers 4

    # Apenas agentes DRL (GNN + MLP)
    python run_experiments.py --agents drl --workers 1

    # Apenas GNN zero-shot
    python run_experiments.py --agents gnn --workers 1
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from grams_env.agents.baselines import ProportionalFairAgent, RoundRobinAgent
from grams_env.core.domain.cell import CellConfig
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env
from grams_env.metrics.exporter import MetricsExporter
from grams_env.metrics.runner import EpisodeRunner

# ============================================================
# Configuração da Matriz Fatorial (conforme artigo)
# ============================================================
UES_LIST = [1, 10, 20, 50, 100, 150, 200]
SPEEDS_LIST = [(0.0,), (3.0,), (20.0,)]
CBR_LIST = [(1000,), (4000,)]
FREQ_LIST = [0.7, 2.0]
SEEDS = list(range(42, 52))  # 10 sementes (42 a 51)
MAX_STEPS = 3600  # 3600 TTIs = 3.6 segundos de simulação

# Caminhos de saída
RESULTS_DIR = Path("results/factorial")
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

# Caminhos dos modelos treinados
GNN_MODEL_PATH = "runs/gnn_ppo/model_gnn_frozen.pt"
MLP_MODEL_PATH = "runs/mlp_ppo/checkpoint_final.pt"

# V usado no treinamento (para verificação do MLP)
TRAIN_NUM_UES = 20


def _load_gnn_policy(device: str = "cpu"):
    """Carrega o modelo GNN treinado com V=20 (pesos congelados)."""
    import torch
    from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic

    policy = GNNActorCritic(
        in_features=3,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_rbs=50,
        device=device,
    )
    policy.load_state_dict(
        torch.load(GNN_MODEL_PATH, map_location=device, weights_only=True)
    )
    policy.eval()
    return policy


def _load_mlp_policy(num_ues: int, device: str = "cpu"):
    """Carrega o modelo MLP treinado com V=20 (pesos congelados).

    IMPORTANTE: O MLP só funciona com num_ues == TRAIN_NUM_UES.
    Para outros valores de V, retorna None (RuntimeError esperado).
    """
    import torch
    from grams_env.agents.mlp.mlp_actor_critic import MLPActorCritic

    if num_ues != TRAIN_NUM_UES:
        return None  # MLP não generaliza para V != V_treino

    policy = MLPActorCritic(
        num_ues=num_ues,
        num_rbs=50,
        device=device,
    )
    policy.load_state_dict(
        torch.load(MLP_MODEL_PATH, map_location=device, weights_only=True)
    )
    policy.eval()
    return policy


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

    # --- Baselines Clássicas ---
    if agent_name == "RoundRobin":
        agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=num_ues)

    elif agent_name == "ProportionalFair":
        agent = ProportionalFairAgent(num_rbs=env.num_rbs, num_ues=num_ues)

    # --- Agentes DRL ---
    elif agent_name == "GNN+PPO":
        agent = _load_gnn_policy()

    elif agent_name == "MLP+PPO":
        agent = _load_mlp_policy(num_ues)
        if agent is None:
            # MLP não pode inferir com V != V_treino → retorna None
            return None

    else:
        raise ValueError(f"Agente '{agent_name}' desconhecido.")

    runner = EpisodeRunner(env, agent, agent_name=agent_name)
    return runner.run(seed=seed, max_steps=max_steps)


def _build_combinations(agent_names: list[str]) -> list[tuple]:
    """Constrói a lista de combinações, filtrando MLP para V != 20."""
    combinations = []
    for agent, ues, speeds, cbr, freq, seed in itertools.product(
        agent_names, UES_LIST, SPEEDS_LIST, CBR_LIST, FREQ_LIST, SEEDS,
    ):
        # MLP só roda com o mesmo V do treinamento
        if agent == "MLP+PPO" and ues != TRAIN_NUM_UES:
            continue
        combinations.append((agent, ues, speeds, cbr, freq, seed))
    return combinations


def _check_models(agent_names: list[str]) -> None:
    """Verifica se os modelos treinados existem antes de iniciar."""
    if "GNN+PPO" in agent_names and not os.path.exists(GNN_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Modelo GNN não encontrado em '{GNN_MODEL_PATH}'.\n"
            f"   Execute ./run_training.sh primeiro para treinar o modelo."
        )
    if "MLP+PPO" in agent_names and not os.path.exists(MLP_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Modelo MLP não encontrado em '{MLP_MODEL_PATH}'.\n"
            f"   Execute ./run_training.sh primeiro para treinar o modelo."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline Fatorial Completo — GRAMS-Env"
    )
    parser.add_argument(
        "--agents",
        choices=["baselines", "drl", "gnn", "all"],
        default="all",
        help=(
            "Agentes a avaliar: "
            "'baselines' (RR+PF), "
            "'drl' (GNN+MLP), "
            "'gnn' (apenas GNN zero-shot), "
            "'all' (todos)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Número de workers no multiprocessing. "
            "Use 1 para agentes DRL (PyTorch não é fork-safe por padrão)."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=MAX_STEPS,
        help="Número de TTIs por episódio.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo PyTorch: 'cpu' ou 'cuda'.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Seleciona agentes
    if args.agents == "baselines":
        agent_names = ["RoundRobin", "ProportionalFair"]
    elif args.agents == "drl":
        agent_names = ["GNN+PPO", "MLP+PPO"]
    elif args.agents == "gnn":
        agent_names = ["GNN+PPO"]
    else:  # all
        agent_names = ["RoundRobin", "ProportionalFair", "GNN+PPO", "MLP+PPO"]

    # Verifica se os modelos existem (para agentes DRL)
    _check_models(agent_names)

    # Constrói combinações (filtra MLP para V != 20)
    combinations = _build_combinations(agent_names)
    total_runs = len(combinations)

    print("=" * 80)
    print(" 🚀 PIPELINE FATORIAL COMPLETO — GRAMS-Env")
    print("=" * 80)
    print(f" Agentes         : {agent_names}")
    print(f" Densidades (V)  : {UES_LIST}")
    print(f" Velocidades     : {[s[0] for s in SPEEDS_LIST]} km/h")
    print(f" CBR             : {[c[0] for c in CBR_LIST]} Bytes")
    print(f" Frequências     : {FREQ_LIST} GHz")
    print(f" Sementes        : {len(SEEDS)} ({SEEDS[0]} a {SEEDS[-1]})")
    print(f" Total de Runs   : {total_runs}")
    print(f" Workers (CPUs)  : {args.workers}")
    print(f" Dispositivo     : {args.device}")
    print(f" Saída CSV       : {SUMMARY_CSV}")
    print("=" * 80)

    # Garante arquivo limpo
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()

    # Executa os experimentos
    first_write = True
    skipped = 0

    if args.workers == 1:
        # Execução sequencial (segura para PyTorch)
        with tqdm(total=total_runs, desc="Simulando Episódios") as pbar:
            for agent, ues, speeds, cbr, freq, seed in combinations:
                result = run_single_episode(
                    agent, ues, speeds, cbr, freq, seed, args.steps,
                )
                if result is None:
                    skipped += 1
                    pbar.update(1)
                    continue

                MetricsExporter.to_summary_csv(
                    result, SUMMARY_CSV, append=not first_write,
                )
                first_write = False
                pbar.update(1)
    else:
        # Execução paralela (apenas para baselines sem PyTorch)
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
                    if result is None:
                        skipped += 1
                        pbar.update(1)
                        continue

                    MetricsExporter.to_summary_csv(
                        result, SUMMARY_CSV, append=not first_write,
                    )
                    first_write = False
                    pbar.update(1)

    print(f"\n✅ Experimentos finalizados com sucesso!")
    print(f"📊 Resultados salvos em: {SUMMARY_CSV}")
    if skipped > 0:
        print(f"⚠️  {skipped} runs do MLP pulados (V ≠ {TRAIN_NUM_UES}).")


if __name__ == "__main__":
    main()
