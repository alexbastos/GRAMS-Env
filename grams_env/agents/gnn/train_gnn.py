"""Script de treinamento — Agente GNN+PPO para GRAMS-Env.

Treina o agente GNNActorCritic com PPO no cenário esparso (V=20 UEs)
e exporta o modelo congelado para avaliação zero-shot.

Uso:
    python -m grams_env.agents.gnn.train_gnn [--num_ues 20] [--iterations 500]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grams_env.agents.common.ppo_trainer import PPOConfig, PPOTrainer
from grams_env.agents.common.ppo_vector_trainer import AsyncVectorPPOTrainer
from grams_env.agents.common.utils import set_seed
from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina agente GNN+PPO no GRAMS-Env.",
    )
    parser.add_argument(
        "--num_ues", type=int, default=20,
        help="Número de UEs para treinamento (default: 20).",
    )
    parser.add_argument(
        "--iterations", type=int, default=500,
        help="Número de iterações PPO (default: 500).",
    )
    parser.add_argument(
        "--rollout_steps", type=int, default=2048,
        help="Steps por rollout (default: 2048).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente aleatória (default: 42).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Dispositivo: 'cpu' ou 'cuda' (default: cpu).",
    )
    parser.add_argument(
        "--save_dir", type=str, default="runs/gnn_ppo",
        help="Diretório para checkpoints e logs.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help=(
            "Número de ambientes paralelos para AsyncVectorEnv. "
            "Use 1 para o PPOTrainer sequencial original (default: 1)."
        ),
    )
    parser.add_argument(
        "--torch_threads", type=int, default=None,
        help=(
            "Número de threads intra-op do PyTorch em CPU. "
            "Útil em HPC para evitar oversubscription."
        ),
    )
    args = parser.parse_args()

    # Reproducibilidade
    set_seed(args.seed)

    # Em treinamento CPU com muitos processos de ambiente, limitar threads do
    # PyTorch evita oversubscription (ex.: 32 envs × BLAS multithread).
    if args.torch_threads is not None:
        import torch
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    # Ambiente
    env = OpenRAN_RBA_Env(num_ues=args.num_ues)
    print("=" * 70)
    print("  GRAMS-Env — Treinamento GNN+PPO")
    print("=" * 70)
    print(f"  UEs           : {args.num_ues}")
    print(f"  RBs           : {env.num_rbs}")
    print(f"  Iterações     : {args.iterations}")
    print(f"  Rollout Steps : {args.rollout_steps}")
    print(f"  Num Envs      : {args.num_envs}")
    if args.num_envs > 1:
        steps_per_env = -(-args.rollout_steps // args.num_envs)
        print(f"  Steps/Env     : {steps_per_env}")
        print(f"  Samples/Roll. : {steps_per_env * args.num_envs}")
    if args.torch_threads is not None:
        print(f"  Torch Threads : {args.torch_threads}")
    print(f"  Seed          : {args.seed}")
    print(f"  Device        : {args.device}")
    print(f"  Save Dir      : {args.save_dir}")
    print("=" * 70)

    # Modelo
    policy = GNNActorCritic(
        in_features=3,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_rbs=env.num_rbs,
        device=args.device,
    )

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parâmetros    : {total_params:,}")
    print("=" * 70)

    # Config PPO
    config = PPOConfig(
        lr=args.lr,
        total_iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        device=args.device,
    )

    # Trainer
    # - num_envs == 1: mantém o comportamento original, com um único ambiente.
    # - num_envs > 1 : usa AsyncVectorEnv para paralelizar env.step() na CPU.
    if args.num_envs == 1:
        trainer = PPOTrainer(
            policy=policy,
            env=env,
            config=config,
            save_dir=Path(args.save_dir),
        )
    else:
        env.close()
        trainer = AsyncVectorPPOTrainer(
            policy=policy,
            num_ues=args.num_ues,
            num_envs=args.num_envs,
            config=config,
            save_dir=Path(args.save_dir),
            seed=args.seed,
        )

    # Treina
    trainer.train()

    # Exporta modelo congelado
    import torch
    frozen_path = Path(args.save_dir) / "model_gnn_frozen.pt"
    torch.save(policy.state_dict(), frozen_path)
    print(f"\n🧊 Modelo congelado salvo em: {frozen_path}")


if __name__ == "__main__":
    main()
