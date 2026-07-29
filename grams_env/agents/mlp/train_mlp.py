"""Script de treinamento — Agente MLP+PPO Baseline para GRAMS-Env.

Treina o baseline MLPActorCritic com PPO no cenário esparso (V=20 UEs).
Serve como comparação direta contra o agente GNN+PPO no artigo.

Uso:
    python -m grams_env.agents.mlp.train_mlp [--num_ues 20] [--iterations 500]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grams_env.agents.common.ppo_trainer import PPOConfig, PPOTrainer
from grams_env.agents.common.utils import set_seed
from grams_env.agents.mlp.mlp_actor_critic import MLPActorCritic
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina agente MLP+PPO baseline no GRAMS-Env.",
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
        "--save_dir", type=str, default="runs/mlp_ppo",
        help="Diretório para checkpoints e logs.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    args = parser.parse_args()

    # Reproducibilidade
    set_seed(args.seed)

    # Ambiente
    env = OpenRAN_RBA_Env(num_ues=args.num_ues)
    print("=" * 70)
    print("  GRAMS-Env — Treinamento MLP+PPO (Baseline)")
    print("=" * 70)
    print(f"  UEs           : {args.num_ues}")
    print(f"  RBs           : {env.num_rbs}")
    print(f"  Iterações     : {args.iterations}")
    print(f"  Rollout Steps : {args.rollout_steps}")
    print(f"  Seed          : {args.seed}")
    print(f"  Device        : {args.device}")
    print(f"  Save Dir      : {args.save_dir}")
    print("=" * 70)

    # Modelo
    policy = MLPActorCritic(
        num_ues=args.num_ues,
        num_rbs=env.num_rbs,
        hidden_dim=256,
        device=args.device,
    )

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parâmetros    : {total_params:,}")
    print(f"  Input dim     : {policy._flat_dim}")
    print("=" * 70)

    # Config PPO
    config = PPOConfig(
        lr=args.lr,
        total_iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        device=args.device,
    )

    # Trainer
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        config=config,
        save_dir=Path(args.save_dir),
    )

    # Treina
    trainer.train()

    # Exporta modelo congelado
    import torch
    frozen_path = Path(args.save_dir) / "model_mlp_frozen.pt"
    torch.save(policy.state_dict(), frozen_path)
    print(f"\n🧊 Modelo congelado salvo em: {frozen_path}")


if __name__ == "__main__":
    main()
