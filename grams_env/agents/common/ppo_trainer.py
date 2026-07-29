"""PPO Trainer genérico — agnóstico ao tipo de rede (GNN ou MLP).

Implementa o loop de treinamento PPO (Proximal Policy Optimization):
    1. Coleta rollout de N steps no ambiente.
    2. Calcula GAE (Generalized Advantage Estimation).
    3. Atualiza a política por K épocas com mini-batches.

O trainer recebe qualquer ActorCritic que implemente a interface
`act(obs)` e `evaluate(obs, action)`.
"""

from __future__ import annotations

import csv
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from grams_env.agents.common.rollout_buffer import RolloutBuffer


# ====================================================================== #
#  Interface ActorCritic                                                   #
# ====================================================================== #

class ActorCriticProtocol(Protocol):
    """Interface que qualquer ActorCritic (GNN ou MLP) deve implementar."""

    def act(
        self,
        obs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, float]:
        """Seleciona uma ação dado o estado atual.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observação do ambiente.

        Returns
        -------
        action : np.ndarray
            Ação selecionada (K,).
        log_prob : float
            Log-probabilidade total da ação.
        value : float
            Estimativa de valor do estado.
        """
        ...

    def evaluate(
        self,
        obs_node_features: list[torch.Tensor],
        obs_adj_matrix: list[torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Avalia ações dadas para um batch de observações.

        Parameters
        ----------
        obs_node_features : list[torch.Tensor]
            Lista de tensores (V_i, 3) — um por amostra do batch.
        obs_adj_matrix : list[torch.Tensor]
            Lista de tensores (V_i, V_i) — um por amostra do batch.
        actions : torch.Tensor
            Ações do batch (B, K).

        Returns
        -------
        log_probs : torch.Tensor (B,)
        values : torch.Tensor (B,)
        entropy : torch.Tensor (B,)
        """
        ...

    def parameters(self) -> Any:
        """Retorna os parâmetros treináveis."""
        ...


# ====================================================================== #
#  Configuração PPO                                                        #
# ====================================================================== #

@dataclass
class PPOConfig:
    """Hiperparâmetros do algoritmo PPO.

    Defaults seguem os valores padrão da literatura
    (Schulman et al. 2017, CleanRL, SB3).
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    rollout_steps: int = 2048
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_iterations: int = 500
    checkpoint_interval: int = 50
    log_interval: int = 1
    device: str = "cpu"


# ====================================================================== #
#  PPO Trainer                                                             #
# ====================================================================== #

class PPOTrainer:
    """Treinador PPO genérico para qualquer ActorCritic.

    Encapsula o loop completo de treinamento:
        collect_rollout → compute_gae → update_policy × epochs

    Parameters
    ----------
    policy : nn.Module
        Modelo ActorCritic (deve implementar act/evaluate).
    env : gym.Env
        Ambiente Gymnasium.
    config : PPOConfig
        Hiperparâmetros.
    save_dir : Path | str
        Diretório para salvar checkpoints e logs.
    """

    def __init__(
        self,
        policy: nn.Module,
        env: gym.Env,
        config: PPOConfig | None = None,
        save_dir: Path | str = "runs",
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        self.policy = policy.to(self.device)
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.lr, eps=1e-5,
        )

        self.buffer = RolloutBuffer(
            rollout_steps=self.config.rollout_steps,
            num_rbs=env.action_space.shape[0],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.device,
        )

        # Estado do ambiente
        self._obs: dict[str, np.ndarray] | None = None
        self._episode_reward: float = 0.0
        self._episode_rewards: list[float] = []

        # Logging
        self._log_path = self.save_dir / "training_log.csv"

    def train(self) -> None:
        """Executa o loop de treinamento completo."""
        self._obs, _ = self.env.reset()

        # Inicializa CSV de log
        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "mean_reward", "policy_loss",
                "value_loss", "entropy", "elapsed_s",
            ])

        for iteration in range(1, self.config.total_iterations + 1):
            t0 = time.time()

            # 1. Coleta rollout
            self._collect_rollout()

            # 2. Atualiza política
            metrics = self._update_policy()

            elapsed = time.time() - t0

            # 3. Logging
            mean_reward = (
                np.mean(self._episode_rewards[-10:])
                if self._episode_rewards
                else 0.0
            )

            if iteration % self.config.log_interval == 0:
                print(
                    f"[Iter {iteration:>4d}]  "
                    f"reward={mean_reward:>10.2f}  "
                    f"pi_loss={metrics['policy_loss']:>8.4f}  "
                    f"vf_loss={metrics['value_loss']:>8.4f}  "
                    f"entropy={metrics['entropy']:>6.4f}  "
                    f"time={elapsed:>5.2f}s"
                )

                with open(self._log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        iteration, mean_reward,
                        metrics["policy_loss"], metrics["value_loss"],
                        metrics["entropy"], elapsed,
                    ])

            # 4. Checkpoint
            if iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(iteration)

        # Salva modelo final
        self._save_checkpoint("final")
        print(f"\n✅ Treinamento concluído. Modelo salvo em {self.save_dir}")

    def _collect_rollout(self) -> None:
        """Coleta rollout_steps transições no ambiente."""
        self.buffer.reset()
        self.policy.eval()

        with torch.no_grad():
            for _ in range(self.config.rollout_steps):
                action, log_prob, value = self.policy.act(self._obs)

                next_obs, reward, terminated, truncated, info = (
                    self.env.step(action)
                )
                done = terminated or truncated

                self.buffer.add(
                    obs=self._obs,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )

                self._episode_reward += reward

                if done:
                    self._episode_rewards.append(self._episode_reward)
                    self._episode_reward = 0.0
                    self._obs, _ = self.env.reset()
                else:
                    self._obs = next_obs

            # Bootstrap value para o último estado
            _, _, last_value = self.policy.act(self._obs)

        self.buffer.compute_gae(last_value)

    def _update_policy(self) -> dict[str, float]:
        """Atualiza a política por K épocas com mini-batches.

        Returns
        -------
        dict[str, float]
            Métricas agregadas: policy_loss, value_loss, entropy.
        """
        self.policy.train()

        total_pi_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.epochs):
            batches = self.buffer.get_batches(self.config.batch_size)

            for batch in batches:
                log_probs, values, entropy = self.policy.evaluate(
                    batch["obs_node_features"],
                    batch["obs_adj_matrix"],
                    batch["actions"],
                )

                advantages = batch["advantages"]
                # Normaliza advantages
                if len(advantages) > 1:
                    advantages = (
                        (advantages - advantages.mean())
                        / (advantages.std() + 1e-8)
                    )

                returns = batch["returns"]
                old_log_probs = batch["old_log_probs"]

                # ---- PPO Clipped Loss ----
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_eps,
                        1.0 + self.config.clip_eps,
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value Loss ----
                value_loss = nn.functional.mse_loss(values, returns)

                # ---- Entropy bonus ----
                entropy_mean = entropy.mean()

                # ---- Total loss ----
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_pi_loss += policy_loss.item()
                total_vf_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                num_updates += 1

        return {
            "policy_loss": total_pi_loss / max(num_updates, 1),
            "value_loss": total_vf_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def _save_checkpoint(self, tag: int | str) -> None:
        """Salva checkpoint do modelo.

        Parameters
        ----------
        tag : int | str
            Identificador do checkpoint (iteração ou 'final').
        """
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
