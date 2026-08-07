"""PPO Trainer vetorizado com AsyncVectorEnv.

Este módulo mantém a mesma lógica PPO do `PPOTrainer` original, mas troca a
coleta sequencial em um único ambiente por coleta paralela em vários ambientes
Gymnasium usando `gymnasium.vector.AsyncVectorEnv`.

Motivação
---------
O ambiente `OpenRAN_RBA_Env` é NumPy/Gymnasium e roda na CPU. Em CPU/HPC, o
maior gargalo costuma ser `env.step()`, não a rede neural. Em vez de fazer:

    1 ambiente × 2048 steps sequenciais

este trainer permite fazer, por exemplo:

    32 ambientes × 64 steps por ambiente = 2048 transições PPO

Assim, a simulação de rádio/mobilidade/tráfego é distribuída entre processos e
usa melhor nós com muitos núcleos.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from grams_env.agents.common.ppo_trainer import PPOConfig
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env


def make_openran_env(
    num_ues: int,
    seed: int,
    max_steps: int = 3600,
) -> Any:
    """Cria uma factory picklable para uso no AsyncVectorEnv.

    `AsyncVectorEnv` executa ambientes em subprocessos. Por isso, ele recebe uma
    lista de funções que constroem ambientes. Cada função abaixo cria um
    `OpenRAN_RBA_Env` independente e faz um reset inicial com seed própria.

    Parameters
    ----------
    num_ues : int
        Número de UEs em cada ambiente.
    seed : int
        Seed específica daquele ambiente. Use `base_seed + rank` para evitar
        trajetórias idênticas entre subprocessos.
    max_steps : int
        Tamanho máximo do episódio em TTIs.

    Returns
    -------
    Callable[[], OpenRAN_RBA_Env]
        Função sem argumentos compatível com `AsyncVectorEnv`.
    """

    def _init() -> OpenRAN_RBA_Env:
        env = OpenRAN_RBA_Env(num_ues=num_ues, max_steps=max_steps)
        env.reset(seed=seed)
        return env

    return _init


class AsyncVectorPPOTrainer:
    """Treinador PPO para múltiplos ambientes CPU em paralelo.

    Diferenças para o `PPOTrainer` original:

    1. Usa `gym.vector.AsyncVectorEnv` para executar N ambientes em subprocessos.
    2. Coleta rollout no formato temporal `(T, N)`, onde:
       - `T = steps_per_env`;
       - `N = num_envs`.
    3. Calcula GAE separadamente por ambiente, preservando a continuidade
       temporal de cada subprocesso.
    4. Depois achata `(T, N)` para `T×N` amostras e aplica o mesmo update PPO.

    Observação importante: a política ainda é avaliada no processo principal.
    Isso é intencional e simples. A paralelização está no gargalo mais caro para
    CPU: `env.step()`.
    """

    def __init__(
        self,
        policy: nn.Module,
        num_ues: int,
        num_envs: int = 32,
        config: PPOConfig | None = None,
        save_dir: Path | str = "runs",
        seed: int = 42,
        max_steps: int = 3600,
    ) -> None:
        if num_envs < 2:
            raise ValueError(
                "AsyncVectorPPOTrainer foi projetado para num_envs >= 2. "
                "Use PPOTrainer para um único ambiente."
            )

        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        self.policy = policy.to(self.device)
        self.num_ues = num_ues
        self.num_envs = num_envs
        self.seed = seed
        self.max_steps = max_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Divide o rollout total entre os ambientes.
        # Ex.: rollout_steps=2048 e num_envs=32 → steps_per_env=64.
        if self.config.rollout_steps % self.num_envs != 0:
            print(
                "⚠️ rollout_steps não é divisível por num_envs; "
                "o trainer usará ceil(rollout_steps / num_envs) e coletará "
                "um pouco mais de transições."
            )
        self.steps_per_env = int(np.ceil(self.config.rollout_steps / self.num_envs))
        self.total_rollout_samples = self.steps_per_env * self.num_envs

        env_fns = [
            make_openran_env(num_ues, seed + rank, max_steps)
            for rank in range(self.num_envs)
        ]
        self.envs = gym.vector.AsyncVectorEnv(env_fns)

        self.num_rbs = int(self.envs.single_action_space.shape[0])
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.lr, eps=1e-5,
        )

        # Estado vetorizado atual. Para Dict observations, Gymnasium retorna:
        #   obs["node_features"]    -> (N, V, 3)
        #   obs["adjacency_matrix"] -> (N, V, V)
        self._obs: dict[str, np.ndarray] | None = None

        # Recompensa acumulada por ambiente, usada só para logging.
        self._episode_rewards_running = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_rewards: list[float] = []

        self._log_path = self.save_dir / "training_log.csv"

        # Arrays do rollout atual. São preenchidos em `_collect_rollout()`.
        self._obs_node_features: list[np.ndarray] = []
        self._obs_adj_matrix: list[np.ndarray] = []
        self._actions: np.ndarray | None = None
        self._old_log_probs: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self._dones: np.ndarray | None = None
        self._advantages: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    def train(self) -> None:
        """Executa o loop PPO completo com ambientes vetorizados."""
        self._obs, _ = self.envs.reset(seed=self.seed)

        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "mean_reward", "policy_loss",
                "value_loss", "entropy", "elapsed_s", "num_envs",
                "steps_per_env", "total_rollout_samples",
            ])

        try:
            for iteration in range(1, self.config.total_iterations + 1):
                t0 = time.time()

                self._collect_rollout()
                metrics = self._update_policy()

                elapsed = time.time() - t0
                mean_reward = (
                    float(np.mean(self._episode_rewards[-10:]))
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
                        f"envs={self.num_envs:>2d}  "
                        f"steps/env={self.steps_per_env:>4d}  "
                        f"time={elapsed:>5.2f}s"
                    )

                    with open(self._log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            iteration, mean_reward,
                            metrics["policy_loss"], metrics["value_loss"],
                            metrics["entropy"], elapsed, self.num_envs,
                            self.steps_per_env, self.total_rollout_samples,
                        ])

                if iteration % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(iteration)

            self._save_checkpoint("final")
            print(f"\n✅ Treinamento vetorizado concluído. Modelo salvo em {self.save_dir}")
        finally:
            # Garante que os subprocessos do AsyncVectorEnv sejam encerrados
            # mesmo se o treinamento for interrompido por erro ou timeout.
            self.envs.close()

    def _collect_rollout(self) -> None:
        """Coleta transições `(T, N)` usando ambientes em paralelo.

        A cada timestep, o processo principal calcula uma ação para cada uma das
        N observações atuais. Em seguida, `envs.step(actions)` dispara os N
        ambientes em subprocessos e retorna batches de rewards/dones/obs.
        """
        assert self._obs is not None, "Chame envs.reset() antes do rollout."
        obs = self._obs

        self.policy.eval()
        self._obs_node_features = []
        self._obs_adj_matrix = []

        actions = np.zeros(
            (self.steps_per_env, self.num_envs, self.num_rbs), dtype=np.int64,
        )
        old_log_probs = np.zeros(
            (self.steps_per_env, self.num_envs), dtype=np.float32,
        )
        values = np.zeros(
            (self.steps_per_env, self.num_envs), dtype=np.float32,
        )
        rewards = np.zeros(
            (self.steps_per_env, self.num_envs), dtype=np.float32,
        )
        dones = np.zeros(
            (self.steps_per_env, self.num_envs), dtype=np.float32,
        )

        with torch.no_grad():
            for t in range(self.steps_per_env):
                # Guarda observações antes do step. Cada índice t contém um
                # batch de N grafos: (N,V,3) e (N,V,V).
                self._obs_node_features.append(obs["node_features"].copy())
                self._obs_adj_matrix.append(obs["adjacency_matrix"].copy())

                action_batch, log_prob_batch, value_batch = self._act_batch(obs)

                next_obs, reward_batch, terminated, truncated, _ = self.envs.step(
                    action_batch,
                )
                done_batch = np.logical_or(terminated, truncated)

                actions[t] = action_batch
                old_log_probs[t] = log_prob_batch
                values[t] = value_batch
                rewards[t] = reward_batch.astype(np.float32)
                dones[t] = done_batch.astype(np.float32)

                self._episode_rewards_running += reward_batch
                for env_idx, done in enumerate(done_batch):
                    if done:
                        self._episode_rewards.append(
                            float(self._episode_rewards_running[env_idx]),
                        )
                        self._episode_rewards_running[env_idx] = 0.0

                # AsyncVectorEnv usa autoreset para ambientes finalizados nas
                # versões modernas do Gymnasium; `next_obs` já é a observação
                # correta para o próximo vector step.
                obs = next_obs

            _, _, last_values = self._act_batch(obs)

        self._obs = obs

        self._actions = actions
        self._old_log_probs = old_log_probs
        self._values = values
        self._rewards = rewards
        self._dones = dones
        self._compute_gae(last_values.astype(np.float32))

    def _act_batch(
        self,
        obs_batch: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula ações para N observações.

        O modelo atual (`GNNActorCritic`) processa uma observação por vez. Este
        método centraliza esse loop e produz um batch de ações compatível com
        `AsyncVectorEnv.step()`.

        Returns
        -------
        actions : np.ndarray
            Array `(N, K)`, onde K é o número de RBs.
        log_probs : np.ndarray
            Log-probabilidade total da ação de cada ambiente, `(N,)`.
        values : np.ndarray
            Valor estimado pelo critic para cada ambiente, `(N,)`.
        """
        action_batch = np.zeros((self.num_envs, self.num_rbs), dtype=np.int64)
        log_prob_batch = np.zeros(self.num_envs, dtype=np.float32)
        value_batch = np.zeros(self.num_envs, dtype=np.float32)

        for env_idx in range(self.num_envs):
            obs_i = {
                "node_features": obs_batch["node_features"][env_idx],
                "adjacency_matrix": obs_batch["adjacency_matrix"][env_idx],
            }
            action, log_prob, value = self.policy.act(obs_i)
            action_batch[env_idx] = action
            log_prob_batch[env_idx] = log_prob
            value_batch[env_idx] = value

        return action_batch, log_prob_batch, value_batch

    def _compute_gae(self, last_values: np.ndarray) -> None:
        """Calcula GAE-λ independentemente para cada ambiente.

        Os arrays têm formato `(T, N)`. O cálculo reverso preserva a estrutura
        temporal de cada ambiente, evitando misturar a transição final de um
        ambiente com a transição inicial de outro.
        """
        assert self._rewards is not None
        assert self._values is not None
        assert self._dones is not None

        advantages = np.zeros_like(self._rewards, dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.steps_per_env)):
            if t == self.steps_per_env - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self._dones[t]
            else:
                next_values = self._values[t + 1]
                next_non_terminal = 1.0 - self._dones[t]

            delta = (
                self._rewards[t]
                + self.config.gamma * next_values * next_non_terminal
                - self._values[t]
            )
            last_gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_non_terminal
                * last_gae
            )
            advantages[t] = last_gae

        self._advantages = advantages
        self._returns = advantages + self._values

    def _update_policy(self) -> dict[str, float]:
        """Atualiza a política PPO usando mini-batches achatados `T×N`."""
        assert self._actions is not None
        assert self._old_log_probs is not None
        assert self._advantages is not None
        assert self._returns is not None

        self.policy.train()

        total_pi_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        total_samples = self.total_rollout_samples
        flat_actions = self._actions.reshape(total_samples, self.num_rbs)
        flat_old_log_probs = self._old_log_probs.reshape(total_samples)
        flat_advantages = self._advantages.reshape(total_samples)
        flat_returns = self._returns.reshape(total_samples)

        # Achata as observações de [(T vezes) arrays (N,V,*)] para lista de
        # T×N grafos. Mantemos lista porque a interface atual do GNNActorCritic
        # já espera lista de tensores por amostra.
        flat_node_features = []
        flat_adj_matrix = []
        for t in range(self.steps_per_env):
            for env_idx in range(self.num_envs):
                flat_node_features.append(self._obs_node_features[t][env_idx])
                flat_adj_matrix.append(self._obs_adj_matrix[t][env_idx])

        for _ in range(self.config.epochs):
            indices = np.random.permutation(total_samples)

            for start in range(0, total_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, total_samples)
                batch_idx = indices[start:end]

                obs_node_features = [
                    torch.as_tensor(
                        flat_node_features[i],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for i in batch_idx
                ]
                obs_adj_matrix = [
                    torch.as_tensor(
                        flat_adj_matrix[i],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for i in batch_idx
                ]
                batch_actions = torch.as_tensor(
                    flat_actions[batch_idx], dtype=torch.long, device=self.device,
                )
                old_log_probs = torch.as_tensor(
                    flat_old_log_probs[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantages = torch.as_tensor(
                    flat_advantages[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                )
                returns = torch.as_tensor(
                    flat_returns[batch_idx], dtype=torch.float32, device=self.device,
                )

                log_probs, values, entropy = self.policy.evaluate(
                    obs_node_features,
                    obs_adj_matrix,
                    batch_actions,
                )

                if len(advantages) > 1:
                    advantages = (
                        (advantages - advantages.mean())
                        / (advantages.std() + 1e-8)
                    )

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
                value_loss = nn.functional.mse_loss(values, returns)
                entropy_mean = entropy.mean()

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
        """Salva checkpoint do modelo e do otimizador."""
        path = self.save_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_envs": self.num_envs,
            "steps_per_env": self.steps_per_env,
            "total_rollout_samples": self.total_rollout_samples,
        }, path)
