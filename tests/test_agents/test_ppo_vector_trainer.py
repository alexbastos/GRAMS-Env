"""Contrato do PPO vetorizado: GAE, SAME_STEP, snapshots e subprocessos."""

import csv
from functools import partial

import gymnasium as gym
import numpy as np
import pytest
import torch

from grams_env.agents.common.ppo_trainer import PPOConfig
from grams_env.agents.common.ppo_vector_trainer import (
    AsyncVectorPPOTrainer,
    compute_vector_gae,
)
from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env


class ToyGraphEnv(gym.Env):
    """Reutiliza buffers; codifica episódio, worker e passo na observação."""

    def __init__(self, rank, horizon, terminate):
        self.rank = rank
        self.horizon = horizon
        self.terminate = terminate
        self.episode = -1
        self.step_index = 0
        self.action_space = gym.spaces.MultiDiscrete([2, 2])
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(
                -np.inf, np.inf, shape=(2, 3), dtype=np.float32,
            ),
            "adjacency_matrix": gym.spaces.Box(
                0, 1, shape=(2, 2), dtype=np.float32,
            ),
        })
        self.features = np.zeros((2, 3), dtype=np.float32)
        self.adjacency = np.eye(2, dtype=np.float32)

    def _observation(self):
        self.features[:, 0] = 100 * self.episode + 10 * self.rank + self.step_index
        self.features[:, 1] = self.step_index
        self.features[:, 2] = self.rank
        return {
            "node_features": self.features,
            "adjacency_matrix": self.adjacency,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_index = 0
        return self._observation(), {}

    def step(self, action):
        expected = np.full(2, self.step_index % 2, dtype=np.int64)
        np.testing.assert_array_equal(action, expected)
        self.step_index += 1
        reward = float(10 * self.rank + self.step_index)
        end = self.step_index == self.horizon
        # A cópia final preserva o contrato Gymnasium mesmo quando reset()
        # reutiliza o buffer no mesmo step do worker.
        obs = {key: value.copy() for key, value in self._observation().items()}
        return obs, reward, end and self.terminate, end and not self.terminate, {}


class DeterministicPolicy(torch.nn.Module):
    """Critic exato e ações observáveis, sem amostragem durante bootstrap."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.act_calls = 0
        self.value_observations = []

    def act(self, obs):
        self.act_calls += 1
        step = int(obs["node_features"][0, 1])
        value = float(obs["node_features"][0, 0])
        return np.full(2, step % 2, dtype=np.int64), 0.0, value

    def get_value(self, obs):
        self.value_observations.append(obs["node_features"].copy())
        return float(obs["node_features"][0, 0])

    def evaluate(self, obs_node_features, obs_adj_matrix, actions):
        values = torch.stack([obs[0, 0] for obs in obs_node_features]) + self.bias
        log_probs = self.bias.expand(actions.shape[0])
        entropy = torch.ones_like(log_probs) + self.bias * 0
        return log_probs, values, entropy


def _toy_trainer(tmp_path, terminate_first=True):
    return AsyncVectorPPOTrainer(
        policy=DeterministicPolicy(), num_ues=2, num_envs=2,
        config=PPOConfig(
            rollout_steps=16, gamma=0.5, gae_lambda=0.5,
            epochs=1, total_iterations=1, device="cpu",
        ),
        save_dir=tmp_path,
        env_fns=[
            partial(ToyGraphEnv, rank=0, horizon=3, terminate=terminate_first),
            partial(ToyGraphEnv, rank=1, horizon=5, terminate=not terminate_first),
        ],
        context="spawn",
    )


def _assert_workers_closed(envs, workers):
    assert envs.closed
    assert len(workers) == 2
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive(), f"Worker {worker.pid} não encerrou"


class TestVectorGAE:
    """Bootstrap e corte da recursão são decisões distintas."""

    def test_manual_termination_truncation_and_rollout_cut(self):
        rewards = np.array([[2, 4], [5, 6], [7, 8]], dtype=np.float32)
        values = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
        bootstrap = np.array([[2, 20], [0, 40], [8, 60]], dtype=np.float32)
        ends = np.array([[False, False], [True, True], [False, False]])

        advantages, returns = compute_vector_gae(
            rewards, values, bootstrap, ends, gamma=0.5, gae_lambda=0.5,
        )

        # t=1: término no env 0 (bootstrap=0), truncamento no env 1
        # (V(final)=40). Ambos cortam GAE antes do próximo episódio em t=2.
        # Deltas: [[2,4], [3,6], [8,8]]. Em t=2 há corte de rollout,
        # não de episódio: usa bootstrap, mas não existe vantagem futura.
        expected = np.array([[2.75, 5.5], [3, 6], [8, 8]])
        assert isinstance(advantages, np.ndarray)
        assert isinstance(returns, np.ndarray)
        assert advantages.shape == returns.shape == (3, 2)
        np.testing.assert_allclose(advantages, expected)
        np.testing.assert_allclose(returns, expected + values)

    def test_environments_are_independent(self):
        rewards = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
        values = rewards / 2
        bootstrap = values + 1
        ends = np.array([[False, True], [True, False], [False, False]])
        original = compute_vector_gae(rewards, values, bootstrap, ends, 0.9, 0.8)
        changed_rewards, changed_values, changed_bootstrap = [
            array.copy() for array in (rewards, values, bootstrap)
        ]
        changed_ends = ends.copy()
        changed_rewards[:, 1] += 1000
        changed_values[:, 1] -= 100
        changed_bootstrap[:, 1] += 500
        changed_ends[:, 1] = ~changed_ends[:, 1]
        changed = compute_vector_gae(
            changed_rewards, changed_values, changed_bootstrap, changed_ends, 0.9, 0.8,
        )
        for before, after in zip(original, changed):
            np.testing.assert_array_equal(before[:, 0], after[:, 0])
            assert not np.allclose(before[:, 1], after[:, 1])
        reversed_result = compute_vector_gae(
            rewards[:, ::-1], values[:, ::-1], bootstrap[:, ::-1],
            ends[:, ::-1], 0.9, 0.8,
        )
        for before, reversed_array in zip(original, reversed_result):
            np.testing.assert_allclose(before, reversed_array[:, ::-1])


class TestVectorRollout:
    @pytest.mark.parametrize("terminate_first", [True, False])
    def test_same_step_bootstrap_actions_and_snapshots(self, tmp_path, terminate_first):
        trainer = _toy_trainer(tmp_path, terminate_first)
        try:
            assert trainer.envs.autoreset_mode == gym.vector.AutoresetMode.SAME_STEP
            trainer._obs = trainer.envs.reset(seed=123)[0]
            trainer._collect_rollout()

            steps = np.arange(8)[:, None]
            horizons = np.array([3, 5])
            ranks = np.array([0, 1])
            local_steps = steps % horizons
            values = 100 * (steps // horizons) + 10 * ranks + local_steps
            ends = local_steps + 1 == horizons
            terminated = ends & np.array([terminate_first, not terminate_first])
            bootstrap = np.where(terminated, 0, values + 1)
            rewards = 10 * ranks + local_steps + 1
            features = np.stack([values, local_steps, np.broadcast_to(ranks, (8, 2))], axis=-1)
            expected_obs = np.repeat(features[:, :, None, :], 2, axis=2)

            np.testing.assert_array_equal(trainer._obs_node_features, expected_obs)
            np.testing.assert_array_equal(trainer._dones, ends)
            np.testing.assert_array_equal(trainer._bootstrap_values, bootstrap)
            np.testing.assert_array_equal(trainer._rewards, rewards)
            np.testing.assert_array_equal(
                trainer._actions, np.repeat((local_steps % 2)[:, :, None], 2, axis=2),
            )
            assert trainer._actions.shape == (8, 2, 2)
            # Exatamente uma ação por transição; bootstrap não chama act().
            assert trainer.policy.act_calls == 16
            assert trainer.policy.value_observations
            truncated = ends & ~terminated
            seen_values = [obs[0, 0] for obs in trainer.policy.value_observations]
            for final_value in bootstrap[truncated]:
                assert final_value in seen_values

            # Oráculo escalar por worker, sem reutilizar compute_vector_gae.
            expected_advantages = np.zeros((8, 2))
            for env_index in range(2):
                carry = 0.0
                for t in range(7, -1, -1):
                    delta = rewards[t, env_index] + 0.5 * bootstrap[t, env_index] - values[t, env_index]
                    carry = delta + 0.25 * (not ends[t, env_index]) * carry
                    expected_advantages[t, env_index] = carry
            np.testing.assert_allclose(trainer._advantages, expected_advantages)
            np.testing.assert_allclose(trainer._returns, expected_advantages + values)

            snapshots = np.asarray(trainer._obs_node_features).copy()
            trainer._obs["node_features"].fill(-999)
            trainer.envs.reset(seed=456)
            np.testing.assert_array_equal(trainer._obs_node_features, snapshots)
        finally:
            trainer.envs.close()

    @pytest.mark.parametrize("num_envs", [0, 1])
    def test_rejects_less_than_two_envs(self, tmp_path, num_envs):
        with pytest.raises(ValueError, match="num_envs"):
            AsyncVectorPPOTrainer(
                policy=DeterministicPolicy(), num_ues=2, num_envs=num_envs,
                save_dir=tmp_path,
            )

    def test_train_closes_workers_when_initial_reset_fails(self, tmp_path, monkeypatch):
        trainer = _toy_trainer(tmp_path)
        workers = list(trainer.envs.processes)

        def fail_reset(*args, **kwargs):
            raise RuntimeError("reset propositalmente inválido")

        try:
            # Falha no reset inicial de train(), antes de qualquer rollout.
            # Injeção no pai evita depender do transporte de erros do Gymnasium.
            monkeypatch.setattr(trainer.envs, "reset", fail_reset)
            with pytest.raises(RuntimeError, match="reset propositalmente inválido"):
                trainer.train()
            _assert_workers_closed(trainer.envs, workers)
        finally:
            trainer.envs.close()


class TestOpenRANVectorEquivalence:
    def test_direct_matches_same_step_spawn(self):
        """Autoreset preserva a trajetória e o RNG do OpenRAN direto."""
        direct_envs = []
        vector_env = None
        try:
            direct_envs = [
                OpenRAN_RBA_Env(num_ues=3, max_steps=3) for _ in range(2)
            ]
            vector_env = gym.vector.AsyncVectorEnv(
                [partial(OpenRAN_RBA_Env, num_ues=3, max_steps=3) for _ in range(2)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
                context="spawn",
            )
            vector_obs, _ = vector_env.reset(seed=[123, 124])
            for rank, (env, seed) in enumerate(zip(direct_envs, [123, 124])):
                direct_obs, _ = env.reset(seed=seed)
                assert vector_obs.keys() == direct_obs.keys()
                for key in direct_obs:
                    np.testing.assert_allclose(
                        vector_obs[key][rank], direct_obs[key], rtol=1e-6, atol=1e-7,
                    )

            # Alocação fixa, distinta por worker e sem consumir RNG de sampling.
            actions = np.stack([
                (np.arange(env.num_rbs, dtype=np.int64) + rank) % 3
                for rank, env in enumerate(direct_envs)
            ])
            reset_counts = np.zeros(2, dtype=np.int64)
            for step in range(8):
                vector_obs, rewards, terminated, truncated, infos = vector_env.step(actions)
                expected_ends = np.full(2, (step + 1) % 3 == 0)
                np.testing.assert_array_equal(terminated, [False, False])
                np.testing.assert_array_equal(truncated, expected_ends)
                np.testing.assert_array_equal(
                    infos.get("_final_obs", np.zeros(2, dtype=bool)), expected_ends,
                )
                for rank, env in enumerate(direct_envs):
                    direct_obs, reward, direct_terminated, direct_truncated, _ = env.step(
                        actions[rank].copy(),
                    )
                    np.testing.assert_allclose(rewards[rank], reward, rtol=1e-6, atol=1e-7)
                    assert terminated[rank] == direct_terminated
                    assert truncated[rank] == direct_truncated
                    if direct_terminated or direct_truncated:
                        final_obs = infos["final_obs"][rank]
                        assert final_obs.keys() == direct_obs.keys()
                        for key in direct_obs:
                            np.testing.assert_allclose(
                                final_obs[key], direct_obs[key], rtol=1e-6, atol=1e-7,
                            )
                        # SAME_STEP retorna a observação do reset; sem seed,
                        # ambos continuam a sequência RNG do episódio anterior.
                        direct_obs, _ = env.reset()
                        reset_counts[rank] += 1
                    for key in direct_obs:
                        np.testing.assert_allclose(
                            vector_obs[key][rank], direct_obs[key], rtol=1e-6, atol=1e-7,
                        )
            np.testing.assert_array_equal(reset_counts, [2, 2])
        finally:
            try:
                if vector_env is not None:
                    vector_env.close()
            finally:
                for env in direct_envs:
                    env.close()


class TestVectorPPOIntegration:
    def test_gnn_train_checkpoint_and_worker_cleanup(self, tmp_path):
        torch.manual_seed(123)
        env = OpenRAN_RBA_Env(num_ues=3, max_steps=3)
        try:
            num_rbs = env.num_rbs
        finally:
            env.close()
        policy = GNNActorCritic(
            in_features=3, hidden_dim=8, num_layers=1,
            num_heads=2, num_rbs=num_rbs,
        )
        before = {name: param.detach().clone() for name, param in policy.named_parameters()}
        trainer = AsyncVectorPPOTrainer(
            policy=policy, num_ues=3, num_envs=2, max_steps=3, seed=123,
            config=PPOConfig(
                lr=1e-3, rollout_steps=16, epochs=1, batch_size=8,
                total_iterations=1, checkpoint_interval=1, device="cpu",
            ),
            save_dir=tmp_path,
            # Omite env_fns e context para cobrir OpenRAN e spawn por default.
        )
        workers = list(trainer.envs.processes)
        try:
            assert all(worker._start_method == "spawn" for worker in workers)
            trainer.train()
            _assert_workers_closed(trainer.envs, workers)
            assert trainer._actions.shape == (8, 2, num_rbs)
            for array in (trainer._rewards, trainer._bootstrap_values, trainer._advantages, trainer._returns):
                assert array.shape == (8, 2)
                assert np.isfinite(array).all()
            assert np.asarray(trainer._dones).any()
            assert all(torch.isfinite(param).all() for param in policy.parameters())
            assert any(
                not torch.equal(before[name], param.detach())
                for name, param in policy.named_parameters()
            )
            for filename in ("checkpoint_1.pt", "checkpoint_final.pt"):
                checkpoint = torch.load(tmp_path / filename, map_location="cpu", weights_only=True)
                assert checkpoint["num_envs"] == 2
                assert checkpoint["steps_per_env"] == 8
                assert checkpoint["total_rollout_samples"] == 16
                assert checkpoint["optimizer_state_dict"]["state"]
                for name, tensor in policy.state_dict().items():
                    torch.testing.assert_close(checkpoint["model_state_dict"][name], tensor)
            with (tmp_path / "training_log.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            assert len(rows) == 1
            for key in ("mean_reward", "policy_loss", "value_loss", "entropy", "elapsed_s"):
                assert np.isfinite(float(rows[0][key])), key
        finally:
            trainer.envs.close()
