"""CLI standalone para comparar checkpoints GNN; não é um teste pytest."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time

GROUPS = ("sequential", "vector2", "vector8")
METRICS = (
    "mean_throughput_bits_per_tti", "throughput_mbps", "queue_delay_p95_ms",
    "total_reward", "entropy_normalized_mean", "max_probability_mean",
    "active_ues_mean", "largest_rb_fraction_mean", "ric_latency_mean_ms",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("deve ser positivo")
    return number


def discover(root: Path) -> dict[str, dict[int, Path]]:
    """Exige a união inteira de seeds em cada grupo, nunca só a interseção."""
    if not root.is_dir():
        raise ValueError(f"Diretório inexistente: {root}")
    runs: dict[str, dict[int, Path]] = {group: {} for group in GROUPS}
    for group in GROUPS:
        for directory in sorted(root.glob(f"{group}_seed*")):
            match = re.fullmatch(rf"{group}_seed(0|[1-9][0-9]*)", directory.name)
            if match is None:
                raise ValueError(f"Nome de run inválido: {directory.name}")
            if not directory.is_dir():
                raise ValueError(f"Run não é diretório: {directory}")
            runs[group][int(match[1])] = directory / "model_gnn_frozen.pt"
    seeds = sorted(set().union(*(set(items) for items in runs.values())))
    if not seeds:
        raise ValueError("Nenhum run sequential_seedN/vector2_seedN/vector8_seedN encontrado")
    missing = []
    for group in GROUPS:
        for seed in seeds:
            checkpoint = runs[group].get(seed)
            if checkpoint is None or not checkpoint.is_file():
                missing.append(f"{group}_seed{seed}/model_gnn_frozen.pt")
    if missing:
        raise ValueError("Matriz incompleta; faltam: " + ", ".join(missing))
    return runs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(episodes: list[dict]) -> list[dict]:
    rows = []
    for group in GROUPS:
        selected = [row for row in episodes if row["group"] == group]
        seeds = sorted({row["train_seed"] for row in selected})
        per_run = []
        for seed in seeds:
            sample = [row for row in selected if row["train_seed"] == seed]
            means = {key: statistics.mean(row[key] for row in sample) for key in METRICS}
            per_run.append(means)
            for key, value in means.items():
                rows.append(dict(level="run", group=group, train_seed=seed,
                                 n_train_seeds=1, n_eval_seeds=len(sample),
                                 metric=key, mean=value, std=""))
        for key in METRICS:
            values = [row[key] for row in per_run]
            rows.append(dict(level="group", group=group, train_seed="",
                             n_train_seeds=len(seeds),
                             n_eval_seeds=len(selected) // len(seeds), metric=key,
                             mean=statistics.mean(values),
                             std=statistics.stdev(values) if len(values) > 1 else ""))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Raiz dos três grupos de runs")
    parser.add_argument("--seeds", required=True, nargs="+", type=int, help="Seeds de avaliação, não de treino")
    parser.add_argument("--steps", type=positive_int, default=3600, help="TTIs por episódio (default: 3600)")
    parser.add_argument("--torch-threads", type=positive_int, default=4, help="Threads intra-op CPU (default: 4)")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("--seeds exige valores únicos entre 0 e 2**32-1")
    root = args.run_dir.resolve()
    try:
        runs = discover(root)
    except ValueError as exc:
        parser.error(str(exc))

    # Imports pesados após parse_args: --help funciona sem dependências de ML.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from grams_env.agents.common.utils import set_seed
    from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic
    from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env
    from grams_env.metrics.collector import MetricsCollector

    class ObservedPolicy(GNNActorCritic):
        def _get_policy_and_value(self, node_features, adj_matrix):
            dist, value = super()._get_policy_and_value(node_features, adj_matrix)
            self.last_distribution = dist
            return dist, value

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    architecture = dict(in_features=3, hidden_dim=64, num_layers=2, num_heads=4, device="cpu")
    env = OpenRAN_RBA_Env(num_ues=20, max_steps=args.steps)
    episodes = []
    checkpoints = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        architecture["num_rbs"] = env.num_rbs
        # Valida todos os state_dicts antes de executar qualquer episódio.
        policies = []
        for group in GROUPS:
            for train_seed, path in sorted(runs[group].items()):
                digest = sha256(path)
                policy = ObservedPolicy(**architecture)
                policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                policy.eval()
                if sha256(path) != digest:
                    raise RuntimeError(f"Checkpoint mudou durante leitura: {path}")
                checkpoints.append(dict(group=group, train_seed=train_seed,
                                        path=str(path.relative_to(root)), sha256=digest))
                policies.append((group, train_seed, policy))
        with torch.inference_mode():
            for group, train_seed, policy in policies:
                for eval_seed in args.seeds:
                    set_seed(eval_seed)
                    obs, _ = env.reset(seed=eval_seed)
                    collector = MetricsCollector(env.config.bandwidth_mhz * 1e6, env.config.tti_s)
                    diagnostics = []
                    for _ in range(args.steps):
                        start = time.perf_counter()
                        action, _, _ = policy.act(obs)
                        latency = time.perf_counter() - start
                        dist = policy.last_distribution
                        counts = np.bincount(action, minlength=env.num_ues)
                        diagnostics.append((dist.entropy().item() / math.log(env.num_ues),
                                            dist.probs.max().item(), int(np.count_nonzero(counts)),
                                            float(counts.max() / env.num_rbs)))
                        obs, reward, terminated, truncated, info = env.step(action)
                        collector.record_step(obs, info, reward, latency)
                        if terminated or truncated:
                            break
                    result = collector.compute()
                    entropy, maxprob, active, fraction = np.mean(diagnostics, axis=0).tolist()
                    row = dict(group=group, train_seed=train_seed, eval_seed=eval_seed,
                               num_steps=result.num_steps, terminated=terminated, truncated=truncated,
                               mean_throughput_bits_per_tti=result.mean_throughput_bits_per_tti,
                               throughput_mbps=result.mean_throughput_bits_per_tti / env.config.tti_s / 1e6,
                               queue_delay_p95_ms=result.queue_delay_p95_ms,
                               total_reward=result.total_reward, entropy_normalized_mean=entropy,
                               max_probability_mean=maxprob, active_ues_mean=active,
                               largest_rb_fraction_mean=fraction,
                               ric_latency_mean_ms=result.ric_latency_mean_ms)
                    if not all(math.isfinite(row[key]) for key in METRICS):
                        raise RuntimeError(f"Métricas não finitas: {row}")
                    episodes.append(row)
                    print(f"{group}_seed{train_seed} eval={eval_seed}: {row['throughput_mbps']:.4f} Mbps", flush=True)
        manifest = dict(
            started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
            run_dir=str(root), evaluation_seeds=args.seeds,
            training_seeds=sorted(runs[GROUPS[0]]), steps=args.steps,
            architecture=architecture, environment=dict(num_ues=20, max_steps=args.steps, cell=asdict(env.config)),
            versions={name: version(name) for name in ("torch", "torch-geometric", "numpy", "gymnasium")},
            python=sys.version, platform=platform.platform(), cpu=platform.processor(),
            cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            thread_environment={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            checkpoints=checkpoints, evaluator_sha256=sha256(Path(__file__)),
            protocol={
                "inference": "CPU, eval(), inference_mode(); stochastic policy.act, no argmax; set_seed and env.reset(seed) each episode; no warmup",
                "order": "serial: group sequential/vector2/vector8, ascending training seed, evaluation seeds in CLI order",
                "horizon": "One episode per checkpoint/eval seed; max_steps=--steps; stop on terminated or truncated; no autoreset or bootstrap",
                "metrics": "Throughput mean bits/TTI divided by tti_s and 1e6 for Mbps; queue delay p95 via MetricsCollector (Little's law, all UEs/TTIs); total reward sum",
                "diagnostics": "Temporal means: categorical entropy/log(20), max UE probability, UEs receiving >=1 RB (not nonempty queues), maximum per-UE RB count/num_rbs per TTI",
                "aggregation": "Equal-weight eval-seed means within each training run, then equal-weight mean and sample std (ddof=1) across training seeds; std blank with <2 training seeds; no pooled eval replicates",
                "timing": "perf_counter around policy.act only, including tensor conversion, actor/critic, sampling, logprob and NumPy output; distribution capture adds an assignment; diagnostics and env.step excluded. Serial evaluation, not parallel training speed. CPU contention, thermal state and order affect timing; run on an idle machine",
                "training_truncation": "Current sequential trainer treats terminated OR truncated as done and zeros bootstrap; vector trainer bootstraps truncation from final_obs, but stops GAE recursion. Thus quality differences cannot be attributed solely to parallelism",
                "limitations": "Checkpoint state_dicts do not prove training config, budget, code version or historical truncation semantics. Group/seed inferred from directory names. Paired evaluation seeds do not create independent training replicates or establish statistical significance",
            },
        )
    finally:
        env.close()
    write_csv(root / "quality_episodes.csv", episodes)
    write_csv(root / "quality_summary.csv", summarize(episodes))
    (root / "quality_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Resultados: {root}/quality_{{episodes.csv,summary.csv,manifest.json}}")


if __name__ == "__main__":
    main()
