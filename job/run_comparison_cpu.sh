#!/usr/bin/env bash
set -Eeuo pipefail

# wait -n requires Bash 4.3 or newer (available on the compute node).
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    echo 'Erro: este script requer Bash >= 4.3.' >&2
    exit 2
fi
if (( $# != 1 )) || [[ -z $1 ]]; then
    echo "Uso: bash $0 ROOT (diretório novo)" >&2
    exit 2
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Resolve ROOT relative to the caller, before switching to the project.
case $1 in
    /*) ROOT=$1 ;;
    *) ROOT="$PWD/$1" ;;
esac
cd -- "$PROJECT_DIR"
export PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
ITERATIONS=${ITERATIONS:-100}
export TORCH_THREADS=4
if [[ ! $ITERATIONS =~ ^[1-9][0-9]*$ ]]; then
    echo 'Erro: ITERATIONS deve ser um inteiro positivo.' >&2
    exit 2
fi
if [[ ! -x $PYTHON_BIN ]]; then
    echo "Erro: Python não executável: $PYTHON_BIN" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# mkdir (without -p for ROOT) atomically refuses any existing destination.
mkdir -p -- "$(dirname -- "$ROOT")"
if ! mkdir -- "$ROOT"; then
    echo "Erro: ROOT deve ser um diretório novo; não será sobrescrito: $ROOT" >&2
    exit 2
fi

lane_pids=()
cleanup() {
    local status=$?
    trap - EXIT
    trap '' INT TERM HUP
    if (( ${#lane_pids[@]} )); then
        # Job control gives each lane its own process group, including workers.
        local pid
        for pid in "${lane_pids[@]}"; do
            kill -TERM -- "-$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in "${lane_pids[@]}"; do
            kill -KILL -- "-$pid" 2>/dev/null || true
        done
        for pid in "${lane_pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

train() {
    local num_envs=$1 label=$2 seed=$3
    echo "Treino: ${label}_seed${seed}, iterations=$ITERATIONS"
    "$PYTHON_BIN" -m grams_env.agents.gnn.train_gnn \
        --device cpu --num_ues 20 --rollout_steps 2048 \
        --iterations "$ITERATIONS" --torch_threads "$TORCH_THREADS" \
        --num_envs "$num_envs" --seed "$seed" \
        --save_dir "$ROOT/${label}_seed${seed}"
}

# Exactly two background lanes; Python runs in the foreground within each.
set -m
(
    trap - EXIT INT TERM HUP
    set +m
    for seed in 42 43 44; do
        train 1 sequential "$seed"
    done
) >"$ROOT/sequential.log" 2>&1 &
lane_pids+=("$!")
(
    trap - EXIT INT TERM HUP
    set +m
    for num_envs in 2 8; do
        for seed in 42 43 44; do
            train "$num_envs" "vector$num_envs" "$seed"
        done
    done
) >"$ROOT/vector.log" 2>&1 &
lane_pids+=("$!")

# Detect an early failure in either lane; EXIT cleanup terminates/reaps both.
status=0
wait -n "${lane_pids[@]}" || status=$?
if (( status != 0 )); then
    echo "Erro: uma lane falhou (status $status); encerrando ambas. Veja $ROOT/*.log" >&2
    exit "$status"
fi
for pid in "${lane_pids[@]}"; do
    status=0
    wait "$pid" || status=$?
    if (( status != 0 )); then
        echo "Erro: lane $pid falhou (status $status). Veja $ROOT/*.log" >&2
        exit "$status"
    fi
done
lane_pids=()
set +m

"$PYTHON_BIN" tests/compare_training_quality.py \
    --run-dir "$ROOT" --seeds 100 101 102 --steps 3600 --torch-threads 4 \
    >"$ROOT/comparison.log" 2>&1
 echo "Comparação concluída: $ROOT/comparison.log"
