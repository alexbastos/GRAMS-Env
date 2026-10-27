#!/bin/bash
# -----------------------------------------------------------------------------
# Script para iniciar os treinamentos dos modelos GNN e MLP no GRAMS-Env.
# -----------------------------------------------------------------------------

usage() {
    echo "Uso: $0 [-d cpu|cuda] [-n NUM_ENVS] [-t TORCH_THREADS] [-i GNN_ITERATIONS]"
    echo ""
    echo "Opções:"
    echo "  -d DEVICE          Device para treinamento: 'cpu' ou 'cuda' (default: cuda)"
    echo "  -n NUM_ENVS        Ambientes paralelos no treino GNN (default: GRAMS_NUM_ENVS ou 1)"
    echo "  -t TORCH_THREADS   Threads intra-op do PyTorch no treino GNN (default: GRAMS_TORCH_THREADS ou não define)"
    echo "  -i GNN_ITERATIONS  Total de iterações GNN (default: GRAMS_GNN_ITERATIONS ou 2000); MLP mantém 500"
    echo "  -h                 Mostra esta ajuda"
}

DEVICE="cuda"
NUM_ENVS="${GRAMS_NUM_ENVS:-1}"
TORCH_THREADS="${GRAMS_TORCH_THREADS:-}"

GNN_ITERATIONS="${GRAMS_GNN_ITERATIONS:-2000}"

while getopts ":d:n:t:i:h" opt; do
    case "$opt" in
        d)
            DEVICE="$OPTARG"
            ;;
        n)
            NUM_ENVS="$OPTARG"
            ;;
        t)
            TORCH_THREADS="$OPTARG"
            ;;
        i)
            GNN_ITERATIONS="$OPTARG"
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "❌ Opção inválida: -$OPTARG"
            usage
            exit 1
            ;;
        :)
            echo "❌ A opção -$OPTARG requer um argumento."
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$GNN_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ GNN_ITERATIONS inválido: '$GNN_ITERATIONS'. Use um inteiro positivo."
    exit 1
fi

if [ "$DEVICE" != "cpu" ] && [ "$DEVICE" != "cuda" ]; then
    echo "❌ Device inválido: '$DEVICE'. Use 'cpu' ou 'cuda'."
    usage
    exit 1
fi

if ! [[ "$NUM_ENVS" =~ ^[0-9]+$ ]] || [ "$NUM_ENVS" -lt 1 ]; then
    echo "❌ NUM_ENVS inválido: '$NUM_ENVS'. Use um inteiro >= 1."
    usage
    exit 1
fi

if [ -n "$TORCH_THREADS" ]; then
    if ! [[ "$TORCH_THREADS" =~ ^[0-9]+$ ]] || [ "$TORCH_THREADS" -lt 1 ]; then
        echo "❌ TORCH_THREADS inválido: '$TORCH_THREADS'. Use um inteiro >= 1."
        usage
        exit 1
    fi
fi

echo "============================================================"
echo " INICIANDO TREINAMENTO DOS AGENTES DE APRENDIZADO POR REFORÇO"
echo "============================================================"
echo "Device selecionado: $DEVICE"
echo "Ambientes GNN   : $NUM_ENVS"
echo "Iterações GNN   : $GNN_ITERATIONS"
echo "Iterações MLP   : 500"
if [ -n "$TORCH_THREADS" ]; then
    echo "Torch threads   : $TORCH_THREADS"
fi
echo ""

# 1. Treinamento da GNN+PPO
echo "[1/2] Iniciando Treinamento: Agente GNN + PPO (Cenário de V=20 UEs)..."
echo "Isso pode demorar várias horas dependendo da máquina."
GNN_CMD=(
    python3 -m grams_env.agents.gnn.train_gnn
    --num_ues 20
    --iterations "$GNN_ITERATIONS"
    --rollout_steps 2048
    --seed 42
    --device "$DEVICE"
    --save_dir runs/gnn_ppo
    --num_envs "$NUM_ENVS"
)

if [ -n "$TORCH_THREADS" ]; then
    GNN_CMD+=(--torch_threads "$TORCH_THREADS")
fi

"${GNN_CMD[@]}"

# Checa se o comando anterior falhou
if [ $? -ne 0 ]; then
    echo "❌ Erro ao treinar o modelo GNN. Abortando script."
    exit 1
fi
echo "✅ Treinamento GNN+PPO finalizado com sucesso!"
echo ""

# 2. Treinamento da Baseline MLP+PPO
echo "[2/2] Iniciando Treinamento: Baseline MLP + PPO (Cenário de V=20 UEs)..."
python3 -m grams_env.agents.mlp.train_mlp \
    --num_ues 20 \
    --iterations 500 \
    --seed 42 \
    --device "$DEVICE" \
    --save_dir runs/mlp_ppo

if [ $? -ne 0 ]; then
    echo "❌ Erro ao treinar o modelo MLP. Abortando script."
    exit 1
fi
echo "✅ Treinamento MLP+PPO finalizado com sucesso!"
echo ""

echo "============================================================"
echo " 🎉 TODOS OS TREINAMENTOS FORAM CONCLUÍDOS!"
echo " Os modelos foram salvos no diretório 'runs/' e já podem"
echo " ser usados para avaliação Zero-Shot."
echo "============================================================"
