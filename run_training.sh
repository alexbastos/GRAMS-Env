#!/bin/bash
# -----------------------------------------------------------------------------
# Script para iniciar os treinamentos dos modelos GNN e MLP no GRAMS-Env.
#
# NOTA: Garanta que você está com o ambiente conda ativado antes de rodar,
# ou defina PYTHON_BIN apontando para o interpretador do ambiente.
# Exemplo: export PYTHON_BIN=/caminho/do/projeto/.venv/bin/python
# -----------------------------------------------------------------------------

usage() {
    echo "Uso: $0 [-d cpu|cuda] [-n NUM_ENVS] [-t TORCH_THREADS]"
    echo ""
    echo "Opções:"
    echo "  -d DEVICE          Device para treinamento: 'cpu' ou 'cuda' (default: cuda)"
    echo "  -n NUM_ENVS        Ambientes paralelos no treino GNN (default: GRAMS_NUM_ENVS ou 1)"
    echo "  -t TORCH_THREADS   Threads intra-op do PyTorch no treino GNN (default: GRAMS_TORCH_THREADS ou não define)"
    echo "  -h                 Mostra esta ajuda"
    echo ""
    echo "Variáveis de ambiente:"
    echo "  PYTHON_BIN          Interpretador Python a usar (default: python)"
}

DEVICE="cuda"
NUM_ENVS="${GRAMS_NUM_ENVS:-1}"
TORCH_THREADS="${GRAMS_TORCH_THREADS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

while getopts ":d:n:t:h" opt; do
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

if [ ! -x "$PYTHON_BIN" ]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "❌ PYTHON_BIN inválido ou não executável: '$PYTHON_BIN'."
        usage
        exit 1
    fi
fi

echo "============================================================"
echo " INICIANDO TREINAMENTO DOS AGENTES DE APRENDIZADO POR REFORÇO"
echo "============================================================"
echo "Python          : $PYTHON_BIN"
echo "Device selecionado: $DEVICE"
echo "Ambientes GNN   : $NUM_ENVS"
if [ -n "$TORCH_THREADS" ]; then
    echo "Torch threads   : $TORCH_THREADS"
fi
echo ""

# 1. Treinamento da GNN+PPO
echo "[1/2] Iniciando Treinamento: Agente GNN + PPO (Cenário de V=20 UEs)..."
echo "Isso pode demorar várias horas dependendo da máquina."
GNN_CMD=(
    "$PYTHON_BIN" -m grams_env.agents.gnn.train_gnn
    --num_ues 20
    --iterations 500
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
"$PYTHON_BIN" -m grams_env.agents.mlp.train_mlp \
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
