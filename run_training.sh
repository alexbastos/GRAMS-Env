#!/bin/bash
# -----------------------------------------------------------------------------
# Script para iniciar os treinamentos dos modelos GNN e MLP no GRAMS-Env.
# 
# NOTA: Garanta que você está com o ambiente conda ativado antes de rodar.
# Comando: conda activate grams
# -----------------------------------------------------------------------------

echo "============================================================"
echo " INICIANDO TREINAMENTO DOS AGENTES DE APRENDIZADO POR REFORÇO"
echo "============================================================"
echo ""

# 1. Treinamento da GNN+PPO
echo "[1/2] Iniciando Treinamento: Agente GNN + PPO (Cenário de V=20 UEs)..."
echo "Isso pode demorar várias horas dependendo da máquina."
python -m grams_env.agents.gnn.train_gnn \
    --num_ues 20 \
    --iterations 500 \
    --rollout_steps 2048 \
    --seed 42 \
    --device cpu \
    --save_dir runs/gnn_ppo

# Checa se o comando anterior falhou
if [ $? -ne 0 ]; then
    echo "❌ Erro ao treinar o modelo GNN. Abortando script."
    exit 1
fi
echo "✅ Treinamento GNN+PPO finalizado com sucesso!"
echo ""

# 2. Treinamento da Baseline MLP+PPO
echo "[2/2] Iniciando Treinamento: Baseline MLP + PPO (Cenário de V=20 UEs)..."
python -m grams_env.agents.mlp.train_mlp \
    --num_ues 20 \
    --iterations 500 \
    --seed 42 \
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
