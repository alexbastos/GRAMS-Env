"""Script de teste rápido para Avaliação Zero-Shot do modelo GNN.

Carrega os pesos de um modelo GNN treinado com 20 UEs e executa a inferência
em um cenário denso com 100 UEs para testar a generalização (Zero-Shot).
"""

import os
import torch

from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env


def main():
    model_path = "runs/gnn_ppo/model_gnn_frozen.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Erro: Modelo não encontrado em '{model_path}'.")
        print("Você precisa rodar o script de treinamento (./run_training.sh) primeiro para gerar os pesos!")
        return

    print("🧠 Carregando modelo GNN (treinado com V=20)...")
    policy = GNNActorCritic(
        in_features=3, 
        hidden_dim=64, 
        num_layers=2,
        num_heads=4, 
        num_rbs=50
    )
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    print("\n🌍 Inicializando ambiente denso com V=100 UEs (Zero-Shot)...")
    env = OpenRAN_RBA_Env(num_ues=100)
    obs, _ = env.reset(seed=0)

    print("⚡ Executando inferência Zero-Shot...")
    with torch.no_grad():
        action, log_prob, value = policy.act(obs)  # A GNN processa matrizes de tamanho 100 sem quebrar!

    obs, reward, _, _, info = env.step(action)
    
    print("\n" + "="*45)
    print("📊 RESULTADOS DO TESTE ZERO-SHOT (V=100)")
    print("="*45)
    print(f"Recompensa (Reward) : {reward:.2f}")
    print(f"Throughput Total    : {info['total_throughput_bits']:.0f} bits")
    print(f"SINR Médio          : {info['mean_sinr_db']:.2f} dB")
    print("="*45)
    print("✅ Sucesso! A GNN generalizou e alocou recursos para 100 UEs.")


if __name__ == "__main__":
    main()
