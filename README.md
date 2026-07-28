# GRAMS-Env: OpenRAN Resource Block Allocation Environment

Ambiente Gymnasium de alto desempenho para simulação da camada MAC e alocação de Resource Blocks (RBA) em redes 6G / Open RAN (O-RAN), construído seguindo os princípios de **Clean Architecture**.

O ambiente simula o enlace descendente de uma célula Urban Macro (UMa) conforme as especificações do **3GPP TR 36.873**, fornecendo observações estruturadas em formato de grafo (`node_features` + `adjacency_matrix`) adequadas para consumo direto por **Graph Neural Networks (GNNs)** e algoritmos de Aprendizado por Reforço (RL).

---

## 🏛️ Arquitetura do Projeto (Clean Architecture)

O projeto foi totalmente refatorado para desacoplar as regras de negócio de telecomunicações, orquestração de RL e infraestrutura de frameworks.

```
GRAMS-Env/
├── grams_env/
│   ├── core/                          # 🟢 CAMADA DE DOMÍNIO (Telecom pura, zero deps externas)
│   │   ├── domain/                    # Entidades imutáveis e dataclasses
│   │   │   ├── cell.py                # CellConfig (parâmetros 3GPP TR 36.873)
│   │   │   ├── user_equipment.py      # UserEquipment (estado do UE)
│   │   │   └── network_state.py       # NetworkState (snapshot por TTI)
│   │   │
│   │   ├── services/                  # Equações e física de telecom (funções puras/serviços)
│   │   │   ├── propagation.py         # TR36873_UMa (Path Loss LOS/NLOS, P_LOS)
│   │   │   ├── channel.py             # ChannelGainCalculator (Direct, Inter-UE, Fading)
│   │   │   ├── link_budget.py         # LinkBudget (SINR, Shannon Capacity)
│   │   │   ├── mobility.py            # MobilityModel (Random Walk com reflexão)
│   │   │   └── traffic.py             # CBRTrafficGenerator (Tráfego CBR)
│   │   │
│   │   └── ports/                     # Interfaces (ABC)
│   │       ├── propagation_port.py    # PropagationModel (Interface)
│   │       └── reward_port.py         # RewardFunction (Interface)
│   │
│   ├── adapters/                      # 🟡 CAMADA DE ADAPTAÇÃO
│   │   ├── graph_builder.py           # Converte NetworkState → Grafo GNN
│   │   └── reward.py                  # ThroughputQueueReward (Função de Recompensa)
│   │
│   └── infrastructure/                # 🔴 CAMADA DE INFRAESTRUTURA
│       ├── gymnasium_env.py           # OpenRAN_RBA_Env (Wrapper Gymnasium fino)
│       └── numpy_rng.py              # NumpyRNG (Adapter de Random Generator)
│
├── tests/                             # 🧪 SUÍTE DE TESTES (Unitários e Integração)
│   ├── test_propagation.py            # Testes de propagação 3GPP (sem Gymnasium)
│   ├── test_link_budget.py            # Testes de SINR e Shannon (sem Gymnasium)
│   ├── test_graph_builder.py          # Testes do construtor de grafos
│   └── test_gymnasium_env.py          # Testes de integração do ambiente Gymnasium
│
├── openran_rba_env.py                 # [Legacy] Arquivo monólito original
├── pyproject.toml
└── README.md
```

---

## 🧠 Alinhamento com a Proposta de Pesquisa (GNN-DRL para 6G)

Este simulador foi construído especificamente para atuar como o **Processo de Decisão de Markov (MDP)** fundacional para agentes baseados em Graph Neural Networks (GNN) integrados a Deep Reinforcement Learning (DRL) aplicados ao ecossistema Open RAN. 

Abaixo estão os pontos de aderência entre a modelagem e a pesquisa teórica:

1. **Modelagem do Grafo de Comunicação $\mathcal{G} = (\mathcal{V}, \mathcal{E})$**:
   * O adaptador `GraphBuilder` (`grams_env/adapters/graph_builder.py`) gera observações estritamente baseadas em grafos.
   * **Nós ( $\mathcal{V}$ ) e Node Features**: Representam os UEs. A matriz `node_features` contém CQI (indicador de qualidade do canal e proxy de localização), tamanho da fila de pacotes e o requisito de QoS (carga CBR).
   * **Arestas ( $\mathcal{E}$ ) e Edge Features**: Representadas pela `adjacency_matrix`, que contém o ganho de interferência contínuo (CSI inter-UE) entre todos os pares de usuários, formando a base exata para a equação de *message passing* da GNN.

2. **Modelagem Física Realista (3GPP)**:
   * O emulador de canal (`TR36873_UMa`) segue o padrão 3GPP TR 36.873 Urban Macro, implementando Perda de Percurso (LOS/NLOS), Sombreamento Log-normal e Desvanecimento Rápido de Rayleigh.
   * A cada episódio, novas topologias aleatórias (*drops*) são geradas com padrões realistas de mobilidade veicular e de pedestres.

3. **Ação Discreta (RBA)**:
   * O ambiente fornece um `action_space` Multi-Discreto onde o agente DRL deve decidir a alocação de $K=50$ blocos de recursos para os $V$ usuários da célula, mimetizando perfeitamente a tomada de decisão da camada MAC em uma unidade O-DU.

4. **Recompensa focada em Eficiência Espectral e Latência (URLLC)**:
   * A função `ThroughputQueueReward` maximiza a Eficiência Espectral global da célula (Soma das Taxas) e ao mesmo tempo penaliza a latência acumulada nas filas, orientando o agente a garantir justiça e baixos atrasos.

5. **Escalabilidade Zero-Shot**:
   * O formato vetorial em forma de grafos do `observation_space` permite independência espacial. O modelo autoriza treinar a GNN-DRL com, por exemplo, 20 UEs, e inferir em 100 UEs na vida real (mudança de topologia abrupta), um requisito essencial e explorado no artigo científico como *Killer Feature* das GNNs.

---

## 🚀 Instalação e Uso Rápido

### Requisitos
- Python 3.10+
- `gymnasium`
- `numpy`
- `pytest` (para testes)

### Exemplo de Uso (Random Agent)

```python
import numpy as np
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

# Inicializa o ambiente com 10 UEs
env = OpenRAN_RBA_Env(num_ues=10)
obs, info = env.reset(seed=42)

print("Estrutura das observações (Grafo):")
print("  Node Features    :", obs["node_features"].shape)      # (10, 3) -> [CQI, Queue_bits, CBR_bytes]
print("  Adjacency Matrix :", obs["adjacency_matrix"].shape)   # (10, 10) -> Interferência Inter-UE

# Executa 1 TTI (1 ms)
action = env.action_space.sample()  # Aloca K=50 RBs entre os 10 UEs
obs, reward, terminated, truncated, info = env.step(action)

print(f"Throughput total : {info['total_throughput_bits']:.0f} bits")
print(f"SINR médio       : {info['mean_sinr_db']:.2f} dB")
print(f"Recompensa       : {reward:.2f}")
```

---

## ⚙️ Especificações Técnicas do Modelo 3GPP TR 36.873 (UMa)

| Parâmetro | Valor Padrão | Descrição |
|---|---|---|
| **Frequência Portadora** | 2.0 GHz | Frequência de operação |
| **Largura de Banda** | 10.0 MHz | Total do sistema |
| **Resource Blocks (K)** | 50 RBs | 180 kHz por RB |
| **TTI (Time Transmission Interval)** | 1 ms | Duração de cada passo (`step`) |
| **Potência gNodeB** | 46 dBm | Potência de transmissão do gNB |
| **Raio da Célula** | 500 m | Célula Urban Macro (UMa) |
| **Limiar de SINR** | 14.8 dB | Mínimo necessário para decodificação |
| **Modelos de Mobilidade** | 0, 3, 20 km/h | Perfis atribuídos por UE |
| **Perfis de Tráfego CBR** | 1000, 4000 B/TTI | Carga constante por UE |

---

## 🧪 Executando os Testes

A suíte de testes abrange testes unitários isolados da física/telecom (sem dependências do Gymnasium) e testes de integração end-to-end.

Para rodar todos os 38 testes:

```bash
pytest tests/ -v
```

---

## 👨‍💻 Créditos

**Alex Vidigal Bastos**  
Universidade Federal de São João del-Rei (UFSJ)