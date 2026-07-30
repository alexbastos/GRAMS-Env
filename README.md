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
│   ├── infrastructure/                # 🔴 CAMADA DE INFRAESTRUTURA
│   │   ├── gymnasium_env.py           # OpenRAN_RBA_Env (Wrapper Gymnasium fino)
│   │   └── numpy_rng.py              # NumpyRNG (Adapter de Random Generator)
│   │
│   └── agents/                        # 🟣 CAMADA DE AGENTES
│       ├── common/                    # Código compartilhado GNN ↔ MLP
│       │   ├── utils.py               # obs→tensor, adj→edge_index, set_seed
│       │   ├── rollout_buffer.py      # Buffer de trajetórias com GAE-λ
│       │   └── ppo_trainer.py         # Loop PPO genérico (ActorCritic agnóstico)
│       ├── gnn/                       # Agente GNN+PPO (artigo principal)
│       │   ├── graph_encoder.py       # GATConv L=2, 4-heads, residual + LayerNorm
│       │   ├── gnn_actor_critic.py    # Actor-Critic invariante ao número de UEs
│       │   └── train_gnn.py           # Script de treinamento (CLI)
│       ├── mlp/                       # Agente MLP+PPO (baseline DRL de comparação)
│       │   ├── mlp_actor_critic.py    # Flatten fixo em V (sem generalização)
│       │   └── train_mlp.py           # Script de treinamento (CLI)
│       └── baselines/                 # 🔵 Baselines Clássicas (branch: baselines)
│           ├── base.py                # BaselineAgent (ABC compatível com ActorCriticProtocol)
│           ├── round_robin.py         # RoundRobinAgent — alocação cíclica determinística
│           └── proportional_fair.py   # ProportionalFairAgent — PF com EWMA
│
├── tests/                             # 🧪 SUÍTE DE TESTES (123 testes)
│   ├── test_propagation.py            # Testes de propagação 3GPP
│   ├── test_link_budget.py            # Testes de SINR e Shannon
│   ├── test_graph_builder.py          # Testes do construtor de grafos
│   ├── test_gymnasium_env.py          # Testes de integração do ambiente
│   └── test_agents/                   # Testes dos modelos de IA e baselines
│       ├── test_graph_encoder.py      # Shapes, invariância, gradientes
│       ├── test_actor_critic.py       # GNN vs MLP (zero-shot proof)
│       ├── test_rollout_buffer.py     # GAE, mini-batches
│       ├── test_ppo_smoke.py          # Smoke test end-to-end
│       └── test_baselines.py          # RR e PF: interface, fairness, integração
│
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

## 🚀 Instalação

### 1. Criar ambiente conda (Python 3.12)

> **Por quê Python 3.12?** O PyTorch e o PyTorch Geometric ainda não têm wheels oficiais para Python 3.13+.

```bash
conda create -n grams python=3.12 -y
conda activate grams
```

### 2. Instalar PyTorch (CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Instalar dependências do projeto

```bash
pip install torch_geometric gymnasium "numpy<2" pytest
```

> **Nota:** `numpy<2` é necessário para compatibilidade com o PyTorch 2.2.x.

### 4. Verificar instalação

```bash
python -c "import torch, torch_geometric, gymnasium; print('OK')"
```

---

## 📖 Uso Rápido — Random Agent

```python
import numpy as np
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

# Inicializa o ambiente com 10 UEs e máximo de 3600 TTIs por episódio
env = OpenRAN_RBA_Env(num_ues=10, max_steps=3600)
obs, info = env.reset(seed=42)

print("Estrutura das observações (Grafo):")
print("  Node Features    :", obs["node_features"].shape)      # (10, 3) -> [CQI, Queue_bits, CBR_bytes]
print("  Adjacency Matrix :", obs["adjacency_matrix"].shape)   # (10, 10) -> Interferência Inter-UE

# Executa 1 TTI (1 ms) com ação aleatória
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

## 🤖 Agentes de Aprendizado por Reforço (branch `agent_GRAMS`)

O pacote `grams_env/agents/` contém a implementação completa de dois agentes PPO para alocação de RBs, além do treinador genérico.

### Arquitetura dos Modelos

#### 1. Extrator de Características GNN — `GraphEncoder`

Rede neural em grafos baseada em **GATConv** (Graph Attention Network) que transforma as observações estruturadas do simulador em *embeddings* por nó:

```
Input (V, 3) → Linear(3→64) → [GATConv(64, heads=4) + ELU + LayerNorm + Residual] × 2 → H^(L) (V, 64)
```

- **L = 2** camadas de convolução em grafos.
- **4 attention heads** com média (não concatenação), mantendo output em `hidden_dim=64`.
- **Residual connections** em cada camada para estabilidade do treinamento.
- **Edge attributes**: o ganho de interferência da `adjacency_matrix` é passado como peso de atenção (`edge_dim=1`), permitindo que a GNN priorize vizinhos com maior interferência co-canal.
- **Invariante ao número de nós**: o *message passing* processa cada nó independentemente do tamanho do grafo, permitindo **generalização zero-shot** — treinar com V=20 UEs e inferir com V=50, 100, 200.

#### 2. Agente GNN+PPO — `GNNActorCritic` *(artigo principal)*

```
Actor:  H^(L) (V, 64) → Linear(64→32) → ReLU → Linear(32→1) → logits (V,) → Categorical → K amostras
Critic: H^(L) (V, 64) → MeanPool → (64,) → Linear(64→32) → ReLU → Linear(32→1) → V(s)
```

- O **Actor** produz uma distribuição `Categorical` sobre os V UEs via logits por nó, depois amostra K=50 vezes para alocar cada RB.
- O **Critic** usa *global mean pooling* dos embeddings para estimar o valor do estado.
- Por ser baseado em GNN, o modelo **funciona com qualquer V** em tempo de inferência.

#### 3. Baseline MLP+PPO — `MLPActorCritic` *(comparação)*

```
Input: flatten(node_features) + upper_triangle(adjacency_matrix) → vetor fixo de dim = V×3 + V×(V-1)/2
Actor:  flat → Linear(→256) → ReLU → Linear(→128) → ReLU → Linear(→K×V) → Categorical por RB
Critic: flat → Linear(→256) → ReLU → Linear(→128) → ReLU → Linear(→1)
```

- O input é um vetor **de tamanho fixo**, determinado por V no momento da criação do modelo.
- **Não pode inferir com V diferente do treinamento** — isso prova a limitação da abordagem MLP frente à GNN no artigo.

#### 4. Treinador PPO Genérico — `PPOTrainer`

Implementação própria do **Proximal Policy Optimization** (Schulman et al. 2017), compatível com qualquer backbone que implemente `act()` e `evaluate()`:

| Hiperparâmetro | Default | Descrição |
|---|---|---|
| `lr` | `3e-4` | Learning rate (Adam) |
| `gamma` | `0.99` | Fator de desconto |
| `gae_lambda` | `0.95` | Parâmetro λ do GAE |
| `clip_eps` | `0.2` | Epsilon do clipping PPO |
| `epochs` | `10` | Épocas de atualização por rollout |
| `batch_size` | `64` | Tamanho do mini-batch |
| `rollout_steps` | `2048` | Steps coletados por iteração |
| `ent_coef` | `0.01` | Coeficiente do bônus de entropia |
| `vf_coef` | `0.5` | Coeficiente da loss de valor |
| `max_grad_norm` | `0.5` | Clipping de gradiente |

Fórmula da loss total:
```
L = L_clip(π) + vf_coef × L_value − ent_coef × H(π)
```

---

### Como Treinar

#### Agente GNN+PPO (cenário de treino esparso: V=20 UEs)

```bash
# Treinamento padrão — 500 iterações com seed 42
conda run -n grams python -m grams_env.agents.gnn.train_gnn \
    --num_ues 20 \
    --iterations 500 \
    --rollout_steps 2048 \
    --seed 42 \
    --device cpu \
    --save_dir runs/gnn_ppo
```

O treinamento salva:
- `runs/gnn_ppo/checkpoint_<iter>.pt` — checkpoints a cada 50 iterações.
- `runs/gnn_ppo/checkpoint_final.pt` — modelo ao final.
- `runs/gnn_ppo/model_gnn_frozen.pt` — pesos congelados para avaliação zero-shot.
- `runs/gnn_ppo/training_log.csv` — curva de aprendizado (reward, losses, entropy).

#### Baseline MLP+PPO (comparação)

```bash
conda run -n grams python -m grams_env.agents.mlp.train_mlp \
    --num_ues 20 \
    --iterations 500 \
    --seed 42 \
    --save_dir runs/mlp_ppo
```

#### Avaliação Zero-Shot (GNN com V diferente do treino)

```python
import torch
from grams_env.agents.gnn.gnn_actor_critic import GNNActorCritic
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

# Carrega modelo treinado com V=20
policy = GNNActorCritic(in_features=3, hidden_dim=64, num_layers=2,
                         num_heads=4, num_rbs=50)
policy.load_state_dict(torch.load("runs/gnn_ppo/model_gnn_frozen.pt"))
policy.eval()

# Infere com V=100 (zero-shot — sem re-treinamento)
env = OpenRAN_RBA_Env(num_ues=100)
obs, _ = env.reset(seed=0)

with torch.no_grad():
    action, log_prob, value = policy.act(obs)  # ✅ funciona com V=100

obs, reward, _, _, info = env.step(action)
print(f"Zero-shot reward: {reward:.2f}")
print(f"Throughput      : {info['total_throughput_bits']:.0f} bits")
```

> **Por que o MLP falha no zero-shot?** O `MLPActorCritic` achata todas as features em um vetor de dimensão fixa `V×3 + V×(V-1)/2`. Com V=100, o vetor tem dimensão diferente do input esperado pelo modelo treinado com V=20 — resultando em `RuntimeError`. A GNN não tem essa limitação.

---

## 📐 Baselines Clássicas (branch `baselines`)

O pacote `grams_env/agents/baselines/` implementa dois escalonadores clássicos de telecomunicações como **pontos de referência de desempenho** para os experimentos fatoriais do artigo. Ambos herdam de `BaselineAgent` e são compatíveis com o mesmo loop de avaliação dos agentes DRL (interface `act(obs) → (action, log_prob, value)`).

### Por que baselines clássicas?

- **Pontos de referência estabelecidos**: PF e RR são os escalonadores padrão da indústria 3GPP (TS 36.213) — qualquer agente DRL deve superá-los para justificar a complexidade adicional.
- **Sem dependência de PyTorch**: implementados em Python puro + NumPy, podem ser executados sem GPU e sem o ambiente conda completo.
- **Mesma interface**: `agent.act(obs)` retorna `(action, 0.0, 0.0)`, permitindo uso idêntico no loop de avaliação.

---

### 1. Round Robin — `RoundRobinAgent`

Distribui os K=50 RBs de forma **cíclica e determinística**, avançando o ponteiro de início em 1 a cada TTI:

```
TTI 0: [0, 1, 2, ..., V-1, 0, 1, ...]   (começa no UE 0)
TTI 1: [1, 2, 3, ..., V-1, 0, 1, ...]   (começa no UE 1)
...
TTI V: [0, 1, 2, ...]                    (ponteiro volta ao início)
```

**Propriedades:**
- Fairness perfeita em número de RBs (cada UE recebe exatamente ⌊K/V⌋ ou ⌈K/V⌉ RBs por TTI).
- Agnóstico ao canal — não usa `node_features` ou `adjacency_matrix`.
- Complexidade O(K) por TTI.

```python
from grams_env.agents.baselines import RoundRobinAgent
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

env = OpenRAN_RBA_Env(num_ues=10)
obs, _ = env.reset(seed=42)

agent = RoundRobinAgent(num_rbs=env.num_rbs, num_ues=10)
agent.reset()  # reinicia ponteiro entre episódios

for _ in range(3600):
    action, _, _ = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        agent.reset()
```

---

### 2. Proportional Fair — `ProportionalFairAgent`

Implementa o escalonador **Proportional Fair** clássico (Viswanath et al., 2002; 3GPP TS 36.213), que balanceia eficiência espectral com justiça temporal:

$$
\text{prioridade}_v(t) = \frac{r_v(t)}{\bar{T}_v(t)}
$$

onde:
- $r_v(t)$ é a **taxa instantânea** estimada via CQI: $r_v = \log_2(1 + \text{SINR}_v)$
- $\bar{T}_v(t)$ é o **throughput médio histórico** por EWMA: $\bar{T}_v(t) = \left(1 - \tfrac{1}{\tau}\right)\bar{T}_v(t{-}1) + \tfrac{1}{\tau}\, r_v(t{-}1) \cdot n_{\text{RBs},v}$
- $\tau$ (`window`) controla a memória do escalonador (padrão: 50 TTIs)

**Mecanismo intra-TTI:** ao alocar múltiplos RBs no mesmo TTI, a prioridade do UE já alocado é reduzida pelo fator $1/(1 + n_{\text{alocados}})$, evitando monopolização por um único UE com canal excelente.

```python
from grams_env.agents.baselines import ProportionalFairAgent
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env

env = OpenRAN_RBA_Env(num_ues=10)
obs, _ = env.reset(seed=42)

agent = ProportionalFairAgent(
    num_rbs=env.num_rbs,
    num_ues=10,
    window=50.0,             # τ: janela EWMA (TTIs)
    initial_avg_throughput=1.0,  # valor inicial (evita divisão por zero)
)
agent.reset()  # reinicia histórico EWMA entre episódios

for _ in range(3600):
    action, _, _ = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        agent.reset()
```

| Parâmetro | Default | Descrição |
|---|---|---|
| `num_rbs` | `50` | Número de Resource Blocks (K) |
| `num_ues` | `10` | Número de UEs (V) |
| `window` | `50.0` | Janela τ da EWMA (TTIs). Maior = adaptação mais lenta |
| `initial_avg_throughput` | `1.0` | Throughput médio inicial (bootstrapping) |

---

### Comparação entre os Escalonadores

| Propriedade | Round Robin | Proportional Fair | GNN+PPO |
|---|---|---|---|
| **Usa informação de canal** | ❌ | ✅ (CQI) | ✅ (CQI + grafo) |
| **Fairness temporal** | ✅ Perfeita | ✅ Adaptativa | Emergente |
| **Eficiência espectral** | Baixa | Média–Alta | Alta |
| **Generalização zero-shot** | ✅ | ✅ | ✅ (GNN) / ❌ (MLP) |
| **Dependência de PyTorch** | ❌ | ❌ | ✅ |
| **Complexidade por TTI** | O(K) | O(K·V) | O(V² + K) |

---

## 🧪 Executando os Testes

A suíte de testes abrange testes unitários da física/telecom (sem dependências de Gymnasium) e testes de integração end-to-end, incluindo os modelos de IA e as baselines clássicas.

Para rodar todos os testes (exceto os que requerem PyTorch):

```bash
# Testes sem dependência de torch (baselines + ambiente + física)
python -m pytest tests/ \
    --ignore=tests/test_agents/test_actor_critic.py \
    --ignore=tests/test_agents/test_graph_encoder.py \
    --ignore=tests/test_agents/test_rollout_buffer.py \
    --ignore=tests/test_agents/test_ppo_smoke.py \
    -v
```

Com o ambiente conda completo (PyTorch instalado):

```bash
conda run -n grams python -m pytest tests/ -v
```

| Suite | Testes | Cobertura |
|---|---|---|
| `test_propagation.py` | 11 | Path Loss LOS/NLOS, P_LOS, multi-frequência |
| `test_link_budget.py` | 7 | SINR, Shannon Capacity |
| `test_graph_builder.py` | 5 | Construção do grafo |
| `test_gymnasium_env.py` | 11 | Ambiente Gymnasium completo |
| `test_traffic.py` | 11 | Tráfego CBR e Poisson |
| `test_agents/test_baselines.py` | **36** | **Interface, RR cíclico, PF fairness, integração V=1..50** |
| `test_agents/test_graph_encoder.py` | 8 | GATConv shapes V=5..100, zero-shot |
| `test_agents/test_actor_critic.py` | 11 | GNN/MLP act/evaluate + zero-shot proof |
| `test_agents/test_rollout_buffer.py` | 9 | GAE, mini-batches, precondições |
| `test_agents/test_ppo_smoke.py` | 2 | Treinamento PPO end-to-end |
| **Total** | **111** | ✅ **0 failed** |

---

## 👨‍💻 Créditos

**Alex Vidigal Bastos**  
Universidade Federal de São João del-Rei (UFSJ)