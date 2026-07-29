Fase 1: Baselines & Sanidade] ➔ [Fase 2: Agente GNN+PPO] ➔ [Fase 3: Execução Fatorial] ➔ [Fase 4: Pós-Processamento]

Aqui está a análise completa do estado atual do projeto, os passos práticos para executar os 72 experimentos definidos na sua matriz fatorial e o levantamento detalhado das implementações que faltam no código.

---

### O Que Falta Implementar (Lacunas no Código)

Embora a fundação física e a Clean Architecture do simulador (`GRAMS-Env`) estejam estruturadas, para rodar os **72 experimentos fatoriais** definidos no `results.tex` (Tabela II), ainda faltam alguns componentes lógicos essenciais:

#### 1. Módulos do Simulador - branch **frequency_multiple**

* **Suporte a Múltiplas Frequências (700 MHz vs 2 GHz):** O modelo de propagação 3GPP `TR36873_UMa` precisa aceitar o parâmetro `carrier_frequency` dinamicamente no `reset()` ou na instanciação, pois os parâmetros de *Path Loss* e a atenução mudam conforme a banda.
* **Injeção de Tráfego Estocástico Variável (CBR 1000 Bytes vs 4000 Bytes):** A classe de tráfego precisa ajustar a taxa média do processo de Poisson ($\lambda_v$) para gerar rajadas correspondentes a esses dois volumes de carga.

#### 2. Modelos de Inteligência Artificial e Agente branch **agent_GRAMS**

* **Extrator de Características GNN (PyTorch Geometric):** Criar a rede neural em grafos (GCN/GAT) que recebe a matriz de adjacência $A(t)$ e as *node features* $X(t)$ do `GRAMS-Env` para gerar os *embeddings* $H^{(L)}$.
* **Politica PPO (Policy Network):** Acoplar a GNN ao algoritmo PPO (usando `Stable Baselines3` ou implementação própria via `PyTorch`) para selecionar a alocação dos 50 RBs.
* **Modelo Baseline MLP-DRL:** Implementar um agente PPO padrão usando redes totalmente conectadas (MLP) para servir como comparação direta no artigo (provando que sem GNN a rede não generaliza).

#### 3. Algoritmos de Comparação (Baselines Clássicas) - branch **baselines**

Classes Python puras que herdam da mesma interface do agente e implementam:

* **Proportional Fair (PF):** Aloca RBs balanceando taxa instantânea e throughput médio histórico.
* **Round Robin (RR):** Alocação cíclica determinística.

#### 4. Pipeline de Experimentos e Métricas

* **Exportador de Telemetria/Resultados:** Módulo para calcular e salvar em arquivos CSV/HDF5 as 3 métricas do artigo:
1. Eficiência Espectral do Sistema ($\text{bps/Hz}$).
2. Atraso de Fila no Percentil 95 ($\text{ms}$).
3. Latência de Inferência do RIC ($\text{ms}$).


* **Gerenciador de Sementes (10 Seeds / Intervalo de Confiança 95%):** Script que executa automaticamente cada uma das 72 configurações com 10 sementes aleatórias distintas para garantir validade estatística.

---

### Roteiro Prático: Passos para Realizar os Experimentos

Para sair do estado atual e chegar aos gráficos finais do artigo, dividimos o trabalho em **4 fases lógicas**:

```
[Fase 1: Baselines & Sanidade] ➔ [Fase 2: Agente GNN+PPO] ➔ [Fase 3: Execução Fatorial] ➔ [Fase 4: Pós-Processamento]

```

#### Fase 1: Sanity Check e Agoritmos Clássicos (1-2 dias)

1. **Teste de Sanidade com Agente Aleatório:**
* Rodar um script simples com `env.action_space.sample()` por $3600\text{ s}$ de simulação.
* Validar se não ocorrem exceções, se as filas não estouram em $V=1$ e se os valores de SINR estão dentro da faixa esperada.


2. **Implementar Baselines PF e RR:**
* Escrever os algoritmos *Proportional Fair* e *Round Robin* interagindo diretamente com o ambiente `gymnasium`.
* Coletar o *baseline* de desempenho tradicional.



#### Fase 2: Treinamento do Agente GNN + DRL (2-3 dias)

1. **Construção da GNN (`torch_geometric`):**
* Escrever o modelo `GraphEncoder(nn.Module)` com $L=2$ camadas de convolução em grafos.


2. **Treinamento no Cenário Base ($V=20$ UEs):**
* Treinar o agente PPO exclusivamente no cenário esparso ($20$ UEs) até a curva de recompensa estabilizar.
* Congelar os pesos do modelo treinado (`model_frozen.pt`).



#### Fase 3: Execução da Matriz de Experimentos (72 Configurações × 10 Seeds)

1. **Automação via Runner Script:**
* Criar um script principal `run_experiments.py` que percorre a grade completa:

$$\text{UEs } (1, 10, 50, 100, 150, 200) \times \text{Velocidades } (0, 3, 20) \times \text{CBR } (1000, 4000) \times \text{Frequências } (700\text{M}, 2\text{G})$$




2. **Execução do Protocolo Zero-Shot:**
* Para os cenários densos ($V = 50, 100, 150, 200$), carregar os pesos congelados do agente treinado com $V=20$ (sem re-treinamento) e rodar em modo de inferência pura para medir a generalização *zero-shot*.



#### Fase 4: Pós-Processamento e Geração dos Gráficos (1 dia)

1. Agregar os dados das 10 sementes independentes por configuração.
2. Calcular a média e o intervalo de confiança de 95%.
3. Gerar as figuras no formato `.pdf` para o LaTeX (ex: *Throughput vs UE Density* e *95th Percentile Delay vs Mobility*).

---

Para darmos sequência às implementações pendentes, por onde você gostaria de começar: **criar o script com as baselines clássicas (PF / RR) para testar a sanidade do simulador** ou **escrever a classe PyTorch da GNN + PPO**?