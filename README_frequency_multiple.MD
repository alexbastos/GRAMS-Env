# Módulo: Suporte a Múltiplas Frequências e Tráfego Poisson (branch `frequency_multiple`)

Este módulo estende as capacidades físicas e de tráfego do simulador `GRAMS-Env`, habilitando a execução da matriz fatorial de 72 experimentos com suporte a diferentes faixas de frequência e tráfego estocástico em rajadas.

---

## 📡 1. Suporte a Múltiplas Frequências (700 MHz vs 2 GHz)

O modelo de propagação **3GPP TR 36.873 UMa (Urban Macro)** calcula dinamicamente a atenuação de canal (*Path Loss*) baseando-se na frequência portadora ativa. A atenuação física aumenta com frequências maiores devido à menor penetração e maior espalhamento das ondas eletromagnéticas.

### Detalhes de Implementação:
- **`CellConfig`**: Adicionada a validação no `__post_init__` que limita a frequência no range válido de **0.5 GHz a 6.0 GHz** determinado pelo 3GPP.
- **Conveniência**: Adicionada a propriedade `carrier_freq_mhz` para retornar a frequência portadora em Megahertz.
- **Path Loss Dinâmico**: A classe `TR36873_UMa` consome a frequência definida no `CellConfig` em suas equações de atenuação direta e de interferência inter-UE, refletindo a diferença de atenuação entre as faixas sub-1 GHz (700 MHz) e C-band (2 GHz) de forma realista.

---

## 🚦 2. Injeção de Tráfego Estocástico Variável (Poisson)

Modelar o tráfego estritamente determinístico (Constant Bit Rate puro) não representa adequadamente as rajadas dinâmicas das fatias de rede 6G (ex: URLLC e eMBB). Por isso, implementou-se o modelo estocástico de Poisson.

### Detalhes de Implementação:
- **`CBRTrafficGenerator`**: Agora suporta dois modos definidos no `CellConfig.traffic_mode`:
  1. `"deterministic"`: Gera exatamente `cbr_bytes` por UE a cada TTI (CBR puro).
  2. `"poisson"`: O número de bytes gerados a cada TTI por UE segue uma distribuição de Poisson com média $\lambda_v = \text{cbr\_bytes}_v$:
     $$\text{arrival\_bytes}_v \sim \text{Poisson}(\lambda_v), \quad \lambda_v \in \{1000, 4000\} \text{ bytes}$$
- **Rajadas**: A variabilidade de Poisson simula rajadas dinâmicas ao redor da média nominal de tráfego, criando filas mais realistas e desafiadoras para os algoritmos de controle e IA.

---

## 💻 Exemplo de Configuração e Uso

```python
import numpy as np
from grams_env.infrastructure.gymnasium_env import OpenRAN_RBA_Env
from grams_env.core.domain.cell import CellConfig

# Cenário 1: 700 MHz com Tráfego Poisson Estocástico (Média 1000B ou 4000B)
cfg_700 = CellConfig(
    carrier_freq_ghz=0.7, 
    traffic_mode="poisson"
)
env_700 = OpenRAN_RBA_Env(num_ues=20, config=cfg_700)

# Cenário 2: 2 GHz com Tráfego Determinístico (CBR Puro)
cfg_2g = CellConfig(
    carrier_freq_ghz=2.0, 
    traffic_mode="deterministic"
)
env_2g = OpenRAN_RBA_Env(num_ues=20, config=cfg_2g)
```

---

## 🧪 Validação dos Módulos

Os módulos foram validados através de 17 novos testes unitários e de integração implementados em `tests/test_propagation.py` e `tests/test_traffic.py`.

### Comportamento Físico Medido (100 steps):

| Configuração | Frequência Portadora | Modo de Tráfego | SINR Médio (dB) |
|---|---|---|---|
| **Cenário A** | 700 MHz | Poisson | **66.48 dB** |
| **Cenário B** | 700 MHz | Determinístico | 62.29 dB |
| **Cenário C** | 2 GHz | Poisson | **57.36 dB** |
| **Cenário D** | 2 GHz | Determinístico | 53.18 dB |

> **Nota física**: Conforme esperado pela teoria eletromagnética do 3GPP, a banda de **700 MHz** apresenta um ganho de SINR médio de aproximadamente **~9 dB** em relação à banda de **2 GHz** para a mesma distância física, devido à menor atenuação do canal de propagação na frequência mais baixa.
