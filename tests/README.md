# Testes

A suíte usa `pytest`, com testes agrupados por componente e fixtures como
`tmp_path` para artefatos temporários. Execute os comandos a partir da raiz do
repositório, em um ambiente com as dependências do projeto e `pytest` instalados.
Para os testes de agentes são necessários também PyTorch e Gymnasium; a suíte
vetorizada requer uma versão do Gymnasium com `AutoresetMode.SAME_STEP`.

## Normalização de features da GNN

`test_agents/test_node_normalization.py` cobre os valores e limites de canal,
fila e carga, monotonia das filas sem clipping de capacidade, observações
intactas e gradientes finitos. Verifica a entrada normalizada do encoder em
`act`, `evaluate` e `get_value`, além da persistência das escalas e da flag
booleana em checkpoints. Checkpoints legados sem ambos os buffers devem
desabilitar a normalização e preservar logits e valor do modelo original.

```sh
python -m pytest tests/test_agents/test_node_normalization.py -v
```

## Comparação de qualidade de checkpoints GNN (CLI standalone)

`compare_training_quality.py` não é um teste pytest e não treina modelos. Execute:

```sh
python tests/compare_training_quality.py --help
.venv/bin/python tests/compare_training_quality.py --run-dir runs/comparacao --seeds 100 101 102 --steps 3600 --torch-threads 4
```

A raiz deve conter `sequential_seed42/43/44`, `vector2_seed42/43/44` e
`vector8_seed42/43/44` (cada combinação é um diretório separado), todos com
`model_gnn_frozen.pt`. Os números de seeds de treino são descobertos, não fixos:
exige-se a mesma coleção não vazia nos três grupos. O glob é imediato, não
recursivo, com nomes estritos `<grupo>_seed<inteiro>`; nomes malformados no
prefixo de um grupo, runs ou checkpoints ausentes causam erro. Não se usa a
interseção para esconder runs incompletos. Uma seed ausente dos três grupos não
pode ser inferida a partir do filesystem. Outros nomes de diretório são ignorados.

A arquitetura é a default de `train_gnn` (3 features, hidden 64, 2 camadas,
4 heads), carregada estritamente em CPU com `weights_only=True`. Usa OpenRAN
com 20 UEs e `CellConfig` default de treino, incluindo tráfego Poisson;
apenas o horizonte é definido por `--steps` (default 3600). Todos os modelos
são validados antes dos episódios. `--help` não importa as dependências ML.

Cada checkpoint é avaliado serialmente nas mesmas seeds de avaliação, com
`set_seed(seed)` e `env.reset(seed=seed)` a cada episódio, `eval()` e
`torch.inference_mode()`. Ações são **amostradas**, como em `policy.act`, não
argmax. O episódio para em término/truncamento, sem autoreset. Threads intra-op
são 4 por padrão e inter-op é 1.

Arquivos produzidos na raiz (substituídos em uma nova execução bem-sucedida;
checkpoints nunca são escritos):

- `quality_episodes.csv`: uma linha por grupo × seed de treino × seed de
  avaliação; passos reais, flags finais e métricas.
- `quality_summary.csv`: formato longo, com linhas `level=run` contendo médias
  entre seeds de avaliação para cada checkpoint e `level=group` contendo média
  e desvio-padrão **amostral entre essas médias de seeds de treino** (`ddof=1`).
  `std` fica vazio nas linhas de run e para grupos com uma única seed de treino.
  Seeds de avaliação não contam como réplicas independentes de treinamento.
- `quality_manifest.json`: configurações efetivas, versões, hashes SHA-256 dos
  checkpoints e evaluator, ordem de execução, threads e protocolo/limitações.

Throughput é a média em bits/TTI e sua conversão para Mbps
`bits_por_TTI / tti_s / 1e6`. A fila usa o P95 do `MetricsCollector`, estimativa
pela Lei de Little sobre todos os UEs e TTIs, não latência medida de pacotes.
Reward é a soma no episódio. Diagnósticos são médias temporais da entropia
categórica dividida por `log(20)`, maior probabilidade de UE, número de UEs
recebendo pelo menos um RB e maior fração de RBs de um UE por TTI. “Ativo” aqui
significa alocado, não necessariamente com fila ou transmissão bem-sucedida.

A latência cronometra `policy.act` (conversão, actor/critic, amostragem e saída),
sem segundo forward; a captura da distribuição acrescenta uma atribuição.
Diagnósticos e `env.step` ficam fora desse intervalo. Não há warmup descartado.
É latência de **avaliação serial**, não speedup de treino. Contenção de CPU com
outros jobs, aquecimento e ordem fixa dos runs afetam timing: execute em máquina
ociosa, não em paralelo com trainers, e não interprete diferenças pequenas como
vantagem intrínseca do checkpoint.

**Limites da comparação:** o trainer sequencial atual zera bootstrap também
no truncamento; o vetorizado usa `V(final_obs)` no truncamento e encerra a
recursão GAE. Há portanto diferença de algoritmo além do paralelismo, e os
checkpoints não permitem confirmar qual versão histórica os gerou. Nomes de
diretório identificam grupos/seeds, mas não provam orçamento, hiperparâmetros
ou código de treino iguais. O avaliador não corrige isso nem estabelece
significância estatística; o resumo não confunde seeds de avaliação com seeds
de treino. Horizontes diferentes também tornam rewards totais incomparáveis.

## PPO vetorizado

`test_agents/test_ppo_vector_trainer.py` especifica o contrato da nova API de
`grams_env.agents.common.ppo_vector_trainer`. Não usa `skip`/`xfail` para esconder
funcionalidades ainda ausentes: enquanto a API não estiver implementada, inclusive
`compute_vector_gae`, a coleta ou execução desses testes pode falhar.

### Por que estes testes existem?

Um treino que termina sem erro não garante que as transições usadas pelo PPO
estejam corretas. Autoreset, truncamentos e memória compartilhada podem gerar
alvos incorretos sem produzir exceções.

- **GAE manual:** verifica números calculados à mão para término verdadeiro,
  truncamento e corte de rollout. O bootstrap já vem zerado no término;
  truncamentos usam `V(observação final)`. Ambos encerram a recursão de GAE,
  mas somente o término elimina o bootstrap. O corte do rollout mantém o
  bootstrap da próxima observação, sem inventar uma vantagem futura.
- **Independência:** alterar recompensas, valores, bootstraps e limites de um
  worker não pode modificar outro; permutar workers apenas permuta resultados.
- **Coleta SAME_STEP:** dois ambientes toy, com horizontes 3 e 5, terminam em
  passos diferentes. Os testes alternam qual termina e qual trunca. Oito passos
  por worker atravessam resets e acabam com episódios ainda em andamento.
  Observações codificam episódio, worker e passo, distinguindo claramente a
  observação final daquela produzida pelo reset.
- **Transições reais:** ações determinísticas são verificadas no próprio ambiente
  e nos buffers, junto com recompensas e observações de cada passo. Isso detecta
  transições artificiais de reset e deslocamentos temporais de `NEXT_STEP`.
  Bootstrap deve usar `get_value()`, sem chamadas extras a `act()`.
- **Snapshots:** buffers reutilizados pelo ambiente e alterações posteriores da
  observação corrente/reset não podem sobrescrever observações armazenadas.
- **Equivalência OpenRAN direto/vetorizado:** compara dois ambientes diretos com
  `AsyncVectorEnv(SAME_STEP, context="spawn")`, seeds iniciais 123/124, ações
  fixas e oito passos com `max_steps=3`. Verifica todas as observações do dict,
  recompensas, `terminated` e `truncated`. Nos passos 3 e 6, compara também
  `final_obs` e sua máscara; depois faz reset manual **sem seed** nos ambientes
  diretos e compara a observação de reset retornada pelo vetor. Assim verifica
  continuidade do RNG e da trajetória através de dois resets por worker, sem
  depender da política ou do trainer.
- **Integração:** GNN pequena em CPU, OpenRAN com `max_steps=3`, dois workers,
  `rollout_steps=16` (8 por worker), uma época e uma iteração. Verifica mudança
  efetiva dos pesos, valores e métricas finitos, checkpoints carregáveis e log.
- **Recursos:** verifica que `train()` fecha seus workers tanto no sucesso quanto
  quando seu reset inicial lança uma exceção. A falha é injetada no processo pai
  para isolar o `try/finally` do trainer, sem depender de como o Gymnasium
  transporta exceções de subprocessos. Não cobre falha durante construção dos
  workers nem falha interna de reset em um worker.

### Contrato exercitado

```python
advantages, returns = compute_vector_gae(
    rewards, values, bootstrap_values, episode_ends, gamma, gae_lambda,
)
```

Entradas temporais e saídas têm formato `(T, N)`. `episode_ends` inclui término
ou truncamento e interrompe apenas a propagação da vantagem:

```text
delta[t] = rewards[t] + gamma * bootstrap_values[t] - values[t]
A[t] = delta[t] + gamma * gae_lambda * (1 - episode_ends[t]) * A[t+1]
returns = advantages + values
```

O trainer aceita `env_fns` opcional, `context="spawn"` por padrão e os argumentos
existentes, exigindo `num_envs >= 2`. As factories toy são definidas no nível do
módulo e parametrizadas com `functools.partial`, compatíveis com `spawn`.
As políticas implementam `act(obs)`, `get_value(obs) -> float` e `evaluate(...)`.
A política toy tem um parâmetro para permitir a construção do otimizador; o
update real é exercitado pela GNN.

Os testes de coleta inicializam explicitamente
`trainer._obs = trainer.envs.reset(seed=...)[0]` antes de `_collect_rollout()` e
inspecionam `_bootstrap_values`, `_dones`, `_advantages`, `_returns`, `_actions`,
`_rewards` e `_obs_node_features`. As vantagens verificadas são as vantagens
brutas de GAE, antes da normalização usada no update PPO.

Os testes usam subprocessos reais, não mocks do `AsyncVectorEnv`. As asserções de
fechamento ocorrem **antes** do `finally` defensivo do teste, para não mascarar
vazamentos do trainer. Checkpoints e logs ficam exclusivamente em `tmp_path`.

### Comandos

Somente cálculo de GAE, sem iniciar workers:

```sh
python -m pytest tests/test_agents/test_ppo_vector_trainer.py::TestVectorGAE -v
```

Coleta toy, validação de quantidade de ambientes e cleanup em erro:

```sh
python -m pytest tests/test_agents/test_ppo_vector_trainer.py::TestVectorRollout -v
```

Equivalência OpenRAN direto vs. vetorizado, sem treino:

```sh
python -m pytest tests/test_agents/test_ppo_vector_trainer.py::TestOpenRANVectorEquivalence -v
```

Integração OpenRAN + GNN:

```sh
python -m pytest tests/test_agents/test_ppo_vector_trainer.py::TestVectorPPOIntegration -v
```

Arquivo completo, testes dos agentes e suíte completa, respectivamente:

```sh
python -m pytest tests/test_agents/test_ppo_vector_trainer.py -v
python -m pytest tests/test_agents -v
python -m pytest tests -v
```

Se usar o ambiente virtual local, substitua `python` por `.venv/bin/python`.
Execute inicialmente sem paralelização adicional do pytest: cada teste
vetorizado já cria dois subprocessos. Para limitar threads numéricas em máquinas
com muitos núcleos:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_agents/test_ppo_vector_trainer.py -v
```

Os oito testes originais passaram na execução reportada pelo usuário. O teste de
equivalência foi adicionado depois e não foi executado durante esta alteração,
conforme solicitado.
