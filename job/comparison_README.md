# Comparação CPU: sequencial × vetorizado

A partir da raiz do projeto no cluster:

```bash
sbatch job/run_comparison_cpu.srm
# Sobrescrever apenas o número de iterações (default: 100):
sbatch --export=ALL,ITERATIONS=200 job/run_comparison_cpu.srm
```

O job usa um nó, uma task, 42 CPUs, 32 GB e 24 horas na partição
`sequana_cpu`. Carrega `python/3.10.16_sequana`, ativa `.venv` e usa
sempre o caminho absoluto `PROJECT_DIR/.venv/bin/python`. Ajuste
`WORKING_DIR` no `.srm` se o checkout estiver em outro local.
Há uma única chamada `srun`, sem reserva exclusiva.

## Execução direta em uma alocação CPU

Requer Bash >= 4.3, `.venv` preparada e o módulo Python carregado:

```bash
module load python/3.10.16_sequana
bash job/run_comparison_cpu.sh runs/comparison_manual
ITERATIONS=200 bash job/run_comparison_cpu.sh runs/comparison_manual_200
```

`ROOT` é o único argumento obrigatório. Caminhos relativos são resolvidos
contra o diretório de onde o shell foi chamado. O destino deve ser novo:
qualquer destino existente é recusado, sem sobrescrever resultados.
No SLURM, o destino é `runs/comparison_JOBID`.

## Treinos e resultados

O shell inicia exatamente duas lanes em background, concorrentes:

- `sequential`: seeds `42`, `43`, `44`, nessa ordem, com `num_envs=1`.
- `vector`: seeds `42`, `43`, `44` com `num_envs=2`; depois as mesmas
  seeds com `num_envs=8`.

Cada lane executa um treino por vez usando a CLI existente
`grams_env.agents.gnn.train_gnn`: CPU, 20 UEs, rollout de 2048,
`ITERATIONS=100` por padrão e `TORCH_THREADS=4` fixo. As variáveis de
threads BLAS/OpenMP são exportadas com valor 1 antes de iniciar Python.
Os trainers não são modificados.

Diretórios: `ROOT/sequential_seed42`, `ROOT/vector2_seed42`,
`ROOT/vector8_seed42`, e equivalentes para seeds 43 e 44.
Logs das lanes: `ROOT/sequential.log` e `ROOT/vector.log`.

O shell espera ambas as lanes. Falhas retornam status não zero e acionam
limpeza dos grupos de processos (incluindo workers), com TERM seguido de
KILL e `wait` dos filhos. INT, TERM e HUP também acionam essa limpeza.
Resultados parciais são preservados; use um novo ROOT para tentar novamente.

Somente após sucesso dos dois grupos, executa em foreground:

```bash
"$PYTHON_BIN" tests/compare_training_quality.py \
  --run-dir "$ROOT" --seeds 100 101 102 --steps 3600 --torch-threads 4
```

A saída e os erros vão para `ROOT/comparison.log`; uma falha do avaliador
é propagada como status do shell/job. O avaliador é implementado
separadamente e precisa estar disponível com essa CLI antes da execução.

Validação de sintaxe:

```bash
bash -n job/run_comparison_cpu.sh job/run_comparison_cpu.srm
```
