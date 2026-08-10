#!/bin/bash
# -----------------------------------------------------------------------------
# Script para preparar o ambiente de treinamento dos modelos GNN e MLP no GRAMS-Env.
# -----------------------------------------------------------------------------

usage() {
    echo "Uso: $0 [-d cpu|cuda] [-e local|sdumont] [-r] [-h]"
    echo ""
    echo "Opções:"
    echo "  -d DEVICE          Device para treinamento: 'cpu' ou 'cuda' (default: cuda)"
    echo "  -e ENVIRONMENT     Ambiente de execução: 'local' ou 'sdumont' (default: sdumont). Caso o ambiente seja 'sdumont', o script irá desativar todos os módulos e carregar o Python."
    echo "  -r                 Recria o ambiente virtual sem perguntar caso .venv já exista"
    echo "  -h                 Mostra esta ajuda"
}

run_step() {
    local description="$1"
    shift

    echo "🛠️ ${description}..."
    if "$@"; then
        echo "✅ ${description} concluído."
    else
        local status=$?
        echo "❌ Falha ao executar: ${description} (código ${status})." >&2
        exit "$status"
    fi
}

confirm_recreate() {
    local answer

    while true; do
        read -r -p "⚠️  O ambiente virtual .venv já existe. Deseja recriá-lo? [s/N] " answer
        case "$answer" in
            [sS]|[sS][iI][mM])
                return 0
                ;;
            ""|[nN]|[nN][aAãÃ][oO])
                return 1
                ;;
            *)
                echo "Responda 's' para sim ou 'n' para não."
                ;;
        esac
    done
}

DEVICE="cuda"
ENVIRONMENT="sdumont"
RECREATE_ENV=false

while getopts ":d:e:rh" opt; do
    case "$opt" in
        d)
            DEVICE="$OPTARG"
            ;;
        e)
            ENVIRONMENT="$OPTARG"
            ;;
        r)
            RECREATE_ENV=true
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

if [ "$ENVIRONMENT" != "local" ] && [ "$ENVIRONMENT" != "sdumont" ]; then
    echo "❌ Ambiente inválido: $ENVIRONMENT. Use 'local' ou 'sdumont'."
    usage
    exit 1
fi

if [ "$DEVICE" != "cpu" ] && [ "$DEVICE" != "cuda" ]; then
    echo "❌ Device inválido: $DEVICE. Use 'cpu' ou 'cuda'."
    usage
    exit 1
fi

if [ "$ENVIRONMENT" == "sdumont" ]; then
    echo "🛠️ Ambiente 'sdumont' selecionado. Desativando módulos e carregando Python."
    run_step "Desativando módulos" module purge
    run_step "Carregando Python" module load python/3.10.16_sequana
fi

if [ -d ".venv" ]; then
    if [ "$RECREATE_ENV" == true ]; then
        echo "⚠️  Ambiente virtual .venv existente será recriado por causa da opção -r."
        run_step "Removendo ambiente virtual existente" rm -rf .venv
    elif confirm_recreate; then
        run_step "Removendo ambiente virtual existente" rm -rf .venv
    else
        echo "ℹ️  Mantendo ambiente virtual existente."
    fi
fi

if [ ! -d ".venv" ]; then
    run_step "Criando ambiente virtual" python3 -m venv .venv
fi

run_step "Ativando ambiente virtual" source .venv/bin/activate

run_step "Atualizando pip" python3 -m pip install --upgrade pip

if [ "$DEVICE" == "cuda" ]; then
    run_step "Instalando dependências do PyTorch para CUDA 12.6" python3 -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
else
    run_step "Instalando dependências do PyTorch para CPU" python3 -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
fi

run_step "Instalando dependências do projeto" python3 -m pip install torch_geometric gymnasium "numpy<2" pytest

run_step "Verificando instalação" python3 -c "import torch, torch_geometric, gymnasium; print('OK'); print(f'torch={torch.__version__}'); print(f'torch_geometric={torch_geometric.__version__}'); print(f'cuda_available={torch.cuda.is_available()}')"
