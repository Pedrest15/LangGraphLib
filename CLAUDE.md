# LangGraphLib - Instruções do Projeto

## Objetivo

Biblioteca Python que abstrai o uso do LangGraph para criação de agentes de IA, incluindo:

- Agentes com tools
- Agentes com memória
- Hierarquização de agentes

## Estrutura do Projeto

O projeto segue uma arquitetura modular com múltiplos arquivos:

```
langgraphlib/
├── __init__.py
├── agent.py          # Classes base de agentes
├── state.py          # Gerenciamento de estado
├── tools.py          # Definição e registro de tools
├── memory.py         # Implementação de memória
├── hierarchy.py      # Hierarquização de agentes
└── utils.py          # Utilitários gerais
```

## Infraestrutura

- **Gerenciador de pacotes**: `uv`
- **Linter/Formatter**: `ruff`
- **Lint**: Configurado via `ruff`

## Comandos Úteis

```bash
# Instalar dependências
uv sync

# Executar linter
uv run ruff check .

# Formatar código
uv run ruff format .

# Corrigir problemas automaticamente
uv run ruff check --fix .
```

## Convenções

- Código em Python 3.10+
- Type hints obrigatórios, usando tipos nativos do Python (`list`, `dict`, `tuple`, `set`, `type | None`) ao invés de `typing.List`, `typing.Dict`, `typing.Optional`, etc.
- Docstrings em português ou inglês (consistente por módulo)
- Testes em `tests/`