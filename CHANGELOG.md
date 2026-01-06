# Changelog

## [0.1.0] - 2026-01-06

### Adicionado

- **Configuração inicial do projeto**
  - Inicialização com `uv init --lib`
  - Configuração do `pyproject.toml` com Python 3.13+
  - Dependências: `langgraph>=0.2`, `langchain-core>=0.3`, `pydantic>=2.0`
  - Dev dependencies: `ruff>=0.8`, `pytest>=8.0`
  - Configuração completa do Ruff (linter/formatter)

- **Módulo `state.py`**
  - `BaseState`: Classe base Pydantic com campo `messages` e reducer
  - `create_state()`: Factory function para criar estados dinâmicos
    - Suporte a campos personalizados via `**fields`
    - Parâmetro `include_messages` para incluir/excluir messages
    - Campos podem ser passados como `tipo` ou `(tipo, default)`
  - `add_messages()`: Reducer para concatenar mensagens
  - `MessagesState`: Estado pré-definido só com messages

- **Documentação**
  - `CLAUDE.md`: Instruções do projeto e convenções de código

### Estrutura atual

```
LangGraphLib/
├── src/langgraphlib/
│   ├── __init__.py
│   ├── state.py          ✅ Implementado
│   └── py.typed
├── pyproject.toml
├── CLAUDE.md
├── CHANGELOG.md
└── README.md
```

### Próximos passos

- [ ] `agent.py` - Classes base de agentes
- [ ] `tools.py` - Decorators e registro de tools
- [ ] `memory.py` - Abstrações de Checkpointer e Store
- [ ] `hierarchy.py` - Supervisor e times de agentes
- [ ] `config.py` - Configurações e model providers