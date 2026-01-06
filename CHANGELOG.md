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

- **Módulo `model.py`**
  - `get_model()`: Obtém modelo de chat no formato `"provider/modelo"`
  - `get_embeddings()`: Obtém modelo de embeddings
  - Suporte a múltiplos providers: OpenAI, Anthropic, Groq, Ollama, Cerebras, Google, Mistral, Cohere, Fireworks, Together
  - Import dinâmico dos providers (só carrega quando usado)
  - Erros amigáveis com instruções de instalação

- **Módulo `agent.py`**
  - `Agent`: Classe para execução de LLMs com state dinâmico
    - `invoke()` / `ainvoke()`: Executa o agente com state fornecido
    - `input_field` / `output_field`: Campos configuráveis do state
    - Suporte a `prompt` via `ChatPromptTemplate`
    - Suporte a `tools` via `bind_tools()`
    - Retorna dict parcial para atualização do state (compatível com LangGraph)
  - Não executa tools - apenas retorna resposta para o Workflow decidir

- **Documentação**
  - `CLAUDE.md`: Instruções do projeto e convenções de código

### Estrutura atual

```
LangGraphLib/
├── src/langgraphlib/
│   ├── __init__.py
│   ├── state.py          ✅ Implementado
│   ├── model.py          ✅ Implementado
│   ├── agent.py          ✅ Implementado
│   └── py.typed
├── pyproject.toml
├── CLAUDE.md
├── CHANGELOG.md
└── README.md
```

### Próximos passos

- [ ] `workflow.py` - Orquestração de agentes com StateGraph
- [ ] `tools.py` - Decorators e registro de tools
- [ ] `memory.py` - Abstrações de Checkpointer e Store
- [ ] `hierarchy.py` - Supervisor e times de agentes