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
    - `input_fields` / `output_fields`: Campos configuráveis do state
      - Suporte a campo único (`str`) ou múltiplos campos (`list[str]`)
      - Exemplo: `input_fields=["context", "question"]`
    - Suporte a `prompt` via `ChatPromptTemplate`
    - Suporte a `tools` via `bind_tools()` - retorna `Command` para `{name}_tools`
    - Suporte a `destinations` para roteamento dinâmico via structured output
      - LLM decide próximo nó entre destinos válidos (Literal)
      - Retorna `Command(goto=destino, update={...})`
      - Schema dinâmico inclui todos os `output_fields` + `goto`
      - Uma única chamada ao LLM (sem dupla chamada)
    - Retorna dict ou `Command` dependendo da configuração

- **Módulo `workflow.py`**
  - `Workflow`: Orquestração de agentes em grafo LangGraph
    - API simplificada com edges como tuplas de strings
    - Edges fixas: `("start", "agent")`, `("agent", "end")`
    - Edges condicionais com strings built-in ou callables:
      - `("agent", "agent_tools", "has_tool_calls")`
      - `("agent", "end", lambda s: s.is_done)`
    - Conversão automática de "start"/"end" para `START`/`END`
    - Criação automática de `ToolNode` com padrão `{agent_name}_tools`
    - Parâmetro `nodes`: dict de nós customizados (funções)
      - Permite adicionar funções como nós do grafo
      - Exemplo: `nodes={"formatter": format_output}`
    - Parâmetro `mode`: escolha entre execução `"sync"` ou `"async"`
    - Suporte a `checkpointer` para persistência de estado
    - `compile()`: Compila o workflow em `CompiledStateGraph`
    - `get_image()`: Retorna imagem do grafo em base64
    - `graph` property: Acesso ao grafo compilado

- **Módulo `edge.py`**
  - `Condition`: Type alias para condições (string ou callable)
  - `Edge`: Type alias para edges (tupla de 2 ou 3 elementos)

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
│   ├── workflow.py       ✅ Implementado
│   ├── edge.py           ✅ Implementado
│   └── py.typed
├── pyproject.toml
├── CLAUDE.md
├── CHANGELOG.md
└── README.md
```

### Próximos passos

- [ ] `tools.py` - Decorators e registro de tools
- [ ] `memory.py` - Abstrações de Checkpointer e Store
- [ ] `hierarchy.py` - Supervisor e times de agentes
- [ ] Atualizar `__init__.py` com exports públicos
- [ ] Testes unitários