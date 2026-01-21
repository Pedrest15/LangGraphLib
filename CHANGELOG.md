# Changelog

## [0.1.0] - 2026-01-06

### Adicionado

- **Módulo `memory.py` - Memória de longo prazo**
  - `MemoryManager`: Gerenciador sync de memória com API simplificada
  - `AsyncMemoryManager`: Versão assíncrona
  - Métodos: `save()`, `get()`, `search()`, `list()`, `delete()`
  - Suporte a namespaces por `user_id` e `application`
  - Busca semântica (quando store tem embeddings)

- **Tools de Memória**
  - `create_remember_tool()`: Tool para salvar memórias (sync)
  - `create_recall_tool()`: Tool para buscar memórias (sync)
  - `create_memory_tools()`: Cria ambas (atalho)
  - `create_async_remember_tool()`: Versão async
  - `create_async_recall_tool()`: Versão async
  - `create_async_memory_tools()`: Versão async

- **Nós de Memória para Workflow**
  - `create_memory_saver_node()`: Extrai e salva memórias automaticamente via LLM
  - `create_memory_retriever_node()`: Busca memórias e injeta no state
  - `create_async_memory_saver_node()`: Versão async
  - `create_async_memory_retriever_node()`: Versão async

- **Exports públicos em `__init__.py`**
  - Todos os componentes principais agora exportados

- **Carregamento automático de variáveis de ambiente**
  - `python-dotenv>=1.0` como dependência
  - `load_dotenv()` automático em `model.py`
  - API keys podem ser definidas via arquivo `.env`

- **Testes de cenários (`tests/test_scenarios.py`)**
  - Cenário 1: Agente simples com messages
  - Cenário 2: Calculadora com tools (add, sub, mult, div)
  - Cenário 3: Supervisor com subnós de funções (destinations)
  - Cenário 3b: Roteamento via conditional_edges (sem destinations)
  - Cenário 4: Grafo supervisor-escritor-revisor com loop

### Corrigido

- **Import de prompts em `agent.py`**
  - Alterado de `langchain.prompts` para `langchain_core.prompts`

- **Conversão de string para AIMessage em structured output**
  - Adicionado método `_process_output_value()` em `Agent`
  - Quando `output_fields` inclui `messages` e o valor é string, converte para `[AIMessage(content=value)]`
  - Corrige `TypeError: can only concatenate list (not "str") to list` com reducer `add_messages`

- **Aviso "wrote to unknown channel branch:to:end"**
  - Adicionado método `_resolve_goto()` em `Agent`
  - Converte string "end" para constante `END` do LangGraph
  - Aplicado em `invoke()` e `ainvoke()`

---

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
│   ├── memory.py         ✅ Implementado
│   └── py.typed
├── tests/
│   └── test_scenarios.py ✅ 5 cenários
├── docs/
│   └── MEMORY_PLAN.md    📋 Plano de memória
├── pyproject.toml
├── CLAUDE.md
├── CHANGELOG.md
└── README.md
```

### Próximos passos

- [ ] `tools.py` - Decorators e registro de tools
- [ ] `hierarchy.py` - Supervisor e times de agentes
- [ ] Testes unitários para memory.py