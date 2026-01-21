# Plano de Implementação de Memória - LangGraphLib

## 1. Visão Geral

O LangGraph possui dois tipos de memória:

| Tipo | Descrição | Escopo | Uso |
|------|-----------|--------|-----|
| **Short-term (Checkpointer)** | Estado do grafo em cada step | Por thread (`thread_id`) | Conversas multi-turn, human-in-the-loop, time-travel |
| **Long-term (Store)** | Dados persistentes do usuário/app | Cross-thread (`user_id`, namespaces) | Preferências do usuário, histórico de interações, memória semântica |

---

## 2. Decisões de Design

### 2.1 Checkpointer

**Decisão**: O `Workflow` recebe diretamente `BaseCheckpointSaver`. A escolha do backend (Postgres, Redis, SQLite, etc.) fica por conta do usuário da API.

**Justificativa**:
- Simplicidade - não precisamos criar wrappers
- Flexibilidade - usuário usa qualquer checkpointer compatível
- Já funciona assim no código atual

```python
# Usuário instancia o checkpointer que preferir
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# Desenvolvimento
checkpointer = InMemorySaver()

# Produção
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# Passa direto para o Workflow
workflow = Workflow(
    state=State,
    agents=[agent],
    edges=[...],
    checkpointer=checkpointer,  # BaseCheckpointSaver
)
```

### 2.2 Store (Long-term Memory)

**Decisão**: Mesma abordagem - `Workflow` recebe `BaseStore` diretamente.

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

store = InMemoryStore()  # ou PostgresStore.from_conn_string(...)

workflow = Workflow(
    state=State,
    agents=[agent],
    edges=[...],
    checkpointer=checkpointer,
    store=store,  # BaseStore
)
```

---

## 3. Arquitetura Proposta

### 3.1 Estrutura de Arquivos

```
src/langgraphlib/
├── memory.py              # MemoryManager (sync e async)
```

Arquivo único e simples, focado apenas no `MemoryManager`.

### 3.2 Classe MemoryManager

```python
"""Gerenciador de memória de longo prazo."""

from langgraph.store.base import BaseStore


class MemoryManager:
    """
    Gerenciador de memória de longo prazo com API simplificada (sync).

    Abstrai o uso de namespaces e operações do Store.

    Examples:
        from langgraph.store.memory import InMemoryStore
        from langgraphlib.memory import MemoryManager

        store = InMemoryStore()
        memory = MemoryManager(store, user_id="user_123")

        # Salvar
        memory.save("preferences", {"theme": "dark"})

        # Buscar
        prefs = memory.get("preferences")

        # Busca semântica (se store tiver embeddings)
        related = memory.search("preferências do usuário", limit=5)
    """

    def __init__(
        self,
        store: BaseStore,
        user_id: str,
        application: str = "default",
    ):
        self._store = store
        self._namespace = (user_id, application)

    @property
    def namespace(self) -> tuple[str, str]:
        """Retorna o namespace atual."""
        return self._namespace

    def save(self, key: str, value: dict) -> None:
        """Salva uma memória."""
        self._store.put(self._namespace, key, value)

    def get(self, key: str) -> dict | None:
        """Recupera uma memória por chave."""
        item = self._store.get(self._namespace, key)
        return item.value if item else None

    def search(
        self,
        query: str,
        limit: int = 10,
        filter: dict | None = None,
    ) -> list[dict]:
        """Busca semântica em memórias."""
        items = self._store.search(
            self._namespace,
            query=query,
            limit=limit,
            filter=filter,
        )
        return [item.value for item in items]

    def list(self, limit: int = 100) -> list[dict]:
        """Lista todas as memórias do namespace."""
        items = self._store.search(self._namespace, limit=limit)
        return [{"key": item.key, "value": item.value} for item in items]

    def delete(self, key: str) -> None:
        """Remove uma memória."""
        self._store.delete(self._namespace, key)


class AsyncMemoryManager:
    """
    Versão assíncrona do MemoryManager.

    Examples:
        memory = AsyncMemoryManager(store, user_id="user_123")

        await memory.save("preferences", {"theme": "dark"})
        prefs = await memory.get("preferences")
    """

    def __init__(
        self,
        store: BaseStore,
        user_id: str,
        application: str = "default",
    ):
        self._store = store
        self._namespace = (user_id, application)

    @property
    def namespace(self) -> tuple[str, str]:
        """Retorna o namespace atual."""
        return self._namespace

    async def save(self, key: str, value: dict) -> None:
        """Salva uma memória."""
        await self._store.aput(self._namespace, key, value)

    async def get(self, key: str) -> dict | None:
        """Recupera uma memória por chave."""
        item = await self._store.aget(self._namespace, key)
        return item.value if item else None

    async def search(
        self,
        query: str,
        limit: int = 10,
        filter: dict | None = None,
    ) -> list[dict]:
        """Busca semântica em memórias."""
        items = await self._store.asearch(
            self._namespace,
            query=query,
            limit=limit,
            filter=filter,
        )
        return [item.value for item in items]

    async def list(self, limit: int = 100) -> list[dict]:
        """Lista todas as memórias do namespace."""
        items = await self._store.asearch(self._namespace, limit=limit)
        return [{"key": item.key, "value": item.value} for item in items]

    async def delete(self, key: str) -> None:
        """Remove uma memória."""
        await self._store.adelete(self._namespace, key)
```

---

## 4. Fluxo de Memória nos Agentes

### 4.1 Quando Inserir Memórias?

Existem três abordagens:

#### Opção A: Tool dedicada para salvar memórias

O agente decide quando salvar usando uma tool.

```python
from langchain_core.tools import tool

@tool
def save_memory(key: str, content: str, category: str = "general") -> str:
    """Salva uma informação importante sobre o usuário para lembrar depois."""
    # A tool recebe o store via closure ou runtime
    memory.save(key, {"content": content, "category": category})
    return f"Memória '{key}' salva com sucesso."

agent = Agent(
    model=model,
    name="assistant",
    prompt="Você é um assistente. Use save_memory para lembrar informações importantes.",
    tools=[save_memory],
)
```

#### Opção B: Nó dedicado para extração de memórias

Um nó separado no grafo analisa a conversa e extrai memórias.

```python
def extract_memories(state: State, *, store: BaseStore) -> dict:
    """Extrai e salva memórias da conversa."""
    user_id = state.user_id  # ou via config
    memory = MemoryManager(store, user_id)

    # Usa LLM para extrair fatos
    extractor = model.with_structured_output(MemoryExtraction)
    result = extractor.invoke(state.messages)

    for fact in result.facts:
        memory.save(f"fact_{uuid4()}", {"text": fact, "source": "conversation"})

    return {}  # Não modifica state

workflow = Workflow(
    state=State,
    agents=[assistant],
    nodes={"memory_extractor": extract_memories},
    edges=[
        ("start", "assistant"),
        ("assistant", "memory_extractor"),
        ("memory_extractor", "end"),
    ],
    store=store,
)
```

#### Opção C: Hook pós-execução no Agent

O Agent pode ter um callback para salvar memórias após cada execução.

```python
class Agent:
    def __init__(
        self,
        ...
        on_response: Callable[[BaseModel, dict], None] | None = None,
    ):
        self._on_response = on_response

    def invoke(self, state: BaseModel) -> dict | Command:
        result = self._call_model(state)

        if self._on_response:
            self._on_response(state, result)

        return result

# Uso
def save_conversation_memory(state, result, memory):
    # Lógica para decidir o que salvar
    memory.save(...)

agent = Agent(
    model=model,
    name="assistant",
    on_response=lambda s, r: save_conversation_memory(s, r, memory),
)
```

### 4.2 Quando Buscar Memórias?

#### Opção A: No prompt do Agent (automático)

O Agent busca memórias relevantes e injeta no contexto.

```python
class Agent:
    def __init__(
        self,
        ...
        memory: MemoryManager | AsyncMemoryManager | None = None,
        memory_query_field: str = "messages",  # Campo usado para query
        memory_limit: int = 5,
    ):
        self._memory = memory
        self._memory_query_field = memory_query_field
        self._memory_limit = memory_limit

    def _get_memory_context(self, state: BaseModel) -> str:
        """Busca memórias relevantes e formata como contexto."""
        if not self._memory:
            return ""

        # Extrai query do state
        query_value = getattr(state, self._memory_query_field, None)
        if not query_value:
            return ""

        # Se for messages, usa última mensagem
        if isinstance(query_value, list) and query_value:
            query = query_value[-1].content
        else:
            query = str(query_value)

        # Busca memórias
        memories = self._memory.search(query, limit=self._memory_limit)

        if not memories:
            return ""

        # Formata como contexto
        context_parts = ["Informações relevantes sobre o usuário:"]
        for mem in memories:
            context_parts.append(f"- {mem.get('content', mem)}")

        return "\n".join(context_parts)

    def invoke(self, state: BaseModel) -> dict | Command:
        # Busca contexto de memória
        memory_context = self._get_memory_context(state)

        # Injeta no prompt
        if memory_context:
            enhanced_prompt = f"{self._prompt}\n\n{memory_context}"
        else:
            enhanced_prompt = self._prompt

        # Continua execução normal...
```

#### Opção B: Tool dedicada para buscar memórias

O agente decide quando buscar usando uma tool.

```python
@tool
def recall_memory(query: str) -> str:
    """Busca informações salvas sobre o usuário."""
    memories = memory.search(query, limit=5)
    if not memories:
        return "Nenhuma memória encontrada."
    return "\n".join([m.get("content", str(m)) for m in memories])

agent = Agent(
    model=model,
    name="assistant",
    prompt="Use recall_memory para buscar informações sobre o usuário quando relevante.",
    tools=[recall_memory],
)
```

#### Opção C: Nó dedicado para enriquecer contexto

```python
def enrich_with_memories(state: State, *, store: BaseStore, config) -> dict:
    """Busca memórias e adiciona ao state."""
    user_id = config["configurable"]["user_id"]
    memory = MemoryManager(store, user_id)

    query = state.messages[-1].content
    memories = memory.search(query, limit=5)

    return {"memory_context": memories}

workflow = Workflow(
    state=State,  # State precisa ter campo memory_context
    agents=[assistant],
    nodes={"enrich": enrich_with_memories},
    edges=[
        ("start", "enrich"),
        ("enrich", "assistant"),
        ("assistant", "end"),
    ],
)
```

### 4.3 Abordagem: Máxima Flexibilidade

Vamos implementar **duas opções** para cada operação, deixando o usuário escolher:

| Operação | Tool | Nó dedicado |
|----------|------|-------------|
| **Inserção** | `create_remember_tool()` | `create_memory_saver_node()` |
| **Busca** | `create_recall_tool()` | `create_memory_retriever_node()` |
| **Ambos** | `create_memory_tools()` | - |

O usuário escolhe a abordagem que melhor se adapta ao seu caso de uso.

---

## 5. API de Memória

### 5.1 Estrutura de Arquivos

```
src/langgraphlib/
├── memory.py              # MemoryManager, tools e nós
```

### 5.2 Tools de Memória

#### 5.2.1 Tools Síncronas

```python
def create_remember_tool(memory: MemoryManager) -> Callable:
    """
    Cria uma tool para salvar memórias.

    Args:
        memory: Instância de MemoryManager

    Returns:
        Tool 'remember' para usar com Agent

    Example:
        memory = MemoryManager(store, user_id="user_123")
        remember = create_remember_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[remember])
    """

    @tool
    def remember(fact: str, category: str = "general") -> str:
        """Salva uma informação importante sobre o usuário para lembrar depois.

        Args:
            fact: O fato ou informação a ser salvo
            category: Categoria da informação (ex: preference, fact, context)
        """
        from uuid import uuid4
        key = f"{category}_{uuid4().hex[:8]}"
        memory.save(key, {"content": fact, "category": category})
        return f"Memória salva: {fact}"

    return remember


def create_recall_tool(memory: MemoryManager) -> Callable:
    """
    Cria uma tool para buscar memórias.

    Args:
        memory: Instância de MemoryManager

    Returns:
        Tool 'recall' para usar com Agent

    Example:
        memory = MemoryManager(store, user_id="user_123")
        recall = create_recall_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[recall])
    """

    @tool
    def recall(query: str, limit: int = 5) -> str:
        """Busca informações salvas sobre o usuário.

        Args:
            query: O que buscar nas memórias
            limit: Número máximo de resultados
        """
        memories = memory.search(query, limit=limit)
        if not memories:
            return "Nenhuma memória encontrada."

        results = []
        for mem in memories:
            content = mem.get("content", str(mem))
            category = mem.get("category", "general")
            results.append(f"[{category}] {content}")

        return "\n".join(results)

    return recall


def create_memory_tools(memory: MemoryManager) -> list[Callable]:
    """
    Cria ambas as tools de memória (remember + recall).

    Atalho para criar as duas tools de uma vez.

    Args:
        memory: Instância de MemoryManager

    Returns:
        Lista com [remember, recall]

    Example:
        memory = MemoryManager(store, user_id="user_123")
        tools = create_memory_tools(memory)

        agent = Agent(model=model, name="assistant", tools=tools)
    """
    return [
        create_remember_tool(memory),
        create_recall_tool(memory),
    ]
```

#### 5.2.2 Tools Assíncronas

```python
def create_async_remember_tool(memory: AsyncMemoryManager) -> Callable:
    """
    Cria uma tool assíncrona para salvar memórias.

    Args:
        memory: Instância de AsyncMemoryManager

    Returns:
        Tool 'remember' assíncrona

    Example:
        memory = AsyncMemoryManager(store, user_id="user_123")
        remember = create_async_remember_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[remember])
    """

    @tool
    async def remember(fact: str, category: str = "general") -> str:
        """Salva uma informação importante sobre o usuário para lembrar depois.

        Args:
            fact: O fato ou informação a ser salvo
            category: Categoria da informação (ex: preference, fact, context)
        """
        from uuid import uuid4
        key = f"{category}_{uuid4().hex[:8]}"
        await memory.save(key, {"content": fact, "category": category})
        return f"Memória salva: {fact}"

    return remember


def create_async_recall_tool(memory: AsyncMemoryManager) -> Callable:
    """
    Cria uma tool assíncrona para buscar memórias.

    Args:
        memory: Instância de AsyncMemoryManager

    Returns:
        Tool 'recall' assíncrona

    Example:
        memory = AsyncMemoryManager(store, user_id="user_123")
        recall = create_async_recall_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[recall])
    """

    @tool
    async def recall(query: str, limit: int = 5) -> str:
        """Busca informações salvas sobre o usuário.

        Args:
            query: O que buscar nas memórias
            limit: Número máximo de resultados
        """
        memories = await memory.search(query, limit=limit)
        if not memories:
            return "Nenhuma memória encontrada."

        results = []
        for mem in memories:
            content = mem.get("content", str(mem))
            category = mem.get("category", "general")
            results.append(f"[{category}] {content}")

        return "\n".join(results)

    return recall


def create_async_memory_tools(memory: AsyncMemoryManager) -> list[Callable]:
    """
    Cria ambas as tools assíncronas de memória (remember + recall).

    Args:
        memory: Instância de AsyncMemoryManager

    Returns:
        Lista com [remember, recall] assíncronas
    """
    return [
        create_async_remember_tool(memory),
        create_async_recall_tool(memory),
    ]
```

### 5.3 Nós de Memória

```python
def create_memory_saver_node(
    model: BaseChatModel,
    memory: MemoryManager,
    extraction_prompt: str | None = None,
) -> Callable:
    """
    Cria um nó que extrai e salva memórias automaticamente.

    O nó usa o LLM para analisar a conversa e extrair fatos relevantes.

    Args:
        model: Modelo de chat para extração
        memory: Instância de MemoryManager
        extraction_prompt: Prompt customizado para extração (opcional)

    Returns:
        Função node para usar no Workflow

    Example:
        memory_saver = create_memory_saver_node(model, memory)

        workflow = Workflow(
            state=State,
            agents=[assistant],
            nodes={"save_memories": memory_saver},
            edges=[
                ("start", "assistant"),
                ("assistant", "save_memories"),
                ("save_memories", "end"),
            ],
        )
    """
    from pydantic import BaseModel as PydanticBaseModel

    class ExtractedMemories(PydanticBaseModel):
        facts: list[str]

    default_prompt = """Analise a conversa e extraia fatos importantes sobre o usuário.
Fatos podem incluir: nome, preferências, profissão, interesses, contexto relevante.
Retorne apenas fatos novos e relevantes. Se não houver fatos novos, retorne lista vazia."""

    prompt = extraction_prompt or default_prompt
    extractor = model.with_structured_output(ExtractedMemories)

    def memory_saver_node(state) -> dict:
        """Extrai e salva memórias da conversa."""
        from uuid import uuid4

        # Prepara mensagens para extração
        messages = getattr(state, "messages", [])
        if not messages:
            return {}

        # Extrai fatos
        extraction_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Conversa:\n{messages}"},
        ]
        result = extractor.invoke(extraction_messages)

        # Salva cada fato
        for fact in result.facts:
            key = f"fact_{uuid4().hex[:8]}"
            memory.save(key, {"content": fact, "source": "auto_extraction"})

        return {}  # Não modifica state

    return memory_saver_node


def create_memory_retriever_node(
    memory: MemoryManager,
    query_field: str = "messages",
    output_field: str = "memory_context",
    limit: int = 5,
) -> Callable:
    """
    Cria um nó que busca memórias e adiciona ao state.

    Args:
        memory: Instância de MemoryManager
        query_field: Campo do state usado como query (default: messages)
        output_field: Campo do state onde salvar o contexto (default: memory_context)
        limit: Número máximo de memórias a buscar

    Returns:
        Função node para usar no Workflow

    Example:
        # State precisa ter o campo output_field
        State = create_state(memory_context=(list, []))

        memory_retriever = create_memory_retriever_node(memory, limit=5)

        workflow = Workflow(
            state=State,
            agents=[assistant],
            nodes={"retrieve_memories": memory_retriever},
            edges=[
                ("start", "retrieve_memories"),
                ("retrieve_memories", "assistant"),
                ("assistant", "end"),
            ],
        )
    """

    def memory_retriever_node(state) -> dict:
        """Busca memórias relevantes e adiciona ao state."""
        # Extrai query do state
        query_value = getattr(state, query_field, None)
        if not query_value:
            return {output_field: []}

        # Se for messages, usa última mensagem
        if isinstance(query_value, list) and query_value:
            last_msg = query_value[-1]
            query = getattr(last_msg, "content", str(last_msg))
        else:
            query = str(query_value)

        # Busca memórias
        memories = memory.search(query, limit=limit)

        return {output_field: memories}

    return memory_retriever_node


# Versões assíncronas

def create_async_memory_saver_node(
    model: BaseChatModel,
    memory: AsyncMemoryManager,
    extraction_prompt: str | None = None,
) -> Callable:
    """Versão assíncrona de create_memory_saver_node."""
    # ... implementação similar com await
    pass


def create_async_memory_retriever_node(
    memory: AsyncMemoryManager,
    query_field: str = "messages",
    output_field: str = "memory_context",
    limit: int = 5,
) -> Callable:
    """Versão assíncrona de create_memory_retriever_node."""
    # ... implementação similar com await
    pass
```

---

## 6. Exemplos de Uso

### 6.1 Abordagem com Tools (LLM decide)

```python
from langgraph.store.memory import InMemoryStore
from langgraphlib import Agent, Workflow, create_state
from langgraphlib.memory import MemoryManager, create_memory_tools
from langgraphlib.model import get_model

# Setup
store = InMemoryStore()
memory = MemoryManager(store, user_id="user_123")
tools = create_memory_tools(memory)

# Agent com tools de memória
model = get_model("openai/gpt-4o")
agent = Agent(
    model=model,
    name="assistant",
    prompt="""Você é um assistente pessoal.

Use 'remember' para salvar informações importantes sobre o usuário.
Use 'recall' para buscar informações salvas quando precisar.""",
    tools=tools,
)

# Workflow
State = create_state()
workflow = Workflow(
    state=State,
    agents=[agent],
    edges=[
        ("start", "assistant"),
        ("assistant", "assistant_tools", "has_tool_calls"),
        ("assistant", "end", "no_tool_calls"),
        ("assistant_tools", "assistant"),
    ],
)

graph = workflow.compile()
```

### 6.2 Abordagem com Nós (Automático)

```python
from langgraph.store.memory import InMemoryStore
from langgraphlib import Agent, Workflow, create_state
from langgraphlib.memory import (
    MemoryManager,
    create_memory_saver_node,
    create_memory_retriever_node,
)
from langgraphlib.model import get_model

# Setup
store = InMemoryStore()
memory = MemoryManager(store, user_id="user_123")
model = get_model("openai/gpt-4o")

# Nós de memória
memory_retriever = create_memory_retriever_node(memory, limit=5)
memory_saver = create_memory_saver_node(model, memory)

# State com campo para contexto de memória
State = create_state(memory_context=(list, []))

# Agent que recebe contexto de memória
agent = Agent(
    model=model,
    name="assistant",
    prompt="""Você é um assistente pessoal.

Contexto do usuário (memórias salvas):
{memory_context}""",
    input_fields=["messages", "memory_context"],
)

# Workflow com nós de memória
workflow = Workflow(
    state=State,
    agents=[agent],
    nodes={
        "retrieve_memories": memory_retriever,
        "save_memories": memory_saver,
    },
    edges=[
        ("start", "retrieve_memories"),
        ("retrieve_memories", "assistant"),
        ("assistant", "save_memories"),
        ("save_memories", "end"),
    ],
)

graph = workflow.compile()
```

### 6.3 Abordagem Híbrida (Tools + Nó de extração)

```python
# Tools para busca manual
tools = create_memory_tools(memory)

# Nó para extração automática ao final
memory_saver = create_memory_saver_node(model, memory)

agent = Agent(
    model=model,
    name="assistant",
    prompt="Use 'recall' para buscar memórias quando precisar.",
    tools=tools,  # Inclui recall para busca manual
)

workflow = Workflow(
    state=State,
    agents=[agent],
    nodes={"save_memories": memory_saver},
    edges=[
        ("start", "assistant"),
        ("assistant", "assistant_tools", "has_tool_calls"),
        ("assistant", "save_memories", "no_tool_calls"),  # Extrai memórias ao final
        ("assistant_tools", "assistant"),
        ("save_memories", "end"),
    ],
)
```

---

## 7. Integração com Workflow

### 7.1 Sem alterações no Workflow

O `Workflow` **não precisa** receber `store` como parâmetro.

**Motivo**: O `MemoryManager` já encapsula o store internamente. As tools e nós acessam o store via closure, não via injeção do LangGraph.

```python
# O store já está no MemoryManager
store = InMemoryStore()
memory = MemoryManager(store, user_id="user_123")

# Tools/nós usam memory (que contém store)
tools = create_memory_tools(memory)
retriever = create_memory_retriever_node(memory)

# Workflow não precisa saber do store
workflow = Workflow(
    state=State,
    agents=[agent],
    nodes={"retrieve": retriever},
    edges=[...],
    checkpointer=checkpointer,  # Apenas checkpointer
)
```

**Nota**: O `Agent` não terá integração direta com memória. A flexibilidade é alcançada através de tools e nós.

---

## 8. Checklist de Implementação

### 8.1 memory.py

- [ ] Criar `src/langgraphlib/memory.py`

**Classes:**
- [ ] `MemoryManager` (sync)
- [ ] `AsyncMemoryManager` (async)

**Tools Sync:**
- [ ] `create_remember_tool()`
- [ ] `create_recall_tool()`
- [ ] `create_memory_tools()` (atalho para ambas)

**Tools Async:**
- [ ] `create_async_remember_tool()`
- [ ] `create_async_recall_tool()`
- [ ] `create_async_memory_tools()` (atalho para ambas)

**Nós Sync:**
- [ ] `create_memory_saver_node()`
- [ ] `create_memory_retriever_node()`

**Nós Async:**
- [ ] `create_async_memory_saver_node()`
- [ ] `create_async_memory_retriever_node()`

**Qualidade:**
- [ ] Testes unitários
- [ ] Documentação com exemplos

### 8.2 Testes de Integração

- [ ] Testar tools com Workflow
- [ ] Testar nós com Workflow
- [ ] Testar abordagem híbrida

### 8.3 Exports e Documentação

- [ ] Atualizar `__init__.py` com exports
- [ ] Atualizar CHANGELOG
- [ ] Exemplos de uso no README

---

## 8. Referências

- [LangGraph Memory Documentation](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph Long-term Memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
