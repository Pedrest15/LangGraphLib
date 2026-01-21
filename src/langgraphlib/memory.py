"""Gerenciamento de memória de longo prazo para agentes."""

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from pydantic import BaseModel

# =============================================================================
# MemoryManager Classes
# =============================================================================


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

        # Buscar por chave
        prefs = memory.get("preferences")

        # Busca semântica (se store tiver embeddings)
        related = memory.search("preferências do usuário", limit=5)

        # Listar todas
        all_memories = memory.list()

        # Deletar
        memory.delete("preferences")
    """

    def __init__(
        self,
        store: BaseStore,
        user_id: str,
        application: str = "default",
    ) -> None:
        """
        Inicializa o MemoryManager.

        Args:
            store: Instância de BaseStore (InMemoryStore, PostgresStore, etc.)
            user_id: ID do usuário (usado como namespace)
            application: Contexto da aplicação (sub-namespace)
        """
        self._store = store
        self._user_id = user_id
        self._application = application
        self._namespace = (user_id, application)

    @property
    def namespace(self) -> tuple[str, str]:
        """Retorna o namespace atual (user_id, application)."""
        return self._namespace

    @property
    def user_id(self) -> str:
        """Retorna o user_id."""
        return self._user_id

    @property
    def store(self) -> BaseStore:
        """Retorna o store subjacente."""
        return self._store

    def save(self, key: str, value: dict[str, Any]) -> None:
        """
        Salva uma memória.

        Args:
            key: Chave única para a memória
            value: Dados a serem salvos (dict)
        """
        self._store.put(self._namespace, key, value)

    def get(self, key: str) -> dict[str, Any] | None:
        """
        Recupera uma memória por chave.

        Args:
            key: Chave da memória

        Returns:
            Dict com os dados ou None se não encontrado
        """
        item = self._store.get(self._namespace, key)
        return item.value if item else None

    def search(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Busca semântica em memórias.

        Requer que o store tenha sido configurado com embeddings.

        Args:
            query: Texto para busca semântica
            limit: Número máximo de resultados
            filter: Filtros adicionais (opcional)

        Returns:
            Lista de memórias encontradas
        """
        items = self._store.search(
            self._namespace,
            query=query,
            limit=limit,
            filter=filter,
        )
        return [item.value for item in items]

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Lista todas as memórias do namespace.

        Args:
            limit: Número máximo de resultados

        Returns:
            Lista de dicts com {key, value}
        """
        items = self._store.search(self._namespace, limit=limit)
        return [{"key": item.key, "value": item.value} for item in items]

    def delete(self, key: str) -> None:
        """
        Remove uma memória.

        Args:
            key: Chave da memória a remover
        """
        self._store.delete(self._namespace, key)


class AsyncMemoryManager:
    """
    Versão assíncrona do MemoryManager.

    Examples:
        memory = AsyncMemoryManager(store, user_id="user_123")

        await memory.save("preferences", {"theme": "dark"})
        prefs = await memory.get("preferences")
        results = await memory.search("preferências", limit=5)
    """

    def __init__(
        self,
        store: BaseStore,
        user_id: str,
        application: str = "default",
    ) -> None:
        """
        Inicializa o AsyncMemoryManager.

        Args:
            store: Instância de BaseStore
            user_id: ID do usuário (usado como namespace)
            application: Contexto da aplicação (sub-namespace)
        """
        self._store = store
        self._user_id = user_id
        self._application = application
        self._namespace = (user_id, application)

    @property
    def namespace(self) -> tuple[str, str]:
        """Retorna o namespace atual (user_id, application)."""
        return self._namespace

    @property
    def user_id(self) -> str:
        """Retorna o user_id."""
        return self._user_id

    @property
    def store(self) -> BaseStore:
        """Retorna o store subjacente."""
        return self._store

    async def save(self, key: str, value: dict[str, Any]) -> None:
        """Salva uma memória."""
        await self._store.aput(self._namespace, key, value)

    async def get(self, key: str) -> dict[str, Any] | None:
        """Recupera uma memória por chave."""
        item = await self._store.aget(self._namespace, key)
        return item.value if item else None

    async def search(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Busca semântica em memórias."""
        items = await self._store.asearch(
            self._namespace,
            query=query,
            limit=limit,
            filter=filter,
        )
        return [item.value for item in items]

    async def list(self, limit: int = 100) -> list[dict[str, Any]]:
        """Lista todas as memórias do namespace."""
        items = await self._store.asearch(self._namespace, limit=limit)
        return [{"key": item.key, "value": item.value} for item in items]

    async def delete(self, key: str) -> None:
        """Remove uma memória."""
        await self._store.adelete(self._namespace, key)


# =============================================================================
# Tools Síncronas
# =============================================================================


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


# =============================================================================
# Tools Assíncronas
# =============================================================================


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


# =============================================================================
# Nós de Memória Síncronos
# =============================================================================


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

    class ExtractedMemories(BaseModel):
        """Memórias extraídas da conversa."""

        facts: list[str]

    default_prompt = (
        "Analise a conversa e extraia fatos importantes sobre o usuário. "
        "Fatos podem incluir: nome, preferências, profissão, interesses, "
        "contexto relevante. Retorne apenas fatos novos e relevantes. "
        "Se não houver fatos novos, retorne lista vazia."
    )

    prompt = extraction_prompt or default_prompt
    extractor = model.with_structured_output(ExtractedMemories)

    def memory_saver_node(state: Any) -> dict:
        """Extrai e salva memórias da conversa."""
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

        return {}

    return memory_saver_node


def create_memory_retriever_node(
    memory: MemoryManager,
    query_field: str = "messages",
    output_field: str = "memory_context",
    limit: int = 5,
) -> Callable:
    """
    Cria um nó que busca memórias e adiciona ao state como string formatada.

    Args:
        memory: Instância de MemoryManager
        query_field: Campo do state usado como query (default: messages)
        output_field: Campo do state onde salvar o contexto (default: memory_context)
        limit: Número máximo de memórias a buscar

    Returns:
        Função node para usar no Workflow

    Example:
        # State precisa ter o campo output_field como string
        State = create_state(memory_context=(str, ""))

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

    def memory_retriever_node(state: Any) -> dict:
        """Busca memórias relevantes e adiciona ao state."""
        query_value = getattr(state, query_field, None)
        if not query_value:
            return {output_field: ""}

        # Se for messages, usa última mensagem
        if isinstance(query_value, list) and query_value:
            last_msg = query_value[-1]
            query = getattr(last_msg, "content", str(last_msg))
        else:
            query = str(query_value)

        # Busca memórias
        memories = memory.search(query, limit=limit)

        # Formata como string
        if not memories:
            return {output_field: "Nenhuma memória encontrada."}

        formatted = []
        for mem in memories:
            content = mem.get("content", str(mem))
            formatted.append(f"- {content}")

        return {output_field: "\n".join(formatted)}

    return memory_retriever_node


# =============================================================================
# Nós de Memória Assíncronos
# =============================================================================


def create_async_memory_saver_node(
    model: BaseChatModel,
    memory: AsyncMemoryManager,
    extraction_prompt: str | None = None,
) -> Callable:
    """
    Cria um nó assíncrono que extrai e salva memórias automaticamente.

    Args:
        model: Modelo de chat para extração
        memory: Instância de AsyncMemoryManager
        extraction_prompt: Prompt customizado para extração (opcional)

    Returns:
        Função node assíncrona para usar no Workflow
    """

    class ExtractedMemories(BaseModel):
        """Memórias extraídas da conversa."""

        facts: list[str]

    default_prompt = (
        "Analise a conversa e extraia fatos importantes sobre o usuário. "
        "Fatos podem incluir: nome, preferências, profissão, interesses, "
        "contexto relevante. Retorne apenas fatos novos e relevantes. "
        "Se não houver fatos novos, retorne lista vazia."
    )

    prompt = extraction_prompt or default_prompt
    extractor = model.with_structured_output(ExtractedMemories)

    async def async_memory_saver_node(state: Any) -> dict:
        """Extrai e salva memórias da conversa."""
        messages = getattr(state, "messages", [])
        if not messages:
            return {}

        # Extrai fatos
        extraction_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Conversa:\n{messages}"},
        ]
        result = await extractor.ainvoke(extraction_messages)

        # Salva cada fato
        for fact in result.facts:
            key = f"fact_{uuid4().hex[:8]}"
            await memory.save(key, {"content": fact, "source": "auto_extraction"})

        return {}

    return async_memory_saver_node


def create_async_memory_retriever_node(
    memory: AsyncMemoryManager,
    query_field: str = "messages",
    output_field: str = "memory_context",
    limit: int = 5,
) -> Callable:
    """
    Cria um nó assíncrono que busca memórias e adiciona ao state como string.

    Args:
        memory: Instância de AsyncMemoryManager
        query_field: Campo do state usado como query (default: messages)
        output_field: Campo do state onde salvar o contexto (default: memory_context)
        limit: Número máximo de memórias a buscar

    Returns:
        Função node assíncrona para usar no Workflow
    """

    async def async_memory_retriever_node(state: Any) -> dict:
        """Busca memórias relevantes e adiciona ao state."""
        query_value = getattr(state, query_field, None)
        if not query_value:
            return {output_field: ""}

        # Se for messages, usa última mensagem
        if isinstance(query_value, list) and query_value:
            last_msg = query_value[-1]
            query = getattr(last_msg, "content", str(last_msg))
        else:
            query = str(query_value)

        # Busca memórias
        memories = await memory.search(query, limit=limit)

        # Formata como string
        if not memories:
            return {output_field: "Nenhuma memória encontrada."}

        formatted = []
        for mem in memories:
            content = mem.get("content", str(mem))
            formatted.append(f"- {content}")

        return {output_field: "\n".join(formatted)}

    return async_memory_retriever_node
