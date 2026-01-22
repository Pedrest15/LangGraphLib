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
    Long-term memory manager with simplified API (sync).

    Abstracts the use of namespaces and Store operations.

    Examples:
        from langgraph.store.memory import InMemoryStore
        from langgraphlib.memory import MemoryManager

        store = InMemoryStore()
        memory = MemoryManager(store, user_id="user_123")

        # Save
        memory.save("preferences", {"theme": "dark"})

        # Get by key
        prefs = memory.get("preferences")

        # Semantic search (if store has embeddings)
        related = memory.search("user preferences", limit=5)

        # List all
        all_memories = memory.list()

        # Delete
        memory.delete("preferences")
    """

    def __init__(
        self,
        store: BaseStore,
        user_id: str,
        application: str = "default",
    ) -> None:
        """
        Initializes the MemoryManager.

        Args:
            store: BaseStore instance (InMemoryStore, PostgresStore, etc.)
            user_id: User ID (used as namespace)
            application: Application context (sub-namespace)
        """
        self._store = store
        self._user_id = user_id
        self._application = application
        self._namespace = (user_id, application)

    @property
    def namespace(self) -> tuple[str, str]:
        """Returns the current namespace (user_id, application)."""
        return self._namespace

    @property
    def user_id(self) -> str:
        """Returns the user_id."""
        return self._user_id

    @property
    def store(self) -> BaseStore:
        """Returns the underlying store."""
        return self._store

    def save(self, key: str, value: dict[str, Any]) -> None:
        """
        Saves a memory.

        Args:
            key: Unique key for the memory
            value: Data to be saved (dict)
        """
        self._store.put(self._namespace, key, value)

    def get(self, key: str) -> dict[str, Any] | None:
        """
        Retrieves a memory by key.

        Args:
            key: Memory key

        Returns:
            Dict with the data or None if not found
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
        Semantic search in memories.

        Requires that the store was configured with embeddings.

        Args:
            query: Text for semantic search
            limit: Maximum number of results
            filter: Additional filters (optional)

        Returns:
            List of found memories
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
        Lists all memories in the namespace.

        Args:
            limit: Maximum number of results

        Returns:
            List of dicts with {key, value}
        """
        items = self._store.search(self._namespace, limit=limit)
        return [{"key": item.key, "value": item.value} for item in items]

    def delete(self, key: str) -> None:
        """
        Removes a memory.

        Args:
            key: Key of the memory to remove
        """
        self._store.delete(self._namespace, key)


class AsyncMemoryManager:
    """
    Asynchronous version of MemoryManager.

    Examples:
        memory = AsyncMemoryManager(store, user_id="user_123")

        await memory.save("preferences", {"theme": "dark"})
        prefs = await memory.get("preferences")
        results = await memory.search("preferences", limit=5)
    """

    def __init__(
        self,
        store: BaseStore,
        user_id: str,
        application: str = "default",
    ) -> None:
        """
        Initializes the AsyncMemoryManager.

        Args:
            store: BaseStore instance
            user_id: User ID (used as namespace)
            application: Application context (sub-namespace)
        """
        self._store = store
        self._user_id = user_id
        self._application = application
        self._namespace = (user_id, application)

    @property
    def namespace(self) -> tuple[str, str]:
        """Returns the current namespace (user_id, application)."""
        return self._namespace

    @property
    def user_id(self) -> str:
        """Returns the user_id."""
        return self._user_id

    @property
    def store(self) -> BaseStore:
        """Returns the underlying store."""
        return self._store

    async def save(self, key: str, value: dict[str, Any]) -> None:
        """Saves a memory."""
        await self._store.aput(self._namespace, key, value)

    async def get(self, key: str) -> dict[str, Any] | None:
        """Retrieves a memory by key."""
        item = await self._store.aget(self._namespace, key)
        return item.value if item else None

    async def search(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search in memories."""
        items = await self._store.asearch(
            self._namespace,
            query=query,
            limit=limit,
            filter=filter,
        )
        return [item.value for item in items]

    async def list(self, limit: int = 100) -> list[dict[str, Any]]:
        """Lists all memories in the namespace."""
        items = await self._store.asearch(self._namespace, limit=limit)
        return [{"key": item.key, "value": item.value} for item in items]

    async def delete(self, key: str) -> None:
        """Removes a memory."""
        await self._store.adelete(self._namespace, key)


# =============================================================================
# Synchronous Tools
# =============================================================================


def create_remember_tool(memory: MemoryManager) -> Callable:
    """
    Creates a tool for saving memories.

    Args:
        memory: MemoryManager instance

    Returns:
        'remember' tool for use with Agent

    Example:
        memory = MemoryManager(store, user_id="user_123")
        remember = create_remember_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[remember])
    """

    @tool
    def remember(fact: str, category: str = "general") -> str:
        """Saves important information about the user to remember later.

        Args:
            fact: The fact or information to be saved
            category: Category of the information (e.g.: preference, fact, context)
        """
        key = f"{category}_{uuid4().hex[:8]}"
        memory.save(key, {"content": fact, "category": category})
        return f"Memory saved: {fact}"

    return remember


def create_recall_tool(memory: MemoryManager) -> Callable:
    """
    Creates a tool for searching memories.

    Args:
        memory: MemoryManager instance

    Returns:
        'recall' tool for use with Agent

    Example:
        memory = MemoryManager(store, user_id="user_123")
        recall = create_recall_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[recall])
    """

    @tool
    def recall(query: str, limit: int = 5) -> str:
        """Searches saved information about the user.

        Args:
            query: What to search in memories
            limit: Maximum number of results
        """
        memories = memory.search(query, limit=limit)
        if not memories:
            return "No memories found."

        results = []
        for mem in memories:
            content = mem.get("content", str(mem))
            category = mem.get("category", "general")
            results.append(f"[{category}] {content}")

        return "\n".join(results)

    return recall


def create_memory_tools(memory: MemoryManager) -> list[Callable]:
    """
    Creates both memory tools (remember + recall).

    Shortcut to create both tools at once.

    Args:
        memory: MemoryManager instance

    Returns:
        List with [remember, recall]

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
# Asynchronous Tools
# =============================================================================


def create_async_remember_tool(memory: AsyncMemoryManager) -> Callable:
    """
    Creates an asynchronous tool for saving memories.

    Args:
        memory: AsyncMemoryManager instance

    Returns:
        Asynchronous 'remember' tool

    Example:
        memory = AsyncMemoryManager(store, user_id="user_123")
        remember = create_async_remember_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[remember])
    """

    @tool
    async def remember(fact: str, category: str = "general") -> str:
        """Saves important information about the user to remember later.

        Args:
            fact: The fact or information to be saved
            category: Category of the information (e.g.: preference, fact, context)
        """
        key = f"{category}_{uuid4().hex[:8]}"
        await memory.save(key, {"content": fact, "category": category})
        return f"Memory saved: {fact}"

    return remember


def create_async_recall_tool(memory: AsyncMemoryManager) -> Callable:
    """
    Creates an asynchronous tool for searching memories.

    Args:
        memory: AsyncMemoryManager instance

    Returns:
        Asynchronous 'recall' tool

    Example:
        memory = AsyncMemoryManager(store, user_id="user_123")
        recall = create_async_recall_tool(memory)

        agent = Agent(model=model, name="assistant", tools=[recall])
    """

    @tool
    async def recall(query: str, limit: int = 5) -> str:
        """Searches saved information about the user.

        Args:
            query: What to search in memories
            limit: Maximum number of results
        """
        memories = await memory.search(query, limit=limit)
        if not memories:
            return "No memories found."

        results = []
        for mem in memories:
            content = mem.get("content", str(mem))
            category = mem.get("category", "general")
            results.append(f"[{category}] {content}")

        return "\n".join(results)

    return recall


def create_async_memory_tools(memory: AsyncMemoryManager) -> list[Callable]:
    """
    Creates both asynchronous memory tools (remember + recall).

    Args:
        memory: AsyncMemoryManager instance

    Returns:
        List with asynchronous [remember, recall]
    """
    return [
        create_async_remember_tool(memory),
        create_async_recall_tool(memory),
    ]


# =============================================================================
# Synchronous Memory Nodes
# =============================================================================


def create_memory_saver_node(
    model: BaseChatModel,
    memory: MemoryManager,
    extraction_prompt: str | None = None,
) -> Callable:
    """
    Creates a node that extracts and saves memories automatically.

    The node uses the LLM to analyze the conversation and extract relevant facts.

    Args:
        model: Chat model for extraction
        memory: MemoryManager instance
        extraction_prompt: Custom extraction prompt (optional)

    Returns:
        Node function for use in Workflow

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
        """Memories extracted from the conversation."""

        facts: list[str]

    default_prompt = (
        "Analyze the conversation and extract important facts about the user. "
        "Facts may include: name, preferences, profession, interests, "
        "relevant context. Return only new and relevant facts. "
        "If there are no new facts, return an empty list."
    )

    prompt = extraction_prompt or default_prompt
    extractor = model.with_structured_output(ExtractedMemories)

    def memory_saver_node(state: Any) -> dict:
        """Extracts and saves memories from the conversation."""
        messages = getattr(state, "messages", [])
        if not messages:
            return {}

        # Extract facts
        extraction_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Conversation:\n{messages}"},
        ]
        result = extractor.invoke(extraction_messages)

        # Save each fact
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
    Creates a node that searches memories and adds to state as formatted string.

    Args:
        memory: MemoryManager instance
        query_field: State field used as query (default: messages)
        output_field: State field to save context (default: memory_context)
        limit: Maximum number of memories to retrieve

    Returns:
        Node function for use in Workflow

    Example:
        # State needs to have the output_field as string
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
        """Searches relevant memories and adds to state."""
        query_value = getattr(state, query_field, None)
        if not query_value:
            return {output_field: ""}

        # If messages, use last message
        if isinstance(query_value, list) and query_value:
            last_msg = query_value[-1]
            query = getattr(last_msg, "content", str(last_msg))
        else:
            query = str(query_value)

        # Search memories
        memories = memory.search(query, limit=limit)

        # Format as string
        if not memories:
            return {output_field: "No memories found."}

        formatted = []
        for mem in memories:
            content = mem.get("content", str(mem))
            formatted.append(f"- {content}")

        return {output_field: "\n".join(formatted)}

    return memory_retriever_node


# =============================================================================
# Asynchronous Memory Nodes
# =============================================================================


def create_async_memory_saver_node(
    model: BaseChatModel,
    memory: AsyncMemoryManager,
    extraction_prompt: str | None = None,
) -> Callable:
    """
    Creates an asynchronous node that extracts and saves memories automatically.

    Args:
        model: Chat model for extraction
        memory: AsyncMemoryManager instance
        extraction_prompt: Custom extraction prompt (optional)

    Returns:
        Asynchronous node function for use in Workflow
    """

    class ExtractedMemories(BaseModel):
        """Memories extracted from the conversation."""

        facts: list[str]

    default_prompt = (
        "Analyze the conversation and extract important facts about the user. "
        "Facts may include: name, preferences, profession, interests, "
        "relevant context. Return only new and relevant facts. "
        "If there are no new facts, return an empty list."
    )

    prompt = extraction_prompt or default_prompt
    extractor = model.with_structured_output(ExtractedMemories)

    async def async_memory_saver_node(state: Any) -> dict:
        """Extracts and saves memories from the conversation."""
        messages = getattr(state, "messages", [])
        if not messages:
            return {}

        # Extract facts
        extraction_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Conversation:\n{messages}"},
        ]
        result = await extractor.ainvoke(extraction_messages)

        # Save each fact
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
    Creates an asynchronous node that searches memories and adds to state as string.

    Args:
        memory: AsyncMemoryManager instance
        query_field: State field used as query (default: messages)
        output_field: State field to save context (default: memory_context)
        limit: Maximum number of memories to retrieve

    Returns:
        Asynchronous node function for use in Workflow
    """

    async def async_memory_retriever_node(state: Any) -> dict:
        """Searches relevant memories and adds to state."""
        query_value = getattr(state, query_field, None)
        if not query_value:
            return {output_field: ""}

        # If messages, use last message
        if isinstance(query_value, list) and query_value:
            last_msg = query_value[-1]
            query = getattr(last_msg, "content", str(last_msg))
        else:
            query = str(query_value)

        # Search memories
        memories = await memory.search(query, limit=limit)

        # Format as string
        if not memories:
            return {output_field: "No memories found."}

        formatted = []
        for mem in memories:
            content = mem.get("content", str(mem))
            formatted.append(f"- {content}")

        return {output_field: "\n".join(formatted)}

    return async_memory_retriever_node
