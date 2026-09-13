from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

# Estado aceito pelo nó: modelo pydantic (create_state) ou dict bruto do LangGraph
ToolInput = BaseModel | dict[str, Any]

# Espelha o tipo aceito por ToolNode(handle_tool_errors=...)
ToolErrorHandler = (
    bool | str | Callable[..., str] | type[Exception] | tuple[type[Exception], ...]
)


class Tool(Runnable[ToolInput, dict[str, Any]]):
    """
    Wrapper for creating a ToolNode from a list of tools.

    The Tool encapsulates the creation of a LangGraph ToolNode, allowing
    users to define tool nodes in a simple way. It is a proper Runnable,
    so LangGraph calls `invoke`/`ainvoke` natively with the run config:
    async tools work in async workflows and `config["configurable"]` reaches
    the tools explicitly.

    NOTE: A Tool only executes inside a compiled graph. ToolNode requires the
    LangGraph runtime, so calling it directly outside a graph raises.

    Examples:
        from langgraphlib.tool import Tool

        # Define tools (sync or async, plain functions or @tool)
        def search(query: str) -> str:
            '''Searches for information.'''
            return f"Results for: {query}"

        async def fetch_profile(user_id: int, *, config: RunnableConfig) -> str:
            '''Fetches a profile using the run config.'''
            tenant = config["configurable"]["tenant"]
            return f"{tenant}:{user_id}"

        # Create Tool with a list of tools
        search_tool = Tool(name="search_tools", tools=[search, fetch_profile])

        # Custom error handling (same options as ToolNode)
        safe_tool = Tool(
            name="safe_tools",
            tools=[search],
            handle_tool_errors="Tool failed, try again.",
        )

        # Use in Workflow
        workflow = Workflow(
            state=MessagesState,
            agents=[agent],
            nodes={
                "agent_tools": search_tool,
            },
            edges=[
                ("start", "agent"),
                ("agent", "agent_tools", "has_tool_calls"),
                ("agent_tools", "agent"),
                ("agent", "end", "no_tool_calls"),
            ],
        )
    """

    def __init__(
        self,
        name: str,
        tools: Sequence[BaseTool | Callable[..., Any]],
        *,
        handle_tool_errors: ToolErrorHandler | None = None,
    ) -> None:
        """
        Initializes the Tool.

        Args:
            name: Name of the tool node (used in traces and graph drawing).
            tools: Tools available to the node. Plain callables, async
                callables or BaseTool instances (e.g. created with @tool).
            handle_tool_errors: Error handling policy forwarded to ToolNode:
                True/False, a fallback message, a callable(exception) -> str,
                an exception type or a tuple of types. None keeps the
                ToolNode default (only tool invocation errors are caught).
        """
        self._name = name
        self._tools = list(tools)
        self._handle_tool_errors = handle_tool_errors
        self._node = self._build_node()

    def _build_node(self) -> ToolNode:
        """Build the underlying ToolNode, forwarding only explicit options."""
        options: dict[str, Any] = {"name": self._name}
        # None significa "usar o default do ToolNode", que e uma funcao privada
        # do langgraph — por isso o kwarg so e repassado quando informado
        if self._handle_tool_errors is not None:
            options["handle_tool_errors"] = self._handle_tool_errors
        return ToolNode(self._tools, **options)

    @property
    def name(self) -> str:
        """Name of the tool node."""
        return self._name

    @property
    def tools(self) -> list[BaseTool | Callable[..., Any]]:
        """List of tools."""
        return self._tools

    @property
    def node(self) -> ToolNode:
        """LangGraph ToolNode."""
        return self._node

    @property
    def handle_tool_errors(self) -> ToolErrorHandler | None:
        """Error handling policy, or None for the ToolNode default."""
        return self._handle_tool_errors

    def invoke(
        self,
        input: ToolInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Executes the ToolNode with the provided state.

        Args:
            input: State instance (or dict) with the messages.
            config: Run config propagated by LangGraph (callbacks,
                configurable, runtime).

        Returns:
            Dict with messages resulting from tool calls.
        """
        return self._node.invoke(input, config, **kwargs)

    async def ainvoke(
        self,
        input: ToolInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Executes the ToolNode asynchronously.

        Args:
            input: State instance (or dict) with the messages.
            config: Run config propagated by LangGraph (callbacks,
                configurable, runtime).

        Returns:
            Dict with messages resulting from tool calls.
        """
        return await self._node.ainvoke(input, config, **kwargs)

    def __call__(
        self, state: ToolInput, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Shortcut for `invoke`, kept for backwards compatibility."""
        return self.invoke(state, config)

    def __repr__(self) -> str:
        """Tool representation."""
        tool_names = [
            getattr(t, "name", None) or getattr(t, "__name__", repr(t))
            for t in self._tools
        ]
        return f"Tool(name={self._name!r}, tools={tool_names})"
