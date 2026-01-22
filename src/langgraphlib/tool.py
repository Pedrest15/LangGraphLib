from collections.abc import Callable
from typing import Any

from langgraph.prebuilt import ToolNode
from pydantic import BaseModel


class Tool:
    """
    Wrapper for creating a ToolNode from a list of callables.

    The Tool encapsulates the creation of a LangGraph ToolNode, allowing
    users to define tool nodes in a simple way.

    Examples:
        from langgraphlib.tool import Tool

        # Define tools
        def search(query: str) -> str:
            '''Searches for information.'''
            return f"Results for: {query}"

        def calculate(expression: str) -> str:
            '''Calculates mathematical expression.'''
            return str(eval(expression))

        # Create Tool with a list of callables
        search_tool = Tool(name="search_tools", tools=[search])
        calc_tool = Tool(name="calc_tools", tools=[calculate])

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
        tools: list[Callable[..., Any]],
    ) -> None:
        """
        Initializes the Tool.

        Args:
            name: Name of the tool node.
            tools: List of callables that will be the available tools.
        """
        self._name = name
        self._tools = tools
        self._node = ToolNode(tools)

    @property
    def name(self) -> str:
        """Name of the tool node."""
        return self._name

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """List of tools."""
        return self._tools

    @property
    def node(self) -> ToolNode:
        """LangGraph ToolNode."""
        return self._node

    def __call__(self, state: BaseModel) -> dict[str, Any]:
        """
        Executes the ToolNode with the provided state.

        Args:
            state: State instance with input data.

        Returns:
            Dict with messages resulting from tool calls.
        """
        return self._node.invoke(state)

    async def ainvoke(self, state: BaseModel) -> dict[str, Any]:
        """
        Executes the ToolNode asynchronously.

        Args:
            state: State instance with input data.

        Returns:
            Dict with messages resulting from tool calls.
        """
        return await self._node.ainvoke(state)

    def __repr__(self) -> str:
        """Tool representation."""
        tool_names = [t.__name__ for t in self._tools]
        return f"Tool(name={self._name!r}, tools={tool_names})"
