from collections.abc import Callable
from typing import Any

from langgraph.prebuilt import ToolNode
from pydantic import BaseModel


class Tool:
    """
    Wrapper para criar um ToolNode a partir de uma lista de callables.

    O Tool encapsula a criação de um ToolNode do LangGraph, permitindo
    que o usuário defina nós de ferramentas de forma simples.

    Examples:
        from langgraphlib.tool import Tool

        # Definir ferramentas
        def search(query: str) -> str:
            '''Busca informações.'''
            return f"Resultados para: {query}"

        def calculate(expression: str) -> str:
            '''Calcula expressão matemática.'''
            return str(eval(expression))

        # Criar Tool com uma lista de callables
        search_tool = Tool(name="search_tools", tools=[search])
        calc_tool = Tool(name="calc_tools", tools=[calculate])

        # Usar no Workflow
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
        Inicializa o Tool.

        Args:
            name: Nome do nó de ferramentas.
            tools: Lista de callables que serão as ferramentas disponíveis.
        """
        self._name = name
        self._tools = tools
        self._node = ToolNode(tools)

    @property
    def name(self) -> str:
        """Nome do nó de ferramentas."""
        return self._name

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Lista de ferramentas."""
        return self._tools

    @property
    def node(self) -> ToolNode:
        """ToolNode do LangGraph."""
        return self._node

    def __call__(self, state: BaseModel) -> dict[str, Any]:
        """
        Executa o ToolNode com o state fornecido.

        Args:
            state: Instância do state com os dados de entrada.

        Returns:
            Dict com as mensagens resultantes das tool calls.
        """
        return self._node.invoke(state)

    async def ainvoke(self, state: BaseModel) -> dict[str, Any]:
        """
        Executa o ToolNode de forma assíncrona.

        Args:
            state: Instância do state com os dados de entrada.

        Returns:
            Dict com as mensagens resultantes das tool calls.
        """
        return await self._node.ainvoke(state)

    def __repr__(self) -> str:
        """Representação do Tool."""
        tool_names = [t.__name__ for t in self._tools]
        return f"Tool(name={self._name!r}, tools={tool_names})"
