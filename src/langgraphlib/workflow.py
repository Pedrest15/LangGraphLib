"""Workflow para orquestração de agentes com LangGraph."""

import base64
from collections.abc import Callable
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from langgraphlib.agent import Agent
from langgraphlib.edge import Condition, Edge


class Workflow:
    """
    Orquestra agentes em um grafo LangGraph.

    Simplifica a criação de grafos permitindo definir edges como tuplas de strings.

    Examples:
        from langgraphlib import Workflow, Agent, create_state
        from langgraphlib.model import get_model

        model = get_model("openai/gpt-4o")
        State = create_state()

        researcher = Agent(model=model, name="researcher", prompt="Pesquise...")
        writer = Agent(model=model, name="writer", prompt="Escreva...")

        # Workflow sequencial
        workflow = Workflow(
            state=State,
            agents=[researcher, writer],
            edges=[
                ("start", "researcher"),
                ("researcher", "writer"),
                ("writer", "end"),
            ],
        )

        graph = workflow.compile()
        result = graph.invoke({"messages": [HumanMessage("Oi")]})

        # Workflow com tools (edge condicional automática)
        agent_tools = Agent(model=model, name="agent", tools=[search])
        workflow = Workflow(
            state=State,
            agents=[agent_tools],
            edges=[
                ("start", "agent"),
                ("agent", "agent_tools", "has_tool_calls"),
                ("agent", "end", "no_tool_calls"),
                ("agent_tools", "agent"),
            ],
        )

        # Múltiplos agentes com tools diferentes
        # Use "{agent_name}_tools" para cada um
        coder = Agent(model=model, name="coder", tools=[run_code])
        searcher = Agent(model=model, name="searcher", tools=[web_search])
        workflow = Workflow(
            state=State,
            agents=[coder, searcher],
            edges=[
                ("start", "coder"),
                ("coder", "coder_tools", "has_tool_calls"),
                ("coder", "searcher", "no_tool_calls"),
                ("coder_tools", "coder"),
                ("searcher", "searcher_tools", "has_tool_calls"),
                ("searcher", "end", "no_tool_calls"),
                ("searcher_tools", "searcher"),
            ],
        )

        # Condição customizada com função
        State = create_state(needs_review=(bool, False))
        workflow = Workflow(
            state=State,
            agents=[writer, reviewer],
            edges=[
                ("start", "writer"),
                ("writer", "reviewer", lambda s: s.needs_review),
                ("writer", "end", lambda s: not s.needs_review),
                ("reviewer", "end"),
            ],
        )
    """

    def __init__(
        self,
        state: type[BaseModel],
        agents: list[Agent],
        edges: list[Edge],
        checkpointer: BaseCheckpointSaver | None = None,
        mode: Literal["sync", "async"] = "sync",
    ) -> None:
        """
        Inicializa o workflow.

        Args:
            state: Classe do state (criada com create_state).
            agents: Lista de agentes do workflow.
            edges: Lista de edges como tuplas:
                - (source, target): edge fixa
                - (source, target, condition): edge condicional
                  Conditions built-in: "has_tool_calls", "no_tool_calls"
                  Condition customizada: função (state) -> bool
            checkpointer: Checkpointer para persistência de estado.
            mode: Modo de execução dos agentes ("sync" ou "async").
        """
        self._state = state
        self._agents = {agent.name: agent for agent in agents}
        self._edges = edges
        self._checkpointer = checkpointer
        self._mode = mode
        self._graph: CompiledStateGraph | None = None

    def _resolve_node_name(self, name: str) -> str | object:
        """Converte 'start'/'end' para constantes do LangGraph."""
        lower = name.lower()
        if lower == "start":
            return START
        if lower == "end":
            return END
        return name

    def _create_agent_node(self, agent: Agent) -> Callable:
        """Cria função node para o agente (sync ou async baseado no mode)."""
        if self._mode == "async":

            async def async_node(state: BaseModel) -> dict[str, Any]:
                return await agent.ainvoke(state)

            return async_node

        def sync_node(state: BaseModel) -> dict[str, Any]:
            return agent.invoke(state)

        return sync_node

    def _has_tool_calls(self, state: BaseModel) -> bool:
        """Verifica se a última mensagem tem tool_calls."""
        messages = getattr(state, "messages", [])
        if not messages:
            return False
        last_message = messages[-1]
        return bool(getattr(last_message, "tool_calls", None))

    def _get_tools_for_agent(self, agent_name: str) -> list[Callable]:
        """Retorna tools de um agente."""
        agent = self._agents.get(agent_name)
        if agent:
            return agent.tools
        return []

    def _build_conditional_edges(
        self, workflow: StateGraph
    ) -> dict[str, list[tuple[str, Condition]]]:
        """
        Agrupa edges condicionais por source node.

        Returns:
            Dict mapeando source -> lista de (target, condition)
        """
        conditional: dict[str, list[tuple[str, Condition]]] = {}

        for edge in self._edges:
            if len(edge) == 3:
                source, target, condition = edge
                source = self._resolve_node_name(source)
                if source not in conditional:
                    conditional[source] = []
                conditional[source].append((target, condition))

        return conditional

    def _evaluate_condition(self, condition: Condition, state: BaseModel) -> bool:
        """Avalia uma condição (string built-in ou callable)."""
        if callable(condition):
            return condition(state)
        if condition == "has_tool_calls":
            return self._has_tool_calls(state)
        if condition == "no_tool_calls":
            return not self._has_tool_calls(state)
        return False

    def compile(self) -> CompiledStateGraph:
        """
        Compila o workflow em um grafo LangGraph.

        Returns:
            Grafo compilado pronto para invoke/ainvoke.
        """
        workflow = StateGraph(self._state)

        # Adiciona nodes dos agentes
        for name, agent in self._agents.items():
            workflow.add_node(name, self._create_agent_node(agent))

        # Adiciona tool nodes se necessário
        # Detecta padrão "{agent_name}_tools" nas edges
        tool_nodes_added: set[str] = set()
        for edge in self._edges:
            if len(edge) >= 2:
                target = edge[1]
                # Verifica se target termina com "_tools"
                if target.endswith("_tools"):
                    agent_name = target[:-6]  # Remove "_tools"
                    if agent_name in self._agents and target not in tool_nodes_added:
                        tools = self._get_tools_for_agent(agent_name)
                        if tools:
                            workflow.add_node(target, ToolNode(tools))
                            tool_nodes_added.add(target)

        # Agrupa edges condicionais
        conditional_edges = self._build_conditional_edges(workflow)

        # Adiciona edges
        for edge in self._edges:
            source = self._resolve_node_name(edge[0])
            target = self._resolve_node_name(edge[1])

            # Edge condicional
            if len(edge) == 3:
                # Será tratada abaixo em bloco
                continue

            # Edge fixa
            workflow.add_edge(source, target)

        # Adiciona edges condicionais agrupadas
        for source, conditions in conditional_edges.items():
            # Cria função de roteamento
            def make_router(
                conds: list[tuple[str, Condition]], wf: "Workflow"
            ) -> Callable:
                def router(state: BaseModel) -> str:
                    for target, condition in conds:
                        if wf._evaluate_condition(condition, state):
                            return wf._resolve_node_name(target)
                    # Fallback para primeiro target
                    return wf._resolve_node_name(conds[0][0])

                return router

            # Cria mapping
            mapping = {}
            for target, _ in conditions:
                resolved = self._resolve_node_name(target)
                mapping[resolved] = resolved

            workflow.add_conditional_edges(
                source, make_router(conditions, self), mapping
            )

        # Compila
        self._graph = workflow.compile(checkpointer=self._checkpointer)
        return self._graph

    def get_image(self, xray: bool = True) -> str:
        """
        Retorna imagem do grafo em base64.

        Args:
            xray: Se True, mostra detalhes internos dos nodes.

        Returns:
            String base64 da imagem PNG.

        Raises:
            ValueError: Se o grafo não foi compilado.
        """
        if not self._graph:
            raise ValueError("Grafo não compilado. Chame compile() primeiro.")

        img_data = self._graph.get_graph(xray=xray).draw_mermaid_png()
        return base64.b64encode(img_data).decode("utf-8")

    @property
    def graph(self) -> CompiledStateGraph | None:
        """Retorna o grafo compilado ou None."""
        return self._graph

    def __repr__(self) -> str:
        """Representação do workflow."""
        agents = list(self._agents.keys())
        return f"Workflow(agents={agents}, edges={len(self._edges)})"
