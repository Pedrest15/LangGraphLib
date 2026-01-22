"""Workflow para orquestração de agentes com LangGraph."""

import base64
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from langgraphlib.agent import Agent
from langgraphlib.edge import Condition, DistributionFunc, Edge
from langgraphlib.streaming import MessageStreamChunk


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

        # Workflow com tools (usando classe Tool)
        from langgraphlib.tool import Tool

        agent_with_tools = Agent(model=model, name="agent", tools=[search])
        agent_tools = Tool(name="agent_tools", tools=[search])

        workflow = Workflow(
            state=State,
            agents=[agent_with_tools],
            nodes={"agent_tools": agent_tools},
            edges=[
                ("start", "agent"),
                ("agent", "agent_tools", "has_tool_calls"),
                ("agent", "end", "no_tool_calls"),
                ("agent_tools", "agent"),
            ],
        )

        # Múltiplos agentes com tools diferentes
        coder = Agent(model=model, name="coder", tools=[run_code])
        searcher = Agent(model=model, name="searcher", tools=[web_search])
        coder_tools = Tool(name="coder_tools", tools=[run_code])
        searcher_tools = Tool(name="searcher_tools", tools=[web_search])

        workflow = Workflow(
            state=State,
            agents=[coder, searcher],
            nodes={
                "coder_tools": coder_tools,
                "searcher_tools": searcher_tools,
            },
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

        # Nós customizados com funções (nodes)
        def format_output(state: State) -> dict:
            return {"output": state.messages[-1].content.upper()}

        workflow = Workflow(
            state=State,
            agents=[writer],
            nodes={"formatter": format_output},
            edges=[
                ("start", "writer"),
                ("writer", "formatter"),
                ("formatter", "end"),
            ],
        )

        # Subgraphs (Workflow aninhado)
        # Um Workflow pode ser usado como nó de outro Workflow
        sub_workflow = Workflow(
            state=State,
            agents=[researcher],
            edges=[
                ("start", "researcher"),
                ("researcher", "end"),
            ],
        )

        main_workflow = Workflow(
            state=State,
            agents=[writer],
            nodes={"research": sub_workflow},  # Workflow como nó
            edges=[
                ("start", "research"),
                ("research", "writer"),
                ("writer", "end"),
            ],
        )

        # Execução paralela (fan-out e fan-in)
        # Múltiplos agentes executam em paralelo e resultados são combinados
        from langgraphlib.state import create_state
        import operator
        from typing import Annotated

        # State com reducer para combinar resultados paralelos
        ParallelState = create_state(
            results=(Annotated[list, operator.add], [])
        )

        researcher = Agent(model=model, name="researcher", prompt="Pesquise...")
        analyst = Agent(model=model, name="analyst", prompt="Analise...")
        writer = Agent(model=model, name="writer", prompt="Escreva...")

        workflow = Workflow(
            state=ParallelState,
            agents=[researcher, analyst, writer],
            edges=[
                # Fan-out: start -> researcher E analyst (paralelo)
                ("start", ["researcher", "analyst"]),
                # Fan-in: ambos convergem para writer
                ("researcher", "writer"),
                ("analyst", "writer"),
                ("writer", "end"),
            ],
        )

        # Sintaxe alternativa sem lista (equivalente)
        workflow = Workflow(
            state=ParallelState,
            agents=[researcher, analyst, writer],
            edges=[
                ("start", "researcher"),
                ("start", "analyst"),
                ("researcher", "writer"),
                ("analyst", "writer"),
                ("writer", "end"),
            ],
        )

        # Map-Reduce com Send (fan-out dinâmico)
        # Número de branches determinado em runtime baseado nos dados
        from langgraphlib import Send, create_state
        import operator
        from typing import Annotated

        # State com lista de itens e resultados acumulados
        MapReduceState = create_state(
            items=(list[str], []),
            results=(Annotated[list[str], operator.add], [])
        )

        # Função de distribuição: cria um Send para cada item
        def distribute_items(state) -> list[Send]:
            return [
                Send("process_item", {"current_item": item, "results": []})
                for item in state.items
            ]

        # Nó que processa cada item individualmente
        def process_item(state) -> dict:
            item = state.get("current_item", "")
            return {"results": [f"Processado: {item}"]}

        workflow = Workflow(
            state=MapReduceState,
            nodes={"process_item": process_item},
            edges=[
                # Fan-out dinâmico: distribui para N instâncias de process_item
                ("start", distribute_items),
                # Fan-in: todas as instâncias convergem para end
                ("process_item", "end"),
            ],
        )

        # Exemplo com subgraph para processamento complexo
        # Cada branch executa um subgraph completo em paralelo
        scraper_subgraph = Workflow(
            state=ScrapingState,
            nodes={"scrape": scrape_site, "filter": filter_results},
            edges=[
                ("start", "scrape"),
                ("scrape", "filter"),
                ("filter", "end"),
            ],
        )

        def distribute_sites(state) -> list[Send]:
            return [
                Send("scraper", {"site_id": site, "results": []})
                for site in state.sites
            ]

        main_workflow = Workflow(
            state=MainState,
            nodes={"scraper": scraper_subgraph, "aggregate": aggregate_results},
            edges=[
                ("start", distribute_sites),  # Distribui para N scrapers
                ("scraper", "aggregate"),      # Todos convergem
                ("aggregate", "end"),
            ],
        )
    """

    def __init__(
        self,
        state: type[BaseModel],
        agents: list[Agent] | None = None,
        edges: list[Edge] | None = None,
        nodes: dict[str, Callable[..., Any]] | None = None,
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
                - (source, [target1, target2, ...]): fan-out estático (paralelo)
                - (source, distribution_func): fan-out dinâmico (map-reduce)
                  distribution_func: (state) -> list[Send]
            nodes: Dict de nós customizados {nome: nó}.
                Pode ser função, Tool, ou outro Workflow (subgraph).
                Funções devem receber state e retornar dict para update.
            checkpointer: Checkpointer para persistência de estado.
            mode: Modo de execução dos agentes ("sync" ou "async").
        """
        self._state = state
        self._agents = {agent.name: agent for agent in (agents or [])}
        self._edges = edges or []
        self._nodes = nodes or {}
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

            async def async_node(
                state: BaseModel | dict[str, Any],
            ) -> dict[str, Any]:
                return await agent.ainvoke(state)

            return async_node

        def sync_node(state: BaseModel | dict[str, Any]) -> dict[str, Any]:
            return agent.invoke(state)

        return sync_node

    def _has_tool_calls(self, state: BaseModel) -> bool:
        """Verifica se a última mensagem tem tool_calls."""
        messages = getattr(state, "messages", [])
        if not messages:
            return False
        last_message = messages[-1]
        return bool(getattr(last_message, "tool_calls", None))

    def _build_conditional_edges(self) -> dict[str, list[tuple[str, Condition]]]:
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

        # Adiciona nós customizados (funções, Tool, ou Workflow/subgraph)
        for name, node in self._nodes.items():
            # Se for um Workflow, compila e usa como subgraph
            if isinstance(node, Workflow):
                subgraph = node.compile()
                workflow.add_node(name, subgraph)
            else:
                workflow.add_node(name, node)

        # Agrupa edges condicionais
        conditional_edges = self._build_conditional_edges()

        # Separa edges de distribuição (map-reduce) das outras
        distribution_edges: list[tuple[str, DistributionFunc]] = []

        # Adiciona edges
        for edge in self._edges:
            source = self._resolve_node_name(edge[0])
            target = edge[1]

            # Edge condicional (3 elementos)
            if len(edge) == 3:
                # Será tratada abaixo em bloco
                continue

            # Fan-out estático: (source, [target1, target2, ...])
            if isinstance(target, list):
                for t in target:
                    resolved_target = self._resolve_node_name(t)
                    workflow.add_edge(source, resolved_target)
                continue

            # Fan-out dinâmico (map-reduce): (source, distribution_func)
            if callable(target):
                distribution_edges.append((source, target))
                continue

            # Edge fixa simples
            resolved_target = self._resolve_node_name(target)
            workflow.add_edge(source, resolved_target)

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

        # Adiciona edges de distribuição (map-reduce com Send)
        for source, distribution_func in distribution_edges:
            workflow.add_conditional_edges(source, distribution_func)

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

    def stream(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        subgraphs: bool = False,
    ) -> Iterator[MessageStreamChunk]:
        """
        Stream de tokens LLM durante execução do workflow.

        Usa stream_mode="messages" para capturar tokens individuais
        do LLM à medida que são gerados.

        Args:
            input: Estado inicial do workflow.
            config: Configuração opcional (thread_id, callbacks, etc).
            subgraphs: Se True, inclui streams de subgraphs.

        Yields:
            Tuplas (AIMessageChunk, metadata) onde:
            - AIMessageChunk: chunk parcial da resposta
            - metadata: dict com info do node (langgraph_node, etc)

        Raises:
            ValueError: Se o grafo não foi compilado.

        Examples:
            workflow = Workflow(...)
            workflow.compile()

            for chunk, metadata in workflow.stream({"messages": [...]}):
                print(chunk.content, end="", flush=True)
        """
        if not self._graph:
            raise ValueError("Grafo não compilado. Chame compile() primeiro.")

        yield from self._graph.stream(
            input,
            config=config,
            stream_mode="messages",
            subgraphs=subgraphs,
        )

    async def astream(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[MessageStreamChunk]:
        """
        Stream assíncrono de tokens LLM durante execução do workflow.

        Usa stream_mode="messages" para capturar tokens individuais
        do LLM à medida que são gerados.

        Args:
            input: Estado inicial do workflow.
            config: Configuração opcional (thread_id, callbacks, etc).
            subgraphs: Se True, inclui streams de subgraphs.

        Yields:
            Tuplas (AIMessageChunk, metadata) onde:
            - AIMessageChunk: chunk parcial da resposta
            - metadata: dict com info do node (langgraph_node, etc)

        Raises:
            ValueError: Se o grafo não foi compilado.

        Examples:
            workflow = Workflow(...)
            workflow.compile()

            async for chunk, metadata in workflow.astream({"messages": [...]}):
                print(chunk.content, end="", flush=True)
        """
        if not self._graph:
            raise ValueError("Grafo não compilado. Chame compile() primeiro.")

        async for item in self._graph.astream(
            input,
            config=config,
            stream_mode="messages",
            subgraphs=subgraphs,
        ):
            yield item
