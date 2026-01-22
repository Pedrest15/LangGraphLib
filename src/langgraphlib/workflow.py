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
    Orchestrates agents in a LangGraph graph.

    Simplifies graph creation by allowing edges to be defined as string tuples.

    Examples:
        from langgraphlib import Workflow, Agent, create_state
        from langgraphlib.model import get_model

        model = get_model("openai/gpt-4o")
        State = create_state()

        researcher = Agent(model=model, name="researcher", prompt="Research...")
        writer = Agent(model=model, name="writer", prompt="Write...")

        # Sequential workflow
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
        result = graph.invoke({"messages": [HumanMessage("Hi")]})

        # Workflow with tools (using Tool class)
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

        # Multiple agents with different tools
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

        # Custom condition with function
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

        # Custom nodes with functions
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

        # Subgraphs (nested Workflow)
        # A Workflow can be used as a node of another Workflow
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
            nodes={"research": sub_workflow},  # Workflow as node
            edges=[
                ("start", "research"),
                ("research", "writer"),
                ("writer", "end"),
            ],
        )

        # Parallel execution (fan-out and fan-in)
        # Multiple agents execute in parallel and results are combined
        from langgraphlib.state import create_state
        import operator
        from typing import Annotated

        # State with reducer to combine parallel results
        ParallelState = create_state(
            results=(Annotated[list, operator.add], [])
        )

        researcher = Agent(model=model, name="researcher", prompt="Research...")
        analyst = Agent(model=model, name="analyst", prompt="Analyze...")
        writer = Agent(model=model, name="writer", prompt="Write...")

        workflow = Workflow(
            state=ParallelState,
            agents=[researcher, analyst, writer],
            edges=[
                # Fan-out: start -> researcher AND analyst (parallel)
                ("start", ["researcher", "analyst"]),
                # Fan-in: both converge to writer
                ("researcher", "writer"),
                ("analyst", "writer"),
                ("writer", "end"),
            ],
        )

        # Alternative syntax without list (equivalent)
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

        # Map-Reduce with Send (dynamic fan-out)
        # Number of branches determined at runtime based on data
        from langgraphlib import Send, create_state
        import operator
        from typing import Annotated

        # State with item list and accumulated results
        MapReduceState = create_state(
            items=(list[str], []),
            results=(Annotated[list[str], operator.add], [])
        )

        # Distribution function: creates a Send for each item
        def distribute_items(state) -> list[Send]:
            return [
                Send("process_item", {"current_item": item, "results": []})
                for item in state.items
            ]

        # Node that processes each item individually
        def process_item(state) -> dict:
            item = state.get("current_item", "")
            return {"results": [f"Processed: {item}"]}

        workflow = Workflow(
            state=MapReduceState,
            nodes={"process_item": process_item},
            edges=[
                # Dynamic fan-out: distributes to N instances of process_item
                ("start", distribute_items),
                # Fan-in: all instances converge to end
                ("process_item", "end"),
            ],
        )

        # Example with subgraph for complex processing
        # Each branch executes a complete subgraph in parallel
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
                ("start", distribute_sites),  # Distributes to N scrapers
                ("scraper", "aggregate"),      # All converge
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
        Initialize the workflow.

        Args:
            state: State class (created with create_state).
            agents: List of workflow agents.
            edges: List of edges as tuples:
                - (source, target): fixed edge
                - (source, target, condition): conditional edge
                  Built-in conditions: "has_tool_calls", "no_tool_calls"
                  Custom condition: function (state) -> bool
                - (source, [target1, target2, ...]): static fan-out (parallel)
                - (source, distribution_func): dynamic fan-out (map-reduce)
                  distribution_func: (state) -> list[Send]
            nodes: Dict of custom nodes {name: node}.
                Can be function, Tool, or another Workflow (subgraph).
                Functions must receive state and return dict for update.
            checkpointer: Checkpointer for state persistence.
            mode: Agent execution mode ("sync" or "async").
        """
        self._state = state
        self._agents = {agent.name: agent for agent in (agents or [])}
        self._edges = edges or []
        self._nodes = nodes or {}
        self._checkpointer = checkpointer
        self._mode = mode
        self._graph: CompiledStateGraph | None = None

    def _resolve_node_name(self, name: str) -> str | object:
        """Convert 'start'/'end' to LangGraph constants."""
        lower = name.lower()
        if lower == "start":
            return START
        if lower == "end":
            return END
        return name

    def _create_agent_node(self, agent: Agent) -> Callable:
        """Create node function for agent (sync or async based on mode)."""
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
        """Check if the last message has tool_calls."""
        messages = getattr(state, "messages", [])
        if not messages:
            return False
        last_message = messages[-1]
        return bool(getattr(last_message, "tool_calls", None))

    def _build_conditional_edges(self) -> dict[str, list[tuple[str, Condition]]]:
        """
        Group conditional edges by source node.

        Returns:
            Dict mapping source -> list of (target, condition)
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
        """Evaluate a condition (built-in string or callable)."""
        if callable(condition):
            return condition(state)
        if condition == "has_tool_calls":
            return self._has_tool_calls(state)
        if condition == "no_tool_calls":
            return not self._has_tool_calls(state)
        return False

    def compile(self) -> CompiledStateGraph:
        """
        Compile the workflow into a LangGraph graph.

        Returns:
            Compiled graph ready for invoke/ainvoke.
        """
        workflow = StateGraph(self._state)

        # Add agent nodes
        for name, agent in self._agents.items():
            workflow.add_node(name, self._create_agent_node(agent))

        # Add custom nodes (functions, Tool, or Workflow/subgraph)
        for name, node in self._nodes.items():
            # If it's a Workflow, compile and use as subgraph
            if isinstance(node, Workflow):
                subgraph = node.compile()
                workflow.add_node(name, subgraph)
            else:
                workflow.add_node(name, node)

        # Group conditional edges
        conditional_edges = self._build_conditional_edges()

        # Separate distribution edges (map-reduce) from others
        distribution_edges: list[tuple[str, DistributionFunc]] = []

        # Add edges
        for edge in self._edges:
            source = self._resolve_node_name(edge[0])
            target = edge[1]

            # Conditional edge (3 elements)
            if len(edge) == 3:
                # Will be handled below in block
                continue

            # Static fan-out: (source, [target1, target2, ...])
            if isinstance(target, list):
                for t in target:
                    resolved_target = self._resolve_node_name(t)
                    workflow.add_edge(source, resolved_target)
                continue

            # Dynamic fan-out (map-reduce): (source, distribution_func)
            if callable(target):
                distribution_edges.append((source, target))
                continue

            # Simple fixed edge
            resolved_target = self._resolve_node_name(target)
            workflow.add_edge(source, resolved_target)

        # Add grouped conditional edges
        for source, conditions in conditional_edges.items():
            # Create routing function
            def make_router(
                conds: list[tuple[str, Condition]], wf: "Workflow"
            ) -> Callable:
                def router(state: BaseModel) -> str:
                    for target, condition in conds:
                        if wf._evaluate_condition(condition, state):
                            return wf._resolve_node_name(target)
                    # Fallback to first target
                    return wf._resolve_node_name(conds[0][0])

                return router

            # Create mapping
            mapping = {}
            for target, _ in conditions:
                resolved = self._resolve_node_name(target)
                mapping[resolved] = resolved

            workflow.add_conditional_edges(
                source, make_router(conditions, self), mapping
            )

        # Add distribution edges (map-reduce with Send)
        for source, distribution_func in distribution_edges:
            workflow.add_conditional_edges(source, distribution_func)

        # Compile
        self._graph = workflow.compile(checkpointer=self._checkpointer)
        return self._graph

    def get_image(self, xray: bool = True) -> str:
        """
        Return graph image as base64.

        Args:
            xray: If True, shows internal node details.

        Returns:
            Base64 string of PNG image.

        Raises:
            ValueError: If graph was not compiled.
        """
        if not self._graph:
            raise ValueError("Graph not compiled. Call compile() first.")

        img_data = self._graph.get_graph(xray=xray).draw_mermaid_png()
        return base64.b64encode(img_data).decode("utf-8")

    @property
    def graph(self) -> CompiledStateGraph | None:
        """Return compiled graph or None."""
        return self._graph

    def __repr__(self) -> str:
        """Workflow representation."""
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
        Stream LLM tokens during workflow execution.

        Uses stream_mode="messages" to capture individual tokens
        from the LLM as they are generated.

        Args:
            input: Initial workflow state.
            config: Optional configuration (thread_id, callbacks, etc).
            subgraphs: If True, includes streams from subgraphs.

        Yields:
            Tuples (AIMessageChunk, metadata) where:
            - AIMessageChunk: partial response chunk
            - metadata: dict with node info (langgraph_node, etc)

        Raises:
            ValueError: If graph was not compiled.

        Examples:
            workflow = Workflow(...)
            workflow.compile()

            for chunk, metadata in workflow.stream({"messages": [...]}):
                print(chunk.content, end="", flush=True)
        """
        if not self._graph:
            raise ValueError("Graph not compiled. Call compile() first.")

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
        Async stream LLM tokens during workflow execution.

        Uses stream_mode="messages" to capture individual tokens
        from the LLM as they are generated.

        Args:
            input: Initial workflow state.
            config: Optional configuration (thread_id, callbacks, etc).
            subgraphs: If True, includes streams from subgraphs.

        Yields:
            Tuples (AIMessageChunk, metadata) where:
            - AIMessageChunk: partial response chunk
            - metadata: dict with node info (langgraph_node, etc)

        Raises:
            ValueError: If graph was not compiled.

        Examples:
            workflow = Workflow(...)
            workflow.compile()

            async for chunk, metadata in workflow.astream({"messages": [...]}):
                print(chunk.content, end="", flush=True)
        """
        if not self._graph:
            raise ValueError("Graph not compiled. Call compile() first.")

        async for item in self._graph.astream(
            input,
            config=config,
            stream_mode="messages",
            subgraphs=subgraphs,
        ):
            yield item
