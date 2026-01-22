# LangGraphLib

A Python library that provides a high-level abstraction over [LangGraph](https://github.com/langchain-ai/langgraph) for building AI agents with minimal boilerplate.

## Features

- **Simple Agent Creation**: Define agents with prompts, tools, and routing in a few lines
- **Declarative Workflows**: Build complex agent graphs using intuitive tuple-based edge definitions
- **Multi-Provider Support**: Use OpenAI, Anthropic, Groq, Ollama, and 7+ other providers
- **Tool Integration**: Seamlessly integrate tools with automatic routing
- **Dynamic Routing**: Let agents decide their next destination using structured outputs
- **Parallel Execution**: Fan-out/fan-in patterns for concurrent agent execution
- **Map-Reduce**: Dynamic branching with `Send` for runtime-determined parallelism
- **Streaming**: Real-time token streaming from LLM responses
- **Long-term Memory**: Built-in memory management with tools and automatic extraction
- **Callbacks**: Logging and tracing handlers for debugging and monitoring
- **Subgraphs**: Nest workflows for modular, reusable components

## Installation

Install directly from GitHub:

```bash
uv add git+https://github.com/your-username/langgraphlib.git
```

Or with pip:

```bash
pip install git+https://github.com/your-username/langgraphlib.git
```

### Provider Dependencies

Install the LangChain integration for your chosen provider:

```bash
# OpenAI
uv add langchain-openai

# Anthropic
uv add langchain-anthropic

# Groq
uv add langchain-groq

# Ollama (local)
uv add langchain-ollama

# Google
uv add langchain-google-genai

# And more: langchain-mistralai, langchain-cohere, langchain-fireworks, langchain-together, langchain-cerebras
```

## Quick Start

### Basic Agent

```python
from langchain_core.messages import HumanMessage
from langgraphlib import Agent, Workflow, MessagesState, get_model

# Get a model (reads API key from environment)
model = get_model("openai/gpt-4o-mini")

# Create an agent
agent = Agent(
    model=model,
    name="assistant",
    prompt="You are a helpful assistant. Be concise.",
)

# Build workflow
workflow = Workflow(
    state=MessagesState,
    agents=[agent],
    edges=[
        ("start", "assistant"),
        ("assistant", "end"),
    ],
)

# Compile and run
graph = workflow.compile()
result = graph.invoke({"messages": [HumanMessage(content="Hello!")]})

print(result["messages"][-1].content)
```

## Core Concepts

### Models

Use `get_model()` to get any supported LLM with a unified interface:

```python
from langgraphlib import get_model

# OpenAI
model = get_model("openai/gpt-4o")
model = get_model("openai/gpt-4o-mini", temperature=0.7)

# Anthropic
model = get_model("anthropic/claude-3-5-sonnet-20241022")

# Groq (fast inference)
model = get_model("groq/llama-3.1-70b-versatile")

# Ollama (local)
model = get_model("ollama/llama3")

# With explicit API key
model = get_model("openai/gpt-4o", api_key="sk-...")

# With streaming enabled
model = get_model("openai/gpt-4o", streaming=True)
```

Supported providers: `openai`, `anthropic`, `groq`, `ollama`, `google`, `mistral`, `cohere`, `fireworks`, `together`, `cerebras`

### State

State holds the data that flows through your workflow. Use `create_state()` to define custom fields:

```python
from langgraphlib import create_state, MessagesState

# Use the built-in MessagesState for simple chat applications
# It has a single field: messages (with add_messages reducer)

# Create custom state with additional fields
MyState = create_state(
    counter=(int, 0),           # field with default value
    status=(str, "pending"),
)

# State without messages
DataState = create_state(
    include_messages=False,
    query=(str, ""),
    result=(str, ""),
)

# State with reducers for parallel execution
import operator
from typing import Annotated

ParallelState = create_state(
    results=(Annotated[list[str], operator.add], [])  # Results are concatenated
)
```

### Agent

The `Agent` class wraps an LLM with configuration for prompts, tools, and routing:

```python
from langgraphlib import Agent

agent = Agent(
    model=model,
    name="researcher",              # Unique identifier
    prompt="You are a researcher.", # System prompt
    tools=[search, calculate],      # Optional tools
    destinations=["writer", "end"], # Optional dynamic routing
    input_fields="messages",        # State field(s) to read
    output_fields="messages",       # State field(s) to write
    max_retries=3,                  # Retry on failure
    timeout=30.0,                   # Timeout in seconds
)
```

#### Custom Input/Output Fields

Agents can read from and write to specific state fields:

```python
State = create_state(
    include_messages=False,
    query=(str, ""),
    answer=(str, ""),
    confidence=(float, 0.0),
)

agent = Agent(
    model=model,
    name="qa",
    prompt="Answer the query concisely.",
    state=State,  # Required for type inference
    input_fields="query",
    output_fields=["answer", "confidence"],
)
```

#### Dynamic Routing with Destinations

Let the agent decide where to go next:

```python
supervisor = Agent(
    model=model,
    name="supervisor",
    prompt="""You are a supervisor. Based on the conversation:
    - If research is needed, go to 'researcher'
    - If writing is needed, go to 'writer'
    - If done, go to 'end'""",
    destinations=["researcher", "writer", "end"],
)

# The agent returns Command(goto="researcher"|"writer"|"end", update={...})
```

### Workflow

The `Workflow` class orchestrates agents into a LangGraph:

```python
from langgraphlib import Workflow

workflow = Workflow(
    state=MyState,           # State class
    agents=[agent1, agent2], # List of agents
    nodes={"custom": func},  # Optional custom nodes
    edges=[...],             # Edge definitions
    checkpointer=saver,      # Optional state persistence
    mode="sync",             # "sync" or "async"
)

graph = workflow.compile()
result = graph.invoke({"messages": [...]})
```

### Edges

Edges define the flow between nodes using tuples:

```python
edges = [
    # Simple edge: source -> target
    ("start", "agent"),
    ("agent", "end"),

    # Conditional edge with built-in conditions
    ("agent", "tools", "has_tool_calls"),
    ("agent", "end", "no_tool_calls"),

    # Conditional edge with custom function
    ("writer", "reviewer", lambda s: s.needs_review),
    ("writer", "end", lambda s: not s.needs_review),

    # Fan-out (parallel execution)
    ("start", ["researcher", "analyst"]),

    # Dynamic fan-out (map-reduce)
    ("start", distribute_items),  # Function returning list[Send]
]
```

## Tools

### Using Tools with Agents

```python
from langchain_core.tools import tool
from langgraphlib import Agent, Tool, Workflow, MessagesState

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)

# Create agent with tools
agent = Agent(
    model=model,
    name="assistant",
    prompt="You are a helpful assistant with access to search and calculate.",
    tools=[search, calculate],
)

# Create tool node (handles tool execution)
agent_tools = Tool(name="assistant_tools", tools=[search, calculate])

workflow = Workflow(
    state=MessagesState,
    agents=[agent],
    nodes={"assistant_tools": agent_tools},
    edges=[
        ("start", "assistant"),
        ("assistant", "assistant_tools", "has_tool_calls"),
        ("assistant", "end", "no_tool_calls"),
        ("assistant_tools", "assistant"),  # Return to agent after tool execution
    ],
)
```

## Advanced Patterns

### Multi-Agent Supervisor

```python
# Define specialized agents
writer = Agent(
    model=model,
    name="writer",
    prompt="Write creative content based on the request.",
)

reviewer = Agent(
    model=model,
    name="reviewer",
    prompt="""Review the content. Respond with:
    - 'APPROVED: [comment]' if good
    - 'REVISE: [suggestions]' if needs work""",
)

supervisor = Agent(
    model=model,
    name="supervisor",
    prompt="""Coordinate the writing process:
    - For new requests, go to 'writer'
    - After 'APPROVED', go to 'end'
    - After 'REVISE', go to 'writer'""",
    destinations=["writer", "end"],
)

workflow = Workflow(
    state=MessagesState,
    agents=[supervisor, writer, reviewer],
    edges=[
        ("start", "supervisor"),
        ("writer", "reviewer"),
        ("reviewer", "supervisor"),
        # supervisor uses destinations to route dynamically
    ],
)
```

### Parallel Execution (Fan-out/Fan-in)

```python
import operator
from typing import Annotated
from langgraphlib import create_state

# State with reducer to combine parallel results
ParallelState = create_state(
    results=(Annotated[list[str], operator.add], [])
)

researcher = Agent(
    model=model,
    name="researcher",
    prompt="Research the topic. Return findings.",
    output_fields="results",
    state=ParallelState,
)

analyst = Agent(
    model=model,
    name="analyst",
    prompt="Analyze the topic. Return analysis.",
    output_fields="results",
    state=ParallelState,
)

summarizer = Agent(
    model=model,
    name="summarizer",
    prompt="Summarize the results from research and analysis.",
    input_fields="results",
    output_fields="results",
    state=ParallelState,
)

workflow = Workflow(
    state=ParallelState,
    agents=[researcher, analyst, summarizer],
    edges=[
        # Fan-out: both run in parallel
        ("start", ["researcher", "analyst"]),
        # Fan-in: both converge to summarizer
        ("researcher", "summarizer"),
        ("analyst", "summarizer"),
        ("summarizer", "end"),
    ],
)
```

### Map-Reduce with Send

Process a dynamic number of items in parallel:

```python
import operator
from typing import Annotated
from langgraphlib import create_state, Send, Workflow

MapReduceState = create_state(
    items=(list[str], []),
    results=(Annotated[list[str], operator.add], []),
)

# Distribution function: creates one Send per item
def distribute_items(state) -> list[Send]:
    return [
        Send("process_item", {"current_item": item, "results": []})
        for item in state.items
    ]

# Processing function
def process_item(state) -> dict:
    item = state.get("current_item", "")
    return {"results": [f"Processed: {item}"]}

workflow = Workflow(
    state=MapReduceState,
    nodes={"process_item": process_item},
    edges=[
        ("start", distribute_items),  # Dynamic fan-out
        ("process_item", "end"),       # All results merge via reducer
    ],
)

graph = workflow.compile()
result = graph.invoke({
    "items": ["apple", "banana", "cherry"],
    "results": []
})
# result["results"] = ["Processed: apple", "Processed: banana", "Processed: cherry"]
```

### Subgraphs

Nest workflows for modularity:

```python
# Create a reusable sub-workflow
research_workflow = Workflow(
    state=MessagesState,
    agents=[researcher],
    edges=[
        ("start", "researcher"),
        ("researcher", "end"),
    ],
)

# Use it as a node in the main workflow
main_workflow = Workflow(
    state=MessagesState,
    agents=[writer],
    nodes={"research": research_workflow},  # Workflow as node
    edges=[
        ("start", "research"),
        ("research", "writer"),
        ("writer", "end"),
    ],
)
```

### Streaming

Stream LLM tokens as they're generated:

```python
workflow = Workflow(
    state=MessagesState,
    agents=[agent],
    edges=[
        ("start", "agent"),
        ("agent", "end"),
    ],
)
workflow.compile()

# Synchronous streaming
for chunk, metadata in workflow.stream({"messages": [...]}):
    if hasattr(chunk, "content") and chunk.content:
        print(chunk.content, end="", flush=True)

# Async streaming
async for chunk, metadata in workflow.astream({"messages": [...]}):
    if hasattr(chunk, "content") and chunk.content:
        print(chunk.content, end="", flush=True)
```

The metadata dict contains information like `langgraph_node` to identify which agent is streaming.

## Memory

LangGraphLib provides tools and nodes for long-term memory management.

### Memory Manager

```python
from langgraph.store.memory import InMemoryStore
from langgraphlib import MemoryManager

store = InMemoryStore()
memory = MemoryManager(store, user_id="user_123")

# Save memories
memory.save("preferences", {"theme": "dark", "language": "en"})

# Retrieve by key
prefs = memory.get("preferences")

# Semantic search (requires store with embeddings)
related = memory.search("user preferences", limit=5)

# List all memories
all_memories = memory.list()

# Delete
memory.delete("preferences")
```

### Memory Tools

Give agents the ability to remember and recall:

```python
from langgraphlib import MemoryManager, create_memory_tools, Agent, Tool

memory = MemoryManager(store, user_id="user_123")
memory_tools = create_memory_tools(memory)  # [remember, recall]

agent = Agent(
    model=model,
    name="assistant",
    prompt="You are a helpful assistant that remembers user preferences.",
    tools=memory_tools,
)

# The agent can now:
# - remember(fact="User prefers dark mode", category="preference")
# - recall(query="user preferences", limit=5)
```

### Automatic Memory Extraction

Use nodes to automatically extract and retrieve memories:

```python
from langgraphlib import (
    MemoryManager,
    create_memory_saver_node,
    create_memory_retriever_node,
    create_state,
)

memory = MemoryManager(store, user_id="user_123")

# Node that extracts facts from conversation
memory_saver = create_memory_saver_node(model, memory)

# Node that retrieves relevant memories
State = create_state(memory_context=(str, ""))
memory_retriever = create_memory_retriever_node(memory)

workflow = Workflow(
    state=State,
    agents=[assistant],
    nodes={
        "retrieve": memory_retriever,
        "save": memory_saver,
    },
    edges=[
        ("start", "retrieve"),          # Get relevant memories
        ("retrieve", "assistant"),      # Agent has context
        ("assistant", "save"),          # Extract new facts
        ("save", "end"),
    ],
)
```

## Callbacks

### Logging Handler

```python
import logging
from langgraphlib import LoggingHandler

logging.basicConfig(level=logging.INFO)

graph = workflow.compile()
result = graph.invoke(
    {"messages": [...]},
    config={"callbacks": [LoggingHandler()]}
)

# Output:
# INFO - [START] assistant
# INFO - [END] (1.23s)

# For more details:
result = graph.invoke(
    {"messages": [...]},
    config={"callbacks": [LoggingHandler(level=logging.DEBUG)]}
)
```

### Trace Handler

Collect structured execution traces:

```python
from langgraphlib import TraceHandler

tracer = TraceHandler()

result = graph.invoke(
    {"messages": [...]},
    config={"callbacks": [tracer]}
)

# Access traces
print(tracer.traces)
# [
#   {"event": "start", "name": "assistant", "timestamp": "...", ...},
#   {"event": "end", "duration_seconds": 1.23, ...},
# ]

# Export as JSON
print(tracer.to_json())

# Clear for next run
tracer.clear()
```

### Multiple Handlers

```python
result = graph.invoke(
    {"messages": [...]},
    config={
        "callbacks": [
            LoggingHandler(),
            TraceHandler(),
        ]
    }
)
```

## Graph Visualization

Get a visual representation of your workflow:

```python
workflow = Workflow(...)
graph = workflow.compile()

# Get base64-encoded PNG
image_b64 = workflow.get_image(xray=True)

# Save to file
import base64
with open("graph.png", "wb") as f:
    f.write(base64.b64decode(image_b64))
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Agent` | Wraps an LLM with prompt, tools, and routing configuration |
| `Workflow` | Orchestrates agents into a LangGraph |
| `Tool` | Creates a tool execution node from callable functions |

### State Functions

| Function | Description |
|----------|-------------|
| `create_state(**fields)` | Create a custom state class |
| `MessagesState` | Pre-built state with messages field |

### Model Functions

| Function | Description |
|----------|-------------|
| `get_model(model, **kwargs)` | Get a chat model instance |

### Memory Classes and Functions

| Name | Description |
|------|-------------|
| `MemoryManager` | Sync memory manager |
| `AsyncMemoryManager` | Async memory manager |
| `create_memory_tools(memory)` | Create remember/recall tools |
| `create_memory_saver_node(model, memory)` | Create auto-extraction node |
| `create_memory_retriever_node(memory)` | Create retrieval node |

### Callback Classes

| Class | Description |
|-------|-------------|
| `LoggingHandler` | Logs execution to Python logger |
| `TraceHandler` | Collects structured execution traces |

### Edge Types

| Type | Description |
|------|-------------|
| `Send` | For dynamic fan-out (map-reduce) |
| `Edge` | Type alias for edge tuples |
| `Condition` | Type alias for edge conditions |

## Requirements

- Python 3.13+
- langgraph >= 0.2
- langchain-core >= 0.3
- pydantic >= 2.0

## Contributing

Contributions are welcome! Feel free to open issues and pull requests.

## License

MIT
