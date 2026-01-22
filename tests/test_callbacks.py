"""Testes de callbacks da langgraphlib."""

import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraphlib.agent import Agent
from langgraphlib.callbacks import LoggingHandler, TraceHandler
from langgraphlib.model import get_model
from langgraphlib.state import MessagesState
from langgraphlib.tool import Tool
from langgraphlib.workflow import Workflow

# Modelo padrão para os testes
MODEL = get_model("openai/gpt-4o-mini")


# =============================================================================
# Cenário 1: Agente simples com callbacks
# =============================================================================


def test_callback_1a_simple_agent_with_logging_handler():
    """
    Agente simples usando LoggingHandler.
    Verifica que o handler loga início/fim dos nós.
    """
    # Configura logging para ver output
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    agent = Agent(
        model=MODEL,
        name="greeter",
        prompt="Você é um assistente amigável. Responda de forma breve e educada.",
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        edges=[
            ("start", "greeter"),
            ("greeter", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 1a: LoggingHandler ===")

    result = graph.invoke(
        {"messages": [HumanMessage(content="Oi, como está?")]},
        config={"callbacks": [LoggingHandler()]},
    )

    assert "messages" in result
    assert len(result["messages"]) >= 2


def test_callback_1b_simple_agent_with_logging_handler_debug():
    """
    Agente simples usando LoggingHandler em modo DEBUG.
    Mostra mais detalhes nos logs.
    """
    # Configura logging para ver output em DEBUG
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

    agent = Agent(
        model=MODEL,
        name="greeter",
        prompt="Você é um assistente amigável. Responda de forma breve e educada.",
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        edges=[
            ("start", "greeter"),
            ("greeter", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 1b: LoggingHandler DEBUG ===")

    # Usando LoggingHandler com nível DEBUG
    result = graph.invoke(
        {"messages": [HumanMessage(content="Oi, como está?")]},
        config={"callbacks": [LoggingHandler(level=logging.DEBUG)]},
    )

    assert "messages" in result
    assert len(result["messages"]) >= 2


def test_callback_1c_simple_agent_with_trace_handler():
    """
    Agente simples usando TraceHandler.
    Verifica que o trace é coletado corretamente.
    """
    agent = Agent(
        model=MODEL,
        name="greeter",
        prompt="Você é um assistente amigável. Responda de forma breve e educada.",
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        edges=[
            ("start", "greeter"),
            ("greeter", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 1c: TraceHandler ===")

    tracer = TraceHandler()
    result = graph.invoke(
        {"messages": [HumanMessage(content="Oi, como está?")]},
        config={"callbacks": [tracer]},
    )

    print(f"Traces coletados: {len(tracer.traces)}")
    print(f"Trace JSON:\n{tracer.to_json()}")

    assert "messages" in result
    assert len(result["messages"]) >= 2
    # Deve ter pelo menos eventos de start e end
    assert len(tracer.traces) >= 2
    # Verifica estrutura do trace
    assert any(t["event"] == "start" for t in tracer.traces)
    assert any(t["event"] == "end" for t in tracer.traces)


# =============================================================================
# Cenário 2: Calculadora com callbacks
# =============================================================================


@tool
def add(a: int, b: int) -> int:
    """Soma dois números."""
    return a + b


@tool
def sub(a: int, b: int) -> int:
    """Subtrai o segundo número do primeiro."""
    return a - b


@tool
def mult(a: int, b: int) -> int:
    """Multiplica dois números."""
    return a * b


@tool
def div(a: int, b: int) -> float:
    """Divide o primeiro número pelo segundo."""
    if b == 0:
        return float("inf")
    return a / b


def test_callback_2a_calculator_with_logging_handler():
    """
    Calculadora com tools usando LoggingHandler.
    Verifica trace de múltiplos nós (agent -> tools -> agent).
    """
    calc_tools = [add, sub, mult, div]

    calculator_agent = Agent(
        model=MODEL,
        name="calculator",
        prompt=(
            "Você é uma calculadora. Use as ferramentas disponíveis para "
            "realizar cálculos. Sempre use a ferramenta apropriada."
        ),
        tools=calc_tools,
    )

    calculator_tools = Tool(name="calculator_tools", tools=calc_tools)

    workflow = Workflow(
        state=MessagesState,
        agents=[calculator_agent],
        nodes={"calculator_tools": calculator_tools},
        edges=[
            ("start", "calculator"),
            ("calculator", "calculator_tools", "has_tool_calls"),
            ("calculator", "end", "no_tool_calls"),
            ("calculator_tools", "calculator"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 2a: Calculadora com LoggingHandler ===")

    result = graph.invoke(
        {"messages": [HumanMessage(content="Quanto é 5 + 3?")]},
        config={"callbacks": [LoggingHandler()]},
    )

    assert "messages" in result


def test_callback_2b_calculator_with_trace_handler():
    """
    Calculadora com tools usando TraceHandler.
    Verifica que todos os nós são rastreados.
    """
    calc_tools = [add, sub, mult, div]

    calculator_agent = Agent(
        model=MODEL,
        name="calculator",
        prompt=(
            "Você é uma calculadora. Use as ferramentas disponíveis para "
            "realizar cálculos. Sempre use a ferramenta apropriada."
        ),
        tools=calc_tools,
    )

    calculator_tools = Tool(name="calculator_tools", tools=calc_tools)

    workflow = Workflow(
        state=MessagesState,
        agents=[calculator_agent],
        nodes={"calculator_tools": calculator_tools},
        edges=[
            ("start", "calculator"),
            ("calculator", "calculator_tools", "has_tool_calls"),
            ("calculator", "end", "no_tool_calls"),
            ("calculator_tools", "calculator"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 2b: Calculadora com TraceHandler ===")

    tracer = TraceHandler()
    result = graph.invoke(
        {"messages": [HumanMessage(content="Quanto é 6 * 7?")]},
        config={"callbacks": [tracer]},
    )

    print(f"Traces coletados: {len(tracer.traces)}")
    for trace in tracer.traces:
        print(f"  {trace['event']}: {trace.get('name', '')}")

    assert "messages" in result
    # Com tools, deve ter mais traces (calculator -> tools -> calculator)
    assert len(tracer.traces) >= 4


# =============================================================================
# Cenário 3: Supervisor com callbacks
# =============================================================================


def test_callback_3a_supervisor_with_logging_handler():
    """
    Supervisor-Escritor-Revisor usando LoggingHandler.
    Verifica trace de fluxo complexo com múltiplos agentes.
    """
    writer = Agent(
        model=MODEL,
        name="writer",
        prompt=(
            "Você é um escritor criativo. Escreva um texto curto (2-3 frases) "
            "baseado na sugestão fornecida. Seja criativo e original."
        ),
    )

    reviewer = Agent(
        model=MODEL,
        name="reviewer",
        prompt=(
            "Você é um revisor de textos. Analise o texto fornecido e dê feedback.\n"
            "Se o texto estiver bom, responda: 'APROVADO: [seu comentário]'\n"
            "Se precisar de melhorias, responda: 'REVISAR: [sugestões específicas]'"
        ),
    )

    supervisor = Agent(
        model=MODEL,
        name="supervisor",
        prompt=(
            "Você é um supervisor de fluxo editorial.\n\n"
            "Analise a conversa e decida o próximo passo:\n"
            "- Se receber uma sugestão de texto inicial, vá para 'writer'\n"
            "- Se o revisor aprovou (contém 'APROVADO'), vá para 'end'\n"
            "- Se o revisor pediu revisão (contém 'REVISAR'), vá para 'writer'\n\n"
            "Responda apenas com uma breve explicação da sua decisão."
        ),
        destinations=["writer", "end"],
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[supervisor, writer, reviewer],
        edges=[
            ("start", "supervisor"),
            ("writer", "reviewer"),
            ("reviewer", "supervisor"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 3a: Supervisor com LoggingHandler ===")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Escreva um texto curto sobre a importância da amizade."
                )
            ]
        },
        config={"callbacks": [LoggingHandler()]},
    )

    assert "messages" in result
    assert len(result["messages"]) >= 4


def test_callback_3b_supervisor_with_trace_handler():
    """
    Supervisor-Escritor-Revisor usando TraceHandler.
    Verifica trace completo do fluxo.
    """
    writer = Agent(
        model=MODEL,
        name="writer",
        prompt=(
            "Você é um escritor criativo. Escreva um texto curto (2-3 frases) "
            "baseado na sugestão fornecida. Seja criativo e original."
        ),
    )

    reviewer = Agent(
        model=MODEL,
        name="reviewer",
        prompt=(
            "Você é um revisor de textos. Analise o texto fornecido e dê feedback.\n"
            "Se o texto estiver bom, responda: 'APROVADO: [seu comentário]'\n"
            "Se precisar de melhorias, responda: 'REVISAR: [sugestões específicas]'"
        ),
    )

    supervisor = Agent(
        model=MODEL,
        name="supervisor",
        prompt=(
            "Você é um supervisor de fluxo editorial.\n\n"
            "Analise a conversa e decida o próximo passo:\n"
            "- Se receber uma sugestão de texto inicial, vá para 'writer'\n"
            "- Se o revisor aprovou (contém 'APROVADO'), vá para 'end'\n"
            "- Se o revisor pediu revisão (contém 'REVISAR'), vá para 'writer'\n\n"
            "Responda apenas com uma breve explicação da sua decisão."
        ),
        destinations=["writer", "end"],
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[supervisor, writer, reviewer],
        edges=[
            ("start", "supervisor"),
            ("writer", "reviewer"),
            ("reviewer", "supervisor"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 3b: Supervisor com TraceHandler ===")

    tracer = TraceHandler()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Escreva um texto curto sobre a importância da amizade."
                )
            ]
        },
        config={"callbacks": [tracer]},
    )

    print(f"Traces coletados: {len(tracer.traces)}")
    for trace in tracer.traces:
        if trace["event"] == "start":
            print(f"  → {trace.get('name', 'unknown')}")
        elif trace["event"] == "end":
            duration = trace.get("duration_seconds", 0)
            print(f"  ✓ ({duration:.2f}s)")

    assert "messages" in result
    # Fluxo complexo deve ter muitos traces
    assert len(tracer.traces) >= 6


# =============================================================================
# Cenário 4: Múltiplos callbacks
# =============================================================================


def test_callback_4_multiple_handlers():
    """
    Agente simples usando múltiplos handlers simultaneamente.
    Verifica que todos os handlers são chamados.
    """
    agent = Agent(
        model=MODEL,
        name="greeter",
        prompt="Você é um assistente amigável. Responda de forma breve e educada.",
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        edges=[
            ("start", "greeter"),
            ("greeter", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 4: Múltiplos Handlers ===")

    tracer = TraceHandler()
    result = graph.invoke(
        {"messages": [HumanMessage(content="Oi!")]},
        config={
            "callbacks": [
                LoggingHandler(),
                tracer,
            ]
        },
    )

    print(f"\nTraces coletados: {len(tracer.traces)}")

    assert "messages" in result
    assert len(tracer.traces) >= 2


# =============================================================================
# Cenário 5: TraceHandler com clear
# =============================================================================


def test_callback_5_trace_handler_clear():
    """
    Testa o método clear do TraceHandler.
    """
    agent = Agent(
        model=MODEL,
        name="greeter",
        prompt="Você é um assistente amigável. Responda de forma breve e educada.",
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        edges=[
            ("start", "greeter"),
            ("greeter", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 5: TraceHandler Clear ===")

    tracer = TraceHandler()

    # Primeira execução
    graph.invoke(
        {"messages": [HumanMessage(content="Oi!")]},
        config={"callbacks": [tracer]},
    )
    traces_before = len(tracer.traces)
    print(f"Traces após primeira execução: {traces_before}")

    # Clear
    tracer.clear()
    print(f"Traces após clear: {len(tracer.traces)}")
    assert len(tracer.traces) == 0

    # Segunda execução
    graph.invoke(
        {"messages": [HumanMessage(content="Tchau!")]},
        config={"callbacks": [tracer]},
    )
    traces_after = len(tracer.traces)
    print(f"Traces após segunda execução: {traces_after}")

    assert traces_after > 0


# =============================================================================
# Execução dos testes
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTES DE CALLBACKS - LangGraphLib")
    print("=" * 60)

    test_callback_1a_simple_agent_with_logging_handler()
    test_callback_1b_simple_agent_with_logging_handler_debug()
    test_callback_1c_simple_agent_with_trace_handler()
    test_callback_2a_calculator_with_logging_handler()
    test_callback_2b_calculator_with_trace_handler()
    test_callback_3a_supervisor_with_logging_handler()
    test_callback_3b_supervisor_with_trace_handler()
    test_callback_4_multiple_handlers()
    test_callback_5_trace_handler_clear()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES DE CALLBACKS CONCLUÍDOS!")
    print("=" * 60)
