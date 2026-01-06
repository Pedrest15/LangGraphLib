"""Testes de cenários da langgraphlib."""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraphlib.agent import Agent
from langgraphlib.model import get_model
from langgraphlib.state import MessagesState, create_state
from langgraphlib.workflow import Workflow

# Modelo padrão para os testes
MODEL = get_model("openai/gpt-4o-mini")


# =============================================================================
# Cenário 1: Agente simples com messages
# =============================================================================
def test_scenario_1_simple_agent():
    """
    Agente que recebe uma pergunta e retorna uma resposta.
    State: apenas messages.
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
    result = graph.invoke({"messages": [HumanMessage(content="Oi, como está?")]})

    print("\n=== Cenário 1: Agente Simples ===")
    print("Input: Oi, como está?")
    print(f"Output: {result['messages'][-1].content}")

    assert "messages" in result
    assert len(result["messages"]) >= 2


# =============================================================================
# Cenário 2: Agente com tools de calculadora
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


def test_scenario_2_calculator_agent():
    """
    Agente com tools de calculadora (add, sub, mult, div).
    State: apenas messages.
    """
    calculator_agent = Agent(
        model=MODEL,
        name="calculator",
        prompt=(
            "Você é uma calculadora. Use as ferramentas disponíveis para "
            "realizar cálculos. Sempre use a ferramenta apropriada."
        ),
        tools=[add, sub, mult, div],
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[calculator_agent],
        edges=[
            ("start", "calculator"),
            ("calculator", "calculator_tools", "has_tool_calls"),
            ("calculator", "end", "no_tool_calls"),
            ("calculator_tools", "calculator"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 2: Calculadora com Tools ===")

    # Teste de adição
    result = graph.invoke({"messages": [HumanMessage(content="Quanto é 5 + 3?")]})
    print(f"5 + 3 = {result['messages'][-1].content}")

    # Teste de subtração
    result = graph.invoke({"messages": [HumanMessage(content="Quanto é 10 - 4?")]})
    print(f"10 - 4 = {result['messages'][-1].content}")

    # Teste de multiplicação
    result = graph.invoke({"messages": [HumanMessage(content="Quanto é 6 * 7?")]})
    print(f"6 * 7 = {result['messages'][-1].content}")

    # Teste de divisão
    result = graph.invoke({"messages": [HumanMessage(content="Quanto é 20 / 4?")]})
    print(f"20 / 4 = {result['messages'][-1].content}")

    assert "messages" in result


# =============================================================================
# Cenário 3: Agente supervisor com subnós de funções
# =============================================================================
def test_scenario_3_supervisor_with_function_nodes():
    """
    Agente supervisor que direciona para subnós de funções (add, sub, mult, div).
    State: query (str), result (int).
    """
    # State customizado (sem messages)
    # - query: entrada do usuário
    # - result: resultado do cálculo (preenchido pelos subnós)
    # - reasoning: raciocínio do supervisor (qual operação identificou)
    MathState = create_state(
        "MathState",
        include_messages=False,
        query=(str, ""),
        result=(int, 0),
        reasoning=(str, ""),
    )

    # Funções dos subnós
    def add_node(state) -> dict:
        """Nó de adição."""
        # Extrai números da query (simplificado)
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2:
            return {"result": numbers[0] + numbers[1]}
        return {"result": 0}

    def sub_node(state) -> dict:
        """Nó de subtração."""
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2:
            return {"result": numbers[0] - numbers[1]}
        return {"result": 0}

    def mult_node(state) -> dict:
        """Nó de multiplicação."""
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2:
            return {"result": numbers[0] * numbers[1]}
        return {"result": 0}

    def div_node(state) -> dict:
        """Nó de divisão."""
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2 and numbers[1] != 0:
            return {"result": numbers[0] // numbers[1]}
        return {"result": 0}

    # Agente supervisor (só roteia, não calcula)
    # output_fields="reasoning" -> supervisor explica qual operação identificou
    # destinations -> LLM escolhe para qual nó ir (add, sub, mult, div)
    supervisor = Agent(
        model=MODEL,
        name="supervisor",
        prompt=(
            "Você é um roteador matemático. Analise a query e decida qual operação "
            "deve ser realizada. Explique brevemente seu raciocínio.\n"
            "Operações disponíveis: add (soma), sub (subtração), "
            "mult (multiplicação), div (divisão)."
        ),
        state=MathState,
        input_fields="query",
        output_fields="reasoning",
        destinations=["add", "sub", "mult", "div"],
    )

    workflow = Workflow(
        state=MathState,
        agents=[supervisor],
        nodes={
            "add": add_node,
            "sub": sub_node,
            "mult": mult_node,
            "div": div_node,
        },
        edges=[
            ("start", "supervisor"),
            ("add", "end"),
            ("sub", "end"),
            ("mult", "end"),
            ("div", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 3: Supervisor com Subnós de Funções ===")

    # Teste de adição
    result = graph.invoke({"query": "Quanto é 5 mais 3?"})
    print(f"5 + 3 = {result['result']}")

    # Teste de subtração
    result = graph.invoke({"query": "Calcule 10 menos 4"})
    print(f"10 - 4 = {result['result']}")

    # Teste de multiplicação
    result = graph.invoke({"query": "Multiplique 6 por 7"})
    print(f"6 * 7 = {result['result']}")

    # Teste de divisão
    result = graph.invoke({"query": "Divida 20 por 4"})
    print(f"20 / 4 = {result['result']}")

    assert "result" in result


# =============================================================================
# Cenário 3b: Mesmo cenário usando conditional_edges (sem destinations)
# =============================================================================
def test_scenario_3b_conditional_edges():
    """
    Mesmo cenário do 3, mas usando conditional_edges ao invés de destinations.
    O agente identifica a operação e salva no state, depois edges condicionais
    direcionam para o nó correto baseado no valor do campo 'operation'.
    """
    # State customizado
    MathState = create_state(
        "MathState",
        include_messages=False,
        query=(str, ""),
        result=(int, 0),
        operation=(str, ""),  # add, sub, mult, div
    )

    # Funções dos subnós (iguais ao cenário 3)
    def add_node(state) -> dict:
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2:
            return {"result": numbers[0] + numbers[1]}
        return {"result": 0}

    def sub_node(state) -> dict:
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2:
            return {"result": numbers[0] - numbers[1]}
        return {"result": 0}

    def mult_node(state) -> dict:
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2:
            return {"result": numbers[0] * numbers[1]}
        return {"result": 0}

    def div_node(state) -> dict:
        import re

        numbers = [int(n) for n in re.findall(r"\d+", state.query)]
        if len(numbers) >= 2 and numbers[1] != 0:
            return {"result": numbers[0] // numbers[1]}
        return {"result": 0}

    # Agente classificador (sem destinations - só identifica a operação)
    # Usa structured output para garantir que 'operation' seja preenchido corretamente
    classifier = Agent(
        model=MODEL,
        name="classifier",
        prompt=(
            "Você é um classificador de operações matemáticas.\n"
            "Analise a query e identifique qual operação deve ser realizada.\n"
            "Retorne APENAS uma das seguintes operações: add, sub, mult, div"
        ),
        state=MathState,
        input_fields="query",
        output_fields="operation",
    )

    # Workflow com conditional edges baseadas no campo 'operation'
    workflow = Workflow(
        state=MathState,
        agents=[classifier],
        nodes={
            "add": add_node,
            "sub": sub_node,
            "mult": mult_node,
            "div": div_node,
        },
        edges=[
            ("start", "classifier"),
            # Conditional edges baseadas no valor de state.operation
            ("classifier", "add", lambda s: s.operation == "add"),
            ("classifier", "sub", lambda s: s.operation == "sub"),
            ("classifier", "mult", lambda s: s.operation == "mult"),
            ("classifier", "div", lambda s: s.operation == "div"),
            ("add", "end"),
            ("sub", "end"),
            ("mult", "end"),
            ("div", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 3b: Conditional Edges (sem destinations) ===")

    # Teste de adição
    result = graph.invoke({"query": "Quanto é 5 mais 3?"})
    print(f"5 + 3 = {result['result']} (operation: {result['operation']})")

    # Teste de subtração
    result = graph.invoke({"query": "Calcule 10 menos 4"})
    print(f"10 - 4 = {result['result']} (operation: {result['operation']})")

    # Teste de multiplicação
    result = graph.invoke({"query": "Multiplique 6 por 7"})
    print(f"6 * 7 = {result['result']} (operation: {result['operation']})")

    # Teste de divisão
    result = graph.invoke({"query": "Divida 20 por 4"})
    print(f"20 / 4 = {result['result']} (operation: {result['operation']})")

    assert "result" in result
    assert "operation" in result


# =============================================================================
# Cenário 4: Grafo supervisor-escritor-revisor
# =============================================================================
def test_scenario_4_supervisor_writer_reviewer():
    """
    Grafo com supervisor, escritor e revisor.
    Fluxo: supervisor -> escritor -> revisor -> supervisor (loop até aprovado) -> end
    State: apenas messages.
    """
    # Agente escritor
    writer = Agent(
        model=MODEL,
        name="writer",
        prompt=(
            "Você é um escritor criativo. Escreva um texto curto (2-3 frases) "
            "baseado na sugestão fornecida. Seja criativo e original."
        ),
    )

    # Agente revisor
    reviewer = Agent(
        model=MODEL,
        name="reviewer",
        prompt=(
            "Você é um revisor de textos. Analise o texto fornecido e dê feedback.\n"
            "Se o texto estiver bom, responda: 'APROVADO: [seu comentário]'\n"
            "Se precisar de melhorias, responda: 'REVISAR: [sugestões específicas]'"
        ),
    )

    # Agente supervisor
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
        destinations=["writer", "reviewer", "end"],
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

    print("\n=== Cenário 4: Supervisor-Escritor-Revisor ===")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Escreva um texto curto sobre a importância da amizade."
                )
            ]
        }
    )

    print("\n--- Conversa Completa ---")
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"{i + 1}. [{role}]: {content}")

    assert "messages" in result
    # Pelo menos: input, writer, reviewer, supervisor
    assert len(result["messages"]) >= 4


# =============================================================================
# Execução dos testes
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTES DE CENÁRIOS - LangGraphLib")
    print("=" * 60)

    test_scenario_1_simple_agent()
    test_scenario_2_calculator_agent()
    test_scenario_3_supervisor_with_function_nodes()
    test_scenario_3b_conditional_edges()
    test_scenario_4_supervisor_writer_reviewer()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES CONCLUÍDOS!")
    print("=" * 60)
