"""Testes de cenários da langgraphlib."""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraphlib import Agent, MessagesState, Tool, Workflow, create_state, get_model

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

    # Tool para as ferramentas de cálculo
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
# Cenário 5: Execução paralela (fan-out e fan-in)
# =============================================================================


def test_scenario_5a_parallel_execution_with_list_syntax():
    """
    Testa execução paralela usando sintaxe de lista para fan-out.
    Dois agentes executam em paralelo e convergem para um terceiro.
    """
    import operator
    from typing import Annotated

    # State com reducer para acumular resultados
    ParallelState = create_state(results=(Annotated[list[str], operator.add], []))

    # Agentes que escrevem em "results"
    researcher = Agent(
        model=MODEL,
        name="researcher",
        prompt="Você é um pesquisador. Retorne apenas: 'Pesquisa concluída'",
        output_fields="results",
        state=ParallelState,
    )

    analyst = Agent(
        model=MODEL,
        name="analyst",
        prompt="Você é um analista. Retorne apenas: 'Análise concluída'",
        output_fields="results",
        state=ParallelState,
    )

    summarizer = Agent(
        model=MODEL,
        name="summarizer",
        prompt=(
            "Você é um sintetizador. Leia os resultados anteriores e "
            "faça um resumo breve. Retorne apenas o resumo."
        ),
        input_fields="results",
        output_fields="results",
        state=ParallelState,
    )

    workflow = Workflow(
        state=ParallelState,
        agents=[researcher, analyst, summarizer],
        edges=[
            # Fan-out: start -> researcher E analyst (paralelo)
            ("start", ["researcher", "analyst"]),
            # Fan-in: ambos convergem para summarizer
            ("researcher", "summarizer"),
            ("analyst", "summarizer"),
            ("summarizer", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 5a: Execução Paralela (sintaxe lista) ===")

    result = graph.invoke({"results": []})

    print(f"Resultados: {result['results']}")

    assert "results" in result
    # Deve ter pelo menos 3 resultados: researcher, analyst, summarizer
    assert len(result["results"]) >= 3


def test_scenario_5b_parallel_execution_with_multiple_edges():
    """
    Testa execução paralela usando múltiplas edges para fan-out.
    Equivalente ao teste anterior, mas com sintaxe de edges separadas.
    """
    import operator
    from typing import Annotated

    # State com reducer para acumular resultados
    ParallelState = create_state(results=(Annotated[list[str], operator.add], []))

    # Agentes que escrevem em "results"
    researcher = Agent(
        model=MODEL,
        name="researcher",
        prompt="Você é um pesquisador. Retorne apenas: 'Pesquisa concluída'",
        output_fields="results",
        state=ParallelState,
    )

    analyst = Agent(
        model=MODEL,
        name="analyst",
        prompt="Você é um analista. Retorne apenas: 'Análise concluída'",
        output_fields="results",
        state=ParallelState,
    )

    summarizer = Agent(
        model=MODEL,
        name="summarizer",
        prompt=(
            "Você é um sintetizador. Leia os resultados anteriores e "
            "faça um resumo breve. Retorne apenas o resumo."
        ),
        input_fields="results",
        output_fields="results",
        state=ParallelState,
    )

    workflow = Workflow(
        state=ParallelState,
        agents=[researcher, analyst, summarizer],
        edges=[
            # Fan-out usando múltiplas edges
            ("start", "researcher"),
            ("start", "analyst"),
            # Fan-in: ambos convergem para summarizer
            ("researcher", "summarizer"),
            ("analyst", "summarizer"),
            ("summarizer", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 5b: Execução Paralela (múltiplas edges) ===")

    result = graph.invoke({"results": []})

    print(f"Resultados: {result['results']}")

    assert "results" in result
    # Deve ter pelo menos 3 resultados: researcher, analyst, summarizer
    assert len(result["results"]) >= 3


def test_scenario_5c_parallel_with_messages():
    """
    Testa execução paralela mantendo messages como state principal.
    Usa MessagesState padrão com reducer add_messages.
    """
    # Agentes simples que respondem brevemente
    fact_checker = Agent(
        model=MODEL,
        name="fact_checker",
        prompt="Você verifica fatos. Responda brevemente: 'Fatos verificados.'",
    )

    grammar_checker = Agent(
        model=MODEL,
        name="grammar_checker",
        prompt="Você verifica gramática. Responda brevemente: 'Gramática OK.'",
    )

    final_reviewer = Agent(
        model=MODEL,
        name="final_reviewer",
        prompt=(
            "Você é o revisor final. Analise as verificações anteriores "
            "e dê um parecer final breve."
        ),
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[fact_checker, grammar_checker, final_reviewer],
        edges=[
            # Fan-out: verificações em paralelo
            ("start", ["fact_checker", "grammar_checker"]),
            # Fan-in: convergem para revisor final
            ("fact_checker", "final_reviewer"),
            ("grammar_checker", "final_reviewer"),
            ("final_reviewer", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 5c: Execução Paralela com Messages ===")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Verifique este texto: O sol é uma estrela.")
            ]
        }
    )

    print(f"Total de mensagens: {len(result['messages'])}")
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  {i + 1}. [{role}]: {content}")

    assert "messages" in result
    # Pelo menos 4: input, fact_checker, grammar_checker, final_reviewer
    assert len(result["messages"]) >= 4


# =============================================================================
# Cenário 6: Map-Reduce com Send (fan-out dinâmico)
# =============================================================================


def test_scenario_6a_map_reduce_with_send():
    """
    Testa map-reduce usando Send para fan-out dinâmico.
    Número de branches determinado em runtime baseado nos dados.
    """
    import operator
    from typing import Annotated

    from langgraphlib import Send

    # State com lista de itens para processar e resultados acumulados
    MapReduceState = create_state(
        items=(list[str], []),
        results=(Annotated[list[str], operator.add], []),
    )

    # Função de distribuição: cria um Send para cada item
    def distribute_items(state) -> list[Send]:
        return [
            Send("process_item", {"items": [], "results": [], "current_item": item})
            for item in state.items
        ]

    # Nó que processa cada item individualmente
    def process_item(state) -> dict:
        item = state.get("current_item", "unknown")
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

    graph = workflow.compile()

    print("\n=== Cenário 6a: Map-Reduce com Send ===")

    # Testa com 3 itens
    result = graph.invoke({"items": ["item_a", "item_b", "item_c"], "results": []})

    print(f"Items processados: {result['results']}")

    assert "results" in result
    # Deve ter 3 resultados (um para cada item)
    assert len(result["results"]) == 3
    assert "Processado: item_a" in result["results"]
    assert "Processado: item_b" in result["results"]
    assert "Processado: item_c" in result["results"]


def test_scenario_6b_map_reduce_with_agent():
    """
    Testa map-reduce usando Send com agentes LLM.
    Cada item é processado por um agente em paralelo.
    """
    import operator
    from typing import Annotated

    from langgraphlib import Send

    # State com lista de tópicos e análises acumuladas
    AnalysisState = create_state(
        topics=(list[str], []),
        analyses=(Annotated[list[str], operator.add], []),
        current_topic=(str, ""),
    )

    # Agente que analisa cada tópico
    analyzer = Agent(
        model=MODEL,
        name="analyzer",
        prompt="Analise brevemente o tópico fornecido em 1 frase.",
        input_fields="current_topic",
        output_fields="analyses",
        state=AnalysisState,
    )

    # Função de distribuição: cria um Send para cada tópico
    def distribute_topics(state) -> list[Send]:
        return [
            Send(
                "analyzer",
                {"topics": [], "analyses": [], "current_topic": topic},
            )
            for topic in state.topics
        ]

    workflow = Workflow(
        state=AnalysisState,
        agents=[analyzer],
        edges=[
            ("start", distribute_topics),
            ("analyzer", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 6b: Map-Reduce com Agent ===")

    # Testa com 2 tópicos para ser rápido
    result = graph.invoke({
        "topics": ["inteligência artificial", "energia renovável"],
        "analyses": [],
        "current_topic": "",
    })

    print(f"Análises: {len(result['analyses'])} tópicos analisados")
    for i, analysis in enumerate(result["analyses"]):
        preview = analysis[:80] + "..." if len(analysis) > 80 else analysis
        print(f"  {i + 1}. {preview}")

    assert "analyses" in result
    # Deve ter 2 análises (uma para cada tópico)
    assert len(result["analyses"]) == 2


def test_scenario_6c_map_reduce_empty_list():
    """
    Testa comportamento quando a lista de itens está vazia.
    Deve ir direto para o end sem processar nada.
    """
    import operator
    from typing import Annotated

    from langgraphlib import Send

    MapReduceState = create_state(
        items=(list[str], []),
        results=(Annotated[list[str], operator.add], []),
    )

    def distribute_items(state) -> list[Send]:
        # Se não há itens, retorna lista vazia de Sends
        return [
            Send("process_item", {"items": [], "results": [], "current_item": item})
            for item in state.items
        ]

    def process_item(state) -> dict:
        item = state.get("current_item", "unknown")
        return {"results": [f"Processado: {item}"]}

    workflow = Workflow(
        state=MapReduceState,
        nodes={"process_item": process_item},
        edges=[
            ("start", distribute_items),
            ("process_item", "end"),
        ],
    )

    graph = workflow.compile()

    print("\n=== Cenário 6c: Map-Reduce com lista vazia ===")

    # Testa com lista vazia
    result = graph.invoke({"items": [], "results": []})

    print(f"Items processados: {result['results']}")

    assert "results" in result
    # Com lista vazia, não deve ter resultados
    assert len(result["results"]) == 0


# =============================================================================
# Cenário 7: Streaming de tokens LLM
# =============================================================================


def test_scenario_7a_streaming_messages():
    """
    Testa streaming de tokens LLM usando stream_mode="messages".
    Verifica que os chunks são recebidos progressivamente.
    """
    agent = Agent(
        model=MODEL,
        name="writer",
        prompt="Escreva uma frase curta sobre programação.",
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        edges=[
            ("start", "writer"),
            ("writer", "end"),
        ],
    )

    workflow.compile()

    print("\n=== Cenário 7a: Streaming de Mensagens ===")

    chunks_received = []
    full_content = ""

    for chunk, _metadata in workflow.stream(
        {"messages": [HumanMessage(content="Oi")]}
    ):
        chunks_received.append(chunk)
        if hasattr(chunk, "content") and chunk.content:
            full_content += chunk.content
            print(chunk.content, end="", flush=True)

    print()  # Nova linha após streaming
    print(f"Total de chunks recebidos: {len(chunks_received)}")
    print(f"Conteúdo final: {full_content[:100]}...")

    # Verificações
    assert len(chunks_received) > 0, "Deve receber pelo menos um chunk"
    assert len(full_content) > 0, "Deve ter conteúdo acumulado"


def test_scenario_7b_streaming_multiple_agents():
    """
    Testa streaming com múltiplos agentes em sequência.
    Verifica que os tokens de cada agente são recebidos.
    """
    researcher = Agent(
        model=MODEL,
        name="researcher",
        prompt="Diga uma frase curta sobre pesquisa.",
    )

    writer = Agent(
        model=MODEL,
        name="writer",
        prompt="Diga uma frase curta sobre escrita.",
    )

    workflow = Workflow(
        state=MessagesState,
        agents=[researcher, writer],
        edges=[
            ("start", "researcher"),
            ("researcher", "writer"),
            ("writer", "end"),
        ],
    )

    workflow.compile()

    print("\n=== Cenário 7b: Streaming com Múltiplos Agentes ===")

    chunks_by_node: dict[str, list] = {}

    for chunk, metadata in workflow.stream(
        {"messages": [HumanMessage(content="Olá")]}
    ):
        node = metadata.get("langgraph_node", "unknown")
        if node not in chunks_by_node:
            chunks_by_node[node] = []
        chunks_by_node[node].append(chunk)

        if hasattr(chunk, "content") and chunk.content:
            print(f"[{node}] {chunk.content}", end="", flush=True)

    print()  # Nova linha
    print(f"Nodes que emitiram chunks: {list(chunks_by_node.keys())}")

    # Verificações
    assert len(chunks_by_node) > 0, "Deve ter chunks de pelo menos um node"


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
    test_scenario_5a_parallel_execution_with_list_syntax()
    test_scenario_5b_parallel_execution_with_multiple_edges()
    test_scenario_5c_parallel_with_messages()
    test_scenario_6a_map_reduce_with_send()
    test_scenario_6b_map_reduce_with_agent()
    test_scenario_6c_map_reduce_empty_list()
    test_scenario_7a_streaming_messages()
    test_scenario_7b_streaming_multiple_agents()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES CONCLUÍDOS!")
    print("=" * 60)
