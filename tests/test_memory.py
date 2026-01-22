"""Testes de memória da langgraphlib."""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langgraphlib import (
    Agent,
    MemoryManager,
    MessagesState,
    Tool,
    Workflow,
    create_memory_retriever_node,
    create_memory_saver_node,
    create_memory_tools,
    create_state,
    get_model,
)

# Modelo padrão para os testes
MODEL = get_model("openai/gpt-4o-mini")


# =============================================================================
# Cenário 1: Agente conversacional com memória
# =============================================================================


def test_memory_1a_conversational_with_tools():
    """
    Agente conversacional usando tools de memória.
    Primeiro: 'Olá, meu nome é Jack Jack'
    Depois: 'Qual o meu nome?'
    """
    store = InMemoryStore()
    checkpointer = MemorySaver()
    memory = MemoryManager(store, user_id="user_test_1a")

    memory_tools = create_memory_tools(memory)

    agent = Agent(
        model=MODEL,
        name="assistant",
        prompt=(
            "Você é um assistente amigável com memória. "
            "SEMPRE use 'remember' para salvar informações importantes sobre o usuário "
            "(como nome, preferências, etc) assim que as receber. "
            "SEMPRE use 'recall' para buscar informações salvas quando o usuário "
            "perguntar sobre algo que pode ter sido mencionado antes."
        ),
        tools=memory_tools,
    )

    # Tool para as ferramentas de memória
    assistant_tools = Tool(name="assistant_tools", tools=memory_tools)

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        nodes={"assistant_tools": assistant_tools},
        checkpointer=checkpointer,
        edges=[
            ("start", "assistant"),
            ("assistant", "assistant_tools", "has_tool_calls"),
            ("assistant", "end", "no_tool_calls"),
            ("assistant_tools", "assistant"),
        ],
    )

    graph = workflow.compile()
    config = {"configurable": {"thread_id": "thread_1a"}}

    print("\n=== Cenário 1a: Conversacional com Tools ===")

    # Primeira mensagem - apresentação
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="Olá, meu nome é Jack Jack")]},
        config=config,
    )
    print("User: Olá, meu nome é Jack Jack")
    print(f"Assistant: {result1['messages'][-1].content}")

    # Segunda mensagem - pergunta sobre o nome
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Qual o meu nome?")]},
        config=config,
    )
    print("User: Qual o meu nome?")
    print(f"Assistant: {result2['messages'][-1].content}")

    # Verifica se a resposta menciona o nome
    assert "messages" in result2
    final_response = result2["messages"][-1].content.lower()
    assert "jack" in final_response, f"Esperava 'jack' na resposta: {final_response}"


def test_memory_1b_conversational_with_nodes():
    """
    Agente conversacional usando nós de memória.
    Primeiro: 'Olá, meu nome é Jack Jack'
    Depois: 'Qual o meu nome?'
    """
    store = InMemoryStore()
    checkpointer = MemorySaver()
    memory = MemoryManager(store, user_id="user_test_1b")

    # State com campo para contexto de memória (como string para o prompt)
    StateWithMemory = create_state(
        "StateWithMemory",
        memory_context=(str, ""),
    )

    # Nós de memória
    memory_retriever = create_memory_retriever_node(memory, limit=5)
    memory_saver = create_memory_saver_node(MODEL, memory)

    agent = Agent(
        model=MODEL,
        name="assistant",
        prompt=(
            "Você é um assistente amigável. "
            "Use o contexto de memória fornecido para personalizar suas respostas.\n"
            "Contexto de memória: {memory_context}"
        ),
        state=StateWithMemory,
        input_fields=["messages", "memory_context"],
    )

    workflow = Workflow(
        state=StateWithMemory,
        agents=[agent],
        nodes={
            "retrieve_memory": memory_retriever,
            "save_memory": memory_saver,
        },
        checkpointer=checkpointer,
        edges=[
            ("start", "retrieve_memory"),
            ("retrieve_memory", "assistant"),
            ("assistant", "save_memory"),
            ("save_memory", "end"),
        ],
    )

    graph = workflow.compile()
    config = {"configurable": {"thread_id": "thread_1b"}}

    print("\n=== Cenário 1b: Conversacional com Nós ===")

    # Primeira mensagem - apresentação
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="Olá, meu nome é Jack Jack")]},
        config=config,
    )
    print("User: Olá, meu nome é Jack Jack")
    print(f"Assistant: {result1['messages'][-1].content}")

    # Segunda mensagem - pergunta sobre o nome
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Qual o meu nome?")]},
        config=config,
    )
    print("User: Qual o meu nome?")
    print(f"Assistant: {result2['messages'][-1].content}")

    # Verifica se a resposta menciona o nome
    assert "messages" in result2
    final_response = result2["messages"][-1].content.lower()
    assert "jack" in final_response, f"Esperava 'jack' na resposta: {final_response}"


# =============================================================================
# Cenário 2: Calculadora com memória
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


def test_memory_2a_calculator_with_tools():
    """
    Calculadora usando tools de memória.
    Primeiro: '7*8?'
    Depois: 'Qual foi minha ultima pergunta?'
    """
    store = InMemoryStore()
    checkpointer = MemorySaver()
    memory = MemoryManager(store, user_id="user_test_2a")

    memory_tools = create_memory_tools(memory)
    calc_tools = [add, sub, mult, div]

    agent = Agent(
        model=MODEL,
        name="calculator",
        prompt=(
            "Você é uma calculadora com memória. "
            "Use as ferramentas de cálculo (add, sub, mult, div) para operações "
            "matemáticas. SEMPRE use 'remember' para salvar cada pergunta do "
            "usuário ANTES de responder. Use 'recall' para buscar perguntas "
            "anteriores quando o usuário perguntar sobre histórico."
        ),
        tools=calc_tools + memory_tools,
    )

    # Tool para as ferramentas da calculadora
    calculator_tools = Tool(name="calculator_tools", tools=calc_tools + memory_tools)

    workflow = Workflow(
        state=MessagesState,
        agents=[agent],
        nodes={"calculator_tools": calculator_tools},
        checkpointer=checkpointer,
        edges=[
            ("start", "calculator"),
            ("calculator", "calculator_tools", "has_tool_calls"),
            ("calculator", "end", "no_tool_calls"),
            ("calculator_tools", "calculator"),
        ],
    )

    graph = workflow.compile()
    config = {"configurable": {"thread_id": "thread_2a"}}

    print("\n=== Cenário 2a: Calculadora com Tools ===")

    # Primeira mensagem - cálculo
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="7*8?")]},
        config=config,
    )
    print("User: 7*8?")
    print(f"Assistant: {result1['messages'][-1].content}")

    # Segunda mensagem - pergunta sobre a última pergunta
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Qual foi minha ultima pergunta?")]},
        config=config,
    )
    print("User: Qual foi minha ultima pergunta?")
    print(f"Assistant: {result2['messages'][-1].content}")

    # Verifica se a resposta menciona a operação anterior
    assert "messages" in result2
    final_response = result2["messages"][-1].content.lower()
    # Deve mencionar 7, 8 ou multiplicação
    assert any(x in final_response for x in ["7", "8", "mult", "×", "*"]), (
        f"Esperava referência ao cálculo anterior: {final_response}"
    )


def test_memory_2b_calculator_with_nodes():
    """
    Calculadora usando nós de memória.
    Primeiro: '7*8?'
    Depois: 'Qual foi minha ultima pergunta?'
    """
    store = InMemoryStore()
    checkpointer = MemorySaver()
    memory = MemoryManager(store, user_id="user_test_2b")

    # State com campo para contexto de memória (string para prompt)
    StateWithMemory = create_state(
        "StateWithMemory",
        memory_context=(str, ""),
    )

    # Nós de memória
    memory_retriever = create_memory_retriever_node(memory, limit=5)
    memory_saver = create_memory_saver_node(MODEL, memory)

    calc_tools = [add, sub, mult, div]

    agent = Agent(
        model=MODEL,
        name="calculator",
        prompt=(
            "Você é uma calculadora com memória. "
            "Use as ferramentas de cálculo para operações matemáticas. "
            "Considere o contexto de memória para responder perguntas sobre interações "
            "anteriores.\n"
            "Contexto de memória: {memory_context}"
        ),
        state=StateWithMemory,
        input_fields=["messages", "memory_context"],
        tools=calc_tools,
    )

    # Tool para as ferramentas de cálculo
    calculator_tools = Tool(name="calculator_tools", tools=calc_tools)

    workflow = Workflow(
        state=StateWithMemory,
        agents=[agent],
        nodes={
            "retrieve_memory": memory_retriever,
            "save_memory": memory_saver,
            "calculator_tools": calculator_tools,
        },
        checkpointer=checkpointer,
        edges=[
            ("start", "retrieve_memory"),
            ("retrieve_memory", "calculator"),
            ("calculator", "calculator_tools", "has_tool_calls"),
            ("calculator", "save_memory", "no_tool_calls"),
            ("calculator_tools", "calculator"),
            ("save_memory", "end"),
        ],
    )

    graph = workflow.compile()
    config = {"configurable": {"thread_id": "thread_2b"}}

    print("\n=== Cenário 2b: Calculadora com Nós ===")

    # Primeira mensagem - cálculo
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="7*8?")]},
        config=config,
    )
    print("User: 7*8?")
    print(f"Assistant: {result1['messages'][-1].content}")

    # Segunda mensagem - pergunta sobre a última pergunta
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Qual foi minha ultima pergunta?")]},
        config=config,
    )
    print("User: Qual foi minha ultima pergunta?")
    print(f"Assistant: {result2['messages'][-1].content}")

    # Verifica se a resposta menciona a operação anterior
    assert "messages" in result2
    final_response = result2["messages"][-1].content.lower()
    assert any(x in final_response for x in ["7", "8", "mult", "×", "*"]), (
        f"Esperava referência ao cálculo anterior: {final_response}"
    )


# =============================================================================
# Cenário 3: Supervisor-Escritor-Revisor com memória
# =============================================================================


def test_memory_3a_supervisor_writer_reviewer_with_tools():
    """
    Grafo supervisor-escritor-revisor usando tools de memória.
    Primeiro: 'Escreva um texto curto sobre a importância da amizade.'
    Depois: 'Qual o assunto do ultimo texto escrito?'

    Usa destinations junto com tools - o Agent faz duas chamadas quando necessário:
    1. Primeira chamada: processa mensagem e pode chamar tools
    2. Segunda chamada (sem tool_calls): decide o destino via structured output
    """
    store = InMemoryStore()
    checkpointer = MemorySaver()
    memory = MemoryManager(store, user_id="user_test_3a")

    memory_tools = create_memory_tools(memory)

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

    # Agente supervisor com memória E destinations
    # O Agent vai usar duas chamadas: uma para tools, outra para routing
    supervisor = Agent(
        model=MODEL,
        name="supervisor",
        prompt=(
            "Você é um supervisor de fluxo editorial com memória.\n\n"
            "Analise a conversa e decida o próximo passo:\n\n"
            "1. Se receber uma sugestão de texto inicial do usuário, vá para 'writer'\n"
            "2. Se o revisor pediu revisão (contém 'REVISAR'), vá para 'writer'\n"
            "3. Se o revisor aprovou (contém 'APROVADO'):\n"
            "   - Se ainda não salvou o assunto, use 'remember' para salvar\n"
            "   - Se já salvou (vê resultado de remember na conversa), vá para 'end'\n"
            "4. Se perguntaram sobre textos anteriores:\n"
            "   - Se ainda não buscou, use 'recall' para buscar\n"
            "   - Se já buscou (vê resultado de recall na conversa), responda e "
            "vá para 'end'\n\n"
            "IMPORTANTE: Nunca chame a mesma tool duas vezes. Se você vê o resultado "
            "de uma tool call na conversa, a tool já foi executada."
        ),
        tools=memory_tools,
        destinations=["writer", "end"],
    )

    # Tool criado para as ferramentas do supervisor
    supervisor_tools = Tool(name="supervisor_tools", tools=memory_tools)

    workflow = Workflow(
        state=MessagesState,
        agents=[supervisor, writer, reviewer],
        nodes={"supervisor_tools": supervisor_tools},
        checkpointer=checkpointer,
        edges=[
            ("start", "supervisor"),
            # Sem edge condicional - o Command(goto=...) do supervisor
            # direciona dinamicamente para supervisor_tools, writer ou end
            ("supervisor_tools", "supervisor"),
            ("writer", "reviewer"),
            ("reviewer", "supervisor"),
        ],
    )

    graph = workflow.compile()
    config = {"configurable": {"thread_id": "thread_3a"}}

    print("\n=== Cenário 3a: Supervisor-Escritor-Revisor com Tools ===")

    # Primeira mensagem - pede texto
    result1 = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Escreva um texto curto sobre a importância da amizade."
                )
            ]
        },
        config=config,
    )
    print("User: Escreva um texto curto sobre a importância da amizade.")
    print(f"Última mensagem: {result1['messages'][-1].content[:200]}...")

    # Segunda mensagem - pergunta sobre o assunto
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Qual o assunto do ultimo texto escrito?")]},
        config=config,
    )
    print("User: Qual o assunto do ultimo texto escrito?")
    print(f"Assistant: {result2['messages'][-1].content}")


def test_memory_3b_supervisor_writer_reviewer_with_nodes():
    """
    Grafo supervisor-escritor-revisor usando nós de memória.
    Primeiro: 'Escreva um texto curto sobre a importância da amizade.'
    Depois: 'Qual o assunto do ultimo texto escrito?'
    """
    store = InMemoryStore()
    checkpointer = MemorySaver()
    memory = MemoryManager(store, user_id="user_test_3b")

    # State com campo para contexto de memória (string para prompt)
    StateWithMemory = create_state(
        "StateWithMemory",
        memory_context=(str, ""),
    )

    # Nós de memória
    memory_retriever = create_memory_retriever_node(memory, limit=5)
    memory_saver = create_memory_saver_node(MODEL, memory)

    # Agente escritor
    writer = Agent(
        model=MODEL,
        name="writer",
        prompt=(
            "Você é um escritor criativo. Escreva um texto curto (2-3 frases) "
            "baseado na sugestão fornecida. Seja criativo e original."
        ),
        state=StateWithMemory,
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
        state=StateWithMemory,
    )

    # Agente supervisor
    supervisor = Agent(
        model=MODEL,
        name="supervisor",
        prompt=(
            "Você é um supervisor de fluxo editorial.\n\n"
            "Considere o contexto de memória para informações sobre textos "
            "anteriores.\nContexto de memória: {memory_context}\n\n"
            "Analise a ÚLTIMA mensagem na conversa e decida o próximo passo:\n"
            "1. Se a última mensagem é do USUÁRIO pedindo para escrever, "
            "vá para 'writer'\n"
            "2. Se a última mensagem contém 'APROVADO' (do revisor), vá para 'end'\n"
            "3. Se a última mensagem contém 'REVISAR' (do revisor), vá para 'writer'\n"
            "4. Se a última mensagem é do USUÁRIO perguntando sobre textos anteriores, "
            "responda usando o contexto de memória e vá para 'end'\n\n"
            "Responda apenas com uma breve explicação da sua decisão."
        ),
        state=StateWithMemory,
        input_fields=["messages", "memory_context"],
        destinations=["writer", "end"],
    )

    workflow = Workflow(
        state=StateWithMemory,
        agents=[supervisor, writer, reviewer],
        nodes={
            "retrieve_memory": memory_retriever,
            "save_memory": memory_saver,
        },
        checkpointer=checkpointer,
        edges=[
            ("start", "retrieve_memory"),
            ("retrieve_memory", "supervisor"),
            ("writer", "reviewer"),
            ("reviewer", "save_memory"),
            ("save_memory", "supervisor"),
        ],
    )

    graph = workflow.compile()
    config = {"configurable": {"thread_id": "thread_3b"}}

    print("\n=== Cenário 3b: Supervisor-Escritor-Revisor com Nós ===")

    # Primeira mensagem - pede texto
    result1 = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Escreva um texto curto sobre a importância da amizade."
                )
            ]
        },
        config=config,
    )
    print("User: Escreva um texto curto sobre a importância da amizade.")
    print(f"Última mensagem: {result1['messages'][-1].content[:200]}...")

    # Segunda mensagem - pergunta sobre o assunto
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Qual o assunto do ultimo texto escrito?")]},
        config=config,
    )
    print("User: Qual o assunto do ultimo texto escrito?")
    print(f"Assistant: {result2['messages'][-1].content}")


# =============================================================================
# Execução dos testes
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTES DE MEMÓRIA - LangGraphLib")
    print("=" * 60)

    test_memory_1a_conversational_with_tools()
    test_memory_1b_conversational_with_nodes()
    test_memory_2a_calculator_with_tools()
    test_memory_2b_calculator_with_nodes()
    test_memory_3a_supervisor_writer_reviewer_with_tools()
    test_memory_3b_supervisor_writer_reviewer_with_nodes()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES DE MEMÓRIA CONCLUÍDOS!")
    print("=" * 60)
