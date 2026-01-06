from collections.abc import Callable
from typing import Any, Literal
from uuid import uuid4

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model


class Agent:
    """
    Agente para execução de LLMs com state dinâmico.

    O Agent encapsula a configuração de um LLM e permite execução
    com states customizados via invoke/ainvoke.

    Funcionalidades:
        - tools: Bind de tools ao LLM, retorna Command para {name}_tools
        - destinations: Roteamento dinâmico via structured output

    Examples:
        from langgraphlib.model import get_model
        from langgraphlib.state import create_state, MessagesState

        model = get_model("openai/gpt-4o")

        # Agente simples com messages
        agent = Agent(model=model, prompt="Você é um assistente útil.")
        result = agent.invoke(MessagesState(messages=[HumanMessage("Oi")]))

        # Agente com campos customizados
        MyState = create_state(
            include_messages=False,
            query=(str, ""),
            response=(str, ""),
        )
        agent = Agent(
            model=model,
            prompt="Responda a query do usuário.",
            input_fields="query",
            output_fields="response",
        )
        result = agent.invoke(MyState(query="Qual a capital do Brasil?"))

        # Agente com múltiplos campos de entrada/saída
        MultiState = create_state(
            include_messages=False,
            context=(str, ""),
            question=(str, ""),
            answer=(str, ""),
            confidence=(str, ""),
        )
        agent = Agent(
            model=model,
            prompt="Responda baseado no contexto.",
            input_fields=["context", "question"],
            output_fields=["answer", "confidence"],
        )

        # Agente com roteamento dinâmico (destinations)
        # O LLM decide para qual nó ir baseado no contexto
        supervisor = Agent(
            model=model,
            name="supervisor",
            prompt="Decida se deve pesquisar ou escrever.",
            destinations=["researcher", "writer", "end"],
        )
        # invoke() retorna Command(goto="researcher"|"writer"|"end", update={...})
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        name: str | None = None,
        prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        destinations: list[str] | None = None,
        input_fields: str | list[str] = "messages",
        output_fields: str | list[str] = "messages",
    ) -> None:
        """
        Inicializa o agente.

        Args:
            model: Instância do modelo LangChain.
            name: Nome do agente (gerado automaticamente se não fornecido).
            prompt: Prompt de sistema para o LLM.
            tools: Lista de ferramentas disponíveis para o agente.
            destinations: Lista de nomes de nós para roteamento dinâmico.
                Se fornecido, o agente usa structured output para decidir
                o próximo nó e retorna Command ao invés de dict.
            input_fields: Campo(s) do state para ler input. Pode ser string
                ou lista de strings para múltiplos campos.
            output_fields: Campo(s) do state para escrever output. Pode ser
                string ou lista de strings para múltiplos campos.
        """
        self._name = name or f"agent_{uuid4().hex[:8]}"
        self._prompt = prompt
        self._tools = tools or []
        self._destinations = destinations

        # Normaliza para lista
        self._input_fields = (
            [input_fields] if isinstance(input_fields, str) else list(input_fields)
        )
        self._output_fields = (
            [output_fields] if isinstance(output_fields, str) else list(output_fields)
        )

        # Cria schema de roteamento se destinations fornecido
        self._router_schema: type[BaseModel] | None = None
        if destinations:
            self._router_schema = self._create_router_schema(destinations)

        # Monta o modelo com prompt e tools
        self._model = self._build_model(model)

    def _create_router_schema(self, destinations: list[str]) -> type[BaseModel]:
        """
        Cria schema Pydantic dinâmico para roteamento.

        O schema força o LLM a retornar outputs E destino em uma única chamada.

        Args:
            destinations: Lista de nomes de nós válidos.

        Returns:
            Classe Pydantic com campos de output e 'goto'.
        """
        # Cria tipo Literal com os destinos permitidos
        destination_type = Literal[tuple(destinations)]  # type: ignore[valid-type]

        # Monta campos dinamicamente para cada output_field
        fields: dict[str, Any] = {}
        for field_name in self._output_fields:
            fields[field_name] = (
                str,
                Field(description=f"Valor para o campo '{field_name}'."),
            )

        # Adiciona campo goto
        fields["goto"] = (
            destination_type,
            Field(description="O próximo nó para onde o fluxo deve ir."),
        )

        return create_model("RouterSchema", **fields)

    def _build_model(self, llm: BaseChatModel) -> BaseChatModel:
        """Constrói modelo com prompt template, tools e/ou structured output."""
        # Bind tools se fornecidas
        if self._tools:
            llm = llm.bind_tools(self._tools)

        # Structured output para roteamento (se destinations e sem tools)
        # Tools têm prioridade - não pode usar structured output junto
        if self._router_schema and not self._tools:
            llm = llm.with_structured_output(self._router_schema)

        # Aplica prompt template se fornecido
        if self._prompt:
            # Monta lista de mensagens com placeholders para cada input field
            messages: list[Any] = [
                SystemMessagePromptTemplate.from_template(self._prompt)
            ]
            for field_name in self._input_fields:
                messages.append(MessagesPlaceholder(variable_name=field_name))

            prompt_template = ChatPromptTemplate.from_messages(messages)
            return prompt_template | llm

        return llm

    @property
    def name(self) -> str:
        """Nome do agente."""
        return self._name

    @property
    def model(self) -> BaseChatModel:
        """Modelo LLM do agente."""
        return self._model

    @property
    def prompt(self) -> str | None:
        """Prompt de sistema."""
        return self._prompt

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Lista de ferramentas do agente."""
        return self._tools

    @property
    def input_fields(self) -> list[str]:
        """Campos do state para input."""
        return self._input_fields

    @property
    def output_fields(self) -> list[str]:
        """Campos do state para output."""
        return self._output_fields

    @property
    def destinations(self) -> list[str] | None:
        """Lista de destinos para roteamento dinâmico."""
        return self._destinations

    def _prepare_input(self, state: BaseModel) -> dict[str, Any]:
        """Prepara input para o modelo a partir do state."""
        input_data: dict[str, Any] = {}

        for field_name in self._input_fields:
            input_value = getattr(state, field_name)

            # Converte string para lista de HumanMessage
            if isinstance(input_value, str):
                input_value = [HumanMessage(content=input_value)]

            input_data[field_name] = input_value

        return input_data

    def invoke(self, state: BaseModel) -> dict[str, Any] | Command:
        """
        Executa o agente com o state fornecido.

        Args:
            state: Instância do state com os dados de entrada.

        Returns:
            - Se destinations configurado: Command com goto e update do state.
            - Se tem tool_calls: Command para {agent_name}_tools.
            - Caso contrário: Dict parcial para atualizar o state.
        """
        input_data = self._prepare_input(state)
        response = self._model.invoke(input_data)

        # Caso 1: destinations (structured output) - response é dict/BaseModel
        if self._destinations and self._router_schema and not self._tools:
            # Extrai goto
            if isinstance(response, dict):
                goto = response.get("goto", self._destinations[0])
            else:
                goto = getattr(response, "goto", self._destinations[0])

            # Extrai cada output field
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                if isinstance(response, dict):
                    state_update[field_name] = response.get(field_name, "")
                else:
                    state_update[field_name] = getattr(response, field_name, "")

            return Command(goto=goto, update=state_update)

        # Caso 2: tools ou padrão - response é AIMessage
        # Primeiro output field recebe a resposta
        primary_field = self._output_fields[0]
        output_value = [response] if primary_field == "messages" else response.content
        state_update = {primary_field: output_value}

        # tool_calls - vai para tool node
        if hasattr(response, "tool_calls") and response.tool_calls:
            return Command(
                goto=f"{self._name}_tools",
                update=state_update,
            )

        # Caso padrão: retorna dict para atualizar state
        return state_update

    async def ainvoke(self, state: BaseModel) -> dict[str, Any] | Command:
        """
        Executa o agente de forma assíncrona.

        Args:
            state: Instância do state com os dados de entrada.

        Returns:
            - Se destinations configurado: Command com goto e update do state.
            - Se tem tool_calls: Command para {agent_name}_tools.
            - Caso contrário: Dict parcial para atualizar o state.
        """
        input_data = self._prepare_input(state)
        response = await self._model.ainvoke(input_data)

        # Caso 1: destinations (structured output) - response é dict/BaseModel
        if self._destinations and self._router_schema and not self._tools:
            # Extrai goto
            if isinstance(response, dict):
                goto = response.get("goto", self._destinations[0])
            else:
                goto = getattr(response, "goto", self._destinations[0])

            # Extrai cada output field
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                if isinstance(response, dict):
                    state_update[field_name] = response.get(field_name, "")
                else:
                    state_update[field_name] = getattr(response, field_name, "")

            return Command(goto=goto, update=state_update)

        # Caso 2: tools ou padrão - response é AIMessage
        # Primeiro output field recebe a resposta
        primary_field = self._output_fields[0]
        output_value = [response] if primary_field == "messages" else response.content
        state_update = {primary_field: output_value}

        # tool_calls - vai para tool node
        if hasattr(response, "tool_calls") and response.tool_calls:
            return Command(
                goto=f"{self._name}_tools",
                update=state_update,
            )

        # Caso padrão: retorna dict para atualizar state
        return state_update

    def __repr__(self) -> str:
        """Representação do agente."""
        return (
            f"Agent(name={self._name!r}, "
            f"input_fields={self._input_fields!r}, "
            f"output_fields={self._output_fields!r}, "
            f"tools={len(self._tools)})"
        )
