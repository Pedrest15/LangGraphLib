from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class Agent:
    """
    Agente para execução de LLMs com state dinâmico.

    O Agent encapsula a configuração de um LLM e permite execução
    com states customizados via invoke/ainvoke. Não executa tools -
    apenas retorna a resposta do LLM para o Workflow decidir.

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
            input_field="query",
            output_field="response",
        )
        result = agent.invoke(MyState(query="Qual a capital do Brasil?"))
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        name: str | None = None,
        prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        input_field: str = "messages",
        output_field: str = "messages",
    ) -> None:
        """
        Inicializa o agente.

        Args:
            model: Instância do modelo LangChain.
            name: Nome do agente (gerado automaticamente se não fornecido).
            prompt: Prompt de sistema para o LLM.
            tools: Lista de ferramentas disponíveis para o agente.
            input_field: Campo do state para ler input.
            output_field: Campo do state para escrever output.
        """
        self._name = name or f"agent_{uuid4().hex[:8]}"
        self._prompt = prompt
        self._tools = tools or []
        self._input_field = input_field
        self._output_field = output_field

        # Monta o modelo com prompt e tools
        self._model = self._build_model(model)

    def _build_model(self, llm: BaseChatModel) -> BaseChatModel:
        """Constrói modelo com prompt template e tools."""
        if self._tools:
            llm = llm.bind_tools(self._tools)

        if self._prompt:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(self._prompt),
                    MessagesPlaceholder(variable_name=self._input_field),
                ]
            )
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
    def input_field(self) -> str:
        """Campo do state para input."""
        return self._input_field

    @property
    def output_field(self) -> str:
        """Campo do state para output."""
        return self._output_field

    def _prepare_input(self, state: BaseModel) -> dict[str, Any]:
        """Prepara input para o modelo a partir do state."""
        input_value = getattr(state, self._input_field)

        # Converte string para lista de HumanMessage
        if isinstance(input_value, str):
            input_value = [HumanMessage(content=input_value)]

        return {self._input_field: input_value}

    def invoke(self, state: BaseModel) -> dict[str, Any]:
        """
        Executa o agente com o state fornecido.

        Args:
            state: Instância do state com os dados de entrada.

        Returns:
            Dict parcial para atualizar o state (apenas output_field).
            O Workflow pode inspecionar a resposta para verificar tool_calls.
        """
        input_data = self._prepare_input(state)
        response = self._model.invoke(input_data)

        # Prepara output baseado no tipo do campo
        if self._output_field == "messages":
            output_value = [response]
        else:
            output_value = response.content

        return {self._output_field: output_value}

    async def ainvoke(self, state: BaseModel) -> dict[str, Any]:
        """
        Executa o agente de forma assíncrona.

        Args:
            state: Instância do state com os dados de entrada.

        Returns:
            Dict parcial para atualizar o state (apenas output_field).
        """
        input_data = self._prepare_input(state)
        response = await self._model.ainvoke(input_data)

        if self._output_field == "messages":
            output_value = [response]
        else:
            output_value = response.content

        return {self._output_field: output_value}

    def __repr__(self) -> str:
        """Representação do agente."""
        return (
            f"Agent(name={self._name!r}, "
            f"input_field={self._input_field!r}, "
            f"output_field={self._output_field!r}, "
            f"tools={len(self._tools)})"
        )
