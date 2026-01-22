import asyncio
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal, get_args, get_origin
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langgraph.graph import END
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

        # Agente com tipagem correta via state
        # Se state é fornecido, os tipos dos output_fields são extraídos dele
        CounterState = create_state(
            include_messages=False,
            query=(str, ""),
            count=(int, 0),
            response=(str, ""),
        )
        agent = Agent(
            model=model,
            prompt="Responda e conte quantas palavras.",
            state=CounterState,  # tipos extraídos: count=int, response=str
            input_fields="query",
            output_fields=["response", "count"],
            destinations=["next", "end"],
        )
        # O LLM retornará count como int, não str
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        name: str | None = None,
        prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        destinations: list[str] | None = None,
        state: type[BaseModel] | None = None,
        input_fields: str | list[str] = "messages",
        output_fields: str | list[str] = "messages",
        max_retries: int = 0,
        timeout: float | None = None,
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
            state: Classe do state para introspecção de tipos.
                Se fornecido junto com destinations, os tipos dos output_fields
                serão extraídos do state para garantir tipagem correta.
            input_fields: Campo(s) do state para ler input. Pode ser string
                ou lista de strings para múltiplos campos.
            output_fields: Campo(s) do state para escrever output. Pode ser
                string ou lista de strings para múltiplos campos.
            max_retries: Número máximo de tentativas em caso de erro (default: 0).
            timeout: Timeout em segundos para execução do modelo (default: None).
        """
        self._name = name or f"agent_{uuid4().hex[:8]}"
        self._prompt = prompt
        self._tools = tools or []
        self._destinations = destinations
        self._state_class = state
        self._max_retries = max_retries
        self._timeout = timeout

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

        # Cria schema de output se necessário (sem destinations)
        self._output_schema: type[BaseModel] | None = None
        if not destinations and self._needs_structured_output():
            self._output_schema = self._create_output_schema()

        # Monta o modelo com prompt e tools
        self._model = self._build_model(model)

        # Modelo separado para roteamento (quando tem tools + destinations)
        self._router_model = self._build_router_model(model)

    def _get_field_type(self, field_name: str) -> type:
        """
        Extrai o tipo de um campo do state.

        Lida com tipos Annotated (ex: Annotated[list[Message], add_messages])
        extraindo o tipo base.

        Args:
            field_name: Nome do campo no state.

        Returns:
            Tipo do campo ou str como fallback.
        """
        if not self._state_class:
            return str

        # Obtém field_info do modelo Pydantic
        field_info = self._state_class.model_fields.get(field_name)
        if not field_info:
            return str

        field_type = field_info.annotation
        if field_type is None:
            return str

        # Se é Annotated, extrai o tipo base (primeiro argumento)
        if get_origin(field_type) is Annotated:
            args = get_args(field_type)
            if args:
                return args[0]

        return field_type

    def _create_router_schema(self, destinations: list[str]) -> type[BaseModel]:
        """
        Cria schema Pydantic dinâmico para roteamento.

        O schema força o LLM a retornar outputs E destino em uma única chamada.
        Se state foi fornecido, usa os tipos reais dos campos.

        Nota: 'messages' é excluído do schema porque é um tipo complexo
        que não é suportado pelo OpenAI structured output. O campo 'messages'
        será tratado separadamente convertendo a resposta para AIMessage.

        Args:
            destinations: Lista de nomes de nós válidos.

        Returns:
            Classe Pydantic com campos de output e 'goto'.
        """
        # Cria tipo Literal com os destinos permitidos
        destination_type = Literal[tuple(destinations)]  # type: ignore[valid-type]

        # Monta campos dinamicamente para cada output_field (exceto messages)
        fields: dict[str, Any] = {}
        has_messages_field = False
        for field_name in self._output_fields:
            # Pula 'messages' - tipo complexo não suportado pelo structured output
            # Será substituído por campo 'response' simples (string)
            if field_name == "messages":
                has_messages_field = True
                continue
            field_type = self._get_field_type(field_name)
            fields[field_name] = (
                field_type,
                Field(description=f"Valor para o campo '{field_name}'."),
            )

        # Se messages está em output_fields, adiciona campo 'response' string
        # que será convertido para [AIMessage(content=response)] no invoke()
        if has_messages_field:
            fields["response"] = (
                str,
                Field(description="Sua resposta ou explicação para o usuário."),
            )

        # Adiciona campo goto
        fields["goto"] = (
            destination_type,
            Field(description="O próximo nó para onde o fluxo deve ir."),
        )

        # Cria modelo com extra="forbid" para gerar additionalProperties=false
        # (necessário para OpenAI structured output)
        return create_model("RouterSchema", __config__={"extra": "forbid"}, **fields)

    def _create_output_schema(self) -> type[BaseModel]:
        """
        Cria schema Pydantic para structured output sem roteamento.

        Usado quando há múltiplos output_fields ou tipos não-string,
        forçando o LLM a respeitar os tipos corretos.

        Returns:
            Classe Pydantic com campos de output tipados.
        """
        fields: dict[str, Any] = {}
        for field_name in self._output_fields:
            field_type = self._get_field_type(field_name)
            fields[field_name] = (
                field_type,
                Field(description=f"Valor para o campo '{field_name}'."),
            )

        # Cria modelo com extra="forbid" para gerar additionalProperties=false
        # (necessário para OpenAI structured output)
        return create_model("OutputSchema", __config__={"extra": "forbid"}, **fields)

    def _needs_structured_output(self) -> bool:
        """
        Determina se structured output deve ser usado (sem destinations).

        Condições:
        - state foi fornecido
        - Pelo menos um output_field tem tipo diferente de str
        - OU há múltiplos output_fields

        Returns:
            True se structured output deve ser aplicado.
        """
        if not self._state_class:
            return False

        # Se há múltiplos output_fields, precisa de structured output
        if len(self._output_fields) > 1:
            return True

        # Se o único campo tem tipo diferente de str, precisa de structured output
        if len(self._output_fields) == 1:
            field_type = self._get_field_type(self._output_fields[0])
            # Ignora 'messages' pois é tratado especialmente
            if self._output_fields[0] != "messages" and field_type is not str:
                return True

        return False

    def _build_model(self, llm: BaseChatModel) -> BaseChatModel:
        """Constrói modelo com prompt template, tools e/ou structured output."""
        # Bind tools se fornecidas
        if self._tools:
            llm = llm.bind_tools(self._tools)

        # Structured output: prioridade router_schema > output_schema
        # Se tem tools E destinations, o router será usado em segunda chamada
        # (não pode misturar bind_tools com with_structured_output)
        if self._router_schema and not self._tools:
            llm = llm.with_structured_output(self._router_schema)
        elif self._output_schema and not self._tools:
            llm = llm.with_structured_output(self._output_schema)

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

    def _build_router_model(self, llm: BaseChatModel) -> BaseChatModel | None:
        """Constrói modelo separado para roteamento quando há tools + destinations."""
        if not (self._tools and self._router_schema):
            return None

        # Modelo apenas com structured output (sem tools)
        router_llm = llm.with_structured_output(self._router_schema)

        # Prompt para decisão de roteamento inclui o prompt original do agente
        # para que o router entenda o contexto e as regras de roteamento
        base_context = self._prompt or ""
        router_prompt = (
            f"{base_context}\n\n"
            "Baseado na conversa abaixo, decida qual deve ser o próximo passo.\n"
            "Analise o contexto e escolha o destino apropriado.\n"
            "IMPORTANTE: Se você já chamou uma tool e ela foi executada (você vê o "
            "resultado na conversa), NÃO chame a mesma tool novamente. Prossiga para "
            "o próximo destino apropriado."
        )

        messages: list[Any] = [SystemMessagePromptTemplate.from_template(router_prompt)]
        for field_name in self._input_fields:
            messages.append(MessagesPlaceholder(variable_name=field_name))

        prompt_template = ChatPromptTemplate.from_messages(messages)
        return prompt_template | router_llm

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

    @property
    def max_retries(self) -> int:
        """Número máximo de tentativas."""
        return self._max_retries

    @property
    def timeout(self) -> float | None:
        """Timeout em segundos."""
        return self._timeout

    def _resolve_goto(self, goto: str) -> str | object:
        """Converte 'end' para constante END do LangGraph."""
        if goto.lower() == "end":
            return END
        return goto

    def _prepare_input(self, state: BaseModel | dict[str, Any]) -> dict[str, Any]:
        """Prepara input para o modelo a partir do state."""
        input_data: dict[str, Any] = {}

        for field_name in self._input_fields:
            # Suporta tanto BaseModel quanto dict (usado por Send)
            if isinstance(state, dict):
                input_value = state.get(field_name)
            else:
                input_value = getattr(state, field_name)

            # Converte string para lista de HumanMessage
            if isinstance(input_value, str):
                input_value = [HumanMessage(content=input_value)]

            input_data[field_name] = input_value

        return input_data

    def _process_output_value(self, field_name: str, value: Any) -> Any:
        """
        Processa valor de output para garantir tipo correto.

        Se o campo é 'messages' e o valor é string, converte para lista de AIMessage.
        Isso é necessário porque structured output retorna strings, mas o reducer
        add_messages espera lista de mensagens.
        """
        if field_name == "messages" and isinstance(value, str):
            return [AIMessage(content=value)]
        return value

    def _invoke_with_retry(self, model: Any, input_data: dict[str, Any]) -> Any:
        """
        Executa o modelo com retry e timeout.

        Args:
            model: Modelo a ser invocado.
            input_data: Dados de entrada.

        Returns:
            Resposta do modelo.

        Raises:
            TimeoutError: Se timeout for atingido.
            Exception: Última exceção após esgotar retries.
        """
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                if self._timeout:
                    start_time = time.time()
                    response = model.invoke(input_data)
                    elapsed = time.time() - start_time
                    if elapsed > self._timeout:
                        raise TimeoutError(
                            f"Timeout de {self._timeout}s atingido "
                            f"(tempo: {elapsed:.2f}s)"
                        )
                    return response
                else:
                    return model.invoke(input_data)
            except TimeoutError:
                raise
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    # Aguarda antes de retry (backoff exponencial)
                    time.sleep(2**attempt * 0.5)
                    continue
                raise

        if last_exception:
            raise last_exception

    async def _ainvoke_with_retry(self, model: Any, input_data: dict[str, Any]) -> Any:
        """
        Executa o modelo de forma assíncrona com retry e timeout.

        Args:
            model: Modelo a ser invocado.
            input_data: Dados de entrada.

        Returns:
            Resposta do modelo.

        Raises:
            TimeoutError: Se timeout for atingido.
            Exception: Última exceção após esgotar retries.
        """
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                if self._timeout:
                    response = await asyncio.wait_for(
                        model.ainvoke(input_data),
                        timeout=self._timeout,
                    )
                    return response
                else:
                    return await model.ainvoke(input_data)
            except TimeoutError:
                raise
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    # Aguarda antes de retry (backoff exponencial)
                    await asyncio.sleep(2**attempt * 0.5)
                    continue
                raise

        if last_exception:
            raise last_exception

    def invoke(self, state: BaseModel | dict[str, Any]) -> dict[str, Any] | Command:
        """
        Executa o agente com o state fornecido.

        Args:
            state: Instância do state ou dict com os dados de entrada.

        Returns:
            - Se destinations configurado: Command com goto e update do state.
            - Se tem tool_calls: Command para {agent_name}_tools.
            - Caso contrário: Dict parcial para atualizar o state.

        Raises:
            TimeoutError: Se timeout for atingido.
            Exception: Se todas as tentativas falharem.
        """
        input_data = self._prepare_input(state)
        response = self._invoke_with_retry(self._model, input_data)

        # Caso 1: destinations (router schema) sem tools - response é dict/BaseModel
        if self._destinations and self._router_schema and not self._tools:
            # Extrai goto
            if isinstance(response, dict):
                goto = response.get("goto", self._destinations[0])
            else:
                goto = getattr(response, "goto", self._destinations[0])

            # Extrai cada output field
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                # 'messages' não está no schema - usa campo 'response' convertido
                if field_name == "messages":
                    if isinstance(response, dict):
                        resp_text = response.get("response", "")
                    else:
                        resp_text = getattr(response, "response", "")
                    state_update["messages"] = [AIMessage(content=resp_text)]
                else:
                    if isinstance(response, dict):
                        value = response.get(field_name, "")
                    else:
                        value = getattr(response, field_name, "")
                    state_update[field_name] = self._process_output_value(
                        field_name, value
                    )

            return Command(goto=self._resolve_goto(goto), update=state_update)

        # Caso 2: output schema (sem destinations) - response é dict/BaseModel
        if self._output_schema and not self._tools:
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                if isinstance(response, dict):
                    value = response.get(field_name, "")
                else:
                    value = getattr(response, field_name, "")
                state_update[field_name] = self._process_output_value(field_name, value)
            return state_update

        # Caso 3: tools - response é AIMessage
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

        # Caso 4: tools + destinations - segunda chamada para decidir roteamento
        if self._router_model and self._destinations:
            # Atualiza input_data com a resposta atual para o router decidir
            router_input = self._prepare_input(state)
            # Adiciona a resposta do agente ao contexto
            if primary_field == "messages":
                router_input["messages"] = router_input.get("messages", []) + [response]

            router_response = self._invoke_with_retry(self._router_model, router_input)

            # Extrai goto
            if isinstance(router_response, dict):
                goto = router_response.get("goto", self._destinations[0])
            else:
                goto = getattr(router_response, "goto", self._destinations[0])

            return Command(goto=self._resolve_goto(goto), update=state_update)

        # Caso padrão: retorna dict para atualizar state
        return state_update

    async def ainvoke(
        self, state: BaseModel | dict[str, Any]
    ) -> dict[str, Any] | Command:
        """
        Executa o agente de forma assíncrona.

        Args:
            state: Instância do state ou dict com os dados de entrada.

        Returns:
            - Se destinations configurado: Command com goto e update do state.
            - Se tem tool_calls: Command para {agent_name}_tools.
            - Caso contrário: Dict parcial para atualizar o state.

        Raises:
            TimeoutError: Se timeout for atingido.
            Exception: Se todas as tentativas falharem.
        """
        input_data = self._prepare_input(state)
        response = await self._ainvoke_with_retry(self._model, input_data)

        # Caso 1: destinations (router schema) sem tools - response é dict/BaseModel
        if self._destinations and self._router_schema and not self._tools:
            # Extrai goto
            if isinstance(response, dict):
                goto = response.get("goto", self._destinations[0])
            else:
                goto = getattr(response, "goto", self._destinations[0])

            # Extrai cada output field
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                # 'messages' não está no schema - usa campo 'response' convertido
                if field_name == "messages":
                    if isinstance(response, dict):
                        resp_text = response.get("response", "")
                    else:
                        resp_text = getattr(response, "response", "")
                    state_update["messages"] = [AIMessage(content=resp_text)]
                else:
                    if isinstance(response, dict):
                        value = response.get(field_name, "")
                    else:
                        value = getattr(response, field_name, "")
                    state_update[field_name] = self._process_output_value(
                        field_name, value
                    )

            return Command(goto=self._resolve_goto(goto), update=state_update)

        # Caso 2: output schema (sem destinations) - response é dict/BaseModel
        if self._output_schema and not self._tools:
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                if isinstance(response, dict):
                    value = response.get(field_name, "")
                else:
                    value = getattr(response, field_name, "")
                state_update[field_name] = self._process_output_value(field_name, value)
            return state_update

        # Caso 3: tools - response é AIMessage
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

        # Caso 4: tools + destinations - segunda chamada para decidir roteamento
        if self._router_model and self._destinations:
            # Atualiza input_data com a resposta atual para o router decidir
            router_input = self._prepare_input(state)
            # Adiciona a resposta do agente ao contexto
            if primary_field == "messages":
                router_input["messages"] = router_input.get("messages", []) + [response]

            router_response = await self._ainvoke_with_retry(
                self._router_model, router_input
            )

            # Extrai goto
            if isinstance(router_response, dict):
                goto = router_response.get("goto", self._destinations[0])
            else:
                goto = getattr(router_response, "goto", self._destinations[0])

            return Command(goto=self._resolve_goto(goto), update=state_update)

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
