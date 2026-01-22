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
    Agent for LLM execution with dynamic state.

    The Agent encapsulates LLM configuration and enables execution
    with custom states via invoke/ainvoke.

    Features:
        - tools: Bind tools to the LLM, returns Command to {name}_tools
        - destinations: Dynamic routing via structured output

    Examples:
        from langgraphlib.model import get_model
        from langgraphlib.state import create_state, MessagesState

        model = get_model("openai/gpt-4o")

        # Simple agent with messages
        agent = Agent(model=model, prompt="You are a helpful assistant.")
        result = agent.invoke(MessagesState(messages=[HumanMessage("Hi")]))

        # Agent with custom fields
        MyState = create_state(
            include_messages=False,
            query=(str, ""),
            response=(str, ""),
        )
        agent = Agent(
            model=model,
            prompt="Answer the user's query.",
            input_fields="query",
            output_fields="response",
        )
        result = agent.invoke(MyState(query="What is the capital of France?"))

        # Agent with multiple input/output fields
        MultiState = create_state(
            include_messages=False,
            context=(str, ""),
            question=(str, ""),
            answer=(str, ""),
            confidence=(str, ""),
        )
        agent = Agent(
            model=model,
            prompt="Answer based on the context.",
            input_fields=["context", "question"],
            output_fields=["answer", "confidence"],
        )

        # Agent with dynamic routing (destinations)
        # The LLM decides which node to go to based on context
        supervisor = Agent(
            model=model,
            name="supervisor",
            prompt="Decide whether to research or write.",
            destinations=["researcher", "writer", "end"],
        )
        # invoke() returns Command(goto="researcher"|"writer"|"end", update={...})

        # Agent with correct typing via state
        # If state is provided, output_fields types are extracted from it
        CounterState = create_state(
            include_messages=False,
            query=(str, ""),
            count=(int, 0),
            response=(str, ""),
        )
        agent = Agent(
            model=model,
            prompt="Answer and count the words.",
            state=CounterState,  # types extracted: count=int, response=str
            input_fields="query",
            output_fields=["response", "count"],
            destinations=["next", "end"],
        )
        # The LLM will return count as int, not str
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
        Initialize the agent.

        Args:
            model: LangChain model instance.
            name: Agent name (auto-generated if not provided).
            prompt: System prompt for the LLM.
            tools: List of tools available to the agent.
            destinations: List of node names for dynamic routing.
                If provided, the agent uses structured output to decide
                the next node and returns Command instead of dict.
            state: State class for type introspection.
                If provided with destinations, output_fields types
                are extracted from state to ensure correct typing.
            input_fields: State field(s) to read input from. Can be string
                or list of strings for multiple fields.
            output_fields: State field(s) to write output to. Can be
                string or list of strings for multiple fields.
            max_retries: Maximum retry attempts on error (default: 0).
            timeout: Timeout in seconds for model execution (default: None).
        """
        self._name = name or f"agent_{uuid4().hex[:8]}"
        self._prompt = prompt
        self._tools = tools or []
        self._destinations = destinations
        self._state_class = state
        self._max_retries = max_retries
        self._timeout = timeout

        # Normalize to list
        self._input_fields = (
            [input_fields] if isinstance(input_fields, str) else list(input_fields)
        )
        self._output_fields = (
            [output_fields] if isinstance(output_fields, str) else list(output_fields)
        )

        # Create routing schema if destinations provided
        self._router_schema: type[BaseModel] | None = None
        if destinations:
            self._router_schema = self._create_router_schema(destinations)

        # Create output schema if needed (without destinations)
        self._output_schema: type[BaseModel] | None = None
        if not destinations and self._needs_structured_output():
            self._output_schema = self._create_output_schema()

        # Build model with prompt and tools
        self._model = self._build_model(model)

        # Separate model for routing (when has tools + destinations)
        self._router_model = self._build_router_model(model)

    def _get_field_type(self, field_name: str) -> type:
        """
        Extract the type of a state field.

        Handles Annotated types (e.g., Annotated[list[Message], add_messages])
        by extracting the base type.

        Args:
            field_name: Field name in the state.

        Returns:
            Field type or str as fallback.
        """
        if not self._state_class:
            return str

        # Get field_info from Pydantic model
        field_info = self._state_class.model_fields.get(field_name)
        if not field_info:
            return str

        field_type = field_info.annotation
        if field_type is None:
            return str

        # If Annotated, extract base type (first argument)
        if get_origin(field_type) is Annotated:
            args = get_args(field_type)
            if args:
                return args[0]

        return field_type

    def _create_router_schema(self, destinations: list[str]) -> type[BaseModel]:
        """
        Create dynamic Pydantic schema for routing.

        The schema forces the LLM to return outputs AND destination in a single call.
        If state was provided, uses actual field types.

        Note: 'messages' is excluded from schema because it's a complex type
        not supported by OpenAI structured output. The 'messages' field
        will be handled separately by converting the response to AIMessage.

        Args:
            destinations: List of valid node names.

        Returns:
            Pydantic class with output fields and 'goto'.
        """
        # Create Literal type with allowed destinations
        destination_type = Literal[tuple(destinations)]  # type: ignore[valid-type]

        # Build fields dynamically for each output_field (except messages)
        fields: dict[str, Any] = {}
        has_messages_field = False
        for field_name in self._output_fields:
            # Skip 'messages' - complex type not supported by structured output
            # Will be replaced by simple 'response' string field
            if field_name == "messages":
                has_messages_field = True
                continue
            field_type = self._get_field_type(field_name)
            fields[field_name] = (
                field_type,
                Field(description=f"Value for field '{field_name}'."),
            )

        # If messages is in output_fields, add 'response' string field
        # which will be converted to [AIMessage(content=response)] in invoke()
        if has_messages_field:
            fields["response"] = (
                str,
                Field(description="Your response or explanation for the user."),
            )

        # Add goto field
        fields["goto"] = (
            destination_type,
            Field(description="The next node the flow should go to."),
        )

        # Create model with extra="forbid" to generate additionalProperties=false
        # (required for OpenAI structured output)
        return create_model("RouterSchema", __config__={"extra": "forbid"}, **fields)

    def _create_output_schema(self) -> type[BaseModel]:
        """
        Create Pydantic schema for structured output without routing.

        Used when there are multiple output_fields or non-string types,
        forcing the LLM to respect correct types.

        Returns:
            Pydantic class with typed output fields.
        """
        fields: dict[str, Any] = {}
        for field_name in self._output_fields:
            field_type = self._get_field_type(field_name)
            fields[field_name] = (
                field_type,
                Field(description=f"Value for field '{field_name}'."),
            )

        # Create model with extra="forbid" to generate additionalProperties=false
        # (required for OpenAI structured output)
        return create_model("OutputSchema", __config__={"extra": "forbid"}, **fields)

    def _needs_structured_output(self) -> bool:
        """
        Determine if structured output should be used (without destinations).

        Conditions:
        - state was provided
        - At least one output_field has a type other than str
        - OR there are multiple output_fields

        Returns:
            True if structured output should be applied.
        """
        if not self._state_class:
            return False

        # If multiple output_fields, need structured output
        if len(self._output_fields) > 1:
            return True

        # If single field has type other than str, need structured output
        if len(self._output_fields) == 1:
            field_type = self._get_field_type(self._output_fields[0])
            # Ignore 'messages' as it's handled specially
            if self._output_fields[0] != "messages" and field_type is not str:
                return True

        return False

    def _build_model(self, llm: BaseChatModel) -> BaseChatModel:
        """Build model with prompt template, tools and/or structured output."""
        # Bind tools if provided
        if self._tools:
            llm = llm.bind_tools(self._tools)

        # Structured output: priority router_schema > output_schema
        # If has tools AND destinations, router will be used in second call
        # (cannot mix bind_tools with with_structured_output)
        if self._router_schema and not self._tools:
            llm = llm.with_structured_output(self._router_schema)
        elif self._output_schema and not self._tools:
            llm = llm.with_structured_output(self._output_schema)

        # Apply prompt template if provided
        if self._prompt:
            # Build message list with placeholders for each input field
            messages: list[Any] = [
                SystemMessagePromptTemplate.from_template(self._prompt)
            ]
            for field_name in self._input_fields:
                messages.append(MessagesPlaceholder(variable_name=field_name))

            prompt_template = ChatPromptTemplate.from_messages(messages)
            return prompt_template | llm

        return llm

    def _build_router_model(self, llm: BaseChatModel) -> BaseChatModel | None:
        """Build separate model for routing when has tools + destinations."""
        if not (self._tools and self._router_schema):
            return None

        # Model with structured output only (no tools)
        router_llm = llm.with_structured_output(self._router_schema)

        # Routing prompt includes original agent prompt
        # so the router understands context and routing rules
        base_context = self._prompt or ""
        router_prompt = (
            f"{base_context}\n\n"
            "Based on the conversation below, decide what the next step should be.\n"
            "Analyze the context and choose the appropriate destination.\n"
            "IMPORTANT: If you already called a tool and it was executed (you see the "
            "result in the conversation), DO NOT call the same tool again. Proceed to "
            "the appropriate next destination."
        )

        messages: list[Any] = [SystemMessagePromptTemplate.from_template(router_prompt)]
        for field_name in self._input_fields:
            messages.append(MessagesPlaceholder(variable_name=field_name))

        prompt_template = ChatPromptTemplate.from_messages(messages)
        return prompt_template | router_llm

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    @property
    def model(self) -> BaseChatModel:
        """Agent's LLM model."""
        return self._model

    @property
    def prompt(self) -> str | None:
        """System prompt."""
        return self._prompt

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Agent's tool list."""
        return self._tools

    @property
    def input_fields(self) -> list[str]:
        """State fields for input."""
        return self._input_fields

    @property
    def output_fields(self) -> list[str]:
        """State fields for output."""
        return self._output_fields

    @property
    def destinations(self) -> list[str] | None:
        """List of destinations for dynamic routing."""
        return self._destinations

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts."""
        return self._max_retries

    @property
    def timeout(self) -> float | None:
        """Timeout in seconds."""
        return self._timeout

    def _resolve_goto(self, goto: str) -> str | object:
        """Convert 'end' to LangGraph END constant."""
        if goto.lower() == "end":
            return END
        return goto

    def _prepare_input(self, state: BaseModel | dict[str, Any]) -> dict[str, Any]:
        """Prepare input for the model from state."""
        input_data: dict[str, Any] = {}

        for field_name in self._input_fields:
            # Support both BaseModel and dict (used by Send)
            if isinstance(state, dict):
                input_value = state.get(field_name)
            else:
                input_value = getattr(state, field_name)

            # Convert string to HumanMessage list
            if isinstance(input_value, str):
                input_value = [HumanMessage(content=input_value)]

            input_data[field_name] = input_value

        return input_data

    def _process_output_value(self, field_name: str, value: Any) -> Any:
        """
        Process output value to ensure correct type.

        If field is 'messages' and value is string, converts to AIMessage list.
        This is needed because structured output returns strings, but the
        add_messages reducer expects a list of messages.
        """
        if field_name == "messages" and isinstance(value, str):
            return [AIMessage(content=value)]
        return value

    def _invoke_with_retry(self, model: Any, input_data: dict[str, Any]) -> Any:
        """
        Execute model with retry and timeout.

        Args:
            model: Model to invoke.
            input_data: Input data.

        Returns:
            Model response.

        Raises:
            TimeoutError: If timeout is reached.
            Exception: Last exception after exhausting retries.
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
                            f"Timeout of {self._timeout}s reached "
                            f"(elapsed: {elapsed:.2f}s)"
                        )
                    return response
                else:
                    return model.invoke(input_data)
            except TimeoutError:
                raise
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    # Wait before retry (exponential backoff)
                    time.sleep(2**attempt * 0.5)
                    continue
                raise

        if last_exception:
            raise last_exception

    async def _ainvoke_with_retry(self, model: Any, input_data: dict[str, Any]) -> Any:
        """
        Execute model asynchronously with retry and timeout.

        Args:
            model: Model to invoke.
            input_data: Input data.

        Returns:
            Model response.

        Raises:
            TimeoutError: If timeout is reached.
            Exception: Last exception after exhausting retries.
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
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2**attempt * 0.5)
                    continue
                raise

        if last_exception:
            raise last_exception

    def invoke(self, state: BaseModel | dict[str, Any]) -> dict[str, Any] | Command:
        """
        Execute the agent with the provided state.

        Args:
            state: State instance or dict with input data.

        Returns:
            - If destinations configured: Command with goto and state update.
            - If has tool_calls: Command to {agent_name}_tools.
            - Otherwise: Partial dict to update state.

        Raises:
            TimeoutError: If timeout is reached.
            Exception: If all attempts fail.
        """
        input_data = self._prepare_input(state)
        response = self._invoke_with_retry(self._model, input_data)

        # Case 1: destinations (router schema) without tools - response is dict/BaseModel
        if self._destinations and self._router_schema and not self._tools:
            # Extract goto
            if isinstance(response, dict):
                goto = response.get("goto", self._destinations[0])
            else:
                goto = getattr(response, "goto", self._destinations[0])

            # Extract each output field
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                # 'messages' not in schema - use converted 'response' field
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

        # Case 2: output schema (without destinations) - response is dict/BaseModel
        if self._output_schema and not self._tools:
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                if isinstance(response, dict):
                    value = response.get(field_name, "")
                else:
                    value = getattr(response, field_name, "")
                state_update[field_name] = self._process_output_value(field_name, value)
            return state_update

        # Case 3: tools - response is AIMessage
        # First output field receives the response
        primary_field = self._output_fields[0]
        output_value = [response] if primary_field == "messages" else response.content
        state_update = {primary_field: output_value}

        # tool_calls - go to tool node
        if hasattr(response, "tool_calls") and response.tool_calls:
            return Command(
                goto=f"{self._name}_tools",
                update=state_update,
            )

        # Case 4: tools + destinations - second call to decide routing
        if self._router_model and self._destinations:
            # Update input_data with current response for router to decide
            router_input = self._prepare_input(state)
            # Add agent response to context
            if primary_field == "messages":
                router_input["messages"] = router_input.get("messages", []) + [response]

            router_response = self._invoke_with_retry(self._router_model, router_input)

            # Extract goto
            if isinstance(router_response, dict):
                goto = router_response.get("goto", self._destinations[0])
            else:
                goto = getattr(router_response, "goto", self._destinations[0])

            return Command(goto=self._resolve_goto(goto), update=state_update)

        # Default case: return dict to update state
        return state_update

    async def ainvoke(
        self, state: BaseModel | dict[str, Any]
    ) -> dict[str, Any] | Command:
        """
        Execute the agent asynchronously.

        Args:
            state: State instance or dict with input data.

        Returns:
            - If destinations configured: Command with goto and state update.
            - If has tool_calls: Command to {agent_name}_tools.
            - Otherwise: Partial dict to update state.

        Raises:
            TimeoutError: If timeout is reached.
            Exception: If all attempts fail.
        """
        input_data = self._prepare_input(state)
        response = await self._ainvoke_with_retry(self._model, input_data)

        # Case 1: destinations (router schema) without tools - response is dict/BaseModel
        if self._destinations and self._router_schema and not self._tools:
            # Extract goto
            if isinstance(response, dict):
                goto = response.get("goto", self._destinations[0])
            else:
                goto = getattr(response, "goto", self._destinations[0])

            # Extract each output field
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                # 'messages' not in schema - use converted 'response' field
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

        # Case 2: output schema (without destinations) - response is dict/BaseModel
        if self._output_schema and not self._tools:
            state_update: dict[str, Any] = {}
            for field_name in self._output_fields:
                if isinstance(response, dict):
                    value = response.get(field_name, "")
                else:
                    value = getattr(response, field_name, "")
                state_update[field_name] = self._process_output_value(field_name, value)
            return state_update

        # Case 3: tools - response is AIMessage
        # First output field receives the response
        primary_field = self._output_fields[0]
        output_value = [response] if primary_field == "messages" else response.content
        state_update = {primary_field: output_value}

        # tool_calls - go to tool node
        if hasattr(response, "tool_calls") and response.tool_calls:
            return Command(
                goto=f"{self._name}_tools",
                update=state_update,
            )

        # Case 4: tools + destinations - second call to decide routing
        if self._router_model and self._destinations:
            # Update input_data with current response for router to decide
            router_input = self._prepare_input(state)
            # Add agent response to context
            if primary_field == "messages":
                router_input["messages"] = router_input.get("messages", []) + [response]

            router_response = await self._ainvoke_with_retry(
                self._router_model, router_input
            )

            # Extract goto
            if isinstance(router_response, dict):
                goto = router_response.get("goto", self._destinations[0])
            else:
                goto = getattr(router_response, "goto", self._destinations[0])

            return Command(goto=self._resolve_goto(goto), update=state_update)

        # Default case: return dict to update state
        return state_update

    def __repr__(self) -> str:
        """Agent representation."""
        return (
            f"Agent(name={self._name!r}, "
            f"input_fields={self._input_fields!r}, "
            f"output_fields={self._output_fields!r}, "
            f"tools={len(self._tools)})"
        )
