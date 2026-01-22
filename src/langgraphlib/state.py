from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, create_model


def add_messages(current: list[AnyMessage], new: list[AnyMessage]) -> list[AnyMessage]:
    """Reducer that appends messages to existing state."""
    return current + new


class BaseState(BaseModel):
    """Base state for agents with messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list[AnyMessage], add_messages] = []


def create_state(
    name: str = "State",
    *,
    include_messages: bool = True,
    **fields: Any,
) -> type[BaseModel]:
    """
    Dynamically creates a State with custom fields.

    Args:
        name: Name of the generated state class.
        include_messages: If True, includes the messages field with reducer.
        **fields: Additional fields in format name=(type, default) or name=type.

    Returns:
        Pydantic class configured as State.

    Examples:
        # Simple state with messages + counter
        MyState = create_state(counter=(int, 0))

        # State without messages
        CustomState = create_state(
            "CustomState", include_messages=False, data=(dict, {})
        )

        # State with type without default
        TaskState = create_state(task_id=str, status=(str, "pending"))
    """
    field_definitions: dict[str, Any] = {}

    if include_messages:
        field_definitions["messages"] = (
            Annotated[list[AnyMessage], add_messages],
            [],
        )

    for field_name, field_spec in fields.items():
        if isinstance(field_spec, tuple):
            field_definitions[field_name] = field_spec
        else:
            field_definitions[field_name] = (field_spec, ...)

    return create_model(
        name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **field_definitions,
    )


# Pre-defined common states
MessagesState = create_state("MessagesState")
