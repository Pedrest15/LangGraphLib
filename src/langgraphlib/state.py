from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, create_model


def add_messages(current: list[AnyMessage], new: list[AnyMessage]) -> list[AnyMessage]:
    """Reducer que adiciona mensagens ao estado existente."""
    return current + new


class BaseState(BaseModel):
    """Estado base para agentes com mensagens."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list[AnyMessage], add_messages] = []


def create_state(
    name: str = "State",
    *,
    include_messages: bool = True,
    **fields: Any,
) -> type[BaseModel]:
    """
    Cria um State dinamicamente com campos personalizados.

    Args:
        name: Nome da classe de estado gerada.
        include_messages: Se True, inclui o campo messages com reducer.
        **fields: Campos adicionais no formato nome=(tipo, default) ou nome=tipo.

    Returns:
        Classe Pydantic configurada como State.

    Examples:
        # State simples com messages + contador
        MyState = create_state(counter=(int, 0))

        # State sem messages
        CustomState = create_state(
            "CustomState", include_messages=False, data=(dict, {})
        )

        # State com tipo sem default
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


# Estados pré-definidos comuns
MessagesState = create_state("MessagesState")
