"""Tipos e aliases para edges do workflow."""

from collections.abc import Callable

from langgraph.types import Send
from pydantic import BaseModel

# Type alias para condições
# - string built-in: "has_tool_calls", "no_tool_calls"
# - função booleana: (state) -> bool
# - função de distribuição (map-reduce): (state) -> list[Send]
Condition = str | Callable[[BaseModel], bool] | Callable[[BaseModel], list[Send]]

# Type alias para função de distribuição (retorna lista de Send para map-reduce)
DistributionFunc = Callable[[BaseModel], list[Send]]

# Type alias para edges
# Formatos suportados:
# - (source, target): edge simples
# - (source, target, condition): edge condicional
# - (source, [target1, target2, ...]): fan-out estático (execução paralela)
# - (source, distribution_func): fan-out dinâmico (map-reduce com Send)
Edge = (
    tuple[str, str]
    | tuple[str, str, Condition]
    | tuple[str, list[str]]
    | tuple[str, DistributionFunc]
)

# Re-export Send para uso direto
__all__ = ["Condition", "DistributionFunc", "Edge", "Send"]
