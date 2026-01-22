from collections.abc import Callable

from langgraph.types import Send
from pydantic import BaseModel

# Type alias for conditions
# - built-in string: "has_tool_calls", "no_tool_calls"
# - boolean function: (state) -> bool
# - distribution function (map-reduce): (state) -> list[Send]
Condition = str | Callable[[BaseModel], bool] | Callable[[BaseModel], list[Send]]

# Type alias for distribution function (returns list of Send for map-reduce)
DistributionFunc = Callable[[BaseModel], list[Send]]

# Type alias for edges
# Supported formats:
# - (source, target): simple edge
# - (source, target, condition): conditional edge
# - (source, [target1, target2, ...]): static fan-out (parallel execution)
# - (source, distribution_func): dynamic fan-out (map-reduce with Send)
Edge = (
    tuple[str, str]
    | tuple[str, str, Condition]
    | tuple[str, list[str]]
    | tuple[str, DistributionFunc]
)

# Re-export Send for direct use
__all__ = ["Condition", "DistributionFunc", "Edge", "Send"]
