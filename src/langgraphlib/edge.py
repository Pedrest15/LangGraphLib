"""Tipos e aliases para edges do workflow."""

from collections.abc import Callable

from pydantic import BaseModel

# Type alias para condições: string built-in ou função customizada
Condition = str | Callable[[BaseModel], bool]

# Type alias para edges
Edge = tuple[str, str] | tuple[str, str, Condition]
