"""Tipos e utilitários para streaming."""

from typing import Any

# Type alias para chunk de streaming de mensagens
# Formato: (AIMessageChunk, metadata_dict)
# - AIMessageChunk: chunk parcial da resposta do LLM
# - metadata_dict: informações como node name, langgraph_node, etc.
MessageStreamChunk = tuple[Any, dict[str, Any]]

__all__ = ["MessageStreamChunk"]
