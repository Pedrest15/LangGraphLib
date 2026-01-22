from typing import Any

# Type alias for message streaming chunk
# Format: (AIMessageChunk, metadata_dict)
# - AIMessageChunk: partial chunk of LLM response
# - metadata_dict: information like node name, langgraph_node, etc.
MessageStreamChunk = tuple[Any, dict[str, Any]]

__all__ = ["MessageStreamChunk"]
