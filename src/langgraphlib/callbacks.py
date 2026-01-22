import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def _get_keys(obj: Any) -> list[str]:
    """Safely extracts keys from an object."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.keys())
    if hasattr(obj, "keys"):
        return list(obj.keys())
    if hasattr(obj, "__dict__"):
        return list(obj.__dict__.keys())
    return []


def _get_name(serialized: dict[str, Any] | None, **kwargs: Any) -> str:
    """
    Safely extracts node name.

    Tries multiple sources:
    1. kwargs["name"] - direct name passed by LangGraph
    2. kwargs["tags"] - tags may contain node name (e.g., "graph:step:1")
    3. serialized["name"] - LangChain serialized name
    4. serialized["id"][-1] - last element of ID
    """
    # 1. Direct name in kwargs
    if kwargs.get("name"):
        return kwargs["name"]

    # 2. Tags - LangGraph puts useful info here
    tags = kwargs.get("tags", [])
    for tag in tags:
        # LangGraph tags may have format "graph:step:N" or be node name
        if isinstance(tag, str) and not tag.startswith("seq:"):
            # Ignore generic sequential tags
            if ":" not in tag:
                return tag

    # 3. Serialized name
    if serialized and isinstance(serialized, dict) and serialized.get("name"):
        return serialized["name"]

    # 4. Last element of ID (fallback)
    if (
        serialized
        and isinstance(serialized, dict)
        and serialized.get("id")
        and isinstance(serialized["id"], list)
    ):
        return serialized["id"][-1]

    return "unknown"


class LoggingHandler(BaseCallbackHandler):
    """
    Handler for graph execution logging.

    Logs start/end of each chain/node with useful information.

    Examples:
        from langgraphlib.callbacks import LoggingHandler

        graph = workflow.compile()
        result = graph.invoke(
            {"messages": [HumanMessage("Hi")]},
            config={"callbacks": [LoggingHandler()]}
        )

        # With DEBUG level for more details
        result = graph.invoke(
            {"messages": [HumanMessage("Hi")]},
            config={"callbacks": [LoggingHandler(level=logging.DEBUG)]}
        )
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        """
        Initializes the handler.

        Args:
            logger: Custom logger. If None, uses default logger.
            level: Logging level (default: INFO).
        """
        self._logger = logger or logging.getLogger("langgraphlib")
        self._level = level
        self._start_times: dict[str, datetime] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node starts."""
        name = _get_name(serialized, **kwargs)
        run_id = kwargs.get("run_id", "")
        self._start_times[str(run_id)] = datetime.now()
        self._logger.log(self._level, f"[START] {name}")

        if self._level <= logging.DEBUG:
            input_keys = _get_keys(inputs)
            self._logger.debug(f"  Input keys: {input_keys}")

    def on_chain_end(
        self,
        outputs: Any,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node finishes."""
        run_id = str(kwargs.get("run_id", ""))
        start_time = self._start_times.pop(run_id, None)

        duration = ""
        if start_time:
            elapsed = (datetime.now() - start_time).total_seconds()
            duration = f" ({elapsed:.2f}s)"

        self._logger.log(self._level, f"[END]{duration}")

        if self._level <= logging.DEBUG:
            output_keys = _get_keys(outputs)
            self._logger.debug(f"  Output keys: {output_keys}")

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node fails."""
        self._logger.error(f"[ERROR] {type(error).__name__}: {error}")


class TraceHandler(BaseCallbackHandler):
    """
    Handler that collects structured execution trace.

    Stores execution history for later analysis.

    Examples:
        from langgraphlib.callbacks import TraceHandler

        tracer = TraceHandler()
        graph = workflow.compile()
        result = graph.invoke(
            {"messages": [HumanMessage("Hi")]},
            config={"callbacks": [tracer]}
        )

        # Access trace after execution
        print(tracer.traces)
        print(tracer.to_json())
    """

    def __init__(self) -> None:
        """Initializes the handler."""
        self._traces: list[dict[str, Any]] = []
        self._start_times: dict[str, datetime] = {}

    @property
    def traces(self) -> list[dict[str, Any]]:
        """List of collected traces."""
        return self._traces

    def clear(self) -> None:
        """Clears collected traces."""
        self._traces = []
        self._start_times = {}

    def to_json(self, indent: int = 2) -> str:
        """Returns traces as JSON string."""
        return json.dumps(self._traces, indent=indent, default=str)

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node starts."""
        run_id = str(kwargs.get("run_id", ""))
        self._start_times[run_id] = datetime.now()

        self._traces.append({
            "event": "start",
            "name": _get_name(serialized, **kwargs),
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "input_keys": _get_keys(inputs),
        })

    def on_chain_end(
        self,
        outputs: Any,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node finishes."""
        run_id = str(kwargs.get("run_id", ""))
        start_time = self._start_times.pop(run_id, None)

        duration = None
        if start_time:
            duration = (datetime.now() - start_time).total_seconds()

        self._traces.append({
            "event": "end",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "output_keys": _get_keys(outputs),
        })

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node fails."""
        run_id = str(kwargs.get("run_id", ""))

        self._traces.append({
            "event": "error",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        })
