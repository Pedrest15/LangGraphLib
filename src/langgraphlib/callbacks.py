"""Callbacks para logging e trace de workflows."""

import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def _get_keys(obj: Any) -> list[str]:
    """Extrai keys de um objeto de forma segura."""
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
    Extrai nome do nó de forma segura.

    Tenta múltiplas fontes:
    1. kwargs["name"] - nome direto passado pelo LangGraph
    2. kwargs["tags"] - tags podem conter nome do nó (ex: "graph:step:1")
    3. serialized["name"] - nome serializado do LangChain
    4. serialized["id"][-1] - último elemento do ID
    """
    # 1. Nome direto nos kwargs
    if kwargs.get("name"):
        return kwargs["name"]

    # 2. Tags - LangGraph coloca info útil aqui
    tags = kwargs.get("tags", [])
    for tag in tags:
        # Tags do LangGraph podem ter formato "graph:step:N" ou ser nome do nó
        if isinstance(tag, str) and not tag.startswith("seq:"):
            # Ignora tags sequenciais genéricas
            if ":" not in tag:
                return tag

    # 3. Serialized name
    if serialized and isinstance(serialized, dict) and serialized.get("name"):
        return serialized["name"]

    # 4. Último elemento do ID (fallback)
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
    Handler para logging de execução do grafo.

    Loga início/fim de cada chain/node com informações úteis.

    Examples:
        from langgraphlib.callbacks import LoggingHandler

        graph = workflow.compile()
        result = graph.invoke(
            {"messages": [HumanMessage("Oi")]},
            config={"callbacks": [LoggingHandler()]}
        )

        # Com nível DEBUG para mais detalhes
        result = graph.invoke(
            {"messages": [HumanMessage("Oi")]},
            config={"callbacks": [LoggingHandler(level=logging.DEBUG)]}
        )
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        """
        Inicializa o handler.

        Args:
            logger: Logger customizado. Se None, usa logger padrão.
            level: Nível de logging (default: INFO).
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
        """Chamado quando uma chain/node inicia."""
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
        """Chamado quando uma chain/node termina."""
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
        """Chamado quando uma chain/node falha."""
        self._logger.error(f"[ERROR] {type(error).__name__}: {error}")


class TraceHandler(BaseCallbackHandler):
    """
    Handler que coleta trace estruturado da execução.

    Armazena histórico de execução para análise posterior.

    Examples:
        from langgraphlib.callbacks import TraceHandler

        tracer = TraceHandler()
        graph = workflow.compile()
        result = graph.invoke(
            {"messages": [HumanMessage("Oi")]},
            config={"callbacks": [tracer]}
        )

        # Acessar trace após execução
        print(tracer.traces)
        print(tracer.to_json())
    """

    def __init__(self) -> None:
        """Inicializa o handler."""
        self._traces: list[dict[str, Any]] = []
        self._start_times: dict[str, datetime] = {}

    @property
    def traces(self) -> list[dict[str, Any]]:
        """Lista de traces coletados."""
        return self._traces

    def clear(self) -> None:
        """Limpa traces coletados."""
        self._traces = []
        self._start_times = {}

    def to_json(self, indent: int = 2) -> str:
        """Retorna traces como JSON string."""
        return json.dumps(self._traces, indent=indent, default=str)

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        **kwargs: Any,
    ) -> None:
        """Chamado quando uma chain/node inicia."""
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
        """Chamado quando uma chain/node termina."""
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
        """Chamado quando uma chain/node falha."""
        run_id = str(kwargs.get("run_id", ""))

        self._traces.append({
            "event": "error",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        })
