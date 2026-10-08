"""Tests offline (sin API key): un LLM falso con respuestas guionadas ejercita el grafo real,
las tools reales y el checkpointer SQLite real."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agente.graph import build_graph
from agente.main import _config


class FakeToolModel(GenericFakeChatModel):
    """Modelo falso que acepta bind_tools() y devuelve mensajes predefinidos."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolModel":
        return self


def _call(name: str, args: dict[str, Any], i: int) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def _modelo(mensajes: list[AIMessage]) -> FakeToolModel:
    it: Iterator[AIMessage] = iter(mensajes)
    return FakeToolModel(messages=it)


def test_multipaso_y_memoria(tmp_path: Path) -> None:
    async def run() -> None:
        db = str(tmp_path / "cp.db")
        llm = _modelo(
            [
                _call("buscar_cliente", {"nombre": "Ana Gómez"}, 1),
                _call("buscar_pedidos", {"cliente_id": 102}, 2),
                AIMessage("Ana Gómez tuvo 3 pedidos por un total de $14.500."),
            ]
        )
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = build_graph(saver, llm)
            out = await graph.ainvoke({"messages": [HumanMessage("pedidos de Ana Gómez")]}, _config("t1"))
        tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2  # razonamiento multi-paso
        assert '"total": 14500.0' in tool_msgs[1].content

        # Reabrimos el .db con otro grafo: el thread conserva el historial.
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = build_graph(saver, _modelo([AIMessage("Hablábamos de Ana Gómez.")]))
            out = await graph.ainvoke({"messages": [HumanMessage("¿de quién hablábamos?")]}, _config("t1"))
            otro = await graph.aget_state(_config("otro-thread"))
        assert len(out["messages"]) == 8  # 6 del turno anterior (H, AI, Tool, AI, Tool, AI) + 2 nuevos
        assert otro.values == {}  # threads aislados

    asyncio.run(run())


def test_error_de_tool_vuelve_al_agente(tmp_path: Path) -> None:
    async def run() -> None:
        llm = _modelo(
            [
                _call("buscar_pedidos", {"cliente_id": 999}, 1),
                _call("buscar_cliente", {"nombre": "999"}, 2),
                AIMessage("No encontré al cliente 999. ¿Me pasás su nombre?"),
            ]
        )
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
            out = await build_graph(saver, llm).ainvoke(
                {"messages": [HumanMessage("cliente 999")]}, _config("t2")
            )
        assert "error" in out["messages"][2].content
        assert out["messages"][-1].content.endswith("?")

    asyncio.run(run())


def test_argumentos_invalidos_no_ejecutan_la_tool(tmp_path: Path) -> None:
    async def run() -> None:
        llm = _modelo(
            [
                _call("buscar_pedidos", {"cliente_id": -5}, 1),  # viola ge=1 (Pydantic)
                _call("buscar_pedidos", {"cliente_id": 102}, 2),  # el agente corrige y reintenta
                AIMessage("Ana Gómez tuvo 3 pedidos."),
            ]
        )
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
            out = await build_graph(saver, llm).ainvoke(
                {"messages": [HumanMessage("pedidos del cliente -5")]}, _config("t4")
            )
        invalido, valido = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        assert invalido.status == "error"
        assert "Argumentos inválidos" in invalido.content
        assert '"pedidos": 3' in valido.content

    asyncio.run(run())


def test_recursion_limit_corta_bucles(tmp_path: Path) -> None:
    async def run() -> None:
        infinito = _modelo([_call("buscar_pedidos", {"cliente_id": 102}, i) for i in range(50)])
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
            with pytest.raises(GraphRecursionError):
                await build_graph(saver, infinito).ainvoke(
                    {"messages": [HumanMessage("loop")]}, _config("t3")
                )

    asyncio.run(run())
