"""Tests offline (sin API key): un LLM falso con respuestas guionadas ejercita el grafo real,
las tools reales y el checkpointer SQLite real.

Están agrupados por criterio de aceptación de la consigna (una clase por criterio).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agente import tools
from agente.graph import build_graph, cerrar_turno_cortado
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


def _invocar(db: Path, llm: FakeToolModel, pregunta: str, thread_id: str) -> list[AnyMessage]:
    """Corre un turno completo del grafo con checkpointer SQLite y devuelve el historial."""

    async def run() -> list[AnyMessage]:
        async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
            out = await build_graph(saver, llm).ainvoke(
                {"messages": [HumanMessage(pregunta)]}, _config(thread_id)
            )
        mensajes: list[AnyMessage] = out["messages"]
        return mensajes

    return asyncio.run(run())


def _observaciones(mensajes: list[AnyMessage]) -> list[ToolMessage]:
    return [m for m in mensajes if isinstance(m, ToolMessage)]


TURNO_ANA = [
    _call("buscar_cliente", {"nombre": "Ana Gómez"}, 1),
    _call("buscar_pedidos", {"cliente_id": 102}, 2),
    AIMessage("Ana Gómez tuvo 3 pedidos por un total de $14.500."),
]


class TestAutonomiaYMultipaso:
    """El agente encadena tools por sí mismo; el ruteo lo decide tools_condition."""

    def test_encadena_dos_tools_para_responder(self, tmp_path: Path) -> None:
        mensajes = _invocar(tmp_path / "cp.db", _modelo(TURNO_ANA), "pedidos de Ana Gómez", "t1")
        buscar_cliente, buscar_pedidos = _observaciones(mensajes)
        assert buscar_cliente.name == "buscar_cliente"
        assert buscar_pedidos.name == "buscar_pedidos"
        assert '"total": 14500.0' in buscar_pedidos.content

    def test_termina_cuando_el_llm_no_pide_tools(self, tmp_path: Path) -> None:
        mensajes = _invocar(tmp_path / "cp.db", _modelo([AIMessage("¡Hola!")]), "hola", "t1")
        assert _observaciones(mensajes) == []
        assert mensajes[-1].content == "¡Hola!"


class TestCicloDeRetorno:
    """Errores e información incompleta vuelven al agente, que reintenta o pide aclaración."""

    def test_error_de_tool_vuelve_al_agente(self, tmp_path: Path) -> None:
        llm = _modelo(
            [
                _call("buscar_pedidos", {"cliente_id": 999}, 1),
                _call("buscar_cliente", {"nombre": "999"}, 2),
                AIMessage("No encontré al cliente 999. ¿Me pasás su nombre?"),
            ]
        )
        mensajes = _invocar(tmp_path / "cp.db", llm, "cliente 999", "t2")
        assert "error" in _observaciones(mensajes)[0].content
        assert str(mensajes[-1].content).endswith("?")

    def test_argumentos_invalidos_no_ejecutan_la_tool(self, tmp_path: Path) -> None:
        llm = _modelo(
            [
                _call("buscar_pedidos", {"cliente_id": -5}, 1),  # viola ge=1 (Pydantic)
                _call("buscar_pedidos", {"cliente_id": 102}, 2),  # el agente corrige y reintenta
                AIMessage("Ana Gómez tuvo 3 pedidos."),
            ]
        )
        invalido, valido = _observaciones(_invocar(tmp_path / "cp.db", llm, "pedidos del -5", "t4"))
        assert invalido.status == "error"
        assert "Argumentos inválidos" in invalido.content
        assert '"pedidos": 3' in valido.content

    def test_timeout_de_la_base_vuelve_como_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tools, "LATENCIA_DB_S", 0.5)
        monkeypatch.setattr(tools, "TIMEOUT_DB_S", 0.01)
        llm = _modelo(
            [
                _call("buscar_pedidos", {"cliente_id": 102}, 1),
                AIMessage("La base no responde, probá en un rato."),
            ]
        )
        (obs,) = _observaciones(_invocar(tmp_path / "cp.db", llm, "pedidos del 102", "t5"))
        assert "no respondió" in obs.content


class TestResilienciaDeEstado:
    """Con el mismo thread_id el agente recuerda la sesión, aun reabriendo la base."""

    def test_recuerda_el_thread_al_reabrir_la_base(self, tmp_path: Path) -> None:
        db = tmp_path / "cp.db"
        _invocar(db, _modelo(TURNO_ANA), "pedidos de Ana Gómez", "t1")
        # Otra conexión y otro grafo (como si el proceso se hubiera reiniciado):
        mensajes = _invocar(db, _modelo([AIMessage("Hablábamos de Ana Gómez.")]), "¿de quién hablábamos?", "t1")
        assert len(mensajes) == 8  # 6 del turno anterior (H, AI, Tool, AI, Tool, AI) + 2 nuevos
        assert mensajes[0].content == "pedidos de Ana Gómez"

    def test_threads_distintos_no_comparten_memoria(self, tmp_path: Path) -> None:
        db = tmp_path / "cp.db"
        _invocar(db, _modelo(TURNO_ANA), "pedidos de Ana Gómez", "t1")
        mensajes = _invocar(db, _modelo([AIMessage("¿De qué cliente?")]), "¿de quién hablábamos?", "otro")
        assert len(mensajes) == 2  # solo la pregunta y la respuesta de este thread


class TestLimiteDeRecursion:
    """recursion_limit corta los bucles y el thread queda usable para el próximo turno."""

    def test_recursion_limit_corta_bucles(self, tmp_path: Path) -> None:
        infinito = _modelo([_call("buscar_pedidos", {"cliente_id": 102}, i) for i in range(50)])
        with pytest.raises(GraphRecursionError):
            _invocar(tmp_path / "cp.db", infinito, "loop", "t3")

    @pytest.mark.parametrize("limite", [10, 9], ids=["limite_par", "limite_impar"])
    def test_turno_cortado_deja_el_thread_consistente(self, tmp_path: Path, limite: int) -> None:
        # Límite par: el corte cae después de `tools`; impar: después de `agent`,
        # con una tool_call sin su ToolMessage.
        async def run() -> None:
            config: RunnableConfig = {"configurable": {"thread_id": "t6"}, "recursion_limit": limite}
            infinito = _modelo([_call("buscar_pedidos", {"cliente_id": 102}, i) for i in range(50)])
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
                graph = build_graph(saver, infinito)
                with pytest.raises(GraphRecursionError):
                    await graph.ainvoke({"messages": [HumanMessage("loop")]}, config)
                await cerrar_turno_cortado(graph, config, "límite de pasos")
                estado = await graph.aget_state(config)

            mensajes = estado.values["messages"]
            llamadas = {tc["id"] for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls}
            assert llamadas == {m.tool_call_id for m in _observaciones(mensajes)}
            assert "límite de pasos" in mensajes[-1].content
            assert estado.next == ()  # el turno quedó cerrado

        asyncio.run(run())
