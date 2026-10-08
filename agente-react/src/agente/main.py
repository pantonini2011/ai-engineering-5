"""Punto de entrada: corre la demo (genera la traza) o un chat interactivo.

Uso:
    python -m agente.main                      # demo guionada → traces/
    python -m agente.main --chat --thread-id t1  # chat interactivo con memoria
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agente.graph import build_graph, cerrar_turno_cortado

RECURSION_LIMIT = 10  # techo de pasos por invocación (evita bucles y costos)
DB_PATH = "checkpoints.db"
TRACES_DIR = Path("traces")

log = logging.getLogger("agente")


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def _describir(msg: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje en un evento legible para la traza."""
    if isinstance(msg, AIMessage) and msg.tool_calls:
        return {
            "tipo": "accion",
            "tool_calls": [{"tool": tc["name"], "args": tc["args"]} for tc in msg.tool_calls],
        }
    if isinstance(msg, ToolMessage):
        try:
            contenido: Any = json.loads(str(msg.content))
        except json.JSONDecodeError:
            contenido = msg.content
        return {"tipo": "observacion", "tool": msg.name, "resultado": contenido}
    if isinstance(msg, AIMessage):
        return {"tipo": "respuesta_final", "texto": msg.text}
    return {"tipo": msg.type, "texto": str(msg.content)}


async def ejecutar_turno(graph: Any, thread_id: str, pregunta: str) -> dict[str, Any]:
    """Corre un turno del usuario y devuelve su traza paso a paso."""
    log.info("[%s] Usuario: %s", thread_id, pregunta)
    pasos: list[dict[str, Any]] = []
    try:
        async for update in graph.astream(
            {"messages": [HumanMessage(pregunta)]}, _config(thread_id), stream_mode="updates"
        ):
            for nodo, salida in update.items():
                for msg in salida["messages"]:
                    evento = {"nodo": nodo, **_describir(msg)}
                    pasos.append(evento)
                    if evento["tipo"] == "accion":
                        for tc in evento["tool_calls"]:
                            log.info("  → El agente decide usar: %s(%s)", tc["tool"], tc["args"])
                    elif evento["tipo"] == "observacion":
                        log.info("  → %s devuelve: %s", evento["tool"], evento["resultado"])
                    else:
                        log.info("  → Respuesta: %s", evento.get("texto"))
    except GraphRecursionError:
        log.warning("  ✖ Se alcanzó recursion_limit=%d; se corta el ciclo.", RECURSION_LIMIT)
        pasos.append({"tipo": "corte", "motivo": f"recursion_limit={RECURSION_LIMIT}"})
        await cerrar_turno_cortado(
            graph, _config(thread_id), f"se alcanzó el límite de {RECURSION_LIMIT} pasos de razonamiento"
        )

    llamadas = sum(len(p["tool_calls"]) for p in pasos if p["tipo"] == "accion")
    return {"thread_id": thread_id, "usuario": pregunta, "llamadas_a_tools": llamadas, "pasos": pasos}


DEMO: list[tuple[str, str]] = [
    # 1) Multi-paso: buscar_cliente → buscar_pedidos (≥2 llamadas)
    ("sesion-ana", "¿Cuántos pedidos tuvo Ana Gómez y cuál fue el total?"),
    # 2) Memoria: "el último" se resuelve con el contexto del thread
    ("sesion-ana", "¿Y el último? ¿Qué productos tenía?"),
    # 3) Ciclo de retorno: id inexistente → error → reintento o pedido de aclaración
    ("sesion-errores", "Pasame el total de pedidos del cliente 999."),
    # 4) Ambigüedad: dos clientes "Ana" → el agente debería pedir aclaración
    ("sesion-errores", "¿Y cuántos pedidos tiene Ana?"),
]


async def demo(llm: BaseChatModel | None = None, db_path: str = DB_PATH) -> list[dict[str, Any]]:
    turnos: list[dict[str, Any]] = []
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(saver, llm)
        for thread_id, pregunta in DEMO:
            turnos.append(await ejecutar_turno(graph, thread_id, pregunta))

    # 5) Persistencia real: se abre una conexión NUEVA al .db (como si el proceso
    #    se hubiera reiniciado) y se retoma el mismo thread_id.
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(saver, llm)
        turnos.append(
            await ejecutar_turno(graph, "sesion-ana", "Recordame: ¿de qué cliente veníamos hablando?")
        )
    return turnos


async def chat(thread_id: str) -> None:
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        graph = build_graph(saver)
        print(f"Chat con memoria (thread_id={thread_id}). Ctrl+C o 'salir' para terminar.")
        while (pregunta := input("\nVos: ").strip()).lower() not in {"salir", "exit"}:
            if pregunta:
                await ejecutar_turno(graph, thread_id, pregunta)


def _configurar_logs(archivo: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if archivo:
        handlers.append(logging.FileHandler(archivo, mode="w", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)
    for ruidoso in ("httpx", "anthropic", "openai"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Agente ReAct con memoria persistente")
    parser.add_argument("--chat", action="store_true", help="modo interactivo")
    parser.add_argument("--thread-id", default="sesion-1")
    args = parser.parse_args()

    if args.chat:
        _configurar_logs(None)
        await chat(args.thread_id)
        return

    TRACES_DIR.mkdir(exist_ok=True)
    _configurar_logs(TRACES_DIR / "traza_ejecucion.log")
    turnos = await demo()
    salida = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "recursion_limit": RECURSION_LIMIT,
        "turnos": turnos,
    }
    (TRACES_DIR / "traza_ejecucion.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Traza guardada en %s/", TRACES_DIR)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
