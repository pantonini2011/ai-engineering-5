"""Definición del StateGraph ReAct: nodo de modelo + nodo de herramientas + arista condicional."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agente.tools import TOOLS

SYSTEM_PROMPT = """Sos un asistente de atención comercial con acceso a la base de pedidos.
Reglas:
- Usá las herramientas para obtener datos; nunca inventes números.
- Si una herramienta devuelve "error" o datos incompletos, corregí los argumentos y reintentá,
  o pedile al usuario la aclaración necesaria.
- Si una búsqueda por nombre devuelve varios clientes, preguntá a cuál se refiere.
- Respondé en español rioplatense, breve, con montos en formato $14.500.
- Usá el historial de la conversación para resolver referencias como "¿y el último?"."""

# Tope de mensajes que se le envían al LLM en cada turno (evita "estado sucio").
MAX_CONTEXT_MESSAGES = 20


class AgentState(MessagesState):
    """Estado del agente. Hereda `messages` de MessagesState, cuyo reducer `add_messages`
    agrega mensajes nuevos en vez de reemplazar la lista (análogo a operator.add,
    pero además deduplica por id)."""


CLAVES_POR_PROVEEDOR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def proveedor_llm() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def verificar_configuracion() -> None:
    """Falla rápido, con un mensaje claro, si el proveedor o su API key no están configurados."""
    provider = proveedor_llm()
    if provider not in CLAVES_POR_PROVEEDOR:
        raise SystemExit(f"✖ LLM_PROVIDER={provider!r} no es válido: usá 'anthropic' u 'openai'.")
    clave = CLAVES_POR_PROVEEDOR[provider]
    if not os.getenv(clave):
        raise SystemExit(f"✖ Falta {clave} (LLM_PROVIDER={provider}). Completala en el .env.")


def build_llm() -> BaseChatModel:
    """Crea el LLM según la variable de entorno LLM_PROVIDER (anthropic | openai)."""
    provider = proveedor_llm()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    from langchain_anthropic import ChatAnthropic

    # Los modelos Claude actuales no aceptan `temperature` distinta del default.
    return ChatAnthropic(model_name=os.getenv("LLM_MODEL", "claude-sonnet-5-5"))  # type: ignore[call-arg]


def _recortar(messages: list[AnyMessage]) -> list[BaseMessage]:
    """Conserva los últimos mensajes sin cortar un par tool_call/tool_result."""
    return trim_messages(
        messages,
        strategy="last",
        token_counter=len,  # cuenta mensajes, no tokens
        max_tokens=MAX_CONTEXT_MESSAGES,
        start_on="human",
        include_system=False,
    )


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    llm: BaseChatModel | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construye y compila el grafo ReAct.

    START → agent ─(tools_condition)─┬─> tools ─> agent  (ciclo)
                                     └─> END
    """
    model = (llm or build_llm()).bind_tools(TOOLS)

    async def agent(state: AgentState) -> dict[str, list[BaseMessage]]:
        mensajes = [SystemMessage(SYSTEM_PROMPT), *_recortar(state["messages"])]
        respuesta = await model.ainvoke(mensajes)
        return {"messages": [respuesta]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    # handle_tool_errors=True: si una tool lanza excepción, el error vuelve al LLM
    # como ToolMessage y el agente puede reintentar en vez de romper el grafo.
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


async def cerrar_turno_cortado(
    graph: CompiledStateGraph[Any, Any, Any, Any], config: RunnableConfig, motivo: str
) -> None:
    """Deja el thread consistente después de un GraphRecursionError.

    Si el corte quedó después de `agent`, hay tool_calls sin su ToolMessage y el
    próximo turno fallaría (los proveedores rechazan ese historial). Se completan
    esas llamadas y se agrega una respuesta final, registrada como salida de `agent`
    para que `tools_condition` cierre el turno en END.
    """
    estado = await graph.aget_state(config)
    mensajes: list[AnyMessage] = estado.values.get("messages", [])
    respondidas = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    pendientes = [
        tc
        for m in mensajes
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
        if tc["id"] not in respondidas
    ]
    cierre: list[BaseMessage] = [
        ToolMessage(
            content=json.dumps({"error": f"No se ejecutó: {motivo}."}, ensure_ascii=False),
            tool_call_id=tc["id"],
            name=tc["name"],
            status="error",
        )
        for tc in pendientes
    ]
    cierre.append(
        AIMessage(f"No pude completar la consulta: {motivo}. ¿Podés reformularla o hacerla más concreta?")
    )
    await graph.aupdate_state(config, {"messages": cierre}, as_node="agent")
