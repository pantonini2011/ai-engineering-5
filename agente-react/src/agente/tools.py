"""Herramientas del agente: simulan una base de datos de clientes y pedidos.

El LLM decide qué herramienta usar leyendo SOLO el nombre, los argumentos tipados
y el docstring. Por eso los docstrings son largos y explícitos.

Los argumentos se validan con Pydantic antes de tocar la "base": si el LLM manda
algo inválido, la tool no se ejecuta y el error vuelve al agente para que corrija.
Cada consulta tiene un timeout: si la "base" no responde a tiempo, la tool devuelve
un error en vez de colgar el grafo.

Mínimo privilegio: las tools son de solo lectura y consultan por id o nombre con
parámetros validados; nunca ejecutan SQL ni código generado por el LLM.
"""

from __future__ import annotations

import asyncio
import json
import unicodedata
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

# --- "Base de datos" simulada -------------------------------------------------

CLIENTES: dict[int, dict[str, Any]] = {
    101: {"nombre": "Juan Pérez", "ciudad": "Rosario"},
    102: {"nombre": "Ana Gómez", "ciudad": "Buenos Aires"},
    103: {"nombre": "Carlos Díaz", "ciudad": "Córdoba"},
    104: {"nombre": "Ana Martínez", "ciudad": "Mendoza"},
}

PEDIDOS: dict[int, list[dict[str, Any]]] = {
    101: [{"pedido_id": 5001, "fecha": "2026-08-02", "total": 3200.0, "estado": "entregado"}],
    102: [
        {"pedido_id": 5010, "fecha": "2026-07-15", "total": 4000.0, "estado": "entregado"},
        {"pedido_id": 5023, "fecha": "2026-08-20", "total": 6300.0, "estado": "entregado"},
        {"pedido_id": 5047, "fecha": "2026-09-30", "total": 4200.0, "estado": "en camino"},
    ],
    103: [],
    104: [{"pedido_id": 5031, "fecha": "2026-09-01", "total": 1500.0, "estado": "cancelado"}],
}

ITEMS: dict[int, list[dict[str, Any]]] = {
    5001: [{"producto": "Teclado", "cantidad": 1, "precio": 3200.0}],
    5010: [{"producto": "Mouse", "cantidad": 2, "precio": 2000.0}],
    5023: [{"producto": "Monitor 24\"", "cantidad": 1, "precio": 6300.0}],
    5047: [
        {"producto": "Auriculares", "cantidad": 1, "precio": 3000.0},
        {"producto": "Cable HDMI", "cantidad": 2, "precio": 600.0},
    ],
    5031: [{"producto": "Webcam", "cantidad": 1, "precio": 1500.0}],
}


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.lower().strip()


# --- Acceso a la "base" con latencia simulada y timeout -----------------------

LATENCIA_DB_S = 0.05  # latencia simulada de cada consulta
TIMEOUT_DB_S = 2.0  # techo por consulta; si se supera, la tool devuelve error

T = TypeVar("T")


async def _consultar(consulta: Callable[[], T]) -> T:
    """Simula una consulta de I/O a la base de datos."""
    await asyncio.sleep(LATENCIA_DB_S)
    return consulta()


async def _con_timeout(consulta: Callable[[], dict[str, Any]], tool_name: str) -> dict[str, Any]:
    """Ejecuta la consulta con `asyncio.wait_for`; ante timeout devuelve un error para el LLM."""
    try:
        return await asyncio.wait_for(_consultar(consulta), timeout=TIMEOUT_DB_S)
    except TimeoutError:
        return {
            "error": f"{tool_name} no respondió en {TIMEOUT_DB_S}s. Reintentá una vez; "
            "si vuelve a fallar, avisale al usuario que la base no está disponible."
        }


# --- Contratos de entrada (Pydantic) -----------------------------------------


class _Entrada(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class BuscarClienteInput(_Entrada):
    nombre: str = Field(
        min_length=2,
        max_length=80,
        description='Nombre completo o parcial del cliente, por ejemplo "Ana Gómez" o "Ana".',
    )


class BuscarPedidosInput(_Entrada):
    cliente_id: int = Field(ge=1, le=99_999, description="Número de cliente (entero, por ejemplo 102).")


class DetallePedidoInput(_Entrada):
    pedido_id: int = Field(ge=1, le=999_999, description="Número de pedido (entero, por ejemplo 5047).")


def _error_de_validacion(error: ValidationError | ValidationErrorV1) -> str:
    """Traduce el ValidationError a un mensaje que el LLM pueda usar para corregir los argumentos."""
    detalles = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'argumentos'}: {e['msg']}" for e in error.errors()
    )
    mensaje = f"Argumentos inválidos ({detalles}). Corregilos y reintentá, o pedí aclaración."
    return json.dumps({"error": mensaje}, ensure_ascii=False)


# --- Herramientas ------------------------------------------------------------


@tool(args_schema=BuscarClienteInput)
async def buscar_cliente(nombre: str) -> dict[str, Any]:
    """Busca clientes por nombre (búsqueda parcial, sin distinguir tildes ni mayúsculas).

    Usala SIEMPRE que el usuario mencione a un cliente por su nombre y no por su
    número de cliente, porque las demás herramientas necesitan el `cliente_id`.

    NO la uses si ya conocés el `cliente_id` (porque el usuario lo dio o porque
    aparece en la conversación): pasá directo a `buscar_pedidos`. Tampoco sirve para
    buscar pedidos ni productos.

    Returns:
        {"resultados": [{"cliente_id": int, "nombre": str, "ciudad": str}, ...]}.
        Si hay más de un resultado, el nombre es ambiguo: pedile al usuario que
        aclare (por ejemplo, la ciudad) en lugar de adivinar.
        Si la lista está vacía, no existe ningún cliente con ese nombre.
    """
    buscado = _normalizar(nombre)

    def consulta() -> dict[str, Any]:
        resultados = [
            {"cliente_id": cid, **datos}
            for cid, datos in CLIENTES.items()
            if buscado in _normalizar(datos["nombre"])
        ]
        return {"resultados": resultados}

    return await _con_timeout(consulta, "buscar_cliente")


@tool(args_schema=BuscarPedidosInput)
async def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """Devuelve el resumen de pedidos de un cliente: cantidad, monto total y la lista
    de pedidos (id, fecha, total, estado) ordenada del más antiguo al más reciente.

    Usala para preguntas sobre cuántos pedidos tuvo un cliente, cuánto gastó en total,
    o cuál fue su primer/último pedido. Requiere el `cliente_id` numérico; si solo
    tenés el nombre, llamá antes a `buscar_cliente`.

    NO la uses para ver qué productos tenía un pedido (para eso está `detalle_pedido`),
    ni la vuelvas a llamar si el resumen de ese cliente ya está en la conversación.

    Returns:
        {"cliente_id": int, "pedidos": int, "total": float, "detalle": [...]}
        o {"error": str} si el cliente no existe. Ante un error, revisá el id
        (por ejemplo buscándolo por nombre) y volvé a intentar, o pedí aclaración.
    """

    def consulta() -> dict[str, Any]:
        if cliente_id not in CLIENTES:
            return {"error": f"No existe el cliente {cliente_id}. Verificá el id con buscar_cliente."}
        pedidos = sorted(PEDIDOS.get(cliente_id, []), key=lambda p: p["fecha"])
        return {
            "cliente_id": cliente_id,
            "pedidos": len(pedidos),
            "total": sum(p["total"] for p in pedidos),
            "detalle": pedidos,
        }

    return await _con_timeout(consulta, "buscar_pedidos")


@tool(args_schema=DetallePedidoInput)
async def detalle_pedido(pedido_id: int) -> dict[str, Any]:
    """Devuelve los ítems (producto, cantidad, precio unitario) de UN pedido puntual.

    Usala cuando el usuario quiera saber qué se compró en un pedido específico
    (por ejemplo "¿qué tenía el último pedido?"). Requiere el `pedido_id`, que se
    obtiene del campo `detalle` de `buscar_pedidos`.

    NO la uses con un número de cliente (son ids distintos) ni inventes un
    `pedido_id`: si no lo tenés, obtenelo primero con `buscar_pedidos`.

    Returns:
        {"pedido_id": int, "items": [...]} o {"error": str} si el pedido no existe.
    """

    def consulta() -> dict[str, Any]:
        if pedido_id not in ITEMS:
            return {"error": f"No existe el pedido {pedido_id}. Obtené ids válidos con buscar_pedidos."}
        return {"pedido_id": pedido_id, "items": ITEMS[pedido_id]}

    return await _con_timeout(consulta, "detalle_pedido")


TOOLS = [buscar_cliente, buscar_pedidos, detalle_pedido]

# Si el LLM manda argumentos inválidos, la tool no se ejecuta y devuelve el error como
# observación (en vez de lanzar), así el agente puede corregir y reintentar.
for _t in TOOLS:
    _t.handle_validation_error = _error_de_validacion
