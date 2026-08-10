"""Dashboard WebSocket push.

Streams a full snapshot on connect and then a delta-free snapshot on an
interval.  Snapshots rather than deltas: the payload is small, and a client
that missed a message must never render stale position data.

The socket is a convenience.  Every value it pushes is also available from the
REST endpoints, so a browser that cannot open a WebSocket (or a deployment
without FastAPI) simply polls instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: set[Any] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)
        log.debug("dashboard websocket connected (%d total)", len(self.connections))

    async def disconnect(self, websocket: Any) -> None:
        async with self._lock:
            self.connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.connections:
            return
        message = json.dumps(payload, default=str)
        async with self._lock:
            targets = list(self.connections)
        dead: list[Any] = []
        for websocket in targets:
            try:
                await websocket.send_text(message)
            except Exception:  # noqa: BLE001 - a dropped client is normal
                dead.append(websocket)
        if dead:
            async with self._lock:
                for websocket in dead:
                    self.connections.discard(websocket)


def register_websocket(app: Any, routes: Any, interval: float = 2.0) -> ConnectionManager:
    """Attach ``/ws`` to a FastAPI app and start the broadcast task."""

    from fastapi import WebSocket, WebSocketDisconnect  # type: ignore

    manager = ConnectionManager()
    app.state.ws_manager = manager

    @app.websocket("/ws")
    async def dashboard_socket(websocket: WebSocket) -> None:  # type: ignore[no-untyped-def]
        await manager.connect(websocket)
        try:
            await websocket.send_text(json.dumps(routes.snapshot(), default=str))
            while True:
                # Client messages are only used as a keep-alive / refresh nudge.
                message = await websocket.receive_text()
                if message.strip() == "refresh":
                    await websocket.send_text(
                        json.dumps(routes.snapshot(), default=str)
                    )
        except WebSocketDisconnect:
            await manager.disconnect(websocket)
        except Exception as exc:  # noqa: BLE001
            log.debug("dashboard websocket error: %s", exc)
            await manager.disconnect(websocket)

    async def _broadcaster() -> None:
        while True:
            await asyncio.sleep(interval)
            if not manager.connections:
                continue
            try:
                await manager.broadcast(routes.snapshot())
            except Exception as exc:  # noqa: BLE001
                log.debug("snapshot broadcast failed: %s", exc)

    @app.on_event("startup")
    async def _start_broadcaster() -> None:  # type: ignore[no-untyped-def]
        app.state.ws_task = asyncio.create_task(_broadcaster(), name="ws-broadcast")

    @app.on_event("shutdown")
    async def _stop_broadcaster() -> None:  # type: ignore[no-untyped-def]
        task = getattr(app.state, "ws_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return manager


__all__ = ["ConnectionManager", "register_websocket"]
