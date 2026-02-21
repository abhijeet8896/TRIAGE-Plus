"""
WebSocket Manager — Real-time bidirectional case updates.
=========================================================
Architecture:
  - Each patient case has a "room" (case_id)
  - PHW and Specialist both join the same room
  - JWT validated on WS handshake
  - Role-bound message types

Message types:
  SPECIALIST_ACKNOWLEDGED — Specialist has seen the case
  SPECIALIST_ADVICE_SUBMITTED — Final advice pushed to PHW
  STATUS_UPDATE — Generic case status change
  PING/PONG — Heartbeat for rural network stability
"""

import json
import asyncio
import logging
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from app.core.security import decode_access_token

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections grouped by case_id "rooms".
    Thread-safe via asyncio primitives.
    """

    def __init__(self):
        # {case_id: {websocket: {user_id, role}}}
        self._rooms: Dict[str, Dict[WebSocket, dict]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        ws: WebSocket,
        case_id: str,
        user_id: str,
        role: str
    ) -> None:
        await ws.accept()
        async with self._lock:
            if case_id not in self._rooms:
                self._rooms[case_id] = {}
            self._rooms[case_id][ws] = {"user_id": user_id, "role": role}

        logger.info(f"WS connected: user={user_id} role={role} case={case_id}")
        await self._send_to_ws(ws, {
            "type": "CONNECTION_ESTABLISHED",
            "case_id": case_id,
            "role": role,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def disconnect(self, ws: WebSocket, case_id: str) -> None:
        async with self._lock:
            if case_id in self._rooms:
                self._rooms[case_id].pop(ws, None)
                if not self._rooms[case_id]:
                    del self._rooms[case_id]
        logger.info(f"WS disconnected: case={case_id}")

    async def broadcast_to_room(self, case_id: str, message: dict) -> None:
        """Send message to all connections in a case room."""
        if case_id not in self._rooms:
            return

        message["timestamp"] = datetime.utcnow().isoformat()
        disconnected = []

        for ws, meta in self._rooms[case_id].items():
            try:
                await self._send_to_ws(ws, message)
            except Exception as e:
                logger.warning(f"WS send failed for {meta['user_id']}: {e}")
                disconnected.append(ws)

        # Cleanup dead connections
        async with self._lock:
            for ws in disconnected:
                self._rooms.get(case_id, {}).pop(ws, None)

    async def send_to_role(
        self, case_id: str, role: str, message: dict
    ) -> None:
        """Send message only to connections with a specific role in the room."""
        if case_id not in self._rooms:
            return

        message["timestamp"] = datetime.utcnow().isoformat()
        for ws, meta in self._rooms[case_id].items():
            if meta["role"] == role:
                try:
                    await self._send_to_ws(ws, message)
                except Exception as e:
                    logger.warning(f"WS role-send failed: {e}")

    async def _send_to_ws(self, ws: WebSocket, message: dict) -> None:
        await ws.send_text(json.dumps(message))

    def get_room_count(self, case_id: str) -> int:
        return len(self._rooms.get(case_id, {}))

    def get_active_case_ids(self) -> list:
        return list(self._rooms.keys())


# Singleton
ws_manager = ConnectionManager()


# ─── WebSocket Endpoint Handler ───────────────────────────────────────────────

async def ws_case_endpoint(websocket: WebSocket, case_id: str):
    """
    Main WebSocket endpoint for case-level real-time communication.
    URL: /ws/case/{case_id}?token=<JWT>

    On connect: validates JWT, joins case room.
    On message: routes by type.
    On disconnect: removes from room.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Token required")
        return

    try:
        payload = decode_access_token(token)
        user_id = payload["sub"]
        role = payload["role"]
    except Exception:
        await websocket.close(code=4003, reason="Invalid token")
        return

    await ws_manager.connect(websocket, case_id, user_id, role)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(raw)
                await _handle_ws_message(websocket, case_id, user_id, role, message)
            except asyncio.TimeoutError:
                # Send heartbeat ping to keep rural connection alive
                await ws_manager._send_to_ws(websocket, {"type": "PING"})

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, case_id)
    except Exception as e:
        logger.error(f"WS error for case={case_id}: {e}")
        await ws_manager.disconnect(websocket, case_id)


async def _handle_ws_message(
    ws: WebSocket, case_id: str, user_id: str, role: str, message: dict
) -> None:
    """Route incoming WebSocket messages by type."""
    msg_type = message.get("type", "")

    if msg_type == "PONG":
        # Heartbeat response — no action needed
        return

    elif msg_type == "SPECIALIST_ACKNOWLEDGED":
        if role not in ("specialist", "admin"):
            await ws_manager._send_to_ws(ws, {
                "type": "ERROR", "message": "Unauthorized action"
            })
            return
        # Broadcast acknowledgement to PHW
        await ws_manager.send_to_role(case_id, "phw", {
            "type": "SPECIALIST_ACKNOWLEDGED",
            "specialist_id": user_id,
            "case_id": case_id,
        })
        logger.info(f"Specialist {user_id} acknowledged case {case_id}")

    elif msg_type == "STATUS_UPDATE":
        # Broadcast status update to entire room
        await ws_manager.broadcast_to_room(case_id, {
            "type": "STATUS_UPDATE",
            "status": message.get("status"),
            "case_id": case_id,
            "updated_by": user_id,
        })

    else:
        await ws_manager._send_to_ws(ws, {
            "type": "ERROR",
            "message": f"Unknown message type: {msg_type}"
        })


# ─── Helper: Push advice to PHW after specialist submits ─────────────────────

async def push_specialist_advice_to_phw(case_id: str, advice: dict) -> None:
    """Called from the specialist advice endpoint after DB save."""
    await ws_manager.send_to_role(case_id, "phw", {
        "type": "SPECIALIST_ADVICE_SUBMITTED",
        "case_id": case_id,
        "advice": advice,
    })
    logger.info(f"Specialist advice pushed via WebSocket to PHW for case {case_id}")
