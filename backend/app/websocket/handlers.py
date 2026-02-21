from fastapi import WebSocket

async def ws_case_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("WebSocket connected (mock handler)")
    await websocket.close()