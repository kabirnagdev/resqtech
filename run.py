"""
Convenience entry point: `python run.py`

Equivalent to:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 3

Do NOT run this with uvicorn's --reload flag (or reload=True here) -- the
capture worker opens the webcam in a background thread on import, and
reload spins up a second process that would fight over the same camera.
"""
from backend import config

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        # A browser tab left open holds the MJPEG video stream open
        # indefinitely from the server's point of view. main.py's
        # /video_feed generator already exits promptly once our own
        # capture thread stops, but this is a belt-and-suspenders cap so
        # Ctrl+C always returns control within a few seconds even if some
        # other connection (a WebSocket, a stalled client) is still open.
        timeout_graceful_shutdown=3,
    )
