"""
Camera capture wrapper with reconnect handling, supporting either a live
webcam device (by index) or a video file (by path) as the source -- the
latter is what lets you test multi-camera tracking, priority scoring, and
the dashboard without owning several physical webcams.

Kept separate from `detector.py` so the capture concern (opening a
device, surviving disconnects, looping a test file) is independent from
*what* reads the frames -- a drone/thermal camera would get its own small
wrapper with the same `read()` / `connected` interface and slot into the
same per-camera worker loop (see main.py's CameraWorker).
"""
from __future__ import annotations

import time

import cv2

from . import config


class Camera:
    def __init__(
        self,
        source_type: str = config.SOURCE_TYPE_WEBCAM,
        source_ref: str = "0",
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        loop_video: bool = True,
    ):
        self.source_type = source_type
        self.source_ref = str(source_ref)
        self.width = width
        self.height = height
        self.loop_video = loop_video

        self.cap: cv2.VideoCapture | None = None
        self.connected = False
        self._last_attempt = 0.0
        self._open()

    # -- internal --------------------------------------------------------
    def _open_target(self):
        """Returns the argument to pass to cv2.VideoCapture for the
        current source_type -- an int device index for a webcam, a string
        path/URL for a file."""
        if self.source_type == config.SOURCE_TYPE_FILE:
            return self.source_ref
        try:
            return int(self.source_ref)
        except ValueError:
            # Allow things like "/dev/video2" or a URL to be used as a
            # "webcam" source too, without forcing an int cast.
            return self.source_ref

    def _open(self) -> bool:
        self._last_attempt = time.time()
        if self.source_type == config.SOURCE_TYPE_FILE:
            cap = cv2.VideoCapture(self._open_target())
        else:
            # CAP_DSHOW avoids the multi-second open delay MSMF sometimes
            # has on Windows. Falling back to CAP_ANY keeps this safe on
            # other OSes and on non-index sources.
            backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
            cap = cv2.VideoCapture(self._open_target(), backend)

        if not cap.isOpened():
            cap.release()
            self.cap = None
            self.connected = False
            return False

        if self.source_type == config.SOURCE_TYPE_WEBCAM:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low

        self.cap = cap
        self.connected = True
        return True

    # -- public ------------------------------------------------------
    def read(self):
        """Returns (ok, frame). Webcams get a throttled auto-reconnect on
        failure; video files loop back to the start at end-of-stream
        instead of being treated as a disconnect (unless loop_video is
        False, in which case EOF behaves like any other read failure)."""
        if self.cap is None or not self.connected:
            if time.time() - self._last_attempt >= config.CAMERA_RECONNECT_DELAY_SEC:
                self._open()
            return False, None

        ok, frame = self.cap.read()

        if not ok and self.source_type == config.SOURCE_TYPE_FILE and self.loop_video:
            # Likely end-of-stream rather than a real failure -- rewind
            # and try once more before giving up.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        if not ok or frame is None:
            self.connected = False
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.connected = False

    def switch(self, source_type: str, source_ref: str) -> bool:
        """Release the current device/file and open a different source."""
        self.release()
        self.source_type = source_type
        self.source_ref = str(source_ref)
        return self._open()

    @property
    def resolution(self) -> tuple[int, int]:
        if self.cap is None:
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (w, h)


def probe_cameras(max_index: int = config.CAMERA_PROBE_MAX_INDEX, skip_index: int | None = None) -> list[int]:
    """Best-effort scan of webcam device indices 0..max_index-1 for ones
    that answer. OpenCV has no OS-agnostic way to list camera names, so
    this just tries opening + reading one frame from each candidate.

    `skip_index` (normally whichever index a running camera already has
    open) is excluded -- opening a second handle on a device that's
    already in use can fail, stall, or steal frames on some backends.
    """
    backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    found: list[int] = []
    for idx in range(max_index):
        if idx == skip_index:
            continue
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        if opened:
            ok, _ = cap.read()
            opened = ok
        cap.release()
        if opened:
            found.append(idx)
    return found
