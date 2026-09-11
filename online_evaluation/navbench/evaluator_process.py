"""Line-delimited control channel for a persistent Isaac scene process."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import socket
import subprocess
import threading
from typing import Callable
import uuid


EVENT_PREFIX = "NAVBENCH_EVENT "


class EvaluatorProcess:
    """Own one evaluator process while its loaded scene remains on the GPU."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        cancelled: Callable[[], bool],
    ) -> None:
        self._events: queue.Queue[dict[str, object]] = queue.Queue()
        self._cancelled = cancelled
        self._control_path = Path("/tmp") / (
            f"navbench-control-{uuid.uuid4().hex}.sock"
        )
        child_env = dict(env)
        child_env["NAVBENCH_CONTROL_SOCKET"] = str(self._control_path)
        self._control_socket: socket.socket | None = None
        self._control = None
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(
            target=self._read_output,
            args=(log_path,),
            name=f"evaluator-output-{self.process.pid}",
            daemon=True,
        )
        self._reader.start()

    def _read_output(self, log_path: Path) -> None:
        assert self.process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log:
            for line in self.process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(EVENT_PREFIX):
                    self._events.put(json.loads(line[len(EVENT_PREFIX):]))

    def _connect_control(self) -> None:
        control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control_socket.connect(str(self._control_path))
        self._control_socket = control_socket
        self._control = control_socket.makefile(
            "w", encoding="utf-8", buffering=1
        )

    def _close_control(self) -> None:
        if self._control is not None:
            self._control.close()
            self._control = None
        if self._control_socket is not None:
            self._control_socket.close()
            self._control_socket = None
        self._control_path.unlink(missing_ok=True)

    def send(self, command: dict[str, object]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"evaluator exited with code {self.process.returncode}"
            )
        if self._control is None:
            raise RuntimeError("evaluator control socket is not connected")
        self._control.write(json.dumps(command, separators=(",", ":")) + "\n")
        self._control.flush()

    def wait_for(self, expected: str) -> dict[str, object]:
        while True:
            if self._cancelled():
                raise RuntimeError("evaluation cancelled")
            try:
                event = self._events.get(timeout=1.0)
            except queue.Empty:
                code = self.process.poll()
                if code is not None:
                    self._close_control()
                    raise RuntimeError(f"evaluator exited with code {code}")
                continue
            if event.get("event") != expected:
                raise RuntimeError(
                    f"expected evaluator event {expected!r}, received {event!r}"
                )
            if expected == "ready":
                self._connect_control()
            return event

    def close(self) -> None:
        code = self.process.poll()
        if code is not None:
            self._close_control()
            if code != 0:
                raise RuntimeError(f"evaluator exited with code {code}")
            return
        self.send({"command": "close"})
        self._close_control()
        code = self.process.wait()
        self._reader.join()
        assert self.process.stdout is not None
        self.process.stdout.close()
        if code != 0:
            raise RuntimeError(f"evaluator exited with code {code}")
