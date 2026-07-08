from __future__ import annotations

import socket
from contextlib import suppress

from .config import JudgeConfig
from .output import serialize_rows
from .types import OutputRow


class JudgeClient:
    """Small TCP client placeholder.

    Replace message strings here after the official judge-box protocol is
    released. The rest of the pipeline only calls start(), send_results(), end().
    """

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def _send(self, payload: str) -> None:
        if not self.config.enabled:
            return
        with socket.create_connection((self.config.host, self.config.port), timeout=self.config.timeout_sec) as sock:
            sock.sendall(payload.encode("utf-8"))

    def start(self) -> None:
        with suppress(Exception):
            self._send("START\n")

    def send_results(self, rows: list[OutputRow]) -> None:
        with suppress(Exception):
            self._send(serialize_rows(rows) + "\n")

    def end(self) -> None:
        with suppress(Exception):
            self._send("END\n")

