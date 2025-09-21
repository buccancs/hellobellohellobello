"""Async UDP Time Synchronization Server (FR3).

- Uses asyncio DatagramProtocol to serve time sync requests.
- Reads bind host and timesync_port from centralized config.json (NFR8).
- On any datagram, responds immediately with time.monotonic_ns() as ASCII bytes.
- Designed to be started as a concurrent task from main application.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

try:
    # Local config loader (added for NFR8)
    from ..config import get as cfg_get
except Exception:  # pragma: no cover - optional import safety

    def cfg_get(key: str, default=None):
        return default


class TimeSyncProtocol(asyncio.DatagramProtocol):
    """Datagram protocol that echoes current monotonic time in nanoseconds."""

    def __init__(self) -> None:
        super().__init__()
        self.transport: asyncio.transports.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.transports.DatagramTransport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        ts_ns = time.monotonic_ns()
        payload = str(ts_ns).encode("ascii")
        if self.transport is not None:
            try:
                self.transport.sendto(payload, addr)
            except Exception:  # pragma: no cover - safety
                pass


class TimeSyncServer:
    """Async UDP server that sends time.monotonic_ns() on any request.

    Typical usage:
        server = TimeSyncServer()
        await server.start()
        ...
        await server.stop()
    """

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self._host = host if host is not None else cfg_get("server_ip", "0.0.0.0")
        self._port = port if port is not None else int(cfg_get("timesync_port", 8081))
        self._transport: asyncio.transports.DatagramTransport | None = None
        self._protocol: TimeSyncProtocol | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def is_running(self) -> bool:
        return self._transport is not None

    async def start(self) -> None:
        if self._transport is not None:
            return
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: TimeSyncProtocol(), local_addr=(self._host, self._port)
        )
        sockname = transport.get_extra_info("sockname")
        if isinstance(sockname, tuple) and len(sockname) >= 2:
            with contextlib.suppress(Exception):
                self._port = int(sockname[1])
        self._transport = transport
        self._protocol = protocol

    async def stop(self) -> None:
        if self._transport is not None:
            try:
                self._transport.close()
            finally:
                self._transport = None
                self._protocol = None
