"""
Main entry point for the PC Controller (Hub) application.

Phase 1 goal: initialize the minimal PyQt6 application shell, start the
NetworkController responsible for Zeroconf discovery and establishing
connections to Android Spokes, and log capability handshakes.

This module intentionally keeps GUI light for Phase 1; the focus is the
core communication layer. GUI will be expanded in later phases.

Additionally, start the async UDP TimeSyncServer (FR3) as a background
thread so Android clients can synchronize clocks via UDP echo of
monotonic_ns. The server reads its port/host from config.json (NFR8).
"""

from __future__ import annotations

import asyncio
import sys
import threading

from PyQt6.QtWidgets import QApplication

from .gui.gui_manager import GUIManager
from .network.network_controller import NetworkController
from .network.tcp_command_server import TCPCommandServer
from .network.time_server import TimeSyncServer


def _time_server_thread(stop_flag: threading.Event) -> None:
    async def _runner() -> None:
        server = TimeSyncServer()
        await server.start()
        try:
            # Poll stop flag periodically without busy waiting
            while not stop_flag.is_set():
                await asyncio.sleep(0.5)
        finally:
            await server.stop()

    try:
        asyncio.run(_runner())
    except Exception:
        return


def main() -> int:
    app = QApplication(sys.argv)

    _stop_flag = threading.Event()
    _ts_thread = threading.Thread(
        target=_time_server_thread, args=(_stop_flag,), daemon=True
    )
    _ts_thread.start()

    tcp_server = TCPCommandServer(host="0.0.0.0", port=8080)
    tcp_server.start()

    network = NetworkController()
    gui = GUIManager(network)
    
    tcp_server.set_device_callbacks(
        device_registered_callback=gui.on_device_registered,
        device_status_callback=gui.on_device_status_updated,
        live_gsr_callback=gui.on_live_gsr_data,
        live_video_callback=gui.on_live_video_frame,
        live_thermal_callback=gui.on_live_thermal_frame
    )

    gui.show()
    code = app.exec()

    network.shutdown()
    tcp_server.stop()
    try:
        _stop_flag.set()
        _ts_thread.join(timeout=2.0)
    except Exception:
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
