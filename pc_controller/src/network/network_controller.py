"""NetworkController for the PC Controller (Hub).

Responsibilities in Phase 1:
- Browse for Android Spoke services via Zeroconf/mDNS.
- Allow connecting to a discovered device over TCP in a background QThread.
- Send an initial JSON command {"id": 1, "command": "query_capabilities"}
  and receive the response, emitting it back to the GUI.

Phase 4 extensions:
- Broadcast start/stop/flash commands with session management.
- Perform NTP-like time synchronization handshake and store per-device offsets.

Note: The Android Spoke advertises service type: _gsr-controller._tcp.local.
"""
from __future__ import annotations

import base64
import json
import os
import random
import socket
import time
from dataclasses import dataclass

from core.device_manager import DeviceManager
from data.data_aggregator import get_local_ip
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from zeroconf import IPVersion, ServiceBrowser, Zeroconf

from config import get as cfg_get
from .file_transfer_server import FileTransferServer
from .protocol import (
    build_v1_cmd,
    build_v1_query_capabilities,
    build_v1_start_recording,
    build_v1_time_sync_req,
    compute_backoff_schedule,
    compute_time_sync,
    compute_time_sync_stats,
    decode_frames,
    encode_frame,
)

# TLS (optional)
try:
    from .tls_utils import create_client_ssl_context
except Exception:  # pragma: no cover - optional import guard
    def create_client_ssl_context():
        return None  # type: ignore

SERVICE_TYPE = "_gsr-controller._tcp.local."


def _connect(host: str, port: int, timeout: float) -> socket.socket:
    """Create a TCP connection and optionally wrap with TLS if configured.

    Uses tls_utils.create_client_ssl_context() when available and PC_TLS_ENABLE=1.
    The returned object behaves like a socket with recv/sendall/close.
    """
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        ctx = None
        try:
            ctx = create_client_ssl_context()
        except Exception:
            ctx = None
        if ctx is not None:
            # If hostname verification is enabled, pass server_hostname
            server_hostname = host if getattr(ctx, "check_hostname", False) else None
            return ctx.wrap_socket(s, server_hostname=server_hostname)
        return s
    except Exception:
        try:
            s.close()
        except Exception:
            pass
        raise


@dataclass(frozen=True)
class DiscoveredDevice:
    name: str
    address: str
    port: int


class _ZeroconfListener:
    """Listener object for Zeroconf service browser."""

    def __init__(self, parent: NetworkController) -> None:
        self._parent = parent

    def remove_service(self, zeroconf: Zeroconf, type_: str, name: str) -> None:  # noqa: D401
        self._parent._on_service_removed(name)

    def add_service(self, zeroconf: Zeroconf, type_: str, name: str) -> None:  # noqa: D401
        info = zeroconf.get_service_info(type_, name, 5000)
        if not info:
            self._parent._emit_log(f"Service resolved empty: {name}")
            return
        addresses = info.parsed_scoped_addresses()
        if not addresses:
            self._parent._emit_log(f"No addresses for: {name}")
            return
        address = addresses[0]
        device = DiscoveredDevice(name=name.rstrip("."), address=address, port=info.port)
        self._parent._on_service_added(device)


class ConnectionWorker(QThread):
    """Background worker to manage a TCP connection to a device.

    On start, connects to the device, sends the query_capabilities command,
    and waits for a single JSON line response. Emits signals for logs and
    capabilities received, then exits.
    """

    log = pyqtSignal(str)
    capabilities = pyqtSignal(str, dict)  # device name, capabilities dict

    def __init__(self, device: DiscoveredDevice, timeout: float = 5.0) -> None:
        super().__init__()
        self._device = device
        self._timeout = timeout

    def run(self) -> None:  # noqa: D401
        try:
            sock = _connect(self._device.address, self._device.port, self._timeout)
            try:
                sock.settimeout(self._timeout)
                # Prefer v1 length-prefixed query, fall back accepted on receive
                v1 = build_v1_query_capabilities(msg_id=int(time.time() * 1000) % 2_000_000_000)
                sock.sendall(encode_frame(v1))
                self.log.emit(f"Sent v1 query_capabilities to {self._device.name}")

                # Receive response: try length-prefixed first, then legacy newline
                buf = b""
                payload = None
                deadline = time.monotonic() + self._timeout
                while time.monotonic() < deadline:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    # Try length-prefixed decode
                    res = decode_frames(buf)
                    if res.messages:
                        payload = res.messages[0]
                        break
                    # If buffer looks like legacy (no numeric length prefix), try newline-delimited
                    nl = buf.find(b"\n")
                    if nl != -1 and not buf[:nl].isdigit():
                        line, buf = buf.split(b"\n", 1)
                        try:
                            payload = json.loads(line.decode("utf-8", errors="replace"))
                            break
                        except Exception:
                            continue
                    # otherwise wait for more data (likely length-prefixed incomplete)
                    buf = res.remainder
                if payload is None:
                    self.log.emit("No response to capabilities query")
                    return
                self.log.emit(f"Received: {payload}")
                self.capabilities.emit(self._device.name, payload)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"Connection error to {self._device.name}: {exc}")


class _BroadcastWorker(QThread):
    """Send commands to all devices in a background thread with retries.

    - Uses v1 length-prefixed framing by default with legacy fallback.
    - Implements exponential backoff with jitter for retries per device.
    """

    log = pyqtSignal(str)

    def __init__(self, devices: dict[str, DiscoveredDevice], command: str, session_id: str | None = None,
                 controller: NetworkController = None, timeout: float = 5.0,
                 receiver_host: str | None = None, receiver_port: int | None = None) -> None:
        super().__init__()
        self._devices = dict(devices)
        self._command = command
        self._session_id = session_id
        self._timeout = timeout
        self._controller = controller
        self._receiver_host = receiver_host
        self._receiver_port = receiver_port
        self._attempts = 3
        self._base_delay_ms = 100

    def _recv_message(self, sock: socket.socket, timeout: float) -> dict | None:
        """Receive a single message using v1 framing first, fallback to legacy."""
        deadline = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < deadline:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            res = decode_frames(buf)
            if res.messages:
                return res.messages[0]
            # Legacy fallback if prefix isn't numeric length
            nl = buf.find(b"\n")
            if nl != -1 and not buf[:nl].isdigit():
                line, buf = buf.split(b"\n", 1)
                try:
                    return json.loads(line.decode("utf-8", errors="replace"))
                except Exception:
                    continue
            buf = res.remainder
        return None

    def _time_sync(self, sock: socket.socket, name: str, trials: int = 12, trim_ratio: float = 0.1) -> None:
        """Perform multiple NTP-like exchanges to robustly estimate clock offset.

        Stores median offset via controller._store_offset and detailed stats via
        controller._store_sync_stats when available.
        """
        try:
            offsets: list[int] = []
            delays: list[int] = []
            for i in range(max(1, int(trials))):
                msg_id = int(time.time() * 1000) % 2_000_000_000
                v1 = build_v1_time_sync_req(msg_id=msg_id, seq=i + 1)
                try:
                    sock.sendall(encode_frame(v1))
                except Exception as exc:
                    self.log.emit(f"Time sync send failed to {name}: {exc}")
                    break
                t0 = int(v1.get("t0", time.time_ns()))
                payload = self._recv_message(sock, self._timeout)
                t3 = time.time_ns()
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") == "error":
                    self.log.emit(f"Time sync error from {name}: {payload}")
                    continue
                # Accept both v1 ack and legacy shape
                try:
                    t1 = int(payload.get("t1", 0))
                    t2 = int(payload.get("t2", 0))
                    offset, delay = compute_time_sync(int(t0), t1, t2, t3)
                    offsets.append(int(offset))
                    delays.append(int(delay))
                except Exception:
                    continue
                # tiny pacing to avoid back-to-back bursts
                time.sleep(0.005)
            if offsets and delays:
                median_off, min_delay, std_dev, used = compute_time_sync_stats(offsets, delays, trim_ratio=trim_ratio)
                if self._controller is not None:
                    self._controller._store_offset(name, int(median_off))
                    try:
                        self._controller._store_sync_stats(name, {
                            "offset_ns": int(median_off),
                            "delay_ns": int(min_delay),
                            "std_dev_ns": int(std_dev),
                            "trials": int(used),
                            "timestamp_ns": int(time.time_ns()),
                        })
                    except Exception:
                        pass
                self.log.emit(f"Time sync {name}: median_offset={median_off}ns min_delay={min_delay}ns std_dev={std_dev}ns trials={used}")
            else:
                self.log.emit(f"Time sync {name}: no valid samples collected")
        except Exception as exc:
            self.log.emit(f"Time sync to {name} failed: {exc}")

    def _send_command(self, sock: socket.socket, name: str) -> bool:
        msg_id = int(time.time() * 1000) % 2_000_000_000
        if self._command == "start_recording" and self._session_id:
            v1 = build_v1_start_recording(msg_id, self._session_id)
        elif self._command == "stop_recording":
            v1 = build_v1_cmd("stop_recording", msg_id)
        elif self._command == "flash_sync":
            v1 = build_v1_cmd("flash_sync", msg_id)
        elif self._command == "transfer_files" and self._session_id and self._receiver_host and self._receiver_port:
            v1 = build_v1_cmd("transfer_files", msg_id, host=self._receiver_host, port=int(self._receiver_port), session_id=self._session_id)
        else:
            self.log.emit(f"Unknown or incomplete command {self._command}")
            return False
        try:
            sock.sendall(encode_frame(v1))
        except Exception as exc:
            self.log.emit(f"Send failed to {name}: {exc}")
            return False
        # Try to receive ack/error but don't require it
        payload = None
        try:
            payload = self._recv_message(sock, self._timeout)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            if payload.get("type") == "ack":
                self.log.emit(f"Ack from {name}: {payload}")
                return True
            if payload.get("type") == "error":
                self.log.emit(f"Error from {name}: {payload}")
                return False
        return True  # consider success if send succeeded

    def run(self) -> None:  # noqa: D401
        try:
            for name, dev in self._devices.items():
                schedule = compute_backoff_schedule(self._base_delay_ms, self._attempts)
                success = False
                for attempt_idx, delay_ms in enumerate(schedule, start=1):
                    try:
                        sock = _connect(dev.address, dev.port, self._timeout)
                        try:
                            sock.settimeout(self._timeout)
                            if self._command == "start_recording":
                                self._time_sync(sock, name)
                                ok = self._send_command(sock, name)
                            elif self._command == "time_sync":
                                self._time_sync(sock, name)
                                ok = True  # time_sync-only: no further command to send
                            else:
                                ok = self._send_command(sock, name)
                            if ok:
                                success = True
                                self.log.emit(f"Attempt {attempt_idx}/{len(schedule)} to {name}: success")
                                break
                            else:
                                self.log.emit(f"Attempt {attempt_idx}/{len(schedule)} to {name}: failed")
                        finally:
                            try:
                                sock.close()
                            except Exception:
                                pass
                    except Exception as exc:
                        self.log.emit(f"Attempt {attempt_idx}/{len(schedule)} to {name} failed to connect: {exc}")
                    # backoff before next attempt if not success
                    if not success and attempt_idx < len(schedule):
                        jitter = random.randint(0, max(1, self._base_delay_ms // 2))
                        time.sleep((delay_ms + jitter) / 1000.0)
                if not success:
                    self.log.emit(f"Broadcast to {name} failed after {len(schedule)} attempts")
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"Broadcast worker error: {exc}")


class PreviewStreamWorker(QThread):
    """Maintains a persistent TCP connection to receive asynchronous messages.

    Specifically listens for preview frame events and rejoin_session notifications.
    Supports both v1 length-prefixed framing and legacy newline-delimited JSON.
    """

    log = pyqtSignal(str)
    frame = pyqtSignal(str, bytes, int)  # device name, jpeg bytes, ts
    rejoin = pyqtSignal(str, dict)  # device name, payload

    def __init__(self, device: DiscoveredDevice, timeout: float = 5.0) -> None:
        super().__init__()
        self._device = device
        self._timeout = timeout
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def _handle_message(self, payload: dict) -> None:
        try:
            if not isinstance(payload, dict):
                return
            # v1 rejoin command from Android
            if payload.get("v") == 1 and payload.get("type") == "cmd" and payload.get("command") == "rejoin_session":
                self.rejoin.emit(self._device.name, dict(payload))
                return
            # v1 event
            if payload.get("v") == 1 and payload.get("type") == "event" and payload.get("name") == "preview_frame":
                b64 = str(payload.get("jpeg_base64", ""))
                ts = int(payload.get("ts", 0))
            # legacy event
            elif payload.get("type") == "preview_frame":
                b64 = str(payload.get("jpeg_base64", ""))
                ts = int(payload.get("ts", 0))
            else:
                return
            data = base64.b64decode(b64)
            self.frame.emit(self._device.name, data, ts)
        except Exception:
            pass

    def run(self) -> None:  # noqa: D401
        while not self._stopped:
            try:
                sock = _connect(self._device.address, self._device.port, self._timeout)
                try:
                    sock.settimeout(self._timeout)
                    buf = b""
                    while not self._stopped:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        # First try to decode v1 length-prefixed frames
                        res = decode_frames(buf)
                        if res.messages:
                            for msg in res.messages:
                                self._handle_message(msg)
                            buf = res.remainder
                            continue
                        # Fallback: only treat as legacy when prefix is not numeric length
                        nl = buf.find(b"\n")
                        if nl != -1 and not buf[:nl].isdigit():
                            line, buf = buf.split(b"\n", 1)
                            try:
                                payload = json.loads(line.decode("utf-8", errors="replace"))
                                self._handle_message(payload)
                            except Exception:
                                continue
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
                # Socket closed; retry after short delay
                time.sleep(1.0)
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"Stream error for {self._device.name}: {exc}")
                time.sleep(2.0)


class NetworkController(QObject):
    """Coordinates network discovery and connections for the PC Hub."""

    device_discovered = pyqtSignal(DiscoveredDevice)
    device_removed = pyqtSignal(str)  # name
    log = pyqtSignal(str)
    device_capabilities = pyqtSignal(str, dict)  # name, payload
    preview_frame = pyqtSignal(str, bytes, int)  # device name, jpeg bytes, ts

    def __init__(self) -> None:
        super().__init__()
        self._zeroconf = Zeroconf(ip_version=IPVersion.All)
        self._browser: ServiceBrowser | None = None
        self._listener = _ZeroconfListener(self)
        self._devices: dict[str, DiscoveredDevice] = {}
        self._workers: dict[str, ConnectionWorker] = {}
        self._stream_workers: dict[str, PreviewStreamWorker] = {}
        self._clock_offsets_ns: dict[str, int] = {}
        self._clock_sync_stats: dict[str, dict] = {}
        # Auto re-sync policy (Priority 2 extension)
        try:
            self._resync_delay_threshold_ns: int = int(os.environ.get("PC_RESYNC_DELAY_THRESHOLD_NS", str(25_000_000)))  # 25 ms
        except Exception:
            self._resync_delay_threshold_ns = 25_000_000
        try:
            self._resync_cooldown_s: float = float(os.environ.get("PC_RESYNC_COOLDOWN_S", "120"))
        except Exception:
            self._resync_cooldown_s = 120.0
        self._last_auto_resync_monotonic: float = 0.0
        self._auto_resync_in_flight: bool = False
        # Session/Device state tracking
        self._active_session_id: str | None = None
        self._is_recording: bool = False
        self._device_manager = DeviceManager()
        # Start FileTransferServer (FR10)
        try:
            base_dir = os.path.join(os.getcwd(), "pc_controller_data")
            os.makedirs(base_dir, exist_ok=True)
            self._file_server = FileTransferServer(base_dir)
            port = int(cfg_get("file_transfer_port", 8082))
            self._file_server.start(port)
            self._emit_log(f"FileTransferServer started on port {port}")
        except Exception as exc:
            self._emit_log(f"FileTransferServer start failed: {exc}")

    def get_clock_offsets(self) -> dict[str, int]:
        """Return a copy of the last known per-device clock offsets (ns)."""
        return dict(self._clock_offsets_ns)

    def get_clock_sync_stats(self) -> dict[str, dict]:
        """Return a copy of the last known per-device clock sync stats.

        Shape per device:
        {"offset_ns": int, "delay_ns": int, "std_dev_ns": int, "trials": int, "timestamp_ns": int}
        """
        try:
            return dict(self._clock_sync_stats)  # type: ignore[attr-defined]
        except Exception:
            return {}

    def start(self) -> None:
        if self._browser is None:
            self._browser = ServiceBrowser(self._zeroconf, SERVICE_TYPE, self._listener)
            self._emit_log("Service discovery started.")

    def shutdown(self) -> None:
        # Stop workers
        for _name, worker in list(self._workers.items()):
            if worker.isRunning():
                worker.quit()
                worker.wait(1000)
        self._workers.clear()
        # Stop preview stream workers
        for _name, sw in list(self._stream_workers.items()):
            try:
                sw.stop()
                sw.wait(1000)
            except Exception:
                pass
        self._stream_workers.clear()
        # Stop FileTransferServer if running (FR10)
        try:
            srv = getattr(self, "_file_server", None)
            if srv is not None:
                srv.stop()
        except Exception:
            pass
        # Stop discovery
        if self._browser is not None:
            self._browser.cancel()
            self._browser = None
        # Close zeroconf
        try:
            self._zeroconf.close()
        except Exception:  # pragma: no cover - safety
            pass

    # Internal events from Zeroconf listener
    def _on_service_added(self, device: DiscoveredDevice) -> None:
        self._devices[device.name] = device
        self.device_discovered.emit(device)
        # Start preview stream worker for this device
        try:
            worker = PreviewStreamWorker(device)
            worker.log.connect(self._emit_log)
            worker.frame.connect(self._on_preview_frame)
            try:
                worker.rejoin.connect(self._on_rejoin)
            except Exception:
                pass
            self._stream_workers[device.name] = worker
            worker.start()
            self._emit_log(f"PreviewStreamWorker started for {device.name}")
        except Exception as exc:
            self._emit_log(f"Failed to start PreviewStreamWorker for {device.name}: {exc}")

    def _on_service_removed(self, name: str) -> None:
        clean_name = name.rstrip(".")
        # Stop preview stream worker if running
        try:
            worker = self._stream_workers.pop(clean_name, None)
            if worker is not None:
                worker.stop()
                worker.wait(1000)
                self._emit_log(f"PreviewStreamWorker stopped for {clean_name}")
        except Exception:
            pass
        if clean_name in self._devices:
            del self._devices[clean_name]
        self.device_removed.emit(clean_name)

    def _emit_log(self, message: str) -> None:
        self.log.emit(message)

    def _on_preview_frame(self, name: str, data: bytes, ts: int) -> None:
        # Re-emit for GUI consumers
        try:
            self.preview_frame.emit(name, data, ts)
        except Exception:
            pass

    def _store_offset(self, name: str, offset_ns: int) -> None:
        self._clock_offsets_ns[name] = offset_ns

    def _store_sync_stats(self, name: str, stats: dict) -> None:
        """Internal: store detailed sync stats for a device and trigger auto re-sync when needed."""
        self._clock_sync_stats[name] = dict(stats)
        # Evaluate auto re-sync policy
        try:
            self._maybe_auto_resync(name, stats)
        except Exception:
            pass

    def _maybe_auto_resync(self, name: str, stats: dict) -> None:
        """If measured delay is high, trigger a time_sync broadcast with cooldown.

        Environment variables to tune behavior:
        - PC_RESYNC_DELAY_THRESHOLD_NS (default 25_000_000 = 25 ms)
        - PC_RESYNC_COOLDOWN_S (default 120 s)
        """
        try:
            delay_ns = int(stats.get("delay_ns", 0))
            now = time.monotonic()
            if (
                delay_ns >= int(self._resync_delay_threshold_ns)
                and (now - float(self._last_auto_resync_monotonic)) >= float(self._resync_cooldown_s)
                and not bool(self._auto_resync_in_flight)
            ):
                self._auto_resync_in_flight = True
                self._last_auto_resync_monotonic = now
                try:
                    self._emit_log(
                        f"High sync delay for {name}: {delay_ns/1e6:.2f} ms >= threshold {self._resync_delay_threshold_ns/1e6:.2f} ms — triggering broadcast_time_sync()"
                    )
                except Exception:
                    pass
                # Launch re-sync broadcast (non-blocking)
                try:
                    self.broadcast_time_sync()
                except Exception:
                    pass
                # Clear in-flight flag immediately; cooldown prevents rapid retriggering
                self._auto_resync_in_flight = False
        except Exception:
            pass

    # Public API
    def connect_to_device(self, name: str, address: str, port: int) -> None:
        device = DiscoveredDevice(name=name, address=address, port=port)
        worker = ConnectionWorker(device)
        worker.log.connect(self._emit_log)
        worker.capabilities.connect(self._on_capabilities)
        self._workers[name] = worker
        worker.start()
        self._emit_log(f"ConnectionWorker started for {name}")

    def _on_capabilities(self, name: str, payload: dict) -> None:
        self.device_capabilities.emit(name, payload)
        self._emit_log(f"Capabilities from {name}: {payload}")

    # Phase 4 broadcast API
    def broadcast_start_recording(self, session_id: str) -> None:
        # Track session state for FR8 rejoin logic
        self._active_session_id = session_id
        self._is_recording = True
        worker = _BroadcastWorker(self._devices, "start_recording", session_id, controller=self)
        worker.log.connect(self._emit_log)
        worker.start()
        self._emit_log(f"Broadcast start_recording for session {session_id}")

    def broadcast_stop_recording(self) -> None:
        # Update session flag but keep _active_session_id for potential FR8 rejoin/transfer
        self._is_recording = False
        worker = _BroadcastWorker(self._devices, "stop_recording", controller=self)
        worker.log.connect(self._emit_log)
        worker.start()
        self._emit_log("Broadcast stop_recording")

    # FR8: Handle rejoin_session notifications from PreviewStreamWorker
    def _on_rejoin(self, device_name: str, payload: dict) -> None:
        try:
            sid = str(payload.get("session_id", ""))
        except Exception:
            sid = ""
        # If we're still recording this session, mark device back to Recording
        if self._is_recording and self._active_session_id and sid == self._active_session_id:
            try:
                self._device_manager.set_status(device_name, "Recording")
            except Exception:
                pass
            self._emit_log(f"Device {device_name} rejoined active session {sid}; status set to Recording")
            return
        # Otherwise, request file transfer for the provided or last active session id
        session_id = sid or (self._active_session_id or "")
        if not session_id:
            self._emit_log(f"Device {device_name} rejoined but no known session id; ignoring")
            return
        try:
            host = get_local_ip()
        except Exception:
            host = "127.0.0.1"
        try:
            port = int(cfg_get("file_transfer_port", 8082))
        except Exception:
            port = 8082
        self._send_transfer_files_to(device_name, session_id, host, port)
        self._emit_log(f"Requested file transfer from {device_name} for session {session_id} to {host}:{port}")

    def _send_transfer_files_to(self, device_name: str, session_id: str, host: str, port: int) -> None:
        dev = self._devices.get(device_name)
        if not dev:
            self._emit_log(f"Cannot transfer_files: device {device_name} not known")
            return
        # Reuse broadcast worker with a single-device map
        devices = {device_name: dev}
        worker = _BroadcastWorker(
            devices,
            "transfer_files",
            session_id=session_id,
            controller=self,
            receiver_host=host,
            receiver_port=port,
        )
        try:
            worker.log.connect(self._emit_log)
        except Exception:
            pass
        worker.start()

    def broadcast_transfer_files(self, host: str, port: int, session_id: str) -> None:
        """Broadcast a transfer_files command with receiver host/port and session id."""
        worker = _BroadcastWorker(
            self._devices,
            "transfer_files",
            session_id=session_id,
            controller=self,
            receiver_host=host,
            receiver_port=port,
        )
        worker.log.connect(self._emit_log)
        worker.start()
        self._emit_log(f"Broadcast transfer_files to {host}:{port} for session {session_id}")

    def broadcast_time_sync(self) -> None:
        """Broadcast a time_sync-only operation to refresh offsets/stats."""
        worker = _BroadcastWorker(self._devices, "time_sync", controller=self)
        worker.log.connect(self._emit_log)
        worker.start()
        self._emit_log("Broadcast time_sync")

    def broadcast_flash_sync(self) -> None:
        worker = _BroadcastWorker(self._devices, "flash_sync", controller=self)
        worker.log.connect(self._emit_log)
        worker.start()
        self._emit_log("Broadcast flash_sync")
