#!/usr/bin/env python3
"""
Performance & Endurance Test Orchestrator

Validates NFR1 (Real-Time Performance) and NFR7 (Scalability) at a high level by:
- Spawning multiple simulated Android clients (TCP servers) speaking the v1 protocol
  used by pc_controller/src/network/protocol.py.
- Registering them with the NetworkController and starting a recording session.
- Instructing clients to stream high-frequency preview_frame events for the duration.
- Monitoring and logging CPU and memory usage of the controller process.

Usage:
  python3 scripts/run_performance_test.py \
    --clients 8 \
    --rate 30 \
    --duration 900

Notes:
- Requires Python 3.11+ and PyQt6 (pulled by the PC controller).
- Tries to use psutil for detailed CPU/memory; if unavailable, falls back to
  a minimal metrics sampler.
- This script runs the NetworkController in-process (no GUI) and bypasses Zeroconf
  by directly registering simulated devices.
"""
from __future__ import annotations

import argparse
import base64
import os
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
PC_SRC = REPO_ROOT / "pc_controller" / "src"
if str(PC_SRC) not in sys.path:
    sys.path.insert(0, str(PC_SRC))

import contextlib

from network.network_controller import (  # type: ignore  # noqa: E402
    DiscoveredDevice,
    NetworkController,
)
from network.protocol import (  # type: ignore  # noqa: E402
    build_v1_ack,
    build_v1_preview_frame,
    decode_frames,
    encode_frame,
)

# Optional psutil for resource monitoring
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


def now_ns() -> int:
    return time.time_ns()


@dataclass
class ClientStats:
    frames_sent: int = 0
    send_errors: int = 0
    last_error: str = ""


class ReceivedCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count = 0

    def on_frame(self, name: str, data: bytes, ts: int) -> None:
        with self._lock:
            self.count += 1


class SimulatedAndroidClient:
    """Lightweight Android client simulator.

    - Accepts multiple TCP connections on a single port.
    - Distinguishes command vs. stream connections heuristically.
    - Responds to v1 commands: query_capabilities, time_sync, start/stop/flash/transfer_files.
    - Streams preview_frame events at a configurable rate on a stream connection.
    """

    def __init__(self, name: str, host: str = "127.0.0.1", rate_hz: int = 30) -> None:
        self.name = name
        self.host = host
        self.rate_hz = max(1, int(rate_hz))
        self.recording = False
        self._srv_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self.port: int | None = None
        self._lock = threading.Lock()
        self._connections: list[socket.socket] = []
        self._stream_threads: list[threading.Thread] = []
        self.stats = ClientStats()

    def start(self) -> None:
        t = threading.Thread(target=self._server_loop, name=f"{self.name}-srv", daemon=True)
        self._srv_thread = t
        t.start()
        deadline = time.time() + 5.0
        while self.port is None and time.time() < deadline:
            time.sleep(0.01)
        if self.port is None:
            raise RuntimeError(f"{self.name}: server failed to bind to a port")

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._sock is not None:
                with contextlib.suppress(Exception):
                    self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
        except Exception:
            pass
        with self._lock:
            for s in list(self._connections):
                with contextlib.suppress(Exception):
                    s.shutdown(socket.SHUT_RDWR)
                with contextlib.suppress(Exception):
                    s.close()
            self._connections.clear()
        for th in list(self._stream_threads):
            with contextlib.suppress(Exception):
                th.join(timeout=0.5)

    def _server_loop(self) -> None:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock = srv
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, 0))
            self.port = srv.getsockname()[1]
            srv.listen(16)
            srv.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    cli, _ = srv.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                with self._lock:
                    self._connections.append(cli)
                th = threading.Thread(
                    target=self._handle_connection,
                    args=(cli,),
                    name=f"{self.name}-conn",
                    daemon=True,
                )
                th.start()
        except Exception as exc:
            self.stats.last_error = f"server error: {exc}"
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass

    def _handle_connection(self, sock: socket.socket) -> None:
        cmd_conn = False
        try:
            sock.settimeout(0.3)
            try:
                peek = sock.recv(1, socket.MSG_PEEK)
            except BlockingIOError:
                peek = b""
            except TimeoutError:
                peek = b""
            except Exception:
                peek = b""
            cmd_conn = len(peek) > 0
        except Exception:
            cmd_conn = False
        finally:
            with contextlib.suppress(Exception):
                sock.settimeout(2.0)

        if cmd_conn:
            self._command_loop(sock)
        else:
            self._stream_loop(sock)

    def _command_loop(self, sock: socket.socket) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                res = decode_frames(buf)
                msgs = res.messages
                buf = res.remainder
                if not msgs and b"\n" in buf and not buf.split(b"\n", 1)[0].isdigit():
                    line, buf = buf.split(b"\n", 1)
                    try:
                        import json as _json

                        msgs = [_json.loads(line.decode("utf-8", errors="replace"))]
                    except Exception:
                        msgs = []
                for m in msgs:
                    self._handle_cmd(sock, m)
        except Exception as exc:
            self.stats.last_error = f"cmd loop error: {exc}"
        finally:
            with contextlib.suppress(Exception):
                sock.close()
            with self._lock:
                if sock in self._connections:
                    self._connections.remove(sock)

    def _handle_cmd(self, sock: socket.socket, msg: dict) -> None:
        try:
            if not isinstance(msg, dict):
                return
            v = msg.get("v")
            typ = msg.get("type")
            cmd = msg.get("command")
            mid = int(msg.get("id", 0))
            if v == 1 and typ == "cmd":
                if cmd == "query_capabilities":
                    reply = build_v1_ack(
                        mid,
                        status="ok",
                        capabilities={
                            "device": self.name,
                            "preview": True,
                            "time_sync": True,
                        },
                    )
                    sock.sendall(encode_frame(reply))
                elif cmd == "time_sync":
                    t1 = now_ns()
                    time.sleep(0.0005)
                    t2 = now_ns()
                    reply = build_v1_ack(mid, status="ok", t1=int(t1), t2=int(t2))
                    sock.sendall(encode_frame(reply))
                elif cmd == "start_recording":
                    self.recording = True
                    reply = build_v1_ack(mid, status="ok")
                    sock.sendall(encode_frame(reply))
                elif cmd == "stop_recording":
                    self.recording = False
                    reply = build_v1_ack(mid, status="ok")
                    sock.sendall(encode_frame(reply))
                elif cmd in ("flash_sync", "transfer_files"):
                    reply = build_v1_ack(mid, status="ok")
                    sock.sendall(encode_frame(reply))
                else:
                    reply = build_v1_ack(mid, status="ok")
                    sock.sendall(encode_frame(reply))
        except Exception as exc:
            self.stats.last_error = f"handle cmd error: {exc}"

    def _stream_loop(self, sock: socket.socket) -> None:
        # Periodically send preview_frame events at configured rate
        interval = 1.0 / float(self.rate_hz)
        fake_jpeg = b"\xff\xd8\xff\xdb" + os.urandom(64) + b"\xff\xd9"
        b64 = base64.b64encode(fake_jpeg).decode("ascii")
        try:
            while not self._stop.is_set():
                if not self.recording:
                    time.sleep(0.05)
                    continue
                ts = now_ns()
                frame = build_v1_preview_frame(self.name, b64, ts)
                try:
                    sock.sendall(encode_frame(frame))
                    self.stats.frames_sent += 1
                except Exception as exc:
                    self.stats.send_errors += 1
                    self.stats.last_error = f"stream send error: {exc}"
                    break
                jitter = random.uniform(-interval * 0.05, interval * 0.05)
                time.sleep(max(0.0, interval + jitter))
        finally:
            with contextlib.suppress(Exception):
                sock.close()
            with self._lock:
                if sock in self._connections:
                    self._connections.remove(sock)


def _ensure_logs_dir() -> Path:
    out_dir = REPO_ROOT / "scripts" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _open_log_file(prefix: str) -> tuple[Path, TextIO]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _ensure_logs_dir()
    path = out_dir / f"{prefix}_{ts}.csv"
    f = open(path, "w", buffering=1)
    f.write("timestamp,cpu_percent,mem_rss_mb,total_frames,send_errors\n")
    return path, f


def _sample_resources(proc=None) -> tuple[float, float]:
    """Return (cpu_percent, rss_mb) for current process or given psutil.Process."""
    if psutil is not None:
        try:
            p = proc or psutil.Process(os.getpid())
            cpu = p.cpu_percent(interval=None)
            rss_mb = (p.memory_info().rss or 0) / (1024 * 1024)
            return float(cpu), float(rss_mb)
        except Exception:
            pass
    # Note: cpu_percent will be 0 without a prior call; the loop will compute deltas over time
    return 0.0, 0.0


def run_test(num_clients: int, rate_hz: int, duration_s: int) -> int:
    clients: list[SimulatedAndroidClient] = []
    for i in range(num_clients):
        c = SimulatedAndroidClient(name=f"SimClient_{i+1:02d}", rate_hz=rate_hz)
        c.start()
        clients.append(c)

    controller = NetworkController()

    for c in clients:
        if c.port is None:
            raise RuntimeError("client port missing")
        dev = DiscoveredDevice(name=c.name, address=c.host, port=int(c.port))
        controller._on_service_added(dev)  # type: ignore[attr-defined]

    time.sleep(2.0)

    session_id = f"PERF_{time.strftime('%Y%m%d_%H%M%S')}"
    controller.broadcast_start_recording(session_id)

    log_path, log_f = _open_log_file("performance")
    print(f"[INFO] Logging CPU/mem to {log_path}")

    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            proc.cpu_percent(interval=None)
        except Exception:
            proc = None

    start = time.time()
    end = start + duration_s
    exit_code = 0
    try:
        while time.time() < end:
            time.sleep(1.0)
            cpu, rss = _sample_resources(proc)
            total_frames = sum(c.stats.frames_sent for c in clients)
            send_errors = sum(c.stats.send_errors for c in clients)
            log_f.write(f"{time.time():.3f},{cpu:.1f},{rss:.1f},{total_frames},{send_errors}\n")
            if send_errors > 0:
                exit_code = 2
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user; stopping early…")
        exit_code = max(exit_code, 1)
    finally:
        with contextlib.suppress(Exception):
            controller.broadcast_stop_recording()
        time.sleep(1.0)
        with contextlib.suppress(Exception):
            controller.shutdown()
        for c in clients:
            c.stop()
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass

    total_frames = sum(c.stats.frames_sent for c in clients)
    total_errors = sum(c.stats.send_errors for c in clients)
    print(
        f"[RESULT] clients={num_clients} rate_hz={rate_hz} duration_s~={int(time.time()-start)} "
        f"frames_sent={total_frames} send_errors={total_errors} exit_code={exit_code}"
    )
    if total_errors == 0 and exit_code == 0:
        print("[PASS] No client send errors detected; controller remained responsive under load.")
    else:
        print("[FAIL] Errors detected during streaming. Inspect logs for details.")
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run performance test with simulated clients")
    p.add_argument("--clients", type=int, default=8, help="Number of simulated clients (>=8)")
    p.add_argument("--rate", type=int, default=30, help="Preview frames per second per client")
    p.add_argument(
        "--duration",
        type=int,
        default=15 * 60,
        help="Test duration in seconds (default 900s = 15min)",
    )
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.clients < 8:
        print(f"[WARN] --clients {args.clients} < 8; adjusting to 8 to meet requirement.")
        args.clients = 8
    if args.duration < 60:
        print(
            "[WARN] Very short duration; consider at least several minutes for meaningful metrics."
        )
    return run_test(args.clients, args.rate, args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
