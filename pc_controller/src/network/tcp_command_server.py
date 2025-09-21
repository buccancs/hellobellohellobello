"""TCP Command Server for PC Controller Hub.

Provides a dedicated TCP server that listens for incoming connections from Android
sensor nodes and handles command/response communication. This ensures robust
bidirectional communication for the multi-modal sensing platform.

Features:
- Multi-threaded connection handling
- JSON command protocol
- Session management
- Real-time status monitoring
- Optional TLS 1.2+ security layer
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# TLS support (optional)
try:
    from .tls_utils import create_server_ssl_context
except Exception:  # pragma: no cover - optional import guard
    def create_server_ssl_context():
        return None

logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Information about a connected client."""

    socket: socket.socket
    address: tuple[str, int]
    device_id: str
    connected_at: float
    last_activity: float
    device_name: str = "Unknown Device"
    device_type: str = "unknown"
    capabilities: list = None
    status: str = "connected"
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class TCPCommandServer:
    """TCP command server for handling Android sensor node connections."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.server_socket: socket.socket | None = None
        self.running = False
        self.clients: dict[str, ClientConnection] = {}
        self.command_handlers: dict[str, Callable] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default command handlers."""
        self.command_handlers.update(
            {
                "query_capabilities": self._handle_query_capabilities,
                "register_device": self._handle_register_device,
                "start_recording": self._handle_start_recording,
                "stop_recording": self._handle_stop_recording,
                "flash_sync": self._handle_flash_sync,
                "heartbeat": self._handle_heartbeat,
                "transfer_files": self._handle_transfer_files,
                "time_sync": self._handle_time_sync,
                "live_data_frame": self._handle_live_data_frame,
                "device_status": self._handle_device_status,
            }
        )

    def start(self) -> bool:
        """Start the TCP command server with optional TLS security."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            
            # Apply TLS wrapping if configured
            tls_context = None
            try:
                tls_context = create_server_ssl_context()
            except Exception as e:
                logger.info(f"TLS context creation failed or disabled: {e}")
            
            if tls_context is not None:
                try:
                    self.server_socket = tls_context.wrap_socket(
                        self.server_socket, server_side=True
                    )
                    logger.info("TLS security layer enabled for TCP Command Server")
                except Exception as e:
                    logger.warning(f"TLS wrapping failed, continuing without TLS: {e}")
            else:
                logger.info("TCP Command Server running without TLS (set PC_TLS_ENABLE=1 to enable)")
            
            self.server_socket.listen(10)

            self.running = True

            accept_thread = threading.Thread(
                target=self._accept_connections, daemon=True
            )
            accept_thread.start()

            logger.info(f"TCP Command Server started on {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start TCP command server: {e}")
            return False

    def stop(self):
        """Stop the TCP command server."""
        self.running = False

        for client in list(self.clients.values()):
            try:
                client.socket.close()
            except Exception:
                pass
        self.clients.clear()

        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        logger.info("TCP Command Server stopped")

    def _accept_connections(self):
        """Accept incoming client connections."""
        while self.running and self.server_socket:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address}")

                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True,
                )
                client_thread.start()

            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
                break

    def _handle_client(self, client_socket: socket.socket, address: tuple[str, int]):
        """Handle communication with a connected client."""
        device_id = f"device_{address[0]}_{address[1]}"
        client = ClientConnection(
            socket=client_socket,
            address=address,
            device_id=device_id,
            connected_at=time.time(),
            last_activity=time.time(),
        )

        client_socket.settimeout(30.0)

        self.clients[device_id] = client

        try:
            while self.running:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break

                    client.last_activity = time.time()

                    try:
                        message = json.loads(data.decode("utf-8"))
                        response = self._process_command(client, message)

                        if response:
                            response_json = json.dumps(response)
                            client_socket.send(response_json.encode("utf-8"))

                    except json.JSONDecodeError as e:
                        error_response = {
                            "type": "error",
                            "message": f"Invalid JSON: {e}",
                            "timestamp": time.time(),
                        }
                        client_socket.send(json.dumps(error_response).encode("utf-8"))
                except TimeoutError:
                    logger.warning(f"Client {device_id} timed out due to inactivity.")
                    break
        except Exception as e:
            logger.error(f"Error handling client {device_id}: {e}")

        finally:
            try:
                client_socket.close()
            except Exception:
                pass

            if device_id in self.clients:
                del self.clients[device_id]

            logger.info(f"Client {device_id} disconnected")

    def _process_command(self, client: ClientConnection, message: dict) -> dict | None:
        """Process a command from a client."""
        command = message.get("command")
        if not command:
            return {
                "type": "error",
                "message": "Missing command field",
                "timestamp": time.time(),
            }

        handler = self.command_handlers.get(command)
        if not handler:
            return {
                "type": "error",
                "message": f"Unknown command: {command}",
                "timestamp": time.time(),
            }

        try:
            return handler(client, message)
        except Exception as e:
            logger.error(f"Error processing command {command}: {e}")
            return {
                "type": "error",
                "message": f"Command processing failed: {e}",
                "timestamp": time.time(),
            }


    def _handle_query_capabilities(
        self, client: ClientConnection, message: dict
    ) -> dict:
        """Handle capability query from Android device."""
        return {
            "type": "capabilities",
            "version": "1.0",
            "supported_commands": list(self.command_handlers.keys()),
            "features": [
                "4K60_recording",
                "raw_capture",
                "tcp_communication",
                "shimmer_integration",
                "thermal_camera",
                "time_sync",
                "file_transfer",
            ],
            "timestamp": time.time(),
        }

    def _handle_start_recording(self, client: ClientConnection, message: dict) -> dict:
        """Handle start recording command."""
        session_id = message.get("session_id", f"session_{int(time.time())}")

        logger.info(f"Starting recording session {session_id} for {client.device_id}")

        return {
            "type": "ack",
            "command": "start_recording",
            "session_id": session_id,
            "status": "recording_started",
            "timestamp": time.time(),
        }

    def _handle_stop_recording(self, client: ClientConnection, message: dict) -> dict:
        """Handle stop recording command."""
        logger.info(f"Stopping recording for {client.device_id}")

        return {
            "type": "ack",
            "command": "stop_recording",
            "status": "recording_stopped",
            "timestamp": time.time(),
        }

    def _handle_flash_sync(self, client: ClientConnection, message: dict) -> dict:
        """Handle flash sync command."""
        flash_timestamp = message.get("timestamp", time.time())

        logger.info(f"Flash sync executed for {client.device_id} at {flash_timestamp}")

        return {
            "type": "ack",
            "command": "flash_sync",
            "flash_timestamp": flash_timestamp,
            "status": "flash_executed",
            "timestamp": time.time(),
        }

    def _handle_heartbeat(self, client: ClientConnection, message: dict) -> dict:
        """Handle heartbeat from Android device."""
        return {
            "type": "heartbeat_ack",
            "server_time": time.time(),
            "client_time": message.get("timestamp", 0),
        }

    def _handle_transfer_files(self, client: ClientConnection, message: dict) -> dict:
        """Handle file transfer command."""
        session_id = message.get("session_id")

        return {
            "type": "ack",
            "command": "transfer_files",
            "session_id": session_id,
            "transfer_port": 8082,
            "status": "ready_for_transfer",
            "timestamp": time.time(),
        }

    def _handle_time_sync(self, client: ClientConnection, message: dict) -> dict:
        """Handle time synchronization request."""
        client_timestamp = message.get("timestamp", 0)
        server_timestamp = time.time()

        return {
            "type": "time_sync_response",
            "server_timestamp": server_timestamp,
            "client_timestamp": client_timestamp,
            "round_trip_time": 0,
            "timestamp": server_timestamp,
        }

    def _handle_register_device(self, client: ClientConnection, message: dict) -> dict:
        """Handle device registration with enhanced device info tracking."""
        device_info = message.get("device_info", {})
        device_type = device_info.get("type", "unknown")
        device_name = device_info.get("name", f"Device_{client.address[0]}")
        capabilities = device_info.get("capabilities", [])
        
        client.device_name = device_name
        client.device_type = device_type
        client.capabilities = capabilities
        client.status = "registered"
        
        logger.info(f"Device registered: {device_name} ({device_type}) with capabilities: {capabilities}")
        
        if hasattr(self, 'device_registered_callback') and self.device_registered_callback:
            self.device_registered_callback(client.device_id, device_name, device_type, capabilities)
        
        return {
            "type": "registration_response",
            "status": "registered",
            "assigned_id": client.device_id,
            "server_capabilities": ["session_management", "live_streaming", "file_transfer", "time_sync"],
            "timestamp": time.time(),
        }

    def _handle_live_data_frame(self, client: ClientConnection, message: dict) -> dict:
        """Handle live data frames from Android devices for real-time display."""
        frame_type = message.get("frame_type", "unknown")
        frame_data = message.get("data")
        timestamp = message.get("timestamp", time.time())
        
        if frame_type == "gsr_sample":
            if hasattr(self, 'live_gsr_callback') and self.live_gsr_callback:
                self.live_gsr_callback(client.device_id, frame_data, timestamp)
        elif frame_type == "video_frame":
            if hasattr(self, 'live_video_callback') and self.live_video_callback:
                self.live_video_callback(client.device_id, frame_data, timestamp)
        elif frame_type == "thermal_frame":
            if hasattr(self, 'live_thermal_callback') and self.live_thermal_callback:
                self.live_thermal_callback(client.device_id, frame_data, timestamp)
        
        return {
            "type": "frame_ack",
            "frame_type": frame_type,
            "timestamp": time.time(),
        }

    def _handle_device_status(self, client: ClientConnection, message: dict) -> dict:
        """Handle device status updates for enhanced session management."""
        status = message.get("status", "unknown")
        session_id = message.get("session_id")
        recording_progress = message.get("recording_progress", 0)
        sensor_status = message.get("sensors", {})
        error_message = message.get("error_message")
        
        client.status = status
        client.last_activity = time.time()
        
        logger.info(f"Device {client.device_id} status update: {status}")
        
        if hasattr(self, 'device_status_callback') and self.device_status_callback:
            self.device_status_callback(
                client.device_id, status, session_id, 
                recording_progress, sensor_status, error_message
            )
        
        return {
            "type": "status_ack",
            "timestamp": time.time(),
        }

    def set_device_callbacks(self, 
                           device_registered_callback: Callable | None = None,
                           device_status_callback: Callable | None = None,
                           live_gsr_callback: Callable | None = None,
                           live_video_callback: Callable | None = None,
                           live_thermal_callback: Callable | None = None):
        """Set callback functions for GUI integration."""
        self.device_registered_callback = device_registered_callback
        self.device_status_callback = device_status_callback
        self.live_gsr_callback = live_gsr_callback
        self.live_video_callback = live_video_callback
        self.live_thermal_callback = live_thermal_callback

    def get_connected_clients(self) -> dict[str, ClientConnection]:
        """Get list of currently connected clients."""
        return dict(self.clients)

    def send_broadcast_command(self, command: dict) -> int:
        """Send a command to all connected clients."""
        success_count = 0
        command_json = json.dumps(command)

        for client in list(self.clients.values()):
            try:
                client.socket.send(command_json.encode("utf-8"))
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to send broadcast to {client.device_id}: {e}")

        return success_count
