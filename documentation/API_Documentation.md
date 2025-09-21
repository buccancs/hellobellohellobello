# API Documentation - Multi-Modal Physiological Sensing Platform

This document provides comprehensive API documentation for the multi-modal physiological sensing platform, covering security (TLS), fault tolerance (Heartbeat), and core communication protocols.

## Table of Contents

1. [TLS Security API](#tls-security-api)
2. [Heartbeat Monitoring API](#heartbeat-monitoring-api)
3. [Core Communication Protocol](#core-communication-protocol)
4. [Integration Examples](#integration-examples)
5. [Configuration Reference](#configuration-reference)

---

## TLS Security API

### Overview

The platform implements enterprise-grade TLS security for secure communication between PC Hub and Android Spokes with certificate-based authentication and encryption.

### Architecture

```
┌─────────────────────────────────────────┐
│           Application Layer             │
├─────────────────────────────────────────┤
│        SecureMessageHandler            │
├─────────────────────────────────────────┤
│       SecureConnectionManager          │
├─────────────────────────────────────────┤
│           TLSConfig                     │
├─────────────────────────────────────────┤
│        Python ssl module               │
└─────────────────────────────────────────┘
```

### Core Components

#### TLSConfig Class

**Location**: `pc_controller/src/security/tls_config.py`

Provides centralized configuration management for all TLS operations.

**Methods**:
```python
class TLSConfig:
    def __init__(self, config_file: Optional[str] = None) -> None
    def create_server_context(self) -> ssl.SSLContext
    def create_client_context(self) -> ssl.SSLContext
    def validate_certificate(self, cert_path: str) -> bool
    def get_cipher_suites(self) -> List[str]
```

**Configuration Options**:
```python
{
    "protocol_version": "TLSv1.3",
    "cipher_suites": ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"],
    "certificate_path": "certs/server.crt",
    "private_key_path": "certs/server.key",
    "ca_certificate_path": "certs/ca.crt",
    "verify_mode": "CERT_REQUIRED",
    "check_hostname": true
}
```

#### SecureConnectionManager Class

**Location**: `pc_controller/src/network/secure_connection.py`

Manages secure connections with automatic TLS handling.

**Key Methods**:
```python
class SecureConnectionManager:
    def __init__(self, tls_config: TLSConfig) -> None
    def create_secure_server(self, host: str, port: int) -> SecureServer
    def connect_secure_client(self, host: str, port: int) -> SecureSocket
    def wrap_socket(self, sock: socket.socket, server_side: bool) -> ssl.SSLSocket
```

### Android TLS Implementation

**Location**: `android_sensor_node/app/src/main/java/security/SecureConnectionManager.kt`

**Key Features**:
- X.509 certificate validation
- TLS 1.3 enforcement with secure cipher suites
- Certificate pinning for enhanced security
- Automatic certificate renewal handling

**Usage Example**:
```kotlin
val secureManager = SecureConnectionManager(context)
val secureSocket = secureManager.createSecureSocket(hubAddress, hubPort)
```

### Security Features

**Certificate Management**:
- Automatic certificate validation and renewal
- Certificate pinning for man-in-the-middle protection
- Support for custom CA certificates
- Certificate revocation list (CRL) checking

**Encryption Standards**:
- TLS 1.3 with forward secrecy
- AES-256-GCM and ChaCha20-Poly1305 encryption
- ECDHE key exchange for perfect forward secrecy
- SHA-384 and SHA-256 message authentication

**Access Control**:
- Mutual TLS authentication (mTLS)
- Device certificate-based authorization
- Session token validation
- IP whitelist support

---

## Heartbeat Monitoring API

### Overview

Provides real-time connection monitoring, automatic reconnection, and device health assessment for fault tolerance in distributed sensor networks.

### Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   PC Hub        │         │  Android Spoke  │
│                 │         │                 │
│ HeartbeatManager│◄────────┤ HeartbeatManager│
│  - Monitor      │  JSON   │  - Send         │
│  - Track Status │  over   │  - Device Info  │
│  - Callbacks    │  TCP    │  - Auto Retry   │
│  - Auto Reconnect│        │  - State Persist│
└─────────────────┘         └─────────────────┘
```

### Core Components

#### HeartbeatManager (Python Hub)

**Location**: `pc_controller/src/network/heartbeat_manager.py`

**Key Methods**:
```python
class HeartbeatManager:
    def __init__(self, config: HeartbeatConfig) -> None
    def start_monitoring(self, device_id: str, connection: SecureSocket) -> None
    def stop_monitoring(self, device_id: str) -> None
    def get_device_status(self, device_id: str) -> DeviceStatus
    def register_status_callback(self, callback: Callable[[str, DeviceStatus], None]) -> None
    def force_reconnect(self, device_id: str) -> bool
```

**Configuration**:
```python
class HeartbeatConfig:
    heartbeat_interval: float = 3.0  # seconds
    timeout_threshold: float = 10.0  # seconds
    max_missed_heartbeats: int = 3
    reconnect_delay: float = 1.0  # seconds
    max_reconnect_attempts: int = 5
    backoff_multiplier: float = 2.0
```

#### Device Status Tracking

**Status Enumerations**:
```python
class DeviceStatus(Enum):
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HEALTHY = "healthy"
    WARNING = "warning"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
```

**Status Information**:
```python
class DeviceInfo:
    device_id: str
    status: DeviceStatus
    last_heartbeat: datetime
    battery_level: int
    storage_free_mb: int
    recording_status: str
    network_latency_ms: float
    consecutive_failures: int
```

#### HeartbeatManager (Android)

**Location**: `android_sensor_node/app/src/main/java/network/HeartbeatManager.kt`

**Key Methods**:
```kotlin
class HeartbeatManager(private val networkClient: NetworkClient) {
    fun startHeartbeat()
    fun stopHeartbeat()
    fun sendImmediateHeartbeat(): Boolean
    fun updateDeviceInfo(info: DeviceInfo)
    fun setConnectionStateListener(listener: ConnectionStateListener)
}
```

### Protocol Specification

#### Heartbeat Message Format

**Heartbeat Request**:
```json
{
    "type": "HEARTBEAT",
    "device_id": "android_spoke_001",
    "timestamp": "2023-12-01T14:30:22.123456Z",
    "sequence_number": 12345,
    "device_info": {
        "battery_level": 85,
        "storage_free_mb": 2048,
        "recording_status": "active",
        "sensor_status": {
            "camera": "active",
            "thermal": "active",
            "gsr": "connected"
        },
        "system_info": {
            "cpu_usage": 15.3,
            "memory_usage": 67.8,
            "temperature": 32.1
        }
    }
}
```

**Heartbeat Response**:
```json
{
    "type": "HEARTBEAT_ACK",
    "device_id": "android_spoke_001",
    "server_timestamp": "2023-12-01T14:30:22.125789Z",
    "sequence_number": 12345,
    "status": "acknowledged",
    "server_info": {
        "active_sessions": 1,
        "connected_devices": 3,
        "system_status": "healthy"
    }
}
```

### Integration Patterns

#### Status Change Callbacks

**Python Hub Example**:
```python
def on_device_status_changed(device_id: str, new_status: DeviceStatus):
    if new_status == DeviceStatus.OFFLINE:
        logger.warning(f"Device {device_id} went offline")
        gui_manager.update_device_status(device_id, "offline")
    elif new_status == DeviceStatus.RECONNECTING:
        logger.info(f"Device {device_id} attempting reconnection")
        gui_manager.update_device_status(device_id, "reconnecting")

heartbeat_manager.register_status_callback(on_device_status_changed)
```

#### Automatic Reconnection

**Android Client Example**:
```kotlin
class NetworkClient {
    private fun handleConnectionLost() {
        heartbeatManager.stopHeartbeat()
        scope.launch {
            reconnectWithBackoff()
        }
    }

    private suspend fun reconnectWithBackoff() {
        var attempt = 1
        var delay = 1000L

        while (attempt <= MAX_RECONNECT_ATTEMPTS) {
            try {
                connect()
                heartbeatManager.startHeartbeat()
                return
            } catch (e: Exception) {
                delay(delay)
                delay *= 2  // Exponential backoff
                attempt++
            }
        }
    }
}
```

---

## Core Communication Protocol

### Message Types

#### Session Management
```json
{
    "type": "SESSION_START",
    "session_id": "session_20231201_143022",
    "timestamp": "2023-12-01T14:30:22.123456Z",
    "parameters": {
        "duration_seconds": 300,
        "sensors": ["camera", "thermal", "gsr"],
        "sync_enabled": true
    }
}

{
    "type": "SESSION_STOP",
    "session_id": "session_20231201_143022",
    "timestamp": "2023-12-01T14:35:22.123456Z"
}
```

#### Device Discovery
```json
{
    "type": "DEVICE_ANNOUNCEMENT",
    "device_id": "android_spoke_001",
    "device_name": "Samsung Galaxy S21",
    "capabilities": ["rgb_camera", "thermal_camera", "gsr_sensor"],
    "protocol_version": "2.0",
    "security_features": ["tls_1_3", "certificate_auth"]
}
```

#### Synchronization
```json
{
    "type": "SYNC_FLASH",
    "flash_id": "flash_001",
    "timestamp": "2023-12-01T14:30:25.456789Z"
}

{
    "type": "TIME_SYNC_REQUEST",
    "client_timestamp": "2023-12-01T14:30:22.123456Z"
}

{
    "type": "TIME_SYNC_RESPONSE",
    "server_timestamp": "2023-12-01T14:30:22.124789Z",
    "client_timestamp": "2023-12-01T14:30:22.123456Z",
    "round_trip_time": 1.234
}
```

---

## Integration Examples

### Complete Secure Connection Setup

**Python Hub**:
```python
# Initialize TLS configuration
tls_config = TLSConfig("config/tls.json")

# Create secure connection manager
secure_manager = SecureConnectionManager(tls_config)

# Initialize heartbeat monitoring
heartbeat_config = HeartbeatConfig(
    heartbeat_interval=3.0,
    timeout_threshold=10.0
)
heartbeat_manager = HeartbeatManager(heartbeat_config)

# Create secure server
server = secure_manager.create_secure_server("0.0.0.0", 8443)

# Accept connections with automatic TLS and heartbeat setup
for connection in server.accept_connections():
    device_id = authenticate_device(connection)
    heartbeat_manager.start_monitoring(device_id, connection)
```

**Android Client**:
```kotlin
// Initialize secure connection
val secureManager = SecureConnectionManager(context)
val connection = secureManager.connect("192.168.1.100", 8443)

// Start heartbeat
val heartbeatManager = HeartbeatManager(connection)
heartbeatManager.startHeartbeat()

// Send session commands
val sessionStart = JsonMessage(
    type = "SESSION_START",
    sessionId = generateSessionId(),
    timestamp = getCurrentTimestamp()
)
connection.sendMessage(sessionStart)
```

---

## Configuration Reference

### TLS Configuration (`tls.json`)
```json
{
    "security": {
        "protocol_version": "TLSv1.3",
        "cipher_suites": [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_AES_128_GCM_SHA256"
        ],
        "certificate_path": "certs/server.crt",
        "private_key_path": "certs/server.key",
        "ca_certificate_path": "certs/ca.crt",
        "verify_mode": "CERT_REQUIRED",
        "check_hostname": true,
        "dh_params_path": "certs/dhparams.pem"
    }
}
```

### Heartbeat Configuration (`heartbeat.json`)
```json
{
    "heartbeat": {
        "interval_seconds": 3.0,
        "timeout_seconds": 10.0,
        "max_missed_beats": 3,
        "reconnect_delay_seconds": 1.0,
        "max_reconnect_attempts": 5,
        "backoff_multiplier": 2.0,
        "jitter_factor": 0.1
    }
}
```

### Network Configuration (`network.json`)
```json
{
    "network": {
        "command_port": 8443,
        "file_transfer_port": 8444,
        "time_sync_port": 8445,
        "discovery_port": 5353,
        "buffer_size": 65536,
        "connection_timeout": 30.0,
        "read_timeout": 5.0
    }
}
```

This comprehensive API documentation provides developers with all necessary information to integrate with the platform's security, monitoring, and communication systems, ensuring robust and secure operation in research environments.
