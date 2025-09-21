# Deployment View Diagram

**Purpose**: Show runtime placement (Android device, PC), ports, discovery, firewalls.

**Placement**: Chapter 3: System Architecture or Deployment section.

**Content**: Android advertises `_gsr-controller._tcp.` on ephemeral port; PC discovers. PC runs UDP time server (port configurable in docs). TCP control channel separate from TCP file upload target.

**Tools**: Mermaid deployment diagram

## Mermaid Diagram

```mermaid
graph TB
  subgraph "Research Environment"
    subgraph "Android Device"
      AndroidApp["Android Sensor Node<br/>com.yourcompany.sensorspoke"]
      AndroidNSD["NSD Service<br/>_gsr-controller._tcp.local.<br/>Port: 8080 (ephemeral)"]
      AndroidStorage["Local Storage<br/>/Android/data/.../files/sessions/"]
      Sensors["Hardware Sensors<br/>• RGB Camera (CameraX)<br/>• Thermal Camera (stub)<br/>• Shimmer GSR (BLE)"]

      AndroidApp --> AndroidNSD
      AndroidApp --> AndroidStorage
      AndroidApp --> Sensors
    end

    subgraph "PC Controller"
      PCApp["Python Controller<br/>PyQt6 GUI"]
      PCNetwork["Network Controller<br/>TCP Client: Auto-discovered port"]
      PCTimeServer["UDP Time Server<br/>Port: 9999 (configurable)"]
      PCFileServer["File Transfer Server<br/>TCP Port: 8090 (configurable)"]
      PCStorage["Session Storage<br/>./sessions/"]
      PCTools["Tools & Scripts<br/>calibration, validation"]

      PCApp --> PCNetwork
      PCApp --> PCTimeServer
      PCApp --> PCFileServer
      PCApp --> PCStorage
      PCApp --> PCTools
    end

    subgraph "Network"
      WiFi["WiFi Network<br/>Local Subnet<br/>192.168.x.x/24"]
      Firewall["Firewall Rules<br/>• Allow mDNS (5353/UDP)<br/>• Allow TCP 8080-8099<br/>• Allow UDP 9999"]
    end
  end

  AndroidNSD -.->|"mDNS Advertise<br/>_gsr-controller._tcp.local."| WiFi
  WiFi -.->|"Zeroconf Discovery"| PCNetwork

  PCNetwork <-->|"TCP Control Channel<br/>JSON Commands/Acks"| AndroidApp
  PCTimeServer <-->|"UDP Time Sync<br/>NTP-like Protocol"| AndroidApp
  AndroidApp -->|"TCP File Transfer<br/>ZIP Stream"| PCFileServer

  AndroidApp -.->|"Preview Events<br/>~6-8 FPS JPEG"| PCNetwork

  WiFi --> Firewall
```

## Network Configuration Details

### Port Assignments
- **Android NSD Service**: Ephemeral port (typically 8080+), advertised via mDNS
- **PC UDP Time Server**: Port 9999 (configurable in `config.py`)
- **PC File Transfer Server**: Port 8090 (configurable in `config.py`)
- **mDNS/Zeroconf**: Port 5353/UDP (standard)

### Firewall Requirements
- **Incoming on PC**:
  - UDP 9999 (time server)
  - TCP 8090 (file transfer server)
  - UDP 5353 (mDNS responses)
- **Outgoing from PC**:
  - TCP 8080+ (dynamic connection to Android devices)
  - UDP 5353 (mDNS queries)
- **Android**: Standard app permissions for network access

### Discovery Flow
1. Android app starts `RecordingService`
2. `NetworkClient.register()` advertises `_gsr-controller._tcp.local.` with current port
3. PC `NetworkController` uses Zeroconf to browse for services
4. PC connects TCP socket to discovered Android device
5. Initial handshake: `query_capabilities` command

### Network Protocols
- **Control Channel**: TCP with length-prefixed JSON framing (v=1) or legacy newline-delimited
- **Time Sync**: UDP echo protocol for offset calculation
- **File Transfer**: TCP connection with JSON header line followed by ZIP stream
- **Preview Stream**: Push events over control channel (base64-encoded JPEG)

## Security Considerations
- Local network operation only (no internet exposure)
- No authentication in current implementation (research environment)
- Future: TLS support available in `tls_enhanced.py` module
- Data integrity via ZIP checksums and session validation
