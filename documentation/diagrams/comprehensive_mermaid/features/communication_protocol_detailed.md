```mermaid
sequenceDiagram
    participant PC as PC Controller Hub
    participant ZC as Zeroconf/mDNS
    participant AND as Android Node
    participant GSR as Shimmer GSR+
    participant TC as TC001 Thermal
    participant CAM as RGB Camera

    Note over PC,CAM: Phase 1: Discovery & Connection
    PC->>ZC: Register service "_sensorhub._tcp"
    AND->>ZC: Advertise "_sensorspoke._tcp"
    ZC->>PC: Device discovered: Android Node
    
    PC->>AND: TLS handshake initiation
    AND->>PC: TLS certificate exchange
    PC->>AND: Secure connection established
    
    Note over PC,CAM: Phase 2: Capability Exchange
    PC->>AND: {"command": "query_capabilities"}
    AND->>PC: {"ack_id": 1, "capabilities": {...}}
    
    AND->>GSR: Scan for Shimmer devices
    GSR->>AND: Device found: MAC address
    AND->>TC: Detect TC001 hardware (VID/PID)
    TC->>AND: Device ready: Thermal camera
    AND->>CAM: Initialize CameraX pipeline
    CAM->>AND: Camera ready: RGB capture
    
    PC->>AND: {"command": "device_status_report"}  
    AND->>PC: {"ack_id": 2, "devices": {"gsr": "ready", "thermal": "ready", "rgb": "ready"}}

    Note over PC,CAM: Phase 3: Time Synchronization
    PC->>AND: NTP-like sync request (T1)
    AND->>PC: Sync response (T2, T3) 
    PC->>AND: Clock offset calculation (T4)
    AND->>PC: Time sync acknowledged
    
    Note over PC,CAM: Phase 4: Session Configuration
    PC->>AND: {"command": "configure_session", "session_id": "20241220_143022"}
    AND->>PC: {"ack_id": 3, "status": "session_configured"}
    
    PC->>AND: {"command": "prepare_recording", "settings": {...}}
    AND->>GSR: Connect to Shimmer (BLE)
    AND->>TC: Initialize TC001 SDK
    AND->>CAM: Setup CameraX dual pipeline
    AND->>PC: {"ack_id": 4, "status": "recording_prepared"}

    Note over PC,CAM: Phase 5: Recording Phase
    PC->>AND: {"command": "start_recording"}
    
    par Parallel sensor data streams
        AND->>GSR: Send start command (0x07)
        GSR->>AND: GSR data stream (continuous)
        AND->>AND: Process GSR (12-bit ADC → μS)
        AND->>AND: Write GSR CSV with timestamps
        
    and
        AND->>TC: Start thermal capture
        TC->>AND: Raw thermal frames
        AND->>AND: Apply TC001 calibration (±2°C)
        AND->>AND: Write thermal CSV + PNG frames
        
    and
        AND->>CAM: Start dual capture (MP4 + JPEG)
        CAM->>AND: Video stream + image frames
        AND->>AND: Save MP4 + timestamped JPEGs
        
    and
        loop Live monitoring
            AND->>PC: Preview frames (base64 encoded)
            PC->>PC: Update dashboard GUI
        end
    end
    
    Note over PC,CAM: Phase 6: Recording Stop
    PC->>AND: {"command": "stop_recording"}
    AND->>GSR: Send stop command (0x20)
    AND->>TC: Stop thermal capture
    AND->>CAM: Stop CameraX pipeline
    AND->>PC: {"ack_id": 5, "status": "recording_stopped"}

    Note over PC,CAM: Phase 7: Data Transfer
    AND->>PC: {"command": "file_transfer_request", "files": [...]}
    PC->>AND: {"ack_id": 6, "status": "transfer_approved"}
    
    loop For each data file
        AND->>PC: File transfer (encrypted)
        PC->>PC: Decrypt & validate file
        PC->>AND: Transfer confirmation
    end
    
    PC->>PC: Aggregate multimodal data
    PC->>PC: Apply temporal alignment
    PC->>PC: Export to HDF5 format
    
    AND->>PC: {"command": "session_complete"}
    PC->>AND: {"ack_id": 7, "status": "session_archived"}
```