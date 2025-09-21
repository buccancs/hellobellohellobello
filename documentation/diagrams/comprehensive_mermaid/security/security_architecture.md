```mermaid
sequenceDiagram
    participant User as Researcher
    participant PC as PC Controller
    participant AND as Android Node
    participant NET as Network Layer
    participant STORE as Data Storage

    Note over User,STORE: Security Architecture Flow

    %% Authentication Phase
    User->>PC: Launch application
    PC->>PC: Load TLS certificates
    PC->>NET: Initialize secure server
    
    AND->>AND: Initialize Android Keystore
    AND->>NET: Advertise secure service
    
    PC->>AND: Discovery request
    AND->>PC: Service advertisement with capabilities
    
    %% TLS Handshake
    PC->>AND: TLS Client Hello
    AND->>PC: TLS Server Hello + Certificate
    PC->>AND: Certificate verification
    AND->>PC: TLS handshake complete
    
    Note over PC,AND: Secure channel established

    %% Authentication
    PC->>AND: Authentication challenge
    AND->>AND: Generate auth token (Android Keystore)
    AND->>PC: Encrypted auth response
    PC->>PC: Validate authentication
    PC->>AND: Authentication successful
    
    %% Session Security
    PC->>AND: Session start (encrypted command)
    AND->>AND: Generate session keys (AES256-GCM)
    
    %% Data Protection
    loop Sensor Data Collection
        AND->>AND: Collect sensor data
        AND->>AND: Timestamp with monotonic clock
        AND->>AND: Encrypt data (AES256-GCM)
        AND->>STORE: Store encrypted locally
        AND->>PC: Send encrypted preview data
        PC->>PC: Decrypt & display preview
    end
    
    %% Secure File Transfer
    PC->>AND: Request data transfer
    AND->>AND: Prepare encrypted data files
    AND->>PC: Transfer encrypted files (TLS)
    PC->>PC: Decrypt & validate data
    PC->>PC: Verify data integrity (checksums)
    
    %% Data Anonymization
    PC->>PC: Apply data anonymization
    PC->>PC: Remove participant identifiers
    PC->>PC: Blur faces in video streams
    PC->>STORE: Store anonymized data
    
    %% Session Cleanup
    PC->>AND: Session end command
    AND->>AND: Clear session keys
    AND->>AND: Cleanup temporary data
    PC->>PC: Archive session securely
    
    %% Audit Trail
    PC->>STORE: Log security events
    AND->>STORE: Log access attempts
    
    Note over User,STORE: All data protected with encryption at rest and in transit

    %% Security layers annotations
    rect rgb(255, 240, 240)
    Note over PC,AND: TLS 1.2+ Transport Security
    end
    
    rect rgb(240, 255, 240) 
    Note over AND,AND: Android Keystore Hardware Security
    end
    
    rect rgb(240, 240, 255)
    Note over PC,STORE: AES256-GCM Data Encryption
    end
    
    rect rgb(255, 255, 240)
    Note over User,STORE: Privacy & Anonymization Layer
    end
```