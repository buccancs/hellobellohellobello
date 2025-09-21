#!/usr/bin/env python3
"""
Mermaid Visualization Generator

This script generates Mermaid diagrams for architectural and process visualizations
as requested by the user. It creates .md files that can be rendered as diagrams.
"""

from pathlib import Path


def ensure_directory_exists(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_system_architecture_mermaid():
    """Generate Chapter 3 high-level system architecture in Mermaid format."""

    mermaid_content = """```mermaid
graph TB
    subgraph "Hub (PC Controller)"
        HUB[PC Controller Hub]
        GUI[PyQt6 GUI Interface]
        NET[Network Controller]
        SYNC[Time Sync Service]
        DATA[Data Aggregator]
        EXPORT[HDF5 Exporter]

        HUB --> GUI
        HUB --> NET
        HUB --> SYNC
        HUB --> DATA
        DATA --> EXPORT
    end

    subgraph "Spoke 1 (Android Device)"
        ANDROID1[Android Sensor Node 1]
        SENSOR1A[RGB Camera]
        SENSOR1B[Thermal Camera]
        SENSOR1C[GSR Sensor]
        STORAGE1[Local Storage]

        ANDROID1 --> SENSOR1A
        ANDROID1 --> SENSOR1B
        ANDROID1 --> SENSOR1C
        ANDROID1 --> STORAGE1
    end

    subgraph "Spoke 2 (Android Device)"
        ANDROID2[Android Sensor Node 2]
        SENSOR2A[RGB Camera]
        SENSOR2B[Thermal Camera]
        SENSOR2C[GSR Sensor]
        STORAGE2[Local Storage]

        ANDROID2 --> SENSOR2A
        ANDROID2 --> SENSOR2B
        ANDROID2 --> SENSOR2C
        ANDROID2 --> STORAGE2
    end

    subgraph "Spoke N (Android Device)"
        ANDROIDN[Android Sensor Node N]
        SENSORNA[RGB Camera]
        SENSORNB[Thermal Camera]
        SENSORNC[GSR Sensor]
        STORAGEN[Local Storage]

        ANDROIDN --> SENSORNA
        ANDROIDN --> SENSORNB
        ANDROIDN --> SENSORNC
        ANDROIDN --> STORAGEN
    end

    NET <==> |TLS 1.2+ TCP/IP| ANDROID1
    NET <==> |TLS 1.2+ TCP/IP| ANDROID2
    NET <==> |TLS 1.2+ TCP/IP| ANDROIDN

    SYNC -.-> |NTP-like Protocol| ANDROID1
    SYNC -.-> |NTP-like Protocol| ANDROID2
    SYNC -.-> |NTP-like Protocol| ANDROIDN

    style HUB fill:#e1f5fe
    style ANDROID1 fill:#f3e5f5
    style ANDROID2 fill:#f3e5f5
    style ANDROIDN fill:#f3e5f5
```"""

    return mermaid_content


def generate_android_architecture_mermaid():
    """Generate Chapter 4 Android application architecture in Mermaid format."""

    mermaid_content = """```mermaid
graph TB
    subgraph "UI Layer"
        ACTIVITY[MainActivity]
        FRAGMENT[Recording Fragment]
        VIEWMODEL[Main ViewModel]
    end

    subgraph "Service Layer"
        SERVICE[Recording Service]
        CONTROLLER[Recording Controller]
    end

    subgraph "Sensor Layer"
        INTERFACE[SensorRecorder Interface]
        RGB[RgbCameraRecorder]
        THERMAL[ThermalCameraRecorder]
        GSR[ShimmerRecorder]
    end

    subgraph "Network Layer"
        CLIENT[Network Client]
        TRANSFER[File Transfer Manager]
        DISCOVERY[Service Discovery]
    end

    subgraph "Data Layer"
        MODELS[Data Models]
        STORAGE[Local Storage]
        REPOSITORY[Repository Pattern]
    end

    subgraph "Hardware Interfaces"
        CAMERAX[CameraX API]
        UVC[UVC Camera Library]
        BLE[Nordic BLE Library]
    end

    ACTIVITY --> VIEWMODEL
    FRAGMENT --> VIEWMODEL
    VIEWMODEL --> SERVICE

    SERVICE --> CONTROLLER
    CONTROLLER --> INTERFACE

    INTERFACE <|-- RGB
    INTERFACE <|-- THERMAL
    INTERFACE <|-- GSR

    RGB --> CAMERAX
    THERMAL --> UVC
    GSR --> BLE

    CONTROLLER --> CLIENT
    CLIENT --> TRANSFER
    CLIENT --> DISCOVERY

    SERVICE --> REPOSITORY
    REPOSITORY --> MODELS
    REPOSITORY --> STORAGE

    style INTERFACE fill:#fff3e0
    style SERVICE fill:#e8f5e8
    style CLIENT fill:#f3e5f5
```"""

    return mermaid_content


def generate_protocol_sequence_mermaid():
    """Generate Chapter 4 protocol sequence diagram in Mermaid format."""

    mermaid_content = """```mermaid
sequenceDiagram
    participant PC as PC Controller Hub
    participant A1 as Android Node 1
    participant A2 as Android Node 2

    Note over PC,A2: Device Discovery Phase
    PC->>A1: mDNS/Zeroconf Discovery
    PC->>A2: mDNS/Zeroconf Discovery
    A1->>PC: Service Advertisement
    A2->>PC: Service Advertisement

    Note over PC,A2: Connection Establishment
    PC->>A1: TLS Connection Request
    PC->>A2: TLS Connection Request
    A1->>PC: TLS Handshake Complete
    A2->>PC: TLS Handshake Complete

    Note over PC,A2: Time Synchronization
    PC->>A1: NTP-like Sync Request (T1)
    PC->>A2: NTP-like Sync Request (T1)
    A1->>PC: Sync Response (T2, T3)
    A2->>PC: Sync Response (T2, T3)
    PC->>A1: Clock Offset Calculation (T4)
    PC->>A2: Clock Offset Calculation (T4)

    Note over PC,A2: Session Preparation
    PC->>A1: Session Config (ID, Settings)
    PC->>A2: Session Config (ID, Settings)
    A1->>PC: Config Acknowledged
    A2->>PC: Config Acknowledged

    Note over PC,A2: Recording Phase
    PC->>A1: START_RECORDING Command
    PC->>A2: START_RECORDING Command

    loop Data Collection
        A1->>PC: Sensor Data Stream
        A2->>PC: Sensor Data Stream
        PC->>A1: Heartbeat/Status Check
        PC->>A2: Heartbeat/Status Check
    end

    PC->>A1: STOP_RECORDING Command
    PC->>A2: STOP_RECORDING Command

    Note over PC,A2: Data Transfer Phase
    A1->>PC: File Transfer Request
    A2->>PC: File Transfer Request
    PC->>A1: Transfer Acknowledgment
    PC->>A2: Transfer Acknowledgment

    A1->>PC: Data File Upload
    A2->>PC: Data File Upload
    PC->>A1: Transfer Complete
    PC->>A2: Transfer Complete
```"""

    return mermaid_content


def generate_data_processing_pipeline_mermaid():
    """Generate Chapter 4 data processing pipeline in Mermaid format."""

    mermaid_content = """```mermaid
flowchart TD
    subgraph "Android Sensor Nodes"
        CAM1[RGB Camera<br/>1080p Video + JPEG]
        THERM1[Thermal Camera<br/>CSV with Timestamps]
        GSR1[GSR Sensor<br/>CSV with μS Values]

        CAM2[RGB Camera<br/>1080p Video + JPEG]
        THERM2[Thermal Camera<br/>CSV with Timestamps]
        GSR2[GSR Sensor<br/>CSV with μS Values]
    end

    subgraph "Local Processing"
        TIMESTAMP1[Timestamp<br/>Assignment]
        COMPRESS1[Data<br/>Compression]
        ENCRYPT1[AES256-GCM<br/>Encryption]

        TIMESTAMP2[Timestamp<br/>Assignment]
        COMPRESS2[Data<br/>Compression]
        ENCRYPT2[AES256-GCM<br/>Encryption]
    end

    subgraph "Network Transfer"
        TLS[TLS 1.2+ Secure<br/>Connection]
        BUFFER[Transfer<br/>Buffer]
    end

    subgraph "PC Controller Hub"
        RECEIVE[Data<br/>Reception]
        DECRYPT[Data<br/>Decryption]
        SYNC[Time<br/>Synchronization]
        ALIGN[Temporal<br/>Alignment]
        AGGREGATE[Data<br/>Aggregation]
    end

    subgraph "Export Pipeline"
        VALIDATE[Data<br/>Validation]
        METADATA[Metadata<br/>Enrichment]
        HDF5[HDF5<br/>Export]
        ANON[Anonymization<br/>& Privacy]
    end

    CAM1 --> TIMESTAMP1
    THERM1 --> TIMESTAMP1
    GSR1 --> TIMESTAMP1

    CAM2 --> TIMESTAMP2
    THERM2 --> TIMESTAMP2
    GSR2 --> TIMESTAMP2

    TIMESTAMP1 --> COMPRESS1
    COMPRESS1 --> ENCRYPT1
    ENCRYPT1 --> TLS

    TIMESTAMP2 --> COMPRESS2
    COMPRESS2 --> ENCRYPT2
    ENCRYPT2 --> TLS

    TLS --> BUFFER
    BUFFER --> RECEIVE

    RECEIVE --> DECRYPT
    DECRYPT --> SYNC
    SYNC --> ALIGN
    ALIGN --> AGGREGATE

    AGGREGATE --> VALIDATE
    VALIDATE --> METADATA
    METADATA --> ANON
    ANON --> HDF5

    style CAM1 fill:#e3f2fd
    style CAM2 fill:#e3f2fd
    style THERM1 fill:#fff3e0
    style THERM2 fill:#fff3e0
    style GSR1 fill:#f3e5f5
    style GSR2 fill:#f3e5f5
    style HDF5 fill:#e8f5e8
```"""

    return mermaid_content


def generate_threading_model_mermaid():
    """Generate Chapter 4 PC Controller threading model in Mermaid format."""

    mermaid_content = """```mermaid
graph TB
    subgraph "Main Thread"
        MAIN[Main Application]
        GUI[PyQt6 GUI]
        EVENT[Event Loop]
    end

    subgraph "Network Layer"
        NET_MAIN[Network Controller]
        TCP_SERVER[TCP Server Thread]
        DISCOVERY[mDNS Discovery Thread]
    end

    subgraph "Worker Threads"
        WORKER1[Device 1 Worker]
        WORKER2[Device 2 Worker]
        WORKERN[Device N Worker]
    end

    subgraph "Data Processing"
        AGGREGATOR[Data Aggregator Thread]
        SYNC_SERVICE[Time Sync Thread]
        FILE_HANDLER[File I/O Thread]
    end

    subgraph "Native Backend (C++)"
        CPP_SHIMMER[Native Shimmer Thread]
        CPP_WEBCAM[Native Webcam Thread]
        LOCK_FREE_Q[Lock-Free Queue]
    end

    MAIN --> GUI
    GUI --> EVENT
    MAIN --> NET_MAIN

    NET_MAIN --> TCP_SERVER
    NET_MAIN --> DISCOVERY

    TCP_SERVER --> WORKER1
    TCP_SERVER --> WORKER2
    TCP_SERVER --> WORKERN

    WORKER1 --> AGGREGATOR
    WORKER2 --> AGGREGATOR
    WORKERN --> AGGREGATOR

    AGGREGATOR --> SYNC_SERVICE
    AGGREGATOR --> FILE_HANDLER

    NET_MAIN --> CPP_SHIMMER
    NET_MAIN --> CPP_WEBCAM

    CPP_SHIMMER --> LOCK_FREE_Q
    CPP_WEBCAM --> LOCK_FREE_Q
    LOCK_FREE_Q --> AGGREGATOR

    style MAIN fill:#e1f5fe
    style TCP_SERVER fill:#f3e5f5
    style AGGREGATOR fill:#e8f5e8
    style CPP_SHIMMER fill:#fff3e0
    style CPP_WEBCAM fill:#fff3e0
```"""

    return mermaid_content


def generate_use_case_diagram_mermaid():
    """Generate Chapter 3 UML use case diagram in Mermaid format."""

    mermaid_content = """```mermaid
graph LR
    subgraph "Actors"
        RESEARCHER[Researcher]
        PARTICIPANT[Study Participant]
        SYSTEM[System Administrator]
    end

    subgraph "System Boundary"
        subgraph "Core Use Cases"
            UC1[Configure Recording Session]
            UC2[Start Multi-Device Recording]
            UC3[Monitor Real-time Data]
            UC4[Stop Recording Session]
            UC5[Export Research Data]
        end

        subgraph "Device Management"
            UC6[Discover Android Devices]
            UC7[Connect Sensor Nodes]
            UC8[Synchronize Device Clocks]
            UC9[Validate Data Integrity]
        end

        subgraph "Data Processing"
            UC10[Aggregate Multimodal Data]
            UC11[Apply Temporal Alignment]
            UC12[Anonymize Participant Data]
            UC13[Generate Quality Reports]
        end

        subgraph "System Administration"
            UC14[Configure Network Settings]
            UC15[Manage User Accounts]
            UC16[Monitor System Health]
            UC17[Backup Research Data]
        end
    end

    RESEARCHER --> UC1
    RESEARCHER --> UC2
    RESEARCHER --> UC3
    RESEARCHER --> UC4
    RESEARCHER --> UC5
    RESEARCHER --> UC9
    RESEARCHER --> UC13

    PARTICIPANT --> UC3

    SYSTEM --> UC14
    SYSTEM --> UC15
    SYSTEM --> UC16
    SYSTEM --> UC17

    UC1 --> UC6
    UC1 --> UC7
    UC2 --> UC8
    UC4 --> UC10
    UC5 --> UC11
    UC5 --> UC12

    style RESEARCHER fill:#e3f2fd
    style PARTICIPANT fill:#f3e5f5
    style SYSTEM fill:#e8f5e8
```"""

    return mermaid_content


def generate_conceptual_overview_mermaid():
    """Generate Chapter 1 conceptual overview diagram in Mermaid format."""

    mermaid_content = """```mermaid
graph TB
    subgraph "Research Context"
        PROBLEM[Stress Detection<br/>in Natural Settings]
        CHALLENGE[Multi-Modal Sensing<br/>Synchronization Challenge]
        SOLUTION[Hub-and-Spoke<br/>Architecture]
    end

    subgraph "Technical Implementation"
        subgraph "Hub (PC Controller)"
            HUB[Central Controller]
            GUI_COMP[Real-time Monitoring]
            SYNC_COMP[Time Synchronization]
            EXPORT_COMP[Data Export]
        end

        subgraph "Spokes (Android Nodes)"
            ANDROID[Mobile Sensor Node]
            RGB[RGB Camera]
            THERMAL[Thermal Imaging]
            GSR[Galvanic Skin Response]
        end
    end

    subgraph "Research Outcomes"
        DATA[Synchronized<br/>Multimodal Dataset]
        ACCURACY[< 5ms Temporal<br/>Accuracy]
        SCALABLE[8+ Device<br/>Scalability]
        PIPELINE[Research-Grade<br/>Export Pipeline]
    end

    PROBLEM --> CHALLENGE
    CHALLENGE --> SOLUTION

    SOLUTION --> HUB
    HUB --> GUI_COMP
    HUB --> SYNC_COMP
    HUB --> EXPORT_COMP

    SOLUTION --> ANDROID
    ANDROID --> RGB
    ANDROID --> THERMAL
    ANDROID --> GSR

    HUB <==> |TLS 1.2+<br/>TCP/IP| ANDROID

    SYNC_COMP --> ACCURACY
    EXPORT_COMP --> DATA
    HUB --> SCALABLE
    DATA --> PIPELINE

    style PROBLEM fill:#ffebee
    style SOLUTION fill:#e8f5e8
    style HUB fill:#e1f5fe
    style ANDROID fill:#f3e5f5
    style ACCURACY fill:#e8f5e8
```"""

    return mermaid_content


def save_mermaid_files():
    """Save all Mermaid diagrams to appropriate directories."""

    base_dir = Path(__file__).parent.parent

    diagrams = {
        'chapter1_introduction': {'conceptual_overview.md': generate_conceptual_overview_mermaid()},
        'chapter3_requirements': {
            'system_architecture.md': generate_system_architecture_mermaid(),
            'use_case_diagram.md': generate_use_case_diagram_mermaid(),
        },
        'chapter4_implementation': {
            'android_architecture.md': generate_android_architecture_mermaid(),
            'protocol_sequence.md': generate_protocol_sequence_mermaid(),
            'data_processing_pipeline.md': generate_data_processing_pipeline_mermaid(),
            'threading_model.md': generate_threading_model_mermaid(),
        },
    }

    for chapter, files in diagrams.items():
        chapter_dir = base_dir / "docs" / "diagrams" / "mermaid" / chapter
        ensure_directory_exists(chapter_dir)

        for filename, content in files.items():
            file_path = chapter_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Generated: {file_path}")


def generate_mermaid_index():
    """Generate an index file for all Mermaid diagrams."""

    index_content = """# Mermaid Diagrams Index

This directory contains all Mermaid diagrams for the Multi-Modal Physiological Sensing Platform thesis.

## Chapter 1: Introduction
- `conceptual_overview.md` - High-level system concept and research context

## Chapter 3: Requirements
- `system_architecture.md` - Hub-and-spoke architecture overview
- `use_case_diagram.md` - User interaction and system use cases

## Chapter 4: Implementation
- `android_architecture.md` - MVVM architecture for Android sensor nodes
- `protocol_sequence.md` - Communication protocol sequence diagram
- `data_processing_pipeline.md` - End-to-end data flow visualization
- `threading_model.md` - PC Controller concurrent processing model

## Usage

### Rendering Mermaid Diagrams

1. **Online Mermaid Editor**: Copy diagram content to https://mermaid.live/
2. **VS Code**: Install Mermaid Preview extension
3. **GitHub**: Diagrams render automatically in `.md` files
4. **Command Line**: Use `@mermaid-js/mermaid-cli` package

### Integration in Documentation

```markdown
```mermaid
[paste diagram content here]
```
```

### Export Options
- PNG/SVG for publications
- PDF for thesis integration
- HTML for interactive documentation

## Benefits of Mermaid Format

✅ **Version Control Friendly** - Text-based format tracks changes easily
✅ **Dynamic Rendering** - Updates automatically when code changes
✅ **Cross-Platform** - Renders in GitHub, GitLab, VS Code, etc.
✅ **Maintenance** - Easier to update than image-based diagrams
✅ **Consistency** - Unified styling across all diagrams
"""

    base_dir = Path(__file__).parent.parent
    index_path = base_dir / "docs" / "diagrams" / "mermaid" / "README.md"

    ensure_directory_exists(index_path.parent)
    with open(index_path, 'w') as f:
        f.write(index_content)

    print(f"✅ Generated Mermaid index: {index_path}")


def main():
    """Generate all Mermaid diagrams and documentation."""

    print("🎨 Generating Mermaid Visualizations")
    print("=" * 50)

    print("\n📁 Creating directory structure...")
    save_mermaid_files()

    print("\n📋 Generating index documentation...")
    generate_mermaid_index()

    print("\n✅ Mermaid diagram generation complete!")
    print("   📍 Location: docs/diagrams/mermaid/")
    print("   🔗 View online: https://mermaid.live/")
    print("   📖 Integration guide: docs/diagrams/mermaid/README.md")


if __name__ == "__main__":
    main()
