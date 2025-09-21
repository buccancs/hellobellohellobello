```mermaid
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
```