```mermaid
stateDiagram-v2
    [*] --> SystemIdle : Application Start
    
    state SystemIdle {
        [*] --> DeviceDiscovery : Scan for devices
        DeviceDiscovery --> DeviceConnected : Device found
        DeviceConnected --> CapabilityQuery : Query sensors
        CapabilityQuery --> DeviceReady : All sensors available
        DeviceReady --> [*] : Ready for session
    }
    
    SystemIdle --> SessionCreation : Create new session
    
    state SessionCreation {
        [*] --> SessionIDGeneration : Generate unique ID
        SessionIDGeneration --> DirectorySetup : Create session directory
        DirectorySetup --> MetadataInit : Initialize metadata
        MetadataInit --> [*] : Session created
    }
    
    SessionCreation --> DevicePreparation : Configure devices
    
    state DevicePreparation {
        [*] --> TimeSync : Synchronize clocks
        TimeSync --> SensorConfig : Configure sensors
        SensorConfig --> HardwareTest : Test connections
        HardwareTest --> [*] : All devices ready
        
        HardwareTest --> SensorConfig : Retry configuration
        TimeSync --> TimeSync : Resync if needed
    }
    
    DevicePreparation --> RecordingActive : Start recording
    
    state RecordingActive {
        [*] --> StreamingStart : Initialize data streams
        
        state StreamingStart {
            [*] --> GSRStart : Start Shimmer GSR+
            [*] --> ThermalStart : Start TC001 thermal
            [*] --> RGBStart : Start CameraX RGB
            
            GSRStart --> GSRStreaming : GSR data flowing
            ThermalStart --> ThermalStreaming : Thermal data flowing  
            RGBStart --> RGBStreaming : RGB data flowing
            
            GSRStreaming --> AllSensorsActive
            ThermalStreaming --> AllSensorsActive
            RGBStreaming --> AllSensorsActive
        }
        
        AllSensorsActive --> DataCollection : Collect sensor data
        
        state DataCollection {
            [*] --> GSRCollection
            [*] --> ThermalCollection
            [*] --> RGBCollection
            [*] --> LiveMonitoring
            
            GSRCollection : Shimmer BLE data
12-bit ADC → μS
            ThermalCollection : TC001 thermal frames
Calibrated temperature
            RGBCollection : CameraX dual pipeline
MP4 + JPEG sequence
            LiveMonitoring : Real-time preview
GUI dashboard updates
            
            GSRCollection --> GSRCollection : Continuous streaming
            ThermalCollection --> ThermalCollection : Frame capture
            RGBCollection --> RGBCollection : Video + photos
            LiveMonitoring --> LiveMonitoring : GUI updates
        }
        
        DataCollection --> StreamingStop : Stop recording command
        
        state StreamingStop {
            AllSensorsActive --> GSRStop : Stop GSR streaming
            AllSensorsActive --> ThermalStop : Stop thermal capture
            AllSensorsActive --> RGBStop : Stop RGB capture
            
            GSRStop --> GSRStopped
            ThermalStop --> ThermalStopped  
            RGBStop --> RGBStopped
            
            GSRStopped --> AllSensorsStopped
            ThermalStopped --> AllSensorsStopped
            RGBStopped --> AllSensorsStopped
        }
        
        StreamingStop --> [*] : Recording stopped
    }
    
    RecordingActive --> DataTransfer : Transfer files
    
    state DataTransfer {
        [*] --> FilePreparation : Prepare files for transfer
        FilePreparation --> Encryption : Encrypt sensitive data
        Encryption --> TransferInit : Initialize file transfer
        TransferInit --> FileTransmission : Send files to PC
        
        state FileTransmission {
            [*] --> GSRTransfer : Transfer GSR CSV
            [*] --> ThermalTransfer : Transfer thermal data
            [*] --> RGBTransfer : Transfer RGB files
            
            GSRTransfer --> GSRComplete : GSR files sent
            ThermalTransfer --> ThermalComplete : Thermal files sent
            RGBTransfer --> RGBComplete : RGB files sent
            
            GSRComplete --> AllFilesTransferred
            ThermalComplete --> AllFilesTransferred
            RGBComplete --> AllFilesTransferred
        }
        
        FileTransmission --> TransferValidation : Verify transfer integrity
        TransferValidation --> [*] : Transfer complete
        
        TransferValidation --> FileTransmission : Retry failed transfers
    }
    
    DataTransfer --> SessionCompletion : Finalize session
    
    state SessionCompletion {
        [*] --> DataAggregation : Aggregate multimodal data
        DataAggregation --> TemporalAlignment : Align timestamps
        TemporalAlignment --> QualityCheck : Validate data quality
        QualityCheck --> MetadataComplete : Complete metadata
        MetadataComplete --> HDF5Export : Export to HDF5
        HDF5Export --> [*] : Session archived
        
        QualityCheck --> DataAggregation : Fix data issues
    }
    
    SessionCompletion --> SystemIdle : Return to idle
    
    %% Error states
    DevicePreparation --> ErrorRecovery : Device failure
    RecordingActive --> ErrorRecovery : Recording error
    DataTransfer --> ErrorRecovery : Transfer failure
    
    state ErrorRecovery {
        [*] --> ErrorDiagnosis : Identify error
        ErrorDiagnosis --> AutoRetry : Automatic retry
        ErrorDiagnosis --> UserIntervention : Manual intervention
        AutoRetry --> DevicePreparation : Retry preparation
        AutoRetry --> RecordingActive : Resume recording
        AutoRetry --> DataTransfer : Retry transfer
        UserIntervention --> SystemIdle : Reset to idle
    }
```