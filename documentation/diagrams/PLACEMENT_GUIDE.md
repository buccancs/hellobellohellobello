# Placement Guide for Report/Thesis Integration

**Purpose**: Guide for organizing diagrams and artifacts within academic report chapters and sections.

## Chapter 3: System Architecture

### 3.1 High-Level Architecture (First Figure)
- **Diagram**: `system_architecture/high_level_architecture.md`
- **Purpose**: End-to-end system overview showing Android + PC components
- **Caption**: "Figure 3.1: Multi-Modal Physiological Sensing Platform Architecture showing Android Sensor Node components (MainActivity, RecordingService, SensorRecorders) communicating with PC Controller via NSD discovery and TCP control protocols."

### 3.2 Deployment View
- **Diagram**: `system_architecture/deployment_view.md`
- **Purpose**: Runtime environment, network topology, port assignments
- **Caption**: "Figure 3.2: Deployment diagram showing Android device advertising _gsr-controller._tcp service, PC discovery via Zeroconf, and bidirectional communication over WiFi network."

### 3.3 PC Controller Components
- **Diagram**: `system_architecture/pc_controller_modules.md`
- **Purpose**: Python application module breakdown
- **Caption**: "Figure 3.3: PC Controller module architecture with PyQt6 GUI, network layer, core services, data processing, and optional C++ native backend."

## Chapter 4: Design and Implementation

### 4.1 Android Application Design

#### 4.1.1 Class Diagram
- **Diagram**: `system_architecture/android_class_diagram.md`
- **Purpose**: Component relationships and dependencies
- **Caption**: "Figure 4.1: Android Sensor Node class diagram showing RecordingController orchestration of SensorRecorder implementations with NetworkClient and utility services."

#### 4.1.2 State Machine
- **Diagram**: `state_machines/recording_controller.md`
- **Purpose**: RecordingController lifecycle and transitions
- **Caption**: "Figure 4.2: RecordingController state machine with IDLE→PREPARING→RECORDING→STOPPING transitions and exception handling paths."

#### 4.1.3 Data Flow
- **Diagram**: `sequences/recording_workflows.md` (RgbCameraRecorder section)
- **Purpose**: Dual pipeline MP4 + JPEG capture with preview
- **Caption**: "Figure 4.3: RGB camera data flow showing parallel VideoCapture (MP4) and ImageCapture (JPEG) with PreviewBus integration for live streaming."

### 4.2 Communication Protocols

#### 4.2.1 Sequence Diagrams
- **Diagram**: `sequences/recording_workflows.md` (Start/Stop section)
- **Purpose**: PC-Android command exchange
- **Caption**: "Figure 4.4: Start/stop recording sequence showing PC command initiation, Android internal coordination, and sensor orchestration."

#### 4.2.2 Protocol Tables
- **Tables**: `protocols/tcp_control_protocol.md`
- **Purpose**: Complete message specification
- **Placement**: Table 4.1: "TCP Control Protocol Commands"
- **Placement**: Table 4.2: "Event Messages and Error Codes"

#### 4.2.3 File Transfer Flow
- **Diagram**: `sequences/recording_workflows.md` (File Transfer section)
- **Purpose**: ZIP streaming from Android to PC
- **Caption**: "Figure 4.5: File transfer sequence with TCP connection, JSON header, and streamed ZIP content from Android FileTransferManager to PC FileTransferServer."

### 4.3 Data Management

#### 4.3.1 Session Directory Structure
- **Diagram**: `data_formats/session_structure.md` (Directory Tree)
- **Purpose**: File organization and naming conventions
- **Caption**: "Figure 4.6: Session directory structure with timestamp-based naming and sensor-specific subdirectories for RGB, thermal, and GSR data."

#### 4.3.2 Data Schemas
- **Tables**: `data_formats/session_structure.md` (CSV sections)
- **Placement**: Table 4.3: "RGB CSV Schema (rgb.csv)"
- **Placement**: Table 4.4: "GSR CSV Schema (gsr.csv)"
- **Placement**: Table 4.5: "Thermal CSV Schema (thermal.csv)"

## Chapter 5: Evaluation and Testing

### 5.1 Performance Analysis

#### 5.1.1 Timing and Synchronization
- **Diagram**: `performance/timing_synchronization.md` (UDP Timeline)
- **Purpose**: Time sync protocol accuracy
- **Caption**: "Figure 5.1: UDP time synchronization protocol showing offset calculation for cross-device clock alignment with <5ms accuracy."

#### 5.1.2 Preview Performance
- **Diagram**: `performance/timing_synchronization.md` (Throttling section)
- **Purpose**: PreviewBus throttling effectiveness
- **Caption**: "Figure 5.2: Preview frame throttling timeline demonstrating 6-8 FPS target rate with 150ms minimum interval to prevent network congestion."

#### 5.1.3 Performance Charts
- **Charts**: `performance/performance_testing.md`
- **Purpose**: CPU usage, transfer speeds, memory consumption
- **Caption**: "Figure 5.3: System performance metrics showing preview FPS vs CPU usage, file transfer throughput, and memory stability."

### 5.2 System Validation

#### 5.2.1 Error Handling Matrix
- **Table**: `performance/performance_testing.md` (Error Handling Matrix)
- **Purpose**: Failure modes and recovery mechanisms
- **Placement**: Table 5.1: "Error Types, Frequency, and Recovery Strategies"

#### 5.2.2 Test Coverage
- **Table**: `performance/performance_testing.md` (Test Coverage Map)
- **Purpose**: Testing strategy completeness
- **Placement**: Table 5.2: "Component Test Coverage Matrix"

#### 5.2.3 Quality Metrics
- **Chart**: `performance/performance_testing.md` (Quality Metrics section)
- **Purpose**: Testing progress and coverage trends
- **Caption**: "Figure 5.4: Test coverage progression over development sprints with quality gates for alpha, beta, and production releases."

## Appendices

### Appendix A: Protocol Reference
- **Content**: Complete `protocols/tcp_control_protocol.md`
- **Purpose**: Comprehensive command reference for implementers
- **Organization**: Commands, events, error codes, examples

### Appendix B: Data Format Specifications
- **Content**: Complete `data_formats/session_structure.md`
- **Purpose**: File format documentation for data analysis
- **Organization**: Directory structure, CSV schemas, file size estimates

### Appendix C: Quick Reference
- **Content**: `examples/quick_reference.md`
- **Purpose**: Copy-paste Mermaid/PlantUML snippets
- **Organization**: System diagrams, protocol examples, styling options

## Figure and Table Numbering

### Recommended Numbering Scheme
- **Chapter 3**: Figures 3.1-3.5, Tables 3.1-3.2
- **Chapter 4**: Figures 4.1-4.8, Tables 4.1-4.6
- **Chapter 5**: Figures 5.1-5.6, Tables 5.1-5.3
- **Appendix**: Figures A.1-A.3, Tables A.1-A.5

### Caption Format Template
```
Figure X.Y: [Diagram Title] showing [key components/relationships].
[Brief description of purpose and what the reader should understand.]
```

### Table Format Template
```
Table X.Y: [Table Title]
[Description of data presented and interpretation guidance.]
```

## Integration Checklist

### System Architecture Chapter
- [ ] High-level system diagram (Android + PC components)
- [ ] Deployment view (network topology, ports)
- [ ] PC Controller module map (Python application structure)
- [ ] Protocol overview (NSD, TCP, UDP protocols)

### Design and Implementation Chapter
- [ ] Android class diagram (component relationships)
- [ ] RecordingController state machine (lifecycle management)
- [ ] Communication sequences (start/stop recording, file transfer)
- [ ] Data format specifications (CSV schemas, directory structure)
- [ ] Protocol tables (command reference, error codes)

### Evaluation and Testing Chapter
- [ ] Performance charts (CPU, memory, throughput)
- [ ] Timing diagrams (synchronization accuracy)
- [ ] Error handling matrix (failure modes, recovery)
- [ ] Test coverage map (component testing strategy)
- [ ] Quality metrics (coverage trends, optimization results)

### Appendices
- [ ] Complete protocol reference (implementer documentation)
- [ ] Data format specifications (analysis tool integration)
- [ ] Quick reference examples (Mermaid/PlantUML snippets)

## Rendering and Export

### For LaTeX Documents
1. Render Mermaid/PlantUML to SVG or PNG
2. Use `\includegraphics{}` with appropriate sizing
3. Apply consistent styling and fonts
4. Reference figures with `\ref{fig:label}`

### For Word Documents
1. Export diagrams as high-resolution PNG (300 DPI)
2. Use Word's native table formatting for protocol tables
3. Apply consistent figure and table styles
4. Use cross-reference features for numbering

### For Markdown/HTML
1. Include Mermaid code blocks directly (GitHub renders natively)
2. Use PlantUML online rendering for complex diagrams
3. Apply CSS styling for consistent appearance
4. Link between sections and diagrams

### Quality Guidelines
- **Resolution**: Minimum 300 DPI for print, vector formats preferred
- **Fonts**: Consistent with document body text (12pt minimum)
- **Colors**: High contrast, colorblind-friendly palette
- **Layout**: Clear spacing, logical flow, readable labels
- **Size**: Fit page width, scale appropriately for readability
