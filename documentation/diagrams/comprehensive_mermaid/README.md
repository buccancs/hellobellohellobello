# Comprehensive Mermaid Charts - Multi-Modal Physiological Sensing Platform

This directory contains detailed, precise Mermaid diagrams that map directly to the actual repository implementation, modules, and features.

## 🏗️ Architecture Diagrams

### `architecture/pc_controller_detailed.md`
**Detailed PC Controller Module Architecture**
- Maps to actual Python modules in `pc_controller/src/`
- Shows real file names and relationships
- Includes GUI, network, core, data, and validation layers
- **Use for:** Understanding PC application structure, developer onboarding

### `architecture/android_detailed_classes.md`  
**Android Application Class Diagram**
- Complete class diagram with actual Kotlin classes
- Shows sensor implementations: ShimmerRecorder, ThermalCameraRecorder, RgbCameraRecorder
- Includes utility classes: TimeManager, PreviewBus, PermissionManager
- **Use for:** Android development, MVVM architecture understanding

### `architecture/repository_module_dependencies.md`
**Repository-wide Module Dependencies**
- Multi-project Gradle structure visualization
- Python and Android dependencies mapped
- Build system and configuration relationships
- **Use for:** Build system understanding, dependency management

## 🔧 Feature Implementation Diagrams

### `features/sensor_integration_features.md`
**Sensor Integration Details**
- Shimmer GSR+ BLE integration with actual commands (0x07, 0x20)
- Topdon TC001 thermal camera SDK integration
- CameraX RGB dual-pipeline implementation
- **Use for:** Hardware integration, sensor development

### `features/communication_protocol_detailed.md`
**Communication Protocol Sequence**
- Complete protocol flow from discovery to data export
- TLS security, time synchronization, file transfer
- Real command formats and message structures
- **Use for:** Protocol implementation, network debugging

### `features/data_synchronization_architecture.md`
**Data Synchronization & Timing**
- NTP-like clock synchronization algorithm
- Multimodal data alignment process
- HDF5 export pipeline with quality validation
- **Use for:** Timing accuracy, data pipeline optimization

## 🔄 Workflow Diagrams

### `workflows/session_lifecycle_state_machine.md`
**Complete Session Lifecycle**
- State machine from device discovery to session completion
- Error handling and recovery states
- Data collection and transfer workflows
- **Use for:** Session management, error handling

## 📊 Usage Guidelines

### Rendering Diagrams
```bash
# VS Code with Mermaid Preview
code --install-extension bierner.markdown-mermaid

# Online editor
# Copy content to https://mermaid.live/

# Command line rendering
npx @mermaid-js/mermaid-cli -i diagram.md -o diagram.png
```

### Integration in Documentation
- **Thesis/Reports:** Copy Mermaid code blocks directly
- **GitHub README:** Diagrams render automatically
- **Development Docs:** Link to specific chart files
- **API Documentation:** Embed relevant architecture diagrams

### Maintenance
- ✅ **Version Controlled:** Text-based format tracks changes
- ✅ **Auto-updating:** Linked to actual code structure
- ✅ **Consistent Styling:** Unified color scheme and formatting
- ✅ **Modular:** Individual charts for specific purposes

## 🎯 Benefits

### For Developers
- **Onboarding:** Visual guide to codebase structure
- **Architecture:** Clear module relationships and dependencies
- **Debugging:** Protocol flows and state machine visualization

### For Research
- **Documentation:** Publication-ready diagrams
- **Validation:** System architecture verification
- **Replication:** Complete implementation specification

### For Academic Work
- **Thesis Integration:** Direct inclusion in academic writing
- **Peer Review:** Clear system visualization for evaluation
- **Technical Communication:** Professional diagram standards

---

*Generated from actual repository implementation - reflects current codebase state*
