# Project Folder Structure

This document describes the rationalized folder structure for the Multi-Modal Physiological Sensing Platform.

## Root Directory Structure

```
.
├── android_sensor_node/          # Android application (Spoke)
├── pc_controller/                # PC application (Hub)
├── documentation/                # All project documentation
├── demos/                        # Demo scripts and visualizations
├── tools/                        # Development and utility scripts
├── gradle/                       # Gradle wrapper files
├── .github/                      # GitHub workflows and templates
├── README.md                     # Main project overview
├── FOLDER_STRUCTURE.md           # This file
└── *.gradle.kts, *.toml, *.ini   # Build and configuration files
```

## Core Components

### `android_sensor_node/`
Complete Android application implementing the sensor node (Spoke) functionality:
- Multi-modal sensor data capture (GSR, RGB camera, thermal camera)
- Real-time data streaming to PC Hub
- Local data storage and management
- User interface for sensor control

### `pc_controller/`
Complete PC application implementing the hub functionality:
- Device discovery and management
- Real-time data visualization
- Session management and recording
- Data export and analysis tools
- Native C++ backend integration

## Supporting Directories

### `documentation/`
Organized project documentation:
- `phases/` - Development phase documentation
- `architecture/` - Technical architecture and protocols
- `development/` - Development guides and processes
- `latex/` - LaTeX thesis documents
- `markdown/` - Markdown documentation
- `diagrams/` - Technical diagrams and visualizations
- `evidence/` - Testing and validation evidence
- `images/` - All project images and screenshots

### `demos/`
Demonstration scripts and visualizations:
- Feature demonstration scripts
- GUI mockups and visualizations
- User experience demos
- System integration examples

### `tools/`
Development and utility scripts:
- Build and deployment scripts
- Data analysis tools
- Testing utilities
- Quality assurance scripts

## Key Benefits of This Structure

1. **Clear Separation**: Core applications are clearly separated from supporting files
2. **Organized Documentation**: All documentation is centralized with logical sub-organization
3. **Consolidated Demos**: All demonstration materials are in one location
4. **Unified Tools**: Development scripts and utilities are consolidated
5. **Reduced Clutter**: Root directory contains only essential files and main folders
6. **Scalable**: Structure can grow with the project without becoming unwieldy

## Migration Notes

This structure consolidates:
- Multiple scattered documentation files → `documentation/`
- Various demo scripts → `demos/`
- Scripts directory → `tools/` (more descriptive name)
- Duplicate image directories → `documentation/images/`
- Redundant test directories → `pc_controller/tests/`

The new structure maintains all functionality while providing better organization and clarity.