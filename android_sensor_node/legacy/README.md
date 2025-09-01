# Legacy Sensor Spoke Implementation

This directory contains the original `com.yourcompany.sensorspoke` implementation that was used before migrating to the IRCamera-based architecture.

## Original Architecture

The legacy sensorspoke implementation was a complex multi-modal physiological sensing platform with the following structure:

- **Package**: `com.yourcompany.sensorspoke`
- **Architecture**: MVVM with 4-tab tabbed interface
- **Features**:
  - RGB Camera recording
  - Thermal Camera (TC001) integration
  - GSR sensor support via Shimmer3 BLE
  - Multi-modal sensor coordination
  - Network communication with PC Hub
  - File transfer and session management

## Migration

This implementation was replaced with a streamlined IRCamera-based architecture:
- **New Package**: `com.topdon.tc001`
- **New Architecture**: IRCamera-style with ViewPager2 navigation
- **Focus**: Simplified thermal camera interface with GSR sensor support

## Files Preserved

The complete original sensorspoke package structure has been preserved here for reference and potential future development.

---

**Note**: These files are no longer part of the active build and are kept for historical reference only.