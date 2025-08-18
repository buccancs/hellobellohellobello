# Enhanced Visualization Features

This document describes the comprehensive visualization enhancements implemented for the PC Controller application.

## Overview

The visualization system has been significantly enhanced to support:
- **Thermal camera visualization** with false-color mapping
- **Multi-channel sensor plotting** (GSR + PPG simultaneous display)
- **Enhanced timeline and annotation system** with visual markers
- **Real-time status dashboard** showing system state
- **Advanced data analysis visualizations** for research insights

## New Device Widget Types

### 1. Thermal Camera Widget (`"thermal"`)

**Features:**
- False-color heatmap display using thermal colormap (blue→cyan→green→yellow→red)
- Real-time temperature overlay showing Min/Max/Mean or ROI values
- ROI (Region of Interest) selection support
- Automatic scaling and normalization

**Usage:**
```python
thermal_widget = DeviceWidget("thermal", "Thermal Camera", parent)
thermal_widget.update_thermal_data(thermal_array, width, height)
thermal_widget.set_thermal_roi(x, y, width, height)  # Optional ROI
```

**Data Format:**
- Input: `numpy.ndarray` of thermal values (raw temperature data)
- Dimensions: typically 256x192 for Topdon TC001 thermal cameras
- Units: Celsius degrees

### 2. Enhanced GSR Widget (Multi-channel)

**Features:**
- Simultaneous display of GSR and PPG channels
- Automatic scaling to show both signals clearly
- Color-coded traces with legend
- Backward compatibility with single-channel data

**Usage:**
```python
gsr_widget = DeviceWidget("gsr", "Shimmer GSR+PPG", parent)
# Multi-channel update
gsr_widget.append_gsr_samples(timestamps, gsr_values, ppg_values)
# Single-channel (legacy)
gsr_widget.append_gsr_samples(timestamps, gsr_values, None)
```

## Status Dashboard Widget

**Features:**
- Recording status indicator (green=recording, red=stopped)
- Device connection count (local + remote devices)
- Network connectivity status
- Live data sample counter with k-formatting

**Display Elements:**
- ● Recording indicator with text
- "Devices: X (Y local, Z remote)"
- "Network: Connected/Discovering..."
- "Data: N samples" or "N.Nk samples"

## Enhanced Timeline & Annotations

### Visual Annotation Markers

**Features:**
- Green dashed vertical lines mark annotation positions
- Text labels showing annotation content
- Highlighted annotations near current playback position
- Persistent storage in `annotations.json`

**Interaction:**
- Click "Add Annotation" to mark current timeline position
- Annotations automatically appear as visual markers on plot
- Timeline cursor (red line) shows current playback position

### Timeline Navigation

**Features:**
- Synchronized video and data plot cursor
- Smooth timeline slider for seeking
- Play/pause controls with 33ms timer precision
- Multiple data stream alignment on common timeline

## Advanced Data Analysis

The `scripts/enhanced_analysis.py` script provides additional visualizations:

### 1. Multi-Modal Correlation Analysis
- Scatter plots of GSR vs Thermal data
- Correlation coefficient calculation
- Time-colored data points to show temporal evolution

### 2. Power Spectral Density Analysis
- Frequency domain analysis of GSR signals
- Welch's method for PSD estimation (with SciPy)
- Low-frequency focus for physiological signals

### 3. Event Detection Visualization
- Gradient-based event detection in GSR signals
- Threshold-based event marking
- Smoothed signal overlay with event highlights

### 4. Statistical Distribution Analysis
- Histograms of signal distributions
- Statistical summaries (mean, std, min, max)
- Multi-modal distribution comparison

## Implementation Details

### Thermal Colormap Algorithm

The thermal visualization uses a custom colormap implementation:

```python
def _apply_thermal_colormap(self, normalized):
    """Blue→Cyan→Green→Yellow→Red mapping"""
    # 0.00-0.25: Blue to Cyan
    # 0.25-0.50: Cyan to Green  
    # 0.50-0.75: Green to Yellow
    # 0.75-1.00: Yellow to Red
```

### Multi-Channel Data Handling

GSR widget handles multiple data formats:
- Legacy: `[(timestamp, gsr_value), ...]`
- Multi-channel: `[(timestamp, gsr_value, ppg_value), ...]`
- Direct arrays: `timestamps, gsr_values, ppg_values`

### Status Updates

Status tracking is integrated throughout the application:
- Recording state changes update the status widget
- Device discovery/removal triggers device count updates
- Data sample counting occurs in real-time during acquisition

## Performance Considerations

### Update Rates
- Video: ~10 FPS render rate (throttled from 30 Hz polling)
- GSR: 20 Hz UI updates (from 128 Hz data acquisition)  
- Thermal: 10 Hz updates (from ~25 Hz thermal camera rate)
- Status: Updates on state changes only

### Memory Management
- Rolling buffers for real-time plots (10 seconds @ 128 Hz)
- Efficient NumPy array operations for thermal processing
- Graceful degradation when PyQtGraph unavailable

## Files Modified/Created

### Core Implementation
- `pc_controller/src/gui/gui_manager.py` - Enhanced DeviceWidget and GUIManager
- `scripts/enhanced_analysis.py` - Advanced analysis visualizations
- `test_visualizations.py` - Unit tests for visualization components

### Demo/Documentation
- `demo_visualizations.py` - Generate demonstration plots
- `enhanced_dashboard_demo.png` - Dashboard layout demonstration
- `enhanced_analysis_demo.png` - Analysis capabilities demonstration

## Usage Examples

### Basic Thermal Display
```python
# In thermal data handler
width, height = 256, 192
thermal_data = get_thermal_frame()  # numpy array of temperatures
thermal_widget.update_thermal_data(thermal_data, width, height)
```

### Multi-Channel GSR Update
```python
# In GSR data handler  
timestamps = np.array([1.0, 1.1, 1.2])
gsr_values = np.array([12.5, 12.7, 12.3])  # microsiemens
ppg_values = np.array([1024, 1030, 1018])  # raw ADC values
gsr_widget.append_gsr_samples(timestamps, gsr_values, ppg_values)
```

### Advanced Analysis
```bash
# Run enhanced analysis on session data
python scripts/enhanced_analysis.py \
  --session /path/to/session_dir \
  --out /path/to/output_plots
```

## Testing

All visualization enhancements include comprehensive testing:
- Unit tests for thermal colormap functions
- Multi-channel data processing validation
- Annotation system persistence testing
- Status tracking logic verification
- Backward compatibility with existing interfaces

Run tests with:
```bash
python test_visualizations.py
```

The enhanced visualization system maintains full backward compatibility while providing significant new capabilities for multi-modal physiological sensing applications.