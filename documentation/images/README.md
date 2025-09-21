# Visualization Examples

This directory contains sample visualizations demonstrating the enhanced visualization capabilities of the multi-modal physiological sensing platform.

## Generated Visualizations

### 1. Multimodal Alignment Plot (`multimodal_alignment_plot.png`)

**Purpose**: Demonstrates synchronized data collection across multiple sensor modalities with flash sync events for temporal alignment.

**Features**:
- **RGB Camera**: Frame capture timeline (~6.67 Hz)
- **Thermal Camera**: Temperature measurements (10 Hz)
- **GSR Sensor**: Galvanic skin response and physiological responses (128 Hz)
- **PPG Signal**: Photoplethysmography from GSR sensor (downsampled for display)
- **Flash Sync Events**: Orange dashed lines marking synchronization points

**Use Case**: Validation of temporal alignment between sensors and demonstration of physiological responses to stimuli.

### 2. Data Quality Dashboard (`data_quality_dashboard.png`)

**Purpose**: Comprehensive data quality assessment for research session validation.

**Components**:
- **Signal Quality Over Time**: Rolling quality scores for all sensor modalities
- **Data Completeness**: Pie charts showing received vs. expected data samples
- **Timing Jitter Analysis**: Histograms of inter-sample intervals vs. target rates
- **Signal-to-Noise Ratio**: GSR signal quality estimation over time

**Use Case**: Post-session data validation, quality control, and research data integrity assessment.

### 3. Performance Telemetry Chart (`performance_telemetry_chart.png`)

**Purpose**: Real-time system performance monitoring during data collection sessions.

**Metrics**:
- **CPU Utilization**: System load with warning/critical thresholds
- **Memory Usage**: RAM consumption with capacity limits
- **Network Throughput**: Data transmission rates for preview streams
- **Preview Frame Rate**: Real-time preview performance vs. target rate

**Use Case**: System health monitoring, performance optimization, and capacity planning for multi-device deployments.

## Generating Custom Visualizations

To generate updated visualizations with your own data:

```bash
# Install required Python packages
pip install matplotlib pandas numpy seaborn

# Run the visualization generator
python3 scripts/generate_sample_visualizations.py
```

## Integration with Documentation

These visualizations are referenced throughout the enhanced documentation:

- **System_Design.md**: Complete architecture with C4 views and component interactions
- **Flows.md**: Runtime behavior and state management
- **Data.md**: Data schemas and quality validation
- **NonFunctional.md**: Performance budgets and timing analysis
- **TEST_PLAN.md**: Comprehensive testing strategy and data quality procedures

## Research Applications

**Temporal Alignment Analysis**: Use multimodal plots to verify synchronization between physiological measurements and experimental stimuli.

**Data Quality Control**: Implement automated quality dashboards for large-scale research studies to identify sessions requiring review.

**Performance Optimization**: Monitor system telemetry during pilot studies to optimize hardware configurations and deployment parameters.

**Publication Graphics**: Use these visualizations as templates for research papers and presentations, customizing with actual experimental data.

## File Formats

All visualizations are generated as high-resolution PNG files (300 DPI) suitable for:
- Research publications and presentations
- Quality control reports
- System documentation
- Technical specifications

For vector graphics or alternative formats, modify the `plt.savefig()` calls in the generation script to use formats like SVG, PDF, or EPS.
