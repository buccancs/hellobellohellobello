#!/usr/bin/env python3
"""
Sample visualization generator for multi-modal physiological sensing platform.

This script generates example plots and dashboards demonstrating the visualization
capabilities described in the enhanced visualization plan.
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_sample_data():
    """Generate sample physiological data for visualization."""

    start_time = datetime.now() - timedelta(minutes=2)
    times_rgb = pd.date_range(start_time, periods=800, freq='150ms')
    times_thermal = pd.date_range(start_time, periods=1200, freq='100ms')
    times_gsr = pd.date_range(start_time, periods=15360, freq='7.8125ms')

    flash_times = [
        start_time + timedelta(seconds=30),
        start_time + timedelta(seconds=60),
        start_time + timedelta(seconds=90),
    ]

    rgb_data = {
        'timestamp': times_rgb,
        'frame_count': range(len(times_rgb)),
        'quality_score': 0.95 + 0.05 * np.random.randn(len(times_rgb)),
    }

    thermal_data = {
        'timestamp': times_thermal,
        'avg_temperature': 36.5
        + 0.5 * np.sin(np.linspace(0, 4 * np.pi, len(times_thermal)))
        + 0.1 * np.random.randn(len(times_thermal)),
        'quality_score': 0.92 + 0.08 * np.random.randn(len(times_thermal)),
    }

    base_gsr = 12.0
    stimulus_response = np.zeros(len(times_gsr))
    for flash_time in flash_times:
        flash_idx = np.abs((times_gsr - flash_time).total_seconds()).argmin()
        response_window = slice(flash_idx, min(flash_idx + 2560, len(times_gsr)))
        t_response = np.linspace(0, 20, len(range(*response_window.indices(len(times_gsr)))))
        stimulus_response[response_window] += (
            3.0 * np.exp(-t_response / 8) * (1 - np.exp(-t_response / 0.5))
        )

    gsr_data = {
        'timestamp': times_gsr,
        'gsr_microsiemens': base_gsr + stimulus_response + 0.2 * np.random.randn(len(times_gsr)),
        'ppg_raw': 2048
        + 200 * np.sin(np.linspace(0, 60 * np.pi, len(times_gsr)))
        + 50 * np.random.randn(len(times_gsr)),
        'quality_score': 0.88 + 0.12 * np.random.randn(len(times_gsr)),
    }

    return {
        'rgb': pd.DataFrame(rgb_data),
        'thermal': pd.DataFrame(thermal_data),
        'gsr': pd.DataFrame(gsr_data),
        'flash_events': flash_times,
    }


def create_multimodal_alignment_plot(data, save_path):
    """Create multimodal timeline alignment plot with flash sync events."""

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].scatter(
        data['rgb']['timestamp'],
        data['rgb']['frame_count'],
        s=20,
        alpha=0.7,
        color='blue',
        label='RGB Frames',
    )
    axes[0].set_ylabel('Frame Count')
    axes[0].set_title('Multi-Modal Data Alignment with Flash Sync Events', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        data['thermal']['timestamp'],
        data['thermal']['avg_temperature'],
        color='red',
        linewidth=1.5,
        label='Average Temperature',
    )
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        data['gsr']['timestamp'],
        data['gsr']['gsr_microsiemens'],
        color='green',
        linewidth=1,
        alpha=0.8,
        label='GSR Conductance',
    )
    axes[2].set_ylabel('GSR (µS)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    ppg_downsampled = data['gsr'][::32]
    axes[3].plot(
        ppg_downsampled['timestamp'],
        ppg_downsampled['ppg_raw'],
        color='purple',
        linewidth=1,
        alpha=0.7,
        label='PPG Signal',
    )
    axes[3].set_ylabel('PPG (ADC)')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    for flash_time in data['flash_events']:
        for ax in axes:
            ax.axvline(
                flash_time,
                color='orange',
                linestyle='--',
                linewidth=2,
                alpha=0.8,
                label='Flash Sync' if ax == axes[0] else '',
            )

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated multimodal alignment plot: {save_path}")


def create_data_quality_dashboard(data, save_path):
    """Create data quality assessment dashboard."""

    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    rgb_quality = data['rgb']['quality_score'].rolling(window=20).mean()
    thermal_quality = data['thermal']['quality_score'].rolling(window=50).mean()
    gsr_quality = data['gsr']['quality_score'].rolling(window=512).mean()

    ax1.plot(data['rgb']['timestamp'], rgb_quality, label='RGB Quality', color='blue', linewidth=2)
    ax1.plot(
        data['thermal']['timestamp'],
        thermal_quality,
        label='Thermal Quality',
        color='red',
        linewidth=2,
    )
    ax1.plot(
        data['gsr']['timestamp'][::128],
        gsr_quality[::128],
        label='GSR Quality',
        color='green',
        linewidth=2,
    )

    ax1.set_ylabel('Quality Score')
    ax1.set_title('Data Quality Assessment Dashboard', fontsize=16, pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 1.0)

    ax2 = fig.add_subplot(gs[1, 0])
    expected_rgb = 800
    actual_rgb = len(data['rgb'])
    rgb_completeness = [actual_rgb, expected_rgb - actual_rgb]
    ax2.pie(
        rgb_completeness,
        labels=['Received', 'Missing'],
        autopct='%1.1f%%',
        colors=['lightblue', 'lightcoral'],
        startangle=90,
    )
    ax2.set_title(f'RGB Completeness\n({actual_rgb}/{expected_rgb} frames)')

    ax3 = fig.add_subplot(gs[1, 1])
    expected_thermal = 1200
    actual_thermal = len(data['thermal'])
    thermal_completeness = [actual_thermal, expected_thermal - actual_thermal]
    ax3.pie(
        thermal_completeness,
        labels=['Received', 'Missing'],
        autopct='%1.1f%%',
        colors=['lightcoral', 'lightgray'],
        startangle=90,
    )
    ax3.set_title(f'Thermal Completeness\n({actual_thermal}/{expected_thermal} samples)')

    ax4 = fig.add_subplot(gs[1, 2])
    expected_gsr = 15360
    actual_gsr = len(data['gsr'])
    gsr_completeness = [actual_gsr, expected_gsr - actual_gsr]
    ax4.pie(
        gsr_completeness,
        labels=['Received', 'Missing'],
        autopct='%1.1f%%',
        colors=['lightgreen', 'lightgray'],
        startangle=90,
    )
    ax4.set_title(f'GSR Completeness\n({actual_gsr}/{expected_gsr} samples)')

    ax5 = fig.add_subplot(gs[2, 0])
    rgb_intervals = data['rgb']['timestamp'].diff().dt.total_seconds() * 1000
    ax5.hist(rgb_intervals.dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax5.axvline(150, color='red', linestyle='--', label='Target (150ms)')
    ax5.set_xlabel('Inter-frame Interval (ms)')
    ax5.set_ylabel('Count')
    ax5.set_title('RGB Timing Jitter')
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 1])
    thermal_intervals = data['thermal']['timestamp'].diff().dt.total_seconds() * 1000
    ax6.hist(thermal_intervals.dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
    ax6.axvline(100, color='blue', linestyle='--', label='Target (100ms)')
    ax6.set_xlabel('Inter-sample Interval (ms)')
    ax6.set_ylabel('Count')
    ax6.set_title('Thermal Timing Jitter')
    ax6.legend()

    ax7 = fig.add_subplot(gs[2, 2])
    gsr_signal = data['gsr']['gsr_microsiemens']
    gsr_smoothed = gsr_signal.rolling(window=128).mean()
    noise_estimate = (gsr_signal - gsr_smoothed).rolling(window=512).std()
    snr_estimate = gsr_smoothed.rolling(window=512).std() / noise_estimate

    ax7.plot(data['gsr']['timestamp'][::128], snr_estimate[::128], color='green', linewidth=2)
    ax7.set_xlabel('Time')
    ax7.set_ylabel('SNR Estimate')
    ax7.set_title('GSR Signal Quality')
    ax7.grid(True, alpha=0.3)

    plt.suptitle('Session Data Quality Report', fontsize=18, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated data quality dashboard: {save_path}")


def create_performance_telemetry_chart(save_path):
    """Create performance monitoring telemetry charts."""

    # Generate sample performance data
    time_points = pd.date_range(datetime.now() - timedelta(minutes=10), periods=600, freq='1s')

    # Simulate performance metrics during a recording session
    cpu_usage = (
        15
        + 20 * np.sin(np.linspace(0, 4 * np.pi, len(time_points)))
        + 5 * np.random.randn(len(time_points))
    )
    cpu_usage = np.clip(cpu_usage, 5, 80)

    memory_usage = 300 + 50 * np.cumsum(np.random.randn(len(time_points)) * 0.01)
    memory_usage = np.clip(memory_usage, 250, 600)

    network_throughput = 8 + 4 * np.random.exponential(1, len(time_points))
    network_throughput = np.clip(network_throughput, 0, 25)

    preview_fps = 6.67 + 0.5 * np.random.randn(len(time_points))
    preview_fps = np.clip(preview_fps, 5, 8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(time_points, cpu_usage, color='red', linewidth=1.5, alpha=0.8)
    axes[0, 0].axhline(60, color='orange', linestyle='--', alpha=0.7, label='Warning (60%)')
    axes[0, 0].axhline(80, color='red', linestyle='--', alpha=0.7, label='Critical (80%)')
    axes[0, 0].set_ylabel('CPU Usage (%)')
    axes[0, 0].set_title('CPU Utilization')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 100)

    axes[0, 1].plot(time_points, memory_usage, color='blue', linewidth=1.5, alpha=0.8)
    axes[0, 1].axhline(500, color='orange', linestyle='--', alpha=0.7, label='Warning (500MB)')
    axes[0, 1].axhline(750, color='red', linestyle='--', alpha=0.7, label='Critical (750MB)')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_title('Memory Utilization')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time_points, network_throughput, color='green', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_ylabel('Throughput (Mbps)')
    axes[1, 0].set_title('Network Throughput')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Time')

    axes[1, 1].plot(time_points, preview_fps, color='purple', linewidth=1.5, alpha=0.8)
    axes[1, 1].axhline(6.67, color='green', linestyle='--', alpha=0.7, label='Target (6.67 FPS)')
    axes[1, 1].set_ylabel('Frame Rate (FPS)')
    axes[1, 1].set_title('Preview Stream Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('Time')

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle('System Performance Telemetry', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated performance telemetry chart: {save_path}")


def main():
    """Generate all sample visualization artifacts."""

    print("Generating sample visualization artifacts for enhanced visualization plan...")

    output_dir = Path(__file__).parent / "../images"
    output_dir.mkdir(exist_ok=True)

    try:
        print("Generating synthetic physiological data...")
        data = generate_sample_data()

        create_multimodal_alignment_plot(data, output_dir / "multimodal_alignment_plot.png")
        create_data_quality_dashboard(data, output_dir / "data_quality_dashboard.png")
        create_performance_telemetry_chart(output_dir / "performance_telemetry_chart.png")

        print("\n🎉 All sample visualizations generated successfully!")
        print(f"📁 Output directory: {output_dir.resolve()}")
        print("\nGenerated files:")
        print(
            "  • multimodal_alignment_plot.png - Demonstrates synchronized data streams with flash events"
        )
        print(
            "  • data_quality_dashboard.png - Shows data completeness, timing, and quality metrics"
        )
        print(
            "  • performance_telemetry_chart.png - System performance monitoring during recording"
        )

    except ImportError as e:
        print(f"❌ Missing required Python package: {e}")
        print("To install required packages:")
        print("  pip install matplotlib pandas numpy seaborn")
        return 1

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
