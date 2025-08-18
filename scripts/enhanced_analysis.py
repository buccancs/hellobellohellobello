#!/usr/bin/env python3
"""
Enhanced Data Analysis and Visualization Script

This script extends the existing analyze_pilot_data.py with additional visualizations:
- Multi-modal correlation plots
- Power spectral density analysis
- Event detection visualization
- Statistical distribution plots

Usage:
  python3 scripts/enhanced_analysis.py --session /path/to/session_dir --out /path/to/output_dir [options]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Insert repo pc_controller/src into path
REPO_ROOT = Path(__file__).resolve().parents[1]
PC_SRC = REPO_ROOT / "pc_controller" / "src"
if str(PC_SRC) not in sys.path:
    sys.path.insert(0, str(PC_SRC))

from data.data_loader import DataLoader  # type: ignore  # noqa: E402

try:
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib  # type: ignore
    HAS_MATPLOTLIB = True
except Exception:
    np = None  # type: ignore
    plt = None  # type: ignore  
    matplotlib = None  # type: ignore
    HAS_MATPLOTLIB = False

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    session_dir: str
    gsr_stats: Dict[str, float]
    thermal_stats: Dict[str, float]
    correlation: float
    plots_created: List[str]

def load_session_data(session_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load GSR and thermal data from session directory."""
    if not np or not HAS_MATPLOTLIB:
        return None, None, None, None
    
    try:
        loader = DataLoader(session_dir)
        session = loader.index_files()
        
        gsr_times, gsr_values = None, None
        thermal_times, thermal_values = None, None
        
        # Load GSR data
        for csv_name in session.csv_files:
            if 'gsr' in csv_name.lower():
                df = loader.load_csv(csv_name)
                if not df.empty and 'gsr_microsiemens' in df.columns:
                    timestamps = df.index.astype('int64') / 1e9  # Convert to seconds
                    gsr_times = timestamps.values - timestamps.values[0]  # Relative time
                    gsr_values = df['gsr_microsiemens'].values
                    break
        
        # Load thermal data
        for csv_name in session.csv_files:
            if 'thermal' in csv_name.lower():
                df = loader.load_csv(csv_name)
                if not df.empty:
                    timestamps = df.index.astype('int64') / 1e9  # Convert to seconds  
                    thermal_times = timestamps.values - timestamps.values[0]  # Relative time
                    
                    # Calculate mean of thermal values (assuming v0, v1, v2, ... columns)
                    value_cols = [col for col in df.columns if col.startswith('v') and col[1:].isdigit()]
                    if value_cols:
                        thermal_values = df[value_cols].mean(axis=1).values
                    break
        
        return gsr_times, gsr_values, thermal_times, thermal_values
        
    except Exception as e:
        print(f"Error loading session data: {e}")
        return None, None, None, None

def create_correlation_plot(gsr_times: np.ndarray, gsr_values: np.ndarray, 
                          thermal_times: np.ndarray, thermal_values: np.ndarray,
                          output_path: str) -> str:
    """Create multi-modal correlation visualization."""
    if not HAS_MATPLOTLIB:
        return ""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Multi-Modal Data Correlation Analysis', fontsize=14)
    
    # Time series plot
    ax1.plot(gsr_times, gsr_values, 'b-', alpha=0.7, label='GSR')
    ax1.set_ylabel('GSR (μS)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(thermal_times, thermal_values, 'r-', alpha=0.7, label='Thermal')
    ax1_twin.set_ylabel('Thermal (°C)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Time Series Overlay')
    
    # Scatter plot for direct correlation
    if len(gsr_values) > 0 and len(thermal_values) > 0:
        # Interpolate to common time base for correlation
        common_times = np.linspace(0, min(gsr_times[-1], thermal_times[-1]), 100)
        gsr_interp = np.interp(common_times, gsr_times, gsr_values)
        thermal_interp = np.interp(common_times, thermal_times, thermal_values)
        
        ax2.scatter(gsr_interp, thermal_interp, alpha=0.6, s=10)
        ax2.set_xlabel('GSR (μS)')
        ax2.set_ylabel('Thermal (°C)')
        ax2.set_title('GSR vs Thermal Correlation')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(gsr_interp, thermal_interp)[0, 1]
        ax2.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Histograms
    if len(gsr_values) > 0:
        ax3.hist(gsr_values, bins=30, alpha=0.7, color='blue', density=True)
        ax3.set_xlabel('GSR (μS)')
        ax3.set_ylabel('Density')
        ax3.set_title('GSR Distribution')
    
    if len(thermal_values) > 0:
        ax4.hist(thermal_values, bins=30, alpha=0.7, color='red', density=True)
        ax4.set_xlabel('Thermal (°C)')
        ax4.set_ylabel('Density') 
        ax4.set_title('Thermal Distribution')
    
    plt.tight_layout()
    
    correlation_path = os.path.join(output_path, 'correlation_analysis.png')
    fig.savefig(correlation_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return correlation_path

def create_spectral_analysis(gsr_times: np.ndarray, gsr_values: np.ndarray, output_path: str) -> str:
    """Create power spectral density analysis."""
    if not HAS_MATPLOTLIB or len(gsr_values) < 10:
        return ""
    
    try:
        # Estimate sampling rate
        dt = np.median(np.diff(gsr_times))
        fs = 1.0 / dt if dt > 0 else 1.0
        
        # Compute power spectral density using Welch's method
        from scipy import signal  # type: ignore
        frequencies, psd = signal.welch(gsr_values, fs, nperseg=min(len(gsr_values)//4, 256))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('GSR Spectral Analysis', fontsize=14)
        
        # Time domain
        ax1.plot(gsr_times, gsr_values, 'b-', linewidth=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('GSR (μS)')
        ax1.set_title('Time Domain Signal')
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain
        ax2.semilogy(frequencies, psd, 'r-', linewidth=1)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.set_title('Frequency Domain Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(2.0, frequencies[-1]))  # Focus on low frequencies
        
        plt.tight_layout()
        
        spectral_path = os.path.join(output_path, 'spectral_analysis.png')
        fig.savefig(spectral_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return spectral_path
        
    except ImportError:
        # Fallback without scipy
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(gsr_times, gsr_values, 'b-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GSR (μS)')
        ax.set_title('GSR Time Series (Spectral analysis requires SciPy)')
        ax.grid(True, alpha=0.3)
        
        spectral_path = os.path.join(output_path, 'time_series_fallback.png')
        fig.savefig(spectral_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return spectral_path

def create_event_detection_plot(gsr_times: np.ndarray, gsr_values: np.ndarray, output_path: str) -> str:
    """Create event detection visualization."""
    if not HAS_MATPLOTLIB or len(gsr_values) < 10:
        return ""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('GSR Event Detection Analysis', fontsize=14)
    
    # Simple event detection based on derivative thresholding
    smooth_window = max(1, len(gsr_values) // 100)  # 1% of data length
    if smooth_window > 1:
        # Simple moving average smoothing
        kernel = np.ones(smooth_window) / smooth_window
        gsr_smooth = np.convolve(gsr_values, kernel, mode='same')
    else:
        gsr_smooth = gsr_values
    
    # Calculate derivative
    gsr_diff = np.gradient(gsr_smooth, gsr_times)
    
    # Detect events (simple threshold-based)
    threshold = np.std(gsr_diff) * 2.0
    events = np.where(np.abs(gsr_diff) > threshold)[0]
    
    # Original signal with smoothed overlay
    ax1.plot(gsr_times, gsr_values, 'lightblue', alpha=0.7, label='Raw GSR')
    ax1.plot(gsr_times, gsr_smooth, 'b-', linewidth=2, label='Smoothed GSR')
    
    # Mark detected events
    if len(events) > 0:
        ax1.scatter(gsr_times[events], gsr_smooth[events], color='red', s=50, 
                   zorder=5, label=f'Events ({len(events)})')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('GSR (μS)')
    ax1.set_title('GSR Signal with Event Detection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Derivative plot
    ax2.plot(gsr_times, gsr_diff, 'g-', linewidth=1, label='GSR Derivative')
    ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold (±{threshold:.3f})')
    ax2.axhline(y=-threshold, color='r', linestyle='--', alpha=0.7)
    
    if len(events) > 0:
        ax2.scatter(gsr_times[events], gsr_diff[events], color='red', s=30, zorder=5)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('dGSR/dt (μS/s)')
    ax2.set_title('GSR Derivative for Event Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    events_path = os.path.join(output_path, 'event_detection.png')
    fig.savefig(events_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return events_path

def run_enhanced_analysis(session_dir: str, output_dir: str) -> AnalysisResult:
    """Run complete enhanced analysis suite."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, cannot create plots")
        return AnalysisResult(session_dir, {}, {}, 0.0, [])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    gsr_times, gsr_values, thermal_times, thermal_values = load_session_data(session_dir)
    
    plots_created = []
    gsr_stats = {}
    thermal_stats = {}
    correlation = 0.0
    
    if gsr_times is not None and gsr_values is not None:
        gsr_stats = {
            'mean': float(np.mean(gsr_values)),
            'std': float(np.std(gsr_values)),
            'min': float(np.min(gsr_values)),
            'max': float(np.max(gsr_values)),
            'duration': float(gsr_times[-1] - gsr_times[0])
        }
        
        # Create spectral analysis
        spectral_plot = create_spectral_analysis(gsr_times, gsr_values, output_dir)
        if spectral_plot:
            plots_created.append(spectral_plot)
        
        # Create event detection
        event_plot = create_event_detection_plot(gsr_times, gsr_values, output_dir)
        if event_plot:
            plots_created.append(event_plot)
    
    if thermal_times is not None and thermal_values is not None:
        thermal_stats = {
            'mean': float(np.mean(thermal_values)),
            'std': float(np.std(thermal_values)),
            'min': float(np.min(thermal_values)),
            'max': float(np.max(thermal_values)),
            'duration': float(thermal_times[-1] - thermal_times[0])
        }
    
    # Cross-modal analysis if both available
    if (gsr_times is not None and gsr_values is not None and 
        thermal_times is not None and thermal_values is not None):
        
        corr_plot = create_correlation_plot(gsr_times, gsr_values, 
                                          thermal_times, thermal_values, output_dir)
        if corr_plot:
            plots_created.append(corr_plot)
        
        # Calculate correlation
        try:
            common_times = np.linspace(0, min(gsr_times[-1], thermal_times[-1]), 100)
            gsr_interp = np.interp(common_times, gsr_times, gsr_values)
            thermal_interp = np.interp(common_times, thermal_times, thermal_values)
            correlation = float(np.corrcoef(gsr_interp, thermal_interp)[0, 1])
        except Exception:
            correlation = 0.0
    
    return AnalysisResult(session_dir, gsr_stats, thermal_stats, correlation, plots_created)

def print_analysis_summary(result: AnalysisResult) -> None:
    """Print analysis summary."""
    print(f"\n=== Enhanced Analysis Summary ===")
    print(f"Session: {result.session_dir}")
    
    if result.gsr_stats:
        print(f"\nGSR Statistics:")
        print(f"  Mean: {result.gsr_stats['mean']:.3f} μS")
        print(f"  Std:  {result.gsr_stats['std']:.3f} μS")
        print(f"  Range: {result.gsr_stats['min']:.3f} - {result.gsr_stats['max']:.3f} μS")
        print(f"  Duration: {result.gsr_stats['duration']:.1f} s")
    
    if result.thermal_stats:
        print(f"\nThermal Statistics:")
        print(f"  Mean: {result.thermal_stats['mean']:.1f} °C")
        print(f"  Std:  {result.thermal_stats['std']:.1f} °C")
        print(f"  Range: {result.thermal_stats['min']:.1f} - {result.thermal_stats['max']:.1f} °C")
        print(f"  Duration: {result.thermal_stats['duration']:.1f} s")
    
    if result.gsr_stats and result.thermal_stats:
        print(f"\nCross-Modal Correlation: {result.correlation:.3f}")
    
    if result.plots_created:
        print(f"\nPlots created:")
        for plot in result.plots_created:
            print(f"  • {os.path.basename(plot)}")
    
    print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced multi-modal data analysis and visualization")
    parser.add_argument("--session", required=True, help="Path to session directory")
    parser.add_argument("--out", required=True, help="Path to output directory for plots")
    parser.add_argument("--dry-run", action="store_true", help="Run without requiring session data")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Enhanced Analysis Script - Dry Run Mode")
        print("This script provides advanced visualizations including:")
        print("  • Multi-modal correlation analysis")
        print("  • Power spectral density plots")
        print("  • Event detection visualization")
        print("  • Statistical distribution analysis")
        print("\nExample usage:")
        print(f"  python {__file__} --session /path/to/session --out /path/to/output")
        return 0
    
    if not os.path.exists(args.session):
        print(f"Error: Session directory not found: {args.session}")
        return 1
    
    print(f"Running enhanced analysis on: {args.session}")
    print(f"Output directory: {args.out}")
    
    result = run_enhanced_analysis(args.session, args.out)
    print_analysis_summary(result)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())