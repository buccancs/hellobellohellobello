#!/usr/bin/env python3
"""
Create visual demonstration plots showing the enhanced visualizations.
This generates example plots demonstrating the new visualization features.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os

def create_device_widgets_demo():
    """Create a demo showing the enhanced device widgets layout."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Enhanced PC Controller Dashboard - New Device Widgets', fontsize=16, fontweight='bold')
    
    # Create a 2x3 grid layout similar to the actual GUI
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. RGB Video Widget (existing)
    ax1 = fig.add_subplot(gs[0, 0])
    # Simulate webcam feed
    webcam_data = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    ax1.imshow(webcam_data)
    ax1.set_title('Local Webcam (RGB Video)', fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 2. NEW: Thermal Camera Widget  
    ax2 = fig.add_subplot(gs[0, 1])
    # Simulate thermal data with false-color mapping
    thermal_data = np.random.randn(192, 256) * 2 + 25 + 5 * np.sin(np.linspace(0, 2*np.pi, 256))
    thermal_normalized = (thermal_data - thermal_data.min()) / (thermal_data.max() - thermal_data.min())
    
    # Apply thermal colormap
    thermal_rgb = plt.cm.hot(thermal_normalized)
    ax2.imshow(thermal_rgb)
    ax2.set_title('Thermal Camera (NEW)', fontweight='bold', color='red')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add temperature overlay
    min_temp, max_temp = thermal_data.min(), thermal_data.max()
    mean_temp = thermal_data.mean()
    ax2.text(0.02, 0.98, f'Mean: {mean_temp:.1f}°C\nMin: {min_temp:.1f}°C\nMax: {max_temp:.1f}°C', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Enhanced GSR Widget (multi-channel)
    ax3 = fig.add_subplot(gs[0, 2])
    t = np.linspace(0, 10, 500)
    gsr_data = 12 + 2 * np.sin(0.1 * t) + 0.5 * np.random.randn(len(t))
    ppg_data = 1000 + 50 * np.sin(2 * t) + 10 * np.random.randn(len(t))
    
    # Scale PPG to be visible with GSR
    ppg_scaled = (ppg_data - ppg_data.min()) / (ppg_data.max() - ppg_data.min()) * 2 + gsr_data.max() + 1
    
    ax3.plot(t, gsr_data, 'b-', linewidth=2, label='GSR (μS)')
    ax3.plot(t, ppg_scaled, 'orange', linewidth=1, label='PPG (scaled)')
    ax3.set_title('Enhanced GSR + PPG (Multi-channel)', fontweight='bold', color='blue')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('GSR (μS)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. NEW: Status Widget
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # Recording status indicator
    circle = patches.Circle((0.15, 0.8), 0.08, color='green', ec='black')
    ax4.add_patch(circle)
    ax4.text(0.35, 0.8, 'Recording', fontsize=12, va='center', fontweight='bold')
    
    # Device count
    ax4.text(0.1, 0.6, 'Devices: 5 (3 local, 2 remote)', fontsize=11, va='center')
    
    # Network status
    ax4.text(0.1, 0.4, 'Network: Connected to 2 devices', fontsize=11, va='center')
    
    # Data stats
    ax4.text(0.1, 0.2, 'Data: 15.2k samples', fontsize=11, va='center')
    
    ax4.set_title('System Status (NEW)', fontweight='bold', color='green')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.patch.set_facecolor('#f0f8ff')
    
    # 5. Remote Device Widget (existing)
    ax5 = fig.add_subplot(gs[1, 1])
    remote_data = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    ax5.imshow(remote_data)
    ax5.set_title('Remote: Android Device 1', fontweight='bold')
    ax5.set_xticks([])
    ax5.set_yticks([])
    
    # 6. Enhanced Timeline Widget
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Simulate timeline with annotations
    timeline = np.linspace(0, 60, 300)
    signal = 10 + 3 * np.sin(0.1 * timeline) + np.random.randn(len(timeline)) * 0.5
    
    ax6.plot(timeline, signal, 'b-', linewidth=1.5, alpha=0.8)
    
    # Add annotation markers
    annotation_times = [15, 30, 45]
    annotation_texts = ['Event A', 'Event B', 'Event C']
    
    for i, (time, text) in enumerate(zip(annotation_times, annotation_texts)):
        ax6.axvline(x=time, color='green', linestyle='--', alpha=0.8, linewidth=2)
        ax6.text(time, signal.max() - 0.5 * i, text, rotation=90, va='bottom', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Current position cursor
    current_pos = 25
    ax6.axvline(x=current_pos, color='red', linewidth=3, alpha=0.9)
    
    ax6.set_title('Enhanced Timeline with Annotations', fontweight='bold', color='purple')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Signal')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_analysis_demo():
    """Create demo of enhanced analysis visualizations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced Data Analysis Visualizations', fontsize=16, fontweight='bold')
    
    # Generate sample data
    t = np.linspace(0, 300, 1000)  # 5 minutes at ~3 Hz
    gsr = 12 + 2 * np.sin(0.02 * t) + 0.8 * np.random.randn(len(t))
    thermal = 25 + 1.5 * np.sin(0.02 * t + 0.5) + 0.3 * np.random.randn(len(t))
    
    # 1. Multi-modal correlation plot
    ax1.scatter(gsr, thermal, alpha=0.6, s=10, c=t, cmap='viridis')
    correlation = np.corrcoef(gsr, thermal)[0, 1]
    ax1.set_xlabel('GSR (μS)')
    ax1.set_ylabel('Thermal (°C)')
    ax1.set_title(f'GSR vs Thermal Correlation (r = {correlation:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for time
    cbar = fig.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label('Time (s)')
    
    # 2. Power spectral density
    try:
        from scipy import signal
        frequencies, psd = signal.welch(gsr, fs=3.0, nperseg=128)
        ax2.semilogy(frequencies, psd, 'b-', linewidth=2)
    except ImportError:
        # Fallback: simple FFT
        fft = np.abs(np.fft.fft(gsr - gsr.mean()))
        freqs = np.fft.fftfreq(len(gsr), 1/3.0)
        ax2.semilogy(freqs[:len(freqs)//2], fft[:len(fft)//2], 'b-', linewidth=2)
        frequencies = freqs[:len(freqs)//2]
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('GSR Frequency Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(0.5, frequencies[-1]))
    
    # 3. Event detection
    gsr_diff = np.gradient(gsr, t)
    threshold = np.std(gsr_diff) * 2
    events = np.where(np.abs(gsr_diff) > threshold)[0]
    
    ax3.plot(t, gsr, 'lightblue', alpha=0.7, label='Raw GSR')
    
    # Apply simple smoothing
    window = 21
    gsr_smooth = np.convolve(gsr, np.ones(window)/window, mode='same')
    ax3.plot(t, gsr_smooth, 'b-', linewidth=2, label='Smoothed GSR')
    
    if len(events) > 0:
        ax3.scatter(t[events], gsr_smooth[events], color='red', s=50, zorder=5, 
                   label=f'Events ({len(events)})')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('GSR (μS)')
    ax3.set_title('Event Detection Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical distributions
    ax4.hist(gsr, bins=30, alpha=0.7, density=True, color='blue', label='GSR')
    ax4.hist(thermal, bins=30, alpha=0.7, density=True, color='red', label='Thermal')
    
    # Add statistics
    gsr_mean, gsr_std = np.mean(gsr), np.std(gsr)
    thermal_mean, thermal_std = np.mean(thermal), np.std(thermal)
    
    ax4.axvline(gsr_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax4.axvline(thermal_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Signal Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add text with statistics
    stats_text = f'GSR: μ={gsr_mean:.2f}, σ={gsr_std:.2f}\nThermal: μ={thermal_mean:.2f}, σ={thermal_std:.2f}'
    ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate demonstration plots."""
    print("Creating visualization demonstration plots...")
    
    # Set matplotlib to use non-GUI backend
    import matplotlib
    matplotlib.use('Agg')
    
    output_dir = '/tmp/visualization_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dashboard demo
    dashboard_fig = create_device_widgets_demo()
    dashboard_path = os.path.join(output_dir, 'enhanced_dashboard_demo.png')
    dashboard_fig.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close(dashboard_fig)
    print(f"Created dashboard demo: {dashboard_path}")
    
    # Generate analysis demo
    analysis_fig = create_analysis_demo() 
    analysis_path = os.path.join(output_dir, 'enhanced_analysis_demo.png')
    analysis_fig.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close(analysis_fig)
    print(f"Created analysis demo: {analysis_path}")
    
    print(f"\nVisualization demos created in: {output_dir}")
    print("These demonstrate the enhanced visualization features implemented:")
    print("  • Thermal camera false-color display")
    print("  • Multi-channel GSR+PPG plotting")
    print("  • Status dashboard widget")
    print("  • Enhanced timeline with annotations")
    print("  • Advanced analysis plots (correlation, spectral, events)")

if __name__ == "__main__":
    main()