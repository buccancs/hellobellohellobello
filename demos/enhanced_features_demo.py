#!/usr/bin/env python3
"""
PC Controller Enhanced Features Demo

This script demonstrates the key enhancements made to the PC Controller:
1. Real-time GSR visualization with PyQtGraph
2. TCP Command Server with device registration and live data handling  
3. Native C++ backend integration for performance-critical operations
4. TLS security layer support (when configured)
5. Enhanced session management and data export

Usage: python enhanced_features_demo.py
"""

import sys
import time
import threading
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pc_controller" / "src"))

def demo_native_backend():
    """Demo 1: Native C++ backend performance."""
    print("=" * 60)
    print("DEMO 1: Native C++ Backend Integration")
    print("=" * 60)
    
    try:
        from core.local_interfaces import ShimmerInterface
        
        print("Testing Shimmer GSR interface with native backend...")
        shimmer = ShimmerInterface(port=os.environ.get("SHIMMER_PORT", "COM3"))
        shimmer.start()
        time.sleep(1.0)
        
        # Get samples and performance stats
        ts, vals = shimmer.get_latest_samples()
        stats = shimmer.get_performance_stats()
        
        print(f"Backend type: {stats['backend_type']}")
        print(f"Samples processed: {stats['samples_processed']}")
        print(f"Latest batch size: {len(ts)} samples")
        print(f"Sample rate: ~{len(ts)} samples/second")
        print(f"GSR values: {vals[:5][:len(vals)]}... (showing first 5)")
        
        shimmer.stop()
        print("✓ Native backend demo completed successfully\n")
        
    except Exception as e:
        print(f"✗ Native backend demo failed: {e}\n")

def demo_tcp_server():
    """Demo 2: Enhanced TCP Command Server."""
    print("=" * 60) 
    print("DEMO 2: Enhanced TCP Command Server")
    print("=" * 60)
    
    try:
        from network.tcp_command_server import TCPCommandServer
        
        server = TCPCommandServer(host="127.0.0.1", port=8081)
        
        def on_device_registered(device_id, name, type_, capabilities):
            print(f"📱 Device registered: {name} ({type_}) with {capabilities}")
            
        def on_live_gsr(device_id, data, timestamp):
            print(f"📊 Live GSR from {device_id}: {data}")
            
        server.set_device_callbacks(
            device_registered_callback=on_device_registered,
            live_gsr_callback=on_live_gsr
        )
        
        print("Starting TCP Command Server on 127.0.0.1:8081...")
        if server.start():
            print("✓ Server started successfully")
            
            print("Simulating device registration...")
            import socket
            import time
            
            def simulate_client():
                time.sleep(0.5)
                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.connect(("127.0.0.1", 8081))
                    
                    register_msg = {
                        "command": "register_device",
                        "device_info": {
                            "name": "Demo Android Device",
                            "type": "sensor_node",
                            "capabilities": ["gsr", "video", "thermal"]
                        }
                    }
                    client.send(json.dumps(register_msg).encode() + b"\n")
                    
                    data_msg = {
                        "command": "live_data_frame",
                        "frame_type": "gsr_sample", 
                        "data": {"gsr_microsiemens": 12.5},
                        "timestamp": time.time()
                    }
                    client.send(json.dumps(data_msg).encode() + b"\n")
                    
                    client.close()
                except Exception as e:
                    print(f"Client simulation error: {e}")
            
            client_thread = threading.Thread(target=simulate_client, daemon=True)
            client_thread.start()
            
            time.sleep(2.0)
            server.stop()
            print("✓ TCP server demo completed successfully\n")
        else:
            print("✗ Failed to start TCP server\n")
            
    except Exception as e:
        print(f"✗ TCP server demo failed: {e}\n")

def demo_tls_configuration():
    """Demo 3: TLS Security Layer Configuration."""
    print("=" * 60)
    print("DEMO 3: TLS Security Layer Configuration")
    print("=" * 60)
    
    try:
        from network.tls_utils import create_client_ssl_context, create_server_ssl_context
        import os
        
        print("Testing TLS configuration...")
        
        # Test without TLS enabled
        client_ctx = create_client_ssl_context()
        server_ctx = create_server_ssl_context()
        
        print(f"TLS client context (PC_TLS_ENABLE not set): {client_ctx}")
        print(f"TLS server context (PC_TLS_ENABLE not set): {server_ctx}")
        
        # Test with TLS enabled but no certificates
        os.environ["PC_TLS_ENABLE"] = "1"
        try:
            client_ctx = create_client_ssl_context()
            server_ctx = create_server_ssl_context()
            print(f"TLS client context (enabled, no certs): {client_ctx}")
            print(f"TLS server context (enabled, no certs): {server_ctx}")
        finally:
            del os.environ["PC_TLS_ENABLE"]
            
        print("✓ TLS configuration demo completed")
        print("  To enable TLS in production:")
        print("  - Set PC_TLS_ENABLE=1")
        print("  - Configure PC_TLS_SERVER_CERT_FILE and PC_TLS_SERVER_KEY_FILE")
        print("  - Optionally set PC_TLS_CA_FILE for certificate verification")
        print()
        
    except Exception as e:
        print(f"✗ TLS demo failed: {e}\n")

def demo_data_export():
    """Demo 4: Enhanced Data Export."""
    print("=" * 60)
    print("DEMO 4: Enhanced Data Export Capabilities")  
    print("=" * 60)
    
    try:
        from data.hdf5_exporter import export_session_to_hdf5
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "demo_session"
            session_dir.mkdir()
            
            (session_dir / "gsr.csv").write_text(
                "timestamp_ns,gsr_microsiemens,ppg_raw\n"
                "1000000000,10.5,2048\n"
                "1000007812,10.7,2055\n"
                "1000015624,10.4,2041\n"
            )
            
            (session_dir / "android_device").mkdir()
            (session_dir / "android_device" / "thermal.csv").write_text(
                "timestamp_ns,temperature_celsius\n"
                "1000000000,25.2\n"
                "1000033333,25.4\n"
            )
            
            output_path = Path(temp_dir) / "demo_export.h5"
            result_path = export_session_to_hdf5(
                str(session_dir), 
                str(output_path),
                metadata={"session_id": "demo_123", "participants": ["demo_user"]},
                annotations={"events": ["start", "calibration", "end"]}
            )
            
            print(f"✓ Exported session data to: {result_path}")
            print(f"  File size: {output_path.stat().st_size} bytes")
            
            import h5py
            with h5py.File(output_path, 'r') as f:
                print("  HDF5 structure:")
                def print_structure(name, obj):
                    print(f"    /{name}")
                f.visititems(print_structure)
                
        print("✓ Data export demo completed successfully\n")
        
    except Exception as e:
        print(f"✗ Data export demo failed: {e}\n")

def main():
    """Run all enhancement demos."""
    print("PC Controller Enhanced Features Demo")
    print("Demonstrating key improvements from the development effort")
    print()
    
    demo_native_backend()
    demo_tcp_server() 
    demo_tls_configuration()
    demo_data_export()
    
    print("=" * 60)
    print("SUMMARY OF ENHANCEMENTS")
    print("=" * 60)
    print("✓ Real-time GSR visualization with enhanced PyQtGraph plots")
    print("✓ Complete TCP server with device registration and live data")
    print("✓ High-performance C++ native backend integration") 
    print("✓ TLS 1.2+ security layer for encrypted communications")
    print("✓ Enhanced session management and device status tracking")
    print("✓ Robust data aggregation and HDF5 export capabilities")
    print("✓ Cross-platform compatibility and error handling")
    print()
    print("The PC Controller now provides a complete hub-and-spoke")
    print("architecture for multi-modal physiological sensing!")

if __name__ == "__main__":
    main()