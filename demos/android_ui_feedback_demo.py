#!/usr/bin/env python3
"""
UI Feedback & User Experience Demo
Demonstrates the comprehensive UI improvements implemented for the Android sensor application.
"""

import time
from typing import Dict, Any
from enum import Enum

class SensorState(Enum):
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ERROR = "error"
    SIMULATED = "simulated"

class UIFeedbackDemo:
    def __init__(self):
        self.sensors = {
            "RGB Camera": {"state": SensorState.OFFLINE, "message": "Offline"},
            "Thermal Camera": {"state": SensorState.OFFLINE, "message": "Offline"},
            "GSR Sensor": {"state": SensorState.OFFLINE, "message": "Disconnected"},
            "PC Link": {"state": SensorState.OFFLINE, "message": "Not Connected"}
        }
        self.recording = False
        self.recording_time = 0
        self.main_status = "Initializing..."
    
    def display_ui_state(self):
        """Display current UI state"""
        print("\n" + "="*60)
        print("📱 SENSOR SPOKE - UI FEEDBACK DEMO")
        print("="*60)
        
        status_color = self.get_status_color(self.main_status)
        print(f"📊 Status: {status_color}{self.main_status}\033[0m")
        
        if self.recording:
            timer = self.format_recording_time(self.recording_time)
            print(f"⏱️  Recording Time: \033[91m{timer}\033[0m")
        
        print("-" * 60)
        
        print("🔍 SENSOR STATUS INDICATORS:")
        for sensor_name, sensor_info in self.sensors.items():
            dot_color = self.get_sensor_dot_color(sensor_info["state"])
            status_msg = sensor_info["message"]
            print(f"   {dot_color}●\033[0m {sensor_name:<15} {status_msg}")
        
        print("-" * 60)
        
        start_btn = "🟢 Start Recording" if not self.recording else "🔴 Recording..."
        stop_btn = "🟢 Stop Recording" if self.recording else "⚫ Stop Recording"
        start_enabled = " (ENABLED)" if not self.recording else " (DISABLED)"
        stop_enabled = " (ENABLED)" if self.recording else " (DISABLED)"
        
        print(f"🎛️  CONTROLS:")
        print(f"   [{start_btn}]{start_enabled}")
        print(f"   [{stop_btn}]{stop_enabled}")
        print("="*60)
    
    def get_status_color(self, status: str) -> str:
        """Get ANSI color code for status text"""
        if any(word in status.lower() for word in ["error", "failed"]):
            return "\033[91m"
        elif "recording" in status.lower():
            return "\033[91m"
        elif any(word in status.lower() for word in ["ready", "connected", "success"]):
            return "\033[92m"
        elif any(word in status.lower() for word in ["checking", "starting", "stopping"]):
            return "\033[94m"
        else:
            return "\033[0m"
    
    def get_sensor_dot_color(self, state: SensorState) -> str:
        """Get colored dot for sensor state"""
        color_map = {
            SensorState.ACTIVE: "\033[92m",
            SensorState.SIMULATED: "\033[93m",
            SensorState.CONNECTING: "\033[94m",
            SensorState.ERROR: "\033[91m",
            SensorState.OFFLINE: "\033[90m"
        }
        return color_map.get(state, "\033[90m")
    
    def format_recording_time(self, seconds: int) -> str:
        """Format recording time as HH:MM:SS"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def simulate_sensor_connections(self):
        """Simulate sensors connecting one by one"""
        print("🚀 Starting UI Feedback Demo...")
        time.sleep(1)
        
        self.main_status = "Ready to connect"
        self.display_ui_state()
        time.sleep(2)
        
        print("\n📹 Connecting RGB Camera...")
        self.sensors["RGB Camera"]["state"] = SensorState.CONNECTING
        self.sensors["RGB Camera"]["message"] = "Connecting..."
        self.main_status = "Connecting RGB Camera..."
        self.display_ui_state()
        time.sleep(2)
        
        self.sensors["RGB Camera"]["state"] = SensorState.ACTIVE
        self.sensors["RGB Camera"]["message"] = "Recording"
        self.main_status = "RGB Camera connected"
        self.display_ui_state()
        time.sleep(2)
        
        # Thermal Camera - simulation mode
        print("\n🌡️  Checking Thermal Camera...")
        self.sensors["Thermal Camera"]["state"] = SensorState.CONNECTING
        self.sensors["Thermal Camera"]["message"] = "Detecting..."
        self.main_status = "Detecting thermal camera..."
        self.display_ui_state()
        time.sleep(2)
        
        self.sensors["Thermal Camera"]["state"] = SensorState.SIMULATED
        self.sensors["Thermal Camera"]["message"] = "Simulated"
        self.main_status = "Thermal camera: Simulation mode"
        self.display_ui_state()
        print("⚠️  NOTIFICATION: Thermal Camera Simulation (device not found)")
        time.sleep(2)
        
        print("\n🔋 Connecting GSR Sensor...")
        self.sensors["GSR Sensor"]["state"] = SensorState.CONNECTING
        self.sensors["GSR Sensor"]["message"] = "Scanning BLE..."
        self.main_status = "Scanning for GSR sensor..."
        self.display_ui_state()
        time.sleep(3)
        
        self.sensors["GSR Sensor"]["state"] = SensorState.ACTIVE
        self.sensors["GSR Sensor"]["message"] = "Connected"
        self.main_status = "GSR sensor connected"
        self.display_ui_state()
        time.sleep(2)
        
        print("\n💻 Connecting to PC Hub...")
        self.sensors["PC Link"]["state"] = SensorState.CONNECTING
        self.sensors["PC Link"]["message"] = "Discovering..."
        self.main_status = "Discovering PC Hub..."
        self.display_ui_state()
        time.sleep(2)
        
        self.sensors["PC Link"]["state"] = SensorState.ACTIVE
        self.sensors["PC Link"]["message"] = "Connected"
        self.main_status = "All sensors ready - Ready to record"
        self.display_ui_state()
        time.sleep(2)
    
    def simulate_recording_session(self):
        """Simulate a recording session"""
        print("\n🎬 Starting Recording Session...")
        
        self.recording = True
        self.recording_time = 0
        self.main_status = "Recording in progress..."
        self.display_ui_state()
        
        for i in range(10):
            time.sleep(1)
            self.recording_time += 1
            self.display_ui_state()
            
            if i == 4:
                print("\n⚠️  TOAST: GSR sensor briefly disconnected...")
                self.sensors["GSR Sensor"]["state"] = SensorState.ERROR
                self.sensors["GSR Sensor"]["message"] = "Reconnecting..."
                self.display_ui_state()
                time.sleep(1)
                
                print("✅ TOAST: GSR sensor reconnected")
                self.sensors["GSR Sensor"]["state"] = SensorState.ACTIVE
                self.sensors["GSR Sensor"]["message"] = "Connected"
        
        print("\n🛑 Stopping Recording...")
        self.main_status = "Stopping recording..."
        self.display_ui_state()
        time.sleep(2)
        
        self.recording = False
        self.recording_time = 0
        self.main_status = "Recording stopped. Files saved to /storage/sessions/session_001"
        self.display_ui_state()
        time.sleep(2)
        
        print("\n📊 RECORDING SUMMARY:")
        print("   Session ID: session_001")
        print("   Duration: 00:00:10")
        print("   GSR samples: 500")
        print("   RGB frames: 300")
        print("   Thermal frames: 30")
        print("   Status: Complete ✅")
    
    def demonstrate_error_handling(self):
        """Demonstrate error handling features"""
        print("\n❌ Demonstrating Error Handling...")
        
        self.main_status = "Error: Camera permission denied"
        self.display_ui_state()
        print("🚨 ERROR DIALOG: Camera permission is required to record RGB video")
        print("   [OK] button to dismiss")
        time.sleep(3)
        
        self.sensors["Thermal Camera"]["state"] = SensorState.ERROR
        self.sensors["Thermal Camera"]["message"] = "Hardware Error"
        self.main_status = "Error: Thermal camera hardware failure"
        self.display_ui_state()
        print("🚨 TOAST: Thermal camera disconnected - data will be incomplete")
        time.sleep(3)
        
        print("\n🔄 Recovering from errors...")
        self.sensors["Thermal Camera"]["state"] = SensorState.SIMULATED
        self.sensors["Thermal Camera"]["message"] = "Simulated"
        self.main_status = "Ready to record (thermal in simulation mode)"
        self.display_ui_state()

    def run_demo(self):
        """Run the complete UI feedback demo"""
        print("🎭 UI FEEDBACK & USER EXPERIENCE DEMO")
        print("=====================================")
        print("This demo shows the comprehensive UI improvements implemented")
        print("for real-time sensor status, recording feedback, and error handling.")
        print("\nPress Enter to start...")
        input()
        
        self.simulate_sensor_connections()
        
        print("\n" + "="*60)
        print("✅ PHASE 1 COMPLETE: All sensors connected and ready")
        print("Press Enter to start recording session...")
        input()
        
        self.simulate_recording_session()
        
        print("\n" + "="*60)
        print("✅ PHASE 2 COMPLETE: Recording session finished")
        print("Press Enter to demonstrate error handling...")
        input()
        
        self.demonstrate_error_handling()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE!")
        print("✅ Real-time sensor status indicators")
        print("✅ Recording timer and state management")
        print("✅ Error dialogs and toast notifications")
        print("✅ Thermal simulation feedback")
        print("✅ Professional UX with color coding")
        print("✅ Comprehensive user feedback at every step")

if __name__ == "__main__":
    demo = UIFeedbackDemo()
    demo.run_demo()