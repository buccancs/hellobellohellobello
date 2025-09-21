Of course. Here is a comprehensive guide on how to set up, run, and test the Multi-Modal Physiological Sensing Platform,
based on the provided documentation.

### **1. Running the PC Controller (Hub) Application**

The PC application acts as the central control hub for the entire system.

**Prerequisites:**

* Python 3.11+ installed.
* A Python virtual environment tool (e.g., `venv`).

**Setup and Execution Steps:**

1. **Initialize the Project:**
    * Clone the repository and navigate to the `/pc_controller/` directory.
    * Initialize a new Git repository if you haven't already.
2. **Set Up Virtual Environment:**
    * Create a Python virtual environment: `python -m venv venv`
    * Activate it:
        * Windows: `venv\Scripts\activate`
        * macOS/Linux: `source venv/bin/activate`
3. **Install Dependencies:**
    * The core dependencies are **PyQt6** for the GUI and **zeroconf** for network discovery.
    * Install them from the `requirements.txt` file: `pip install -r requirements.txt`
4. **Launch the Application:**
    * Run the main entry point script: `python src/main.py`
    * The application's GUI will launch, showing the Dashboard, Logs, and Playback tabs. The Dashboard will begin
      searching for Android devices on the network using Zeroconf.

---

### **2. Running the Android Sensor Node (Spoke) Application**

The Android app turns a smartphone into a remote sensor node.

**Prerequisites:**

* Android Studio installed.
* An Android device with a modern API level.

**Setup and Execution Steps:**

1. **Open the Project:**
    * Open Android Studio and select the `asd-8f44e41db0ea99bb940edb74662504a627767cb6` directory. Gradle will
      automatically sync and build the project.
2. **Configure Permissions:**
    * Ensure the `AndroidManifest.xml` file includes the necessary permissions: `INTERNET`, `ACCESS_WIFI_STATE`, and
      `CHANGE_WIFI_MULTICAST_STATE` to allow network communication and service discovery.
3. **Build and Run:**
    * Connect your Android device to your computer via USB.
    * Click the "Run" button in Android Studio to build and deploy the app to your device.
4. **Start the Service:**
    * Once the app is running, it will automatically start advertising itself on the local network using Zeroconf (mDNS)
      under the service type `_gsr-controller._tcp.local.`.
    * A foreground service will start a `ServerSocket` to listen for incoming connections from the PC Hub, ensuring the
      connection remains active even if the app is in the background.

---

### **3. Connecting the Sensors**

The platform supports a Shimmer GSR+ sensor and a Topdon thermal camera.

**A. Connecting the Shimmer3 GSR+ Sensor**

There are two modes for connecting the Shimmer sensor:

* **High-Integrity Mode (Wired to PC):**
    1. Dock the Shimmer sensor in its base and connect it to the PC via a USB serial port (e.g., "COM3" on Windows).
    2. The PC Controller's `NativeShimmer` C++ backend will automatically connect to this sensor.
    3. The live GSR data will appear in the `PyQtGraph` plot on the PC Dashboard.

* **High-Mobility Mode (Wireless to Android):**
    1. Ensure Bluetooth is enabled on the Android device.
    2. Pair the Shimmer sensor with the Android device through the device's Bluetooth settings.
    3. The Android app, using the Nordic BLE library, will establish a connection to the sensor.
    4. The app will send a start command (`0x07`) to the sensor to begin streaming GSR and PPG data at 128 Hz. The data
       is then forwarded to the PC Hub over Wi-Fi.

**B. Connecting the Topdon TC001 Thermal Camera**

1. Connect the Topdon TC001 camera to the Android device's USB-C port.
2. The Android app will detect the USB device attachment.
3. The `ThermalCameraRecorder` module, using the `UVCCamera` library, will initialize the camera.
4. A frame callback will be registered to receive raw thermal data, which is then written to a CSV file, with each row
   containing a timestamp and 49,152 temperature values.

---

### **4. How to Run Tests**

The system is designed with a multi-layered testing strategy to ensure reliability and accuracy.

**A. Virtual Test (Simulation Mode)**

If a physical Shimmer sensor is not available, you can use the built-in simulation mode.

* **How to Run:**
    1. On the PC Controller application, if no Shimmer device is detected, an option to "Enable Simulation Mode" will be
       available in the UI.
    2. Activate this mode. The system will begin generating continuous dummy GSR sensor data, which will be displayed on
       the live plot.
    3. This allows you to test the full data pipeline, including network communication, session management, and UI
       responsiveness, without needing any hardware.

**B. Hardware Test (End-to-End Recording Session)**

This test verifies the entire system's functionality in a real-world scenario.

* **How to Run:**
    1. **Setup:** Ensure the PC Controller and at least one Android Spoke are running and on the same Wi-Fi network.
       Connect all hardware sensors as described above.
    2. **Discovery & Connection:** The PC app will automatically discover the Android device(s). Select a device from
       the list in the UI to establish a TCP/IP connection. A handshake will occur where the Android device sends its
       capabilities (e.g., available cameras) to the PC.
    3. **Start Recording:** Click the "Start Recording" button on the PC GUI. This sends a JSON command to all connected
       Android devices to begin capturing data simultaneously.
    4. **Monitor:** Observe the live video previews and real-time GSR plots on the PC Dashboard to confirm data is
       streaming correctly.
    5. **Synchronization Check (Optional):** Use the "Flash Sync" command on the PC. This will cause all connected
       Android screens to flash white simultaneously, providing a visual marker in the recorded videos to verify
       temporal alignment.
    6. **Stop Recording:** Click "Stop Recording." All devices will stop their captures and finalize their data files.
    7. **Data Transfer:** The Android app will automatically compress its session data into a ZIP archive and transfer
       it to the PC, where it is saved in a unique session folder.
    8. **Verification:** Check the session folder on the PC to ensure all data files (MP4 video, thermal CSV, GSR CSV,
       etc.) are present and correctly formatted.

**C. Unit, Integration, and System Tests**

The project specifies a formal testing framework for developers.

* **Unit Tests:**
    * **PC:** Use the `pytest` framework. Tests are located in the `/pc_controller/tests/` directory. Run them from the
      command line: `pytest`.
    * **Android:** Use `JUnit` and `Robolectric`. Tests are located within the standard Android project structure. Run
      them from Android Studio or using Gradle: `./gradlew test`.
* **Integration Tests:**
    * These tests verify the communication protocol and multi-device synchronization. A `DeviceSimulator` and a
      JSON-based message protocol are used to validate interactions without requiring physical hardware.
* **System Tests:**
    * **Endurance Testing:** The system is validated through 8-hour endurance tests to detect memory leaks and ensure
      stability over long recording sessions.
    * **Performance Monitoring:** CPU usage, memory consumption, and network throughput are tracked during tests to
      ensure the system meets its performance requirements and does not drop frames or data samples.