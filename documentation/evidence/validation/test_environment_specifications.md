# Test Environment Specifications

## Hardware Environment

### PC Controller Test Platform
- **Operating System:** Ubuntu 22.04 LTS (Linux kernel 5.15.0)
- **Processor:** Intel Core i7-8700K @ 3.70GHz (6 cores, 12 threads)
- **Memory:** 32GB DDR4 RAM @ 2666MHz
- **Storage:** Samsung 970 EVO Plus 1TB NVMe SSD
- **Network Interface:**
  - Ethernet: Intel I219-V Gigabit (primary)
  - WiFi: Intel AX200 802.11ax (testing/backup)
- **Graphics:** NVIDIA GeForce GTX 1070 (for UI acceleration)
- **Python Environment:** Python 3.12.3 with virtual environment

### Android Device Test Platform
- **Primary Devices:**
  - Samsung Galaxy S21 (Android 13, API level 33)
  - Google Pixel 6 (Android 14, API level 34)
  - OnePlus Nord N20 (Android 12, API level 31)
- **Secondary Devices:**
  - Samsung Galaxy Tab S7 (Android 13, tablet form factor)
  - Google Pixel 4a (Android 11, older API compatibility)

### Network Infrastructure
- **Primary Network:** University research lab WiFi (802.11ac, 5GHz)
- **Secondary Network:** Dedicated 2.4GHz hotspot for testing
- **Test Router:** ASUS AX6000 with controlled QoS settings
- **Bandwidth:** 1Gbps symmetric fiber connection
- **Latency:** <2ms to local servers, <15ms to internet

## Software Environment

### PC Controller Dependencies
```
Python: 3.12.3
PyQt6: 6.5.0
OpenCV: 4.8.0
NumPy: 1.24.3
Pandas: 2.0.2
Cryptography: 41.0.1
Pytest: 7.4.0
Coverage: 7.2.7
```

### Android Application Stack
- **Target SDK:** Android 13 (API level 33)
- **Minimum SDK:** Android 8.0 (API level 26)
- **Build System:** Gradle 8.0 with Kotlin 1.8.21
- **Key Libraries:**
  - AndroidX Camera2: 1.3.0
  - Kotlin Coroutines: 1.7.1
  - OkHttp3: 4.11.0
  - Gson: 2.10.1

### Development and Testing Tools
- **IDE:** IntelliJ IDEA Community 2023.2 + Android Studio Giraffe
- **Version Control:** Git 2.40.1
- **CI/CD:** GitHub Actions with Ubuntu 22.04 runners
- **Code Quality:** SonarQube Community Edition 9.9
- **Performance Profiling:** Android Profiler, Python cProfile

## Test Data and Scenarios

### Synthetic Test Data
- **GSR Simulation:** Physiologically plausible conductance ranges (1-50 [UNICODE]S)
- **Thermal Simulation:** Body temperature variations (32-37degC)
- **RGB Video:** 1920x1080 @ 30fps H.264 encoded test patterns
- **Network Latency:** Controlled delays from 1ms to 500ms
- **Jitter Simulation:** Random timing variations up to +/-10ms

### Real-World Test Scenarios
- **Multi-Device:** Up to 8 simultaneous Android connections
- **Extended Duration:** 8-hour continuous operation tests
- **Network Stress:** Bandwidth limitation, packet loss injection
- **Battery Drain:** Low battery condition simulation
- **Environmental:** WiFi interference, Bluetooth congestion

## Measurement and Calibration

### Timing Accuracy
- **PC Clock Source:** NTP synchronized system clock (+/-1ms accuracy)
- **Android Clock Source:** System.nanoTime() for high-resolution timestamps
- **Synchronization Reference:** PC controller as master time source
- **Drift Measurement:** Statistical analysis over 1000+ sample points

### Network Performance
- **Latency Measurement:** Round-trip time analysis with iperf3
- **Bandwidth Testing:** Sustained throughput measurement
- **Packet Loss:** Controlled injection at 0.1%, 1.0%, 5.0% rates
- **Connection Quality:** Signal strength, interference monitoring

### System Resource Monitoring
- **Memory Profiling:** Linux /proc/meminfo and Android Debug Bridge
- **CPU Utilization:** htop, Android Profiler continuous monitoring
- **Disk I/O:** iostat, storage performance analysis
- **Network Usage:** netstat, per-application bandwidth tracking

## Quality Assurance Procedures

### Code Quality Standards
- **Unit Test Coverage:** Minimum 80% line coverage requirement
- **Integration Test Coverage:** All API endpoints and critical user paths
- **Static Analysis:** SonarQube quality gate (Grade A rating)
- **Linting:** Consistent code formatting with automated checks

### Performance Standards
- **Memory Leak Detection:** No growth >5MB over 4-hour sessions
- **CPU Usage:** Average <25%, peak <60% during normal operation
- **Response Time:** UI interactions <200ms, network calls <2000ms
- **Synchronization:** 95% of measurements within +/-10ms target

### Security Validation
- **TLS Implementation:** OpenSSL 3.0.2 with FIPS 140-2 compliance
- **Certificate Management:** Self-signed cert generation and validation
- **Data Protection:** AES-256 encryption for sensitive data
- **Network Security:** Port scanning, vulnerability assessment

## Reproducibility Standards

### Environment Isolation
- **Virtual Environments:** Isolated Python virtualenv for all PC testing
- **Docker Containers:** Containerized test environments for CI/CD
- **Android Emulators:** AVD configurations with specific API levels
- **Network Isolation:** Dedicated test VLAN for controlled conditions

### Documentation Standards
- **Test Case Specification:** Detailed pre-conditions, steps, expected outcomes
- **Configuration Management:** All environment settings version controlled
- **Data Provenance:** Complete traceability of test data and results
- **Audit Trail:** All test executions logged with timestamps and parameters

### Version Control
- **Software Versions:** All dependency versions pinned and documented
- **Test Data Versioning:** Synthetic datasets tagged and archived
- **Configuration Snapshots:** System configurations saved before each test
- **Result Archiving:** All test outputs preserved with metadata

## Limitations and Constraints

### Hardware Limitations
- **Sensor Hardware:** Shimmer3 GSR+ and Topdon TC001 not available for testing
- **Bluetooth Testing:** Limited to Android Bluetooth stack simulation
- **Multi-Platform:** Windows and macOS testing not included in current scope
- **Resource Constraints:** Testing limited to available hardware configuration

### Software Limitations
- **Simulation Fidelity:** Hardware sensor behavior approximated, not replicated
- **Scale Testing:** Maximum 8 concurrent devices due to hardware constraints
- **Long-Term Testing:** Extended tests limited to 8-hour maximum duration
- **Real-World Variability:** Laboratory conditions may not reflect deployment

### Methodology Constraints
- **User Testing:** Limited to 3 participants, not statistically representative
- **Network Conditions:** Controlled lab environment, limited real-world variation
- **Error Scenarios:** Fault injection simulated, may not cover all real failures
- **Performance Metrics:** Based on available monitoring tools, not specialized equipment

---

**Note:** This test environment specification represents the actual conditions under which evidence was collected. Results should be interpreted within these constraints and limitations. Production deployment may encounter different conditions requiring additional validation.
