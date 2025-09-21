# Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide provides systematic diagnostic procedures for the Multi-Modal Sensor Platform. It covers common issues, advanced debugging techniques, and recovery procedures for production environments.

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Connection and Network Issues](#connection-and-network-issues)
3. [TLS and Security Issues](#tls-and-security-issues)
4. [Heartbeat and Fault Tolerance Issues](#heartbeat-and-fault-tolerance-issues)
5. [Android Device Issues](#android-device-issues)
6. [Performance and Resource Issues](#performance-and-resource-issues)
7. [Data Collection and Storage Issues](#data-collection-and-storage-issues)
8. [GUI and User Interface Issues](#gui-and-user-interface-issues)
9. [Advanced Debugging Techniques](#advanced-debugging-techniques)
10. [Recovery Procedures](#recovery-procedures)
11. [Log Analysis](#log-analysis)
12. [Preventive Measures](#preventive-measures)

## Quick Diagnostic Checklist

### Initial System Health Check

Run this checklist when experiencing any issues with the platform:

```bash
#!/bin/bash
# quick-health-check.sh - Rapid system diagnostics

echo "=== Multi-Modal Sensor Platform Health Check ==="
echo "Timestamp: $(date)"
echo

# 1. Check if platform service is running
echo "1. Platform Service Status:"
if pgrep -f "pc_controller" > /dev/null; then
    echo "✅ Platform service is running"
    ps aux | grep pc_controller | grep -v grep
else
    echo "❌ Platform service is NOT running"
fi
echo

# 2. Check port availability
echo "2. Port Availability:"
for port in 8080 8443 9001 8765; do
    if netstat -ln | grep ":$port " > /dev/null; then
        echo "✅ Port $port is open"
    else
        echo "❌ Port $port is NOT open"
    fi
done
echo

# 3. Check network connectivity
echo "3. Network Connectivity:"
if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
    echo "✅ Internet connectivity available"
else
    echo "❌ No internet connectivity"
fi
echo

# 4. Check disk space
echo "4. Disk Space:"
df -h | grep -E "/$|/app|/var" | while read line; do
    usage=$(echo $line | awk '{print $5}' | sed 's/%//')
    mount=$(echo $line | awk '{print $6}')
    if [ "$usage" -gt 90 ]; then
        echo "❌ Disk space critical on $mount: ${usage}%"
    elif [ "$usage" -gt 80 ]; then
        echo "⚠️  Disk space warning on $mount: ${usage}%"
    else
        echo "✅ Disk space OK on $mount: ${usage}%"
    fi
done
echo

# 5. Check memory usage
echo "5. Memory Usage:"
mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
if (( $(echo "$mem_usage > 90" | bc -l) )); then
    echo "❌ Memory usage critical: ${mem_usage}%"
elif (( $(echo "$mem_usage > 80" | bc -l) )); then
    echo "⚠️  Memory usage warning: ${mem_usage}%"
else
    echo "✅ Memory usage OK: ${mem_usage}%"
fi
echo

# 6. Check log files for recent errors
echo "6. Recent Error Log Check:"
if [ -f "/app/logs/platform.log" ]; then
    error_count=$(tail -n 100 /app/logs/platform.log | grep -i "error\|exception\|failed" | wc -l)
    if [ "$error_count" -gt 0 ]; then
        echo "⚠️  Found $error_count recent errors in logs"
        echo "Recent errors:"
        tail -n 100 /app/logs/platform.log | grep -i "error\|exception\|failed" | tail -5
    else
        echo "✅ No recent errors in logs"
    fi
else
    echo "❌ Log file not found"
fi
echo

# 7. Check TLS certificate validity
echo "7. TLS Certificate Check:"
if [ -f "/etc/ssl/certs/server.crt" ]; then
    expiry_date=$(openssl x509 -in /etc/ssl/certs/server.crt -noout -enddate | cut -d= -f2)
    expiry_epoch=$(date -d "$expiry_date" +%s)
    current_epoch=$(date +%s)
    days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))

    if [ "$days_until_expiry" -lt 7 ]; then
        echo "❌ TLS certificate expires in $days_until_expiry days"
    elif [ "$days_until_expiry" -lt 30 ]; then
        echo "⚠️  TLS certificate expires in $days_until_expiry days"
    else
        echo "✅ TLS certificate valid for $days_until_expiry days"
    fi
else
    echo "❌ TLS certificate file not found"
fi
echo

echo "=== Health Check Complete ==="
```

### Device Connection Quick Test

```python
# device_connection_test.py - Quick device connectivity test
import socket
import json
import time
import sys

def test_device_connection(device_ip, device_port=8080):
    """Test basic connectivity to Android device"""
    print(f"Testing connection to {device_ip}:{device_port}")

    try:
        # Test basic TCP connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((device_ip, device_port))

        if result == 0:
            print("✅ TCP connection successful")

            # Test device query
            query = {
                "v": 1,
                "type": "query_capabilities",
                "timestamp_ns": time.time_ns()
            }

            query_json = json.dumps(query) + "\n"
            sock.send(query_json.encode())

            # Try to receive response
            sock.settimeout(10)
            response = sock.recv(1024)

            if response:
                print("✅ Device responded to query")
                try:
                    response_data = json.loads(response.decode().strip())
                    print(f"Device ID: {response_data.get('device_id', 'Unknown')}")
                    print(f"Status: {response_data.get('status', 'Unknown')}")
                except json.JSONDecodeError:
                    print("⚠️  Received non-JSON response")
            else:
                print("❌ No response from device")
        else:
            print(f"❌ TCP connection failed: {result}")

    except socket.timeout:
        print("❌ Connection timeout")
    except Exception as e:
        print(f"❌ Connection error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python device_connection_test.py <device_ip>")
        sys.exit(1)

    device_ip = sys.argv[1]
    test_device_connection(device_ip)
```

## Connection and Network Issues

### Issue: Devices Cannot Connect to Hub

#### Symptoms
- Android devices show "Connection Failed" or similar errors
- Hub shows no incoming connections
- Network timeouts during connection attempts

#### Diagnostic Steps

```bash
# 1. Verify hub is listening on correct ports
netstat -tlnp | grep -E ":8080|:8443|:9001|:8765"

# 2. Check if firewall is blocking connections
sudo iptables -L -n | grep -E "8080|8443|9001|8765"

# 3. Test port accessibility from device network
# Run from a device on the same network
telnet <hub_ip> 8080
telnet <hub_ip> 8443

# 4. Check network routing
traceroute <hub_ip>
ping <hub_ip>

# 5. Verify DNS resolution (if using hostnames)
nslookup sensor-hub.local
```

#### Solutions

**1. Firewall Configuration**
```bash
# Ubuntu/Debian
sudo ufw allow 8080/tcp
sudo ufw allow 8443/tcp
sudo ufw allow 9001/tcp
sudo ufw allow 8765/udp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=8443/tcp
sudo firewall-cmd --permanent --add-port=9001/tcp
sudo firewall-cmd --permanent --add-port=8765/udp
sudo firewall-cmd --reload
```

**2. Network Interface Binding**
```python
# Check if hub is binding to correct interface
# In pc_controller/src/main.py, ensure:
server_socket.bind(('0.0.0.0', 8080))  # Listen on all interfaces
# Not:
# server_socket.bind(('127.0.0.1', 8080))  # Only localhost
```

**3. Router/Switch Configuration**
```bash
# Ensure VLAN configuration allows inter-VLAN communication
# Check switch configuration for port isolation
# Verify that device and hub networks can communicate
```

### Issue: Intermittent Connection Drops

#### Symptoms
- Devices connect successfully but disconnect randomly
- Connection drops during data transfer
- Heartbeat timeouts

#### Diagnostic Tools

```python
# connection_monitor.py - Monitor connection stability
import socket
import time
import threading
import json
from datetime import datetime

class ConnectionMonitor:
    def __init__(self, hub_ip, hub_port=8080):
        self.hub_ip = hub_ip
        self.hub_port = hub_port
        self.connection_stats = {
            'successful_connections': 0,
            'failed_connections': 0,
            'disconnections': 0,
            'connection_durations': []
        }

    def test_connection_stability(self, duration_minutes=10, test_interval=30):
        """Test connection stability over time"""
        print(f"Testing connection stability for {duration_minutes} minutes")
        end_time = time.time() + (duration_minutes * 60)

        while time.time() < end_time:
            start_time = time.time()
            try:
                # Attempt connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((self.hub_ip, self.hub_port))

                self.connection_stats['successful_connections'] += 1
                print(f"{datetime.now()}: ✅ Connection successful")

                # Keep connection alive and monitor
                self._monitor_connection(sock, start_time)

            except Exception as e:
                self.connection_stats['failed_connections'] += 1
                print(f"{datetime.now()}: ❌ Connection failed: {e}")

            time.sleep(test_interval)

        self._print_stats()

    def _monitor_connection(self, sock, start_time):
        """Monitor an active connection"""
        try:
            # Send periodic keep-alive messages
            for i in range(5):  # Monitor for ~25 seconds
                message = {
                    "v": 1,
                    "type": "ping",
                    "timestamp_ns": time.time_ns()
                }
                sock.send(json.dumps(message).encode() + b'\n')

                # Try to receive response
                sock.settimeout(5)
                response = sock.recv(1024)
                if not response:
                    break

                time.sleep(5)

            # Connection survived monitoring period
            duration = time.time() - start_time
            self.connection_stats['connection_durations'].append(duration)

        except Exception as e:
            self.connection_stats['disconnections'] += 1
            print(f"{datetime.now()}: ⚠️  Connection dropped: {e}")
        finally:
            sock.close()

    def _print_stats(self):
        """Print connection statistics"""
        stats = self.connection_stats
        print("\n=== Connection Stability Report ===")
        print(f"Successful connections: {stats['successful_connections']}")
        print(f"Failed connections: {stats['failed_connections']}")
        print(f"Unexpected disconnections: {stats['disconnections']}")

        if stats['connection_durations']:
            avg_duration = sum(stats['connection_durations']) / len(stats['connection_durations'])
            print(f"Average connection duration: {avg_duration:.2f} seconds")
            print(f"Max connection duration: {max(stats['connection_durations']):.2f} seconds")
            print(f"Min connection duration: {min(stats['connection_durations']):.2f} seconds")

if __name__ == "__main__":
    monitor = ConnectionMonitor("192.168.1.100")
    monitor.test_connection_stability(duration_minutes=5, test_interval=10)
```

#### Solutions

**1. Network Infrastructure Issues**
```bash
# Check for network congestion
iftop -i eth0  # Monitor network traffic

# Check for packet loss
ping -c 100 <hub_ip> | tail -2

# Monitor network interface errors
watch -n 1 'cat /proc/net/dev | grep eth0'

# Check WiFi signal strength (Android devices)
# Use Android WiFi Analyzer or similar tools
```

**2. Power Management Issues**
```kotlin
// Android: Prevent WiFi sleep
// In AndroidManifest.xml:
<uses-permission android:name="android.permission.WAKE_LOCK" />

// In your service:
private fun acquireWakeLock() {
    val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
    wakeLock = powerManager.newWakeLock(
        PowerManager.PARTIAL_WAKE_LOCK,
        "SensorPlatform::WakeLock"
    )
    wakeLock?.acquire(10*60*1000L /*10 minutes*/)
}

// WiFi lock to prevent WiFi sleep
private fun acquireWifiLock() {
    val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
    wifiLock = wifiManager.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "SensorPlatform")
    wifiLock?.acquire()
}
```

**3. TCP Keep-Alive Configuration**
```python
# Python socket configuration for better connection stability
import socket

def configure_socket_keepalive(sock):
    """Configure TCP keep-alive for better connection stability"""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # Platform-specific keep-alive settings
    import platform
    if platform.system() == 'Linux':
        # TCP_KEEPIDLE: seconds before starting keep-alive probes
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        # TCP_KEEPINTVL: interval between keep-alive probes
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        # TCP_KEEPCNT: number of failed probes before declaring dead
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)
    elif platform.system() == 'Windows':
        # Windows uses different socket options
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60000, 10000))
```

### Issue: High Latency or Slow Data Transfer

#### Symptoms
- Slow file transfers between devices and hub
- High latency in real-time data streaming
- Timeouts during large data operations

#### Diagnostic Tools

```python
# network_performance_test.py - Network performance diagnostics
import socket
import time
import threading
import statistics

class NetworkPerformanceTest:
    def __init__(self, hub_ip, hub_port=9001):
        self.hub_ip = hub_ip
        self.hub_port = hub_port

    def test_throughput(self, data_size_mb=10, chunk_size=8192):
        """Test network throughput"""
        print(f"Testing throughput with {data_size_mb}MB of data")

        # Generate test data
        test_data = b'A' * chunk_size
        total_bytes = data_size_mb * 1024 * 1024
        chunks_to_send = total_bytes // chunk_size

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.hub_ip, self.hub_port))

            start_time = time.time()
            bytes_sent = 0

            for i in range(chunks_to_send):
                bytes_sent += sock.send(test_data)

                # Progress reporting
                if i % 100 == 0:
                    progress = (i / chunks_to_send) * 100
                    elapsed = time.time() - start_time
                    current_speed = (bytes_sent / elapsed) / (1024 * 1024)  # MB/s
                    print(f"Progress: {progress:.1f}% - Speed: {current_speed:.2f} MB/s")

            total_time = time.time() - start_time
            throughput = (bytes_sent / total_time) / (1024 * 1024)  # MB/s

            print(f"Throughput test completed:")
            print(f"  Total bytes sent: {bytes_sent:,}")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Throughput: {throughput:.2f} MB/s")

            sock.close()
            return throughput

        except Exception as e:
            print(f"Throughput test failed: {e}")
            return 0

    def test_latency(self, num_tests=100):
        """Test network latency"""
        print(f"Testing latency with {num_tests} ping tests")

        latencies = []

        for i in range(num_tests):
            try:
                start_time = time.time()

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((self.hub_ip, self.hub_port))

                # Send small ping message
                ping_msg = b'PING\n'
                sock.send(ping_msg)

                # Wait for any response
                response = sock.recv(4)

                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)

                sock.close()

                if i % 10 == 0:
                    print(f"Ping {i+1}/{num_tests}: {latency:.2f} ms")

            except Exception as e:
                print(f"Ping {i+1} failed: {e}")

        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0

            print(f"Latency test results:")
            print(f"  Average: {avg_latency:.2f} ms")
            print(f"  Minimum: {min_latency:.2f} ms")
            print(f"  Maximum: {max_latency:.2f} ms")
            print(f"  Std Dev: {std_dev:.2f} ms")
            print(f"  Success rate: {len(latencies)}/{num_tests} ({len(latencies)/num_tests*100:.1f}%)")

        return latencies

if __name__ == "__main__":
    tester = NetworkPerformanceTest("192.168.1.100")

    # Test latency first
    latencies = tester.test_latency(50)

    # Test throughput
    throughput = tester.test_throughput(5)  # 5MB test
```

#### Solutions

**1. Network Optimization**
```bash
# Increase network buffer sizes
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' >> /etc/sysctl.conf
sysctl -p

# Optimize TCP congestion control
echo 'net.ipv4.tcp_congestion_control = bbr' >> /etc/sysctl.conf
```

**2. Application-Level Optimization**
```python
# Optimize socket buffer sizes
def optimize_socket_buffers(sock, buffer_size=65536):
    """Optimize socket buffer sizes for better performance"""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)

    # Disable Nagle's algorithm for real-time data
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
```

**3. QoS Configuration**
```bash
# Traffic shaping with tc (Traffic Control)
# Prioritize platform traffic

# Create classes
tc qdisc add dev eth0 root handle 1: htb default 30

# High priority for platform control traffic
tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit
tc class add dev eth0 parent 1:1 classid 1:10 htb rate 20mbit ceil 50mbit prio 1
tc class add dev eth0 parent 1:1 classid 1:20 htb rate 60mbit ceil 80mbit prio 2
tc class add dev eth0 parent 1:1 classid 1:30 htb rate 20mbit ceil 30mbit prio 3

# Filter platform traffic to high priority
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dport 8080 0xffff flowid 1:10
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dport 8443 0xffff flowid 1:10
tc filter add dev eth0 protocol ip parent 1:0 prio 2 u32 match ip dport 9001 0xffff flowid 1:20
```

## TLS and Security Issues

### Issue: TLS Certificate Validation Failures

#### Symptoms
- "Certificate verify failed" errors
- Clients unable to establish secure connections
- SSL handshake failures

#### Diagnostic Commands

```bash
# 1. Check certificate validity
openssl x509 -in /etc/ssl/certs/server.crt -text -noout

# 2. Verify certificate chain
openssl verify -CAfile /etc/ssl/certs/ca.crt /etc/ssl/certs/server.crt

# 3. Check certificate expiration
openssl x509 -in /etc/ssl/certs/server.crt -noout -dates

# 4. Test TLS connection
openssl s_client -connect localhost:8443 -CAfile /etc/ssl/certs/ca.crt

# 5. Check private key matches certificate
cert_modulus=$(openssl x509 -noout -modulus -in /etc/ssl/certs/server.crt | openssl md5)
key_modulus=$(openssl rsa -noout -modulus -in /etc/ssl/private/server.key | openssl md5)
echo "Certificate: $cert_modulus"
echo "Private Key: $key_modulus"
```

#### Advanced TLS Diagnostics

```python
# tls_diagnostics.py - Comprehensive TLS testing
import ssl
import socket
import datetime
import OpenSSL.crypto

class TLSDiagnostics:
    def __init__(self, hostname, port=8443):
        self.hostname = hostname
        self.port = port

    def test_tls_connection(self):
        """Test TLS connection and report details"""
        print(f"Testing TLS connection to {self.hostname}:{self.port}")

        try:
            # Create SSL context
            context = ssl.create_default_context()

            # Test connection
            with socket.create_connection((self.hostname, self.port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=self.hostname) as ssock:
                    print("✅ TLS connection successful")

                    # Get certificate information
                    cert = ssock.getpeercert()
                    cert_der = ssock.getpeercert(binary_form=True)

                    self._analyze_certificate(cert, cert_der)
                    self._analyze_connection(ssock)

        except ssl.SSLError as e:
            print(f"❌ SSL Error: {e}")
            self._diagnose_ssl_error(e)
        except socket.error as e:
            print(f"❌ Socket Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")

    def _analyze_certificate(self, cert, cert_der):
        """Analyze certificate details"""
        print("\n--- Certificate Analysis ---")

        # Basic certificate info
        print(f"Subject: {cert.get('subject', 'Unknown')}")
        print(f"Issuer: {cert.get('issuer', 'Unknown')}")
        print(f"Version: {cert.get('version', 'Unknown')}")
        print(f"Serial Number: {cert.get('serialNumber', 'Unknown')}")

        # Validity dates
        not_before = cert.get('notBefore')
        not_after = cert.get('notAfter')

        if not_before and not_after:
            not_before_dt = datetime.datetime.strptime(not_before, '%b %d %H:%M:%S %Y %Z')
            not_after_dt = datetime.datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
            now = datetime.datetime.now()

            print(f"Valid from: {not_before_dt}")
            print(f"Valid until: {not_after_dt}")

            if now < not_before_dt:
                print("❌ Certificate not yet valid")
            elif now > not_after_dt:
                print("❌ Certificate has expired")
            else:
                days_remaining = (not_after_dt - now).days
                print(f"✅ Certificate valid ({days_remaining} days remaining)")

        # Subject Alternative Names
        san_list = cert.get('subjectAltName', [])
        if san_list:
            print("Subject Alternative Names:")
            for san_type, san_value in san_list:
                print(f"  {san_type}: {san_value}")

        # Advanced certificate analysis using pyOpenSSL
        try:
            x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_ASN1, cert_der)

            # Check key size
            public_key = x509.get_pubkey()
            key_size = public_key.bits()
            print(f"Key size: {key_size} bits")

            if key_size < 2048:
                print("⚠️  Key size below recommended minimum (2048 bits)")

            # Check signature algorithm
            sig_alg = x509.get_signature_algorithm().decode()
            print(f"Signature algorithm: {sig_alg}")

            if 'sha1' in sig_alg.lower():
                print("⚠️  SHA-1 signature algorithm is deprecated")

        except ImportError:
            print("pyOpenSSL not available for advanced certificate analysis")
        except Exception as e:
            print(f"Certificate analysis error: {e}")

    def _analyze_connection(self, ssock):
        """Analyze TLS connection details"""
        print("\n--- Connection Analysis ---")

        # TLS version
        version = ssock.version()
        print(f"TLS Version: {version}")

        if version in ['TLSv1', 'TLSv1.1']:
            print("⚠️  TLS version is deprecated")

        # Cipher suite
        cipher = ssock.cipher()
        if cipher:
            cipher_name, cipher_version, cipher_bits = cipher
            print(f"Cipher: {cipher_name}")
            print(f"Cipher version: {cipher_version}")
            print(f"Cipher strength: {cipher_bits} bits")

            # Check for weak ciphers
            weak_ciphers = ['RC4', 'DES', 'NULL', 'EXPORT']
            if any(weak in cipher_name for weak in weak_ciphers):
                print("❌ Weak cipher detected")

        # Certificate chain
        cert_chain = ssock.getpeercert_chain()
        if cert_chain:
            print(f"Certificate chain length: {len(cert_chain)}")

    def _diagnose_ssl_error(self, ssl_error):
        """Provide specific guidance for SSL errors"""
        error_msg = str(ssl_error).lower()

        print("\n--- SSL Error Diagnosis ---")

        if "certificate verify failed" in error_msg:
            print("Certificate verification failed. Possible causes:")
            print("- Certificate has expired")
            print("- Certificate is self-signed without proper CA")
            print("- Hostname doesn't match certificate")
            print("- Certificate chain is incomplete")
            print("- System time is incorrect")

        elif "wrong version number" in error_msg:
            print("TLS version mismatch. Possible causes:")
            print("- Server doesn't support TLS (plain HTTP on HTTPS port)")
            print("- Client and server TLS versions incompatible")
            print("- Firewall or proxy interfering with connection")

        elif "handshake failure" in error_msg:
            print("TLS handshake failed. Possible causes:")
            print("- No compatible cipher suites")
            print("- Server certificate configuration error")
            print("- Client certificate required but not provided")

        elif "connection refused" in error_msg:
            print("Connection refused. Possible causes:")
            print("- Server not running on specified port")
            print("- Firewall blocking connection")
            print("- Incorrect hostname or port")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tls_diagnostics.py <hostname>")
        sys.exit(1)

    hostname = sys.argv[1]
    diagnostics = TLSDiagnostics(hostname)
    diagnostics.test_tls_connection()
```

#### Solutions

**1. Certificate Issues**
```bash
# Regenerate expired certificate
cd /etc/pki/platform-ca
openssl genrsa -out private/server-new.key 2048
openssl req -config openssl.cnf -key private/server-new.key -new -sha256 -out certs/server-new.csr

# Update SAN extensions
cat > certs/server-new.ext << 'EOF'
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = sensor-hub.research.local
DNS.2 = *.sensor-hub.research.local
DNS.3 = localhost
IP.1 = 192.168.1.100
IP.2 = 127.0.0.1
EOF

# Sign new certificate
openssl ca -config openssl.cnf -extensions server_cert -days 365 -notext -md sha256 -in certs/server-new.csr -out certs/server-new.crt

# Backup old certificates and install new ones
cp /etc/ssl/certs/server.crt /etc/ssl/certs/server.crt.backup
cp /etc/ssl/private/server.key /etc/ssl/private/server.key.backup
cp certs/server-new.crt /etc/ssl/certs/server.crt
cp private/server-new.key /etc/ssl/private/server.key

# Restart platform service
systemctl restart sensor-platform
```

**2. Client Certificate Configuration (Android)**
```kotlin
// Android: Install client certificate for mTLS
private fun installClientCertificate() {
    try {
        // Load client certificate from assets
        val certInputStream = assets.open("client.p12")
        val certBytes = certInputStream.readBytes()

        // Install certificate
        val intent = KeyChain.createInstallIntent()
        intent.putExtra(KeyChain.EXTRA_PKCS12, certBytes)
        intent.putExtra(KeyChain.EXTRA_NAME, "Sensor Platform Client")
        startActivityForResult(intent, REQUEST_CERT_INSTALL)

    } catch (e: Exception) {
        Log.e("TLS", "Failed to install client certificate", e)
    }
}

// Custom TrustManager for development (DO NOT USE IN PRODUCTION)
private fun createTrustAllManager(): X509TrustManager {
    return object : X509TrustManager {
        override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) {}
        override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {}
        override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
    }
}

// Proper TrustManager for production
private fun createCustomTrustManager(): X509TrustManager {
    val trustStore = KeyStore.getInstance("BKS")
    val caInputStream = assets.open("ca.crt")

    // Load CA certificate
    val cf = CertificateFactory.getInstance("X.509")
    val caCert = cf.generateCertificate(caInputStream) as X509Certificate

    trustStore.load(null, null)
    trustStore.setCertificateEntry("ca", caCert)

    val tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
    tmf.init(trustStore)

    return tmf.trustManagers[0] as X509TrustManager
}
```

### Issue: TLS Performance Problems

#### Symptoms
- Slow TLS handshakes
- High CPU usage during TLS operations
- Connection timeouts

#### Performance Optimization

```python
# tls_performance_optimizer.py
import ssl
import time
import cProfile
import pstats

class TLSPerformanceOptimizer:
    def __init__(self):
        self.optimization_results = {}

    def create_optimized_context(self, server_side=True):
        """Create performance-optimized SSL context"""
        if server_side:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        # Performance optimizations
        context.options |= ssl.OP_NO_COMPRESSION  # Disable compression for speed
        context.options |= ssl.OP_SINGLE_DH_USE   # Security and performance
        context.options |= ssl.OP_SINGLE_ECDH_USE # Security and performance

        # Fast cipher suites (prioritize ECDHE for perfect forward secrecy)
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS")

        # Set minimum TLS version
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        # Enable session resumption for clients
        if not server_side:
            context.options |= ssl.OP_NO_TICKET  # Use session IDs instead of tickets

        return context

    def benchmark_tls_handshake(self, hostname, port, num_connections=100):
        """Benchmark TLS handshake performance"""
        print(f"Benchmarking TLS handshake performance ({num_connections} connections)")

        context = self.create_optimized_context(server_side=False)
        handshake_times = []

        for i in range(num_connections):
            try:
                start_time = time.perf_counter()

                with socket.create_connection((hostname, port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        handshake_time = time.perf_counter() - start_time
                        handshake_times.append(handshake_time)

                if i % 10 == 0:
                    avg_so_far = sum(handshake_times) / len(handshake_times)
                    print(f"Progress: {i+1}/{num_connections}, Avg: {avg_so_far*1000:.2f}ms")

            except Exception as e:
                print(f"Connection {i+1} failed: {e}")

        if handshake_times:
            avg_time = sum(handshake_times) / len(handshake_times)
            min_time = min(handshake_times)
            max_time = max(handshake_times)

            print(f"TLS Handshake Performance:")
            print(f"  Average: {avg_time*1000:.2f}ms")
            print(f"  Minimum: {min_time*1000:.2f}ms")
            print(f"  Maximum: {max_time*1000:.2f}ms")
            print(f"  Success rate: {len(handshake_times)}/{num_connections}")

            return avg_time

        return None

    def profile_tls_operations(self, hostname, port):
        """Profile TLS operations to identify bottlenecks"""
        print("Profiling TLS operations...")

        profiler = cProfile.Profile()
        profiler.enable()

        # Perform various TLS operations
        context = self.create_optimized_context(server_side=False)

        try:
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    # Send some data
                    ssock.send(b"GET /health HTTP/1.1\r\nHost: " + hostname.encode() + b"\r\n\r\n")
                    response = ssock.recv(1024)

        except Exception as e:
            print(f"Profiling failed: {e}")

        profiler.disable()

        # Analyze profile
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')

        print("Top TLS operations by time:")
        stats.print_stats(10)

        return stats

if __name__ == "__main__":
    optimizer = TLSPerformanceOptimizer()

    # Benchmark handshake performance
    avg_time = optimizer.benchmark_tls_handshake("localhost", 8443, 50)

    # Profile operations
    if avg_time:
        optimizer.profile_tls_operations("localhost", 8443)
```

#### Solutions

**1. Server-Side Optimizations**
```python
# Optimized TLS server configuration
def create_high_performance_tls_server():
    """Create high-performance TLS server"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # Load certificates
    context.load_cert_chain('/etc/ssl/certs/server.crt', '/etc/ssl/private/server.key')

    # Performance optimizations
    context.options |= ssl.OP_NO_COMPRESSION
    context.options |= ssl.OP_SINGLE_DH_USE
    context.options |= ssl.OP_SINGLE_ECDH_USE

    # Use ECDH for better performance
    context.set_ecdh_curve('prime256v1')

    # Fast cipher suites
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!DSS')

    # Enable session caching
    context.set_session_cache_size(1000)

    return context
```

**2. Session Resumption**
```python
# Enable TLS session resumption
class TLSSessionCache:
    def __init__(self, max_size=1000, timeout=3600):
        self.cache = {}
        self.max_size = max_size
        self.timeout = timeout

    def get_session(self, session_id):
        """Get cached session"""
        if session_id in self.cache:
            session_data, timestamp = self.cache[session_id]
            if time.time() - timestamp < self.timeout:
                return session_data
            else:
                del self.cache[session_id]
        return None

    def store_session(self, session_id, session_data):
        """Store session in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest session
            oldest_id = min(self.cache.keys(),
                          key=lambda k: self.cache[k][1])
            del self.cache[oldest_id]

        self.cache[session_id] = (session_data, time.time())
```

## Heartbeat and Fault Tolerance Issues

### Issue: Devices Not Sending Heartbeats

#### Symptoms
- Devices show as "UNKNOWN" status in hub
- No heartbeat messages in logs
- Devices appear offline despite being connected

#### Diagnostic Steps

```python
# heartbeat_diagnostics.py - Comprehensive heartbeat testing
import json
import time
import socket
import threading
from datetime import datetime

class HeartbeatDiagnostics:
    def __init__(self, hub_ip, hub_port=8080):
        self.hub_ip = hub_ip
        self.hub_port = hub_port
        self.received_heartbeats = []
        self.listening = False

    def simulate_android_heartbeat(self, device_id="test_device"):
        """Simulate Android device sending heartbeats"""
        print(f"Simulating heartbeat from {device_id}")

        heartbeat = {
            "v": 1,
            "type": "heartbeat",
            "timestamp_ns": time.time_ns(),
            "device_id": device_id,
            "device_info": {
                "battery_level": 85,
                "available_storage_mb": 2048,
                "is_recording": False,
                "uptime_ms": 3600000,
                "network_type": "wifi"
            }
        }

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((self.hub_ip, self.hub_port))

            message = json.dumps(heartbeat) + "\n"
            sock.send(message.encode())

            # Wait for response
            response = sock.recv(1024)
            if response:
                print(f"✅ Hub responded: {response.decode().strip()}")
            else:
                print("❌ No response from hub")

            sock.close()
            return True

        except Exception as e:
            print(f"❌ Failed to send heartbeat: {e}")
            return False

    def listen_for_heartbeats(self, duration=60):
        """Listen for incoming heartbeat messages"""
        print(f"Listening for heartbeats for {duration} seconds...")

        self.listening = True
        self.received_heartbeats = []

        def listener():
            try:
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind(('0.0.0.0', 8081))  # Different port to avoid conflicts
                server_sock.listen(5)
                server_sock.settimeout(1)  # Short timeout for checking self.listening

                print("Heartbeat listener started on port 8081")

                while self.listening:
                    try:
                        client_sock, addr = server_sock.accept()
                        data = client_sock.recv(1024)

                        if data:
                            try:
                                message = json.loads(data.decode().strip())
                                if message.get('type') == 'heartbeat':
                                    self.received_heartbeats.append({
                                        'timestamp': datetime.now(),
                                        'source': addr[0],
                                        'message': message
                                    })
                                    print(f"Received heartbeat from {addr[0]}: {message.get('device_id')}")
                            except json.JSONDecodeError:
                                print(f"Received non-JSON data from {addr[0]}")

                        client_sock.close()

                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.listening:
                            print(f"Listener error: {e}")

                server_sock.close()

            except Exception as e:
                print(f"Failed to start heartbeat listener: {e}")

        listener_thread = threading.Thread(target=listener)
        listener_thread.start()

        # Wait for specified duration
        time.sleep(duration)
        self.listening = False
        listener_thread.join()

        print(f"Heartbeat listening complete. Received {len(self.received_heartbeats)} heartbeats")

        for hb in self.received_heartbeats:
            device_id = hb['message'].get('device_id', 'Unknown')
            battery = hb['message'].get('device_info', {}).get('battery_level', 'Unknown')
            print(f"  {hb['timestamp']}: {device_id} (Battery: {battery}%)")

    def test_heartbeat_format(self):
        """Test heartbeat message format validation"""
        print("Testing heartbeat message formats...")

        # Valid heartbeat
        valid_heartbeat = {
            "v": 1,
            "type": "heartbeat",
            "timestamp_ns": time.time_ns(),
            "device_id": "test_device",
            "device_info": {
                "battery_level": 75,
                "available_storage_mb": 1024,
                "is_recording": True,
                "uptime_ms": 1800000,
                "network_type": "cellular"
            }
        }

        print("Valid heartbeat test:")
        self._send_test_heartbeat(valid_heartbeat)

        # Test invalid formats
        invalid_tests = [
            # Missing required fields
            {
                "v": 1,
                "type": "heartbeat",
                "device_id": "test_device"
                # Missing timestamp_ns and device_info
            },
            # Invalid version
            {
                "v": 2,
                "type": "heartbeat",
                "timestamp_ns": time.time_ns(),
                "device_id": "test_device",
                "device_info": {"battery_level": 75}
            },
            # Invalid device_info
            {
                "v": 1,
                "type": "heartbeat",
                "timestamp_ns": time.time_ns(),
                "device_id": "test_device",
                "device_info": {
                    "battery_level": 150,  # Invalid battery level
                    "available_storage_mb": -100  # Invalid storage
                }
            }
        ]

        for i, invalid_hb in enumerate(invalid_tests, 1):
            print(f"Invalid heartbeat test {i}:")
            self._send_test_heartbeat(invalid_hb)

    def _send_test_heartbeat(self, heartbeat):
        """Send a test heartbeat and check response"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.hub_ip, self.hub_port))

            message = json.dumps(heartbeat) + "\n"
            sock.send(message.encode())

            response = sock.recv(1024)
            if response:
                print(f"  Response: {response.decode().strip()}")
            else:
                print("  No response received")

            sock.close()

        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    diagnostics = HeartbeatDiagnostics("192.168.1.100")

    # Test sending heartbeat
    print("=== Testing Heartbeat Transmission ===")
    diagnostics.simulate_android_heartbeat("diagnostic_device")

    # Test message formats
    print("\n=== Testing Message Formats ===")
    diagnostics.test_heartbeat_format()

    # Listen for heartbeats (uncomment to test)
    # print("\n=== Listening for Heartbeats ===")
    # diagnostics.listen_for_heartbeats(30)
```

#### Solutions

**1. Android Heartbeat Service Issues**
```kotlin
// Fix Android heartbeat service
class HeartbeatService : Service() {
    private var heartbeatTimer: Timer? = null
    private var heartbeatTask: TimerTask? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startHeartbeats()
        return START_STICKY  // Restart if killed
    }

    private fun startHeartbeats() {
        stopHeartbeats()  // Stop any existing timer

        heartbeatTimer = Timer()
        heartbeatTask = object : TimerTask() {
            override fun run() {
                sendHeartbeat()
            }
        }

        // Start immediately, then every 3 seconds
        heartbeatTimer?.scheduleAtFixedRate(heartbeatTask, 0, 3000)

        Log.i("HeartbeatService", "Heartbeat service started")
    }

    private fun sendHeartbeat() {
        try {
            val heartbeat = createHeartbeatMessage()
            val success = networkClient.sendMessage(heartbeat)

            if (!success) {
                Log.w("HeartbeatService", "Failed to send heartbeat, attempting reconnect")
                // Attempt reconnection
                if (networkClient.reconnect()) {
                    Log.i("HeartbeatService", "Reconnected successfully")
                    networkClient.sendMessage(heartbeat)  // Retry
                }
            }

        } catch (e: Exception) {
            Log.e("HeartbeatService", "Heartbeat error", e)
        }
    }

    private fun createHeartbeatMessage(): String {
        val deviceInfo = mapOf(
            "battery_level" to getBatteryLevel(),
            "available_storage_mb" to getAvailableStorageMB(),
            "is_recording" to isRecording(),
            "uptime_ms" to SystemClock.elapsedRealtime(),
            "network_type" to getNetworkType()
        )

        val heartbeat = mapOf(
            "v" to 1,
            "type" to "heartbeat",
            "timestamp_ns" to System.nanoTime(),
            "device_id" to getDeviceId(),
            "device_info" to deviceInfo
        )

        return JSONObject(heartbeat).toString()
    }

    // Implement missing methods
    private fun getBatteryLevel(): Int {
        val batteryManager = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        return batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
    }

    private fun getAvailableStorageMB(): Long {
        val statsFs = StatFs(filesDir.absolutePath)
        return statsFs.availableBytes / (1024 * 1024)
    }

    private fun isRecording(): Boolean {
        // Check if recording service is active
        return RecordingController.isRecording()
    }

    private fun getNetworkType(): String {
        val connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val networkInfo = connectivityManager.activeNetworkInfo
        return when (networkInfo?.type) {
            ConnectivityManager.TYPE_WIFI -> "wifi"
            ConnectivityManager.TYPE_MOBILE -> "cellular"
            else -> "unknown"
        }
    }
}
```

**2. Battery Optimization Issues**
```kotlin
// Handle Android battery optimization
class BatteryOptimizationHelper(private val context: Context) {

    fun requestBatteryOptimizationExemption() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val intent = Intent()
            val packageName = context.packageName
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager

            if (!pm.isIgnoringBatteryOptimizations(packageName)) {
                intent.action = Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
                intent.data = Uri.parse("package:$packageName")
                context.startActivity(intent)
            }
        }
    }

    fun isOptimizationIgnored(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            pm.isIgnoringBatteryOptimizations(context.packageName)
        } else {
            true
        }
    }
}

// In your main activity
private fun setupBatteryOptimization() {
    val batteryHelper = BatteryOptimizationHelper(this)

    if (!batteryHelper.isOptimizationIgnored()) {
        AlertDialog.Builder(this)
            .setTitle("Battery Optimization")
            .setMessage("To ensure reliable heartbeat transmission, please disable battery optimization for this app.")
            .setPositiveButton("Settings") { _, _ ->
                batteryHelper.requestBatteryOptimizationExemption()
            }
            .setNegativeButton("Skip", null)
            .show()
    }
}
```

### Issue: Frequent False Disconnections

#### Symptoms
- Devices marked as offline despite being connected
- Rapid online/offline status changes
- Heartbeat timeouts despite network connectivity

#### Solutions

**1. Adjust Timeout Configuration**
```python
# Optimized heartbeat manager configuration
class OptimizedHeartbeatManager(HeartbeatManager):
    def __init__(self):
        super().__init__(
            heartbeat_interval_s=5.0,        # Longer interval for stability
            timeout_multiplier=4,            # More lenient timeout
            max_reconnect_attempts=3,        # Fewer attempts to avoid spam
            reconnect_base_delay_s=2.0,      # Longer base delay
            reconnect_max_delay_s=60.0       # Longer max delay
        )

    def is_device_offline(self, device_id: str) -> bool:
        """More conservative offline detection"""
        status = self.get_device_status(device_id)
        if not status or not status.last_heartbeat_time:
            return False  # Don't mark as offline if never received heartbeat

        time_since_last = time.time() - status.last_heartbeat_time
        grace_period = self.heartbeat_interval_s * self.timeout_multiplier * 1.5  # Extra grace

        return time_since_last > grace_period
```

**2. Network Quality Assessment**
```python
# Adaptive heartbeat timing based on network quality
class AdaptiveHeartbeatManager(HeartbeatManager):
    def __init__(self):
        super().__init__()
        self.network_quality_samples = {}
        self.quality_window_size = 10

    def process_heartbeat(self, message: dict) -> bool:
        """Process heartbeat with network quality tracking"""
        device_id = message.get('device_id')
        if not device_id:
            return False

        # Calculate network delay
        received_time = time.time_ns()
        sent_time = message.get('timestamp_ns', received_time)
        network_delay = (received_time - sent_time) / 1_000_000  # Convert to ms

        # Track network quality
        self._update_network_quality(device_id, network_delay)

        # Adjust timeout based on network quality
        self._adjust_timeout_for_device(device_id)

        return super().process_heartbeat(message)

    def _update_network_quality(self, device_id: str, delay_ms: float):
        """Update network quality metrics for device"""
        if device_id not in self.network_quality_samples:
            self.network_quality_samples[device_id] = []

        samples = self.network_quality_samples[device_id]
        samples.append(delay_ms)

        # Keep only recent samples
        if len(samples) > self.quality_window_size:
            samples.pop(0)

    def _adjust_timeout_for_device(self, device_id: str):
        """Adjust timeout based on observed network quality"""
        samples = self.network_quality_samples.get(device_id, [])
        if len(samples) < 3:
            return  # Not enough data

        avg_delay = sum(samples) / len(samples)

        # Adjust timeout multiplier based on network quality
        if avg_delay > 1000:  # High latency network
            multiplier = 6
        elif avg_delay > 500:  # Medium latency
            multiplier = 4
        else:  # Low latency
            multiplier = 3

        # Store per-device timeout (you'd need to modify HeartbeatManager to support this)
        self._device_timeout_multipliers[device_id] = multiplier
```

This comprehensive troubleshooting guide provides systematic diagnostic procedures and solutions for all major issues that can occur with the Multi-Modal Sensor Platform. The guide emphasizes practical, step-by-step approaches to identifying and resolving problems in production environments.
