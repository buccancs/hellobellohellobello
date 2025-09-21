# Production Deployment Guide

## Overview

This comprehensive guide covers the deployment of the Multi-Modal Sensor Platform in production research environments. It includes security hardening, infrastructure setup, monitoring configuration, and operational procedures for enterprise-grade research deployments.

## Table of Contents

1. [Prerequisites and System Requirements](#prerequisites-and-system-requirements)
2. [Security Configuration](#security-configuration)
3. [Infrastructure Setup](#infrastructure-setup)
4. [TLS Certificate Management](#tls-certificate-management)
5. [Network Configuration](#network-configuration)
6. [Application Deployment](#application-deployment)
7. [Database and Storage Configuration](#database-and-storage-configuration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Performance Tuning](#performance-tuning)
11. [Operational Procedures](#operational-procedures)
12. [Maintenance and Updates](#maintenance-and-updates)

## Prerequisites and System Requirements

### Hardware Requirements

#### PC Hub (Server)
- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB recommended for large deployments
- **Storage**: 1TB SSD minimum, RAID 1 configuration recommended
- **Network**: Gigabit Ethernet, WiFi 6 capability
- **GPU**: Optional, NVIDIA GTX 1060 or better for data processing acceleration

#### Android Devices (Sensors)
- **OS**: Android 8.0 (API 26) or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 64GB minimum, 128GB recommended
- **Camera**: 1080p front and rear cameras
- **Network**: WiFi 802.11n or better, cellular backup recommended
- **Battery**: 4000mAh minimum for extended recording sessions

### Software Requirements

#### Operating System Support
- **Primary**: Windows 10/11 Enterprise, Ubuntu 20.04/22.04 LTS
- **Secondary**: macOS 12.0+, CentOS 8, RHEL 8+

#### Dependencies
```bash
# Python Requirements
Python 3.11+
PyQt6 6.5+
numpy 1.24+
pandas 2.0+
asyncio
cryptography 40.0+

# System Dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libssl-dev libffi-dev
sudo apt-get install -y qt6-base-dev qt6-tools-dev
sudo apt-get install -y nginx postgresql-client-14

# System Dependencies (RHEL/CentOS)
sudo dnf install -y python3.11 python3.11-devel
sudo dnf install -y gcc gcc-c++ cmake
sudo dnf install -y openssl-devel libffi-devel
sudo dnf install -y qt6-qtbase-devel
sudo dnf install -y nginx postgresql
```

### Network Requirements

#### Firewall Configuration
```bash
# PC Hub Ports
8080/tcp   # Main application port
8443/tcp   # TLS secure port
9001/tcp   # File transfer port
8765/tcp   # Time sync UDP port

# Android Device Ports
Dynamic outbound connections to hub
5555/tcp   # ADB debugging (development only)
```

#### Bandwidth Requirements
- **Per Device**: 2-5 Mbps sustained for video streaming
- **Total Network**: Scale linearly with device count
- **Recommended**: Gigabit network for 10+ devices

## Security Configuration

### TLS Certificate Setup

#### Production Certificate Authority

Create a dedicated Certificate Authority for the research platform:

```bash
# Create CA directory structure
sudo mkdir -p /etc/pki/platform-ca/{certs,crl,newcerts,private}
sudo chmod 700 /etc/pki/platform-ca/private
cd /etc/pki/platform-ca

# Create CA configuration
cat > openssl.cnf << 'EOF'
[ ca ]
default_ca = CA_default

[ CA_default ]
dir = /etc/pki/platform-ca
certs = $dir/certs
crl_dir = $dir/crl
database = $dir/index.txt
new_certs_dir = $dir/newcerts
certificate = $dir/certs/ca.crt
serial = $dir/serial
crlnumber = $dir/crlnumber
crl = $dir/crl/ca.crl
private_key = $dir/private/ca.key
RANDFILE = $dir/private/.rand

name_opt = ca_default
cert_opt = ca_default
default_days = 365
default_crl_days = 30
default_md = sha256
preserve = no
policy = policy_strict

[ policy_strict ]
countryName = match
stateOrProvinceName = match
organizationName = match
organizationalUnitName = optional
commonName = supplied
emailAddress = optional

[ req ]
default_bits = 4096
distinguished_name = req_distinguished_name
string_mask = utf8only
default_md = sha256
x509_extensions = v3_ca

[ req_distinguished_name ]
countryName = Country Name (2 letter code)
stateOrProvinceName = State or Province Name
localityName = Locality Name
0.organizationName = Organization Name
organizationalUnitName = Organizational Unit Name
commonName = Common Name
emailAddress = Email Address

[ v3_ca ]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer
basicConstraints = critical,CA:true
keyUsage = critical, digitalSignature, cRLSign, keyCertSign

[ server_cert ]
basicConstraints = CA:FALSE
nsCertType = server
nsComment = "Research Platform Server Certificate"
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer:always
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
EOF

# Initialize CA
echo 1000 > serial
touch index.txt
echo 1000 > crlnumber

# Generate CA private key
openssl genrsa -aes256 -out private/ca.key 4096
chmod 400 private/ca.key

# Generate CA certificate
openssl req -config openssl.cnf -key private/ca.key -new -x509 -days 7300 -sha256 -extensions v3_ca -out certs/ca.crt
```

#### Server Certificate Generation

```bash
# Generate server private key
openssl genrsa -out private/server.key 2048
chmod 400 private/server.key

# Create certificate request
openssl req -config openssl.cnf -key private/server.key -new -sha256 -out certs/server.csr

# Create server certificate extensions
cat > certs/server.ext << 'EOF'
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = sensor-hub.research.local
DNS.2 = *.sensor-hub.research.local
DNS.3 = localhost
IP.1 = 192.168.1.100
IP.2 = 10.0.0.100
IP.3 = 127.0.0.1
EOF

# Sign server certificate
openssl ca -config openssl.cnf -extensions server_cert -days 375 -notext -md sha256 -in certs/server.csr -out certs/server.crt

# Verify certificate
openssl x509 -noout -text -in certs/server.crt
openssl verify -CAfile certs/ca.crt certs/server.crt
```

#### Client Certificate Generation (for mTLS)

```bash
# Generate client private key
openssl genrsa -out private/android-client.key 2048

# Create client certificate request
openssl req -config openssl.cnf -key private/android-client.key -new -sha256 -out certs/android-client.csr

# Create client certificate extensions
cat > certs/client.ext << 'EOF'
basicConstraints = CA:FALSE
nsCertType = client, email
nsComment = "Research Platform Android Client Certificate"
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
keyUsage = critical, nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth, emailProtection
EOF

# Sign client certificate
openssl ca -config openssl.cnf -extensions usr_cert -days 375 -notext -md sha256 -in certs/android-client.csr -out certs/android-client.crt

# Create PKCS#12 bundle for Android
openssl pkcs12 -export -out certs/android-client.p12 -inkey private/android-client.key -in certs/android-client.crt -certfile certs/ca.crt
```

### User Authentication and Authorization

#### Database Setup for User Management

```sql
-- PostgreSQL user management schema
CREATE DATABASE platform_users;

-- Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'researcher',
    organization VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP NULL
);

-- Create sessions table
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    session_id VARCHAR(128) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT true
);

-- Create audit log table
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    resource VARCHAR(100),
    details JSONB,
    ip_address INET,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create roles and permissions
CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB
);

INSERT INTO roles (name, description, permissions) VALUES
('admin', 'Full system administrator', '["*"]'),
('researcher', 'Research scientist with recording privileges', '["session.create", "session.manage", "data.view", "data.export"]'),
('analyst', 'Data analyst with read-only access', '["data.view", "data.export"]'),
('viewer', 'Read-only access to completed sessions', '["data.view"]');
```

#### Authentication Implementation

```python
# authentication.py - Production user authentication
import hashlib
import secrets
import hmac
import time
from datetime import datetime, timedelta
import jwt
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash

class UserAuthenticationManager:
    def __init__(self, db_config: dict, jwt_secret: str):
        self.db_config = db_config
        self.jwt_secret = jwt_secret
        self.session_timeout = timedelta(hours=8)
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)

    def authenticate_user(self, username: str, password: str, ip_address: str) -> dict:
        """Authenticate user with rate limiting and lockout"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Check user existence and lockout status
                cur.execute("""
                    SELECT id, username, password_hash, role, failed_login_attempts, locked_until
                    FROM users
                    WHERE username = %s AND is_active = true
                """, (username,))

                user = cur.fetchone()
                if not user:
                    self._log_authentication_attempt(None, username, False, ip_address, "User not found")
                    return {"success": False, "error": "Invalid credentials"}

                user_id, username, password_hash, role, failed_attempts, locked_until = user

                # Check if account is locked
                if locked_until and datetime.now() < locked_until:
                    self._log_authentication_attempt(user_id, username, False, ip_address, "Account locked")
                    return {"success": False, "error": "Account temporarily locked"}

                # Verify password
                if not check_password_hash(password_hash, password):
                    # Increment failed attempts
                    failed_attempts += 1
                    if failed_attempts >= self.max_failed_attempts:
                        lockout_time = datetime.now() + self.lockout_duration
                        cur.execute("""
                            UPDATE users
                            SET failed_login_attempts = %s, locked_until = %s
                            WHERE id = %s
                        """, (failed_attempts, lockout_time, user_id))
                    else:
                        cur.execute("""
                            UPDATE users
                            SET failed_login_attempts = %s
                            WHERE id = %s
                        """, (failed_attempts, user_id))

                    self._log_authentication_attempt(user_id, username, False, ip_address, "Invalid password")
                    return {"success": False, "error": "Invalid credentials"}

                # Successful authentication - reset failed attempts
                cur.execute("""
                    UPDATE users
                    SET failed_login_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (user_id,))

                # Create session token
                session_token = self._create_session_token(user_id, username, role)
                self._store_session(user_id, session_token, ip_address, cur)

                self._log_authentication_attempt(user_id, username, True, ip_address, "Successful login")

                return {
                    "success": True,
                    "token": session_token,
                    "user": {
                        "id": user_id,
                        "username": username,
                        "role": role
                    }
                }

    def _create_session_token(self, user_id: int, username: str, role: str) -> str:
        """Create JWT session token"""
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "iat": time.time(),
            "exp": time.time() + self.session_timeout.total_seconds()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def validate_session(self, token: str) -> dict:
        """Validate session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            # Check if session exists in database
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT user_id FROM user_sessions
                        WHERE session_id = %s AND is_active = true AND expires_at > CURRENT_TIMESTAMP
                    """, (token,))

                    if not cur.fetchone():
                        return {"valid": False, "error": "Session not found or expired"}

            return {"valid": True, "user": payload}

        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"valid": False, "error": "Invalid token"}
```

### Application-Level Security

#### Input Validation and Sanitization

```python
# security_validation.py
import re
import json
import html
from typing import Any, Dict, List
import jsonschema

class SecurityValidator:
    def __init__(self):
        self.session_id_pattern = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
        self.device_id_pattern = re.compile(r'^[a-zA-Z0-9_-]{1,32}$')
        self.filename_pattern = re.compile(r'^[a-zA-Z0-9._-]{1,255}$')

        # JSON schemas for API validation
        self.heartbeat_schema = {
            "type": "object",
            "required": ["v", "type", "timestamp_ns", "device_id", "device_info"],
            "properties": {
                "v": {"type": "integer", "enum": [1]},
                "type": {"type": "string", "enum": ["heartbeat"]},
                "timestamp_ns": {"type": "integer", "minimum": 0},
                "device_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,32}$"},
                "device_info": {
                    "type": "object",
                    "required": ["battery_level", "available_storage_mb", "is_recording"],
                    "properties": {
                        "battery_level": {"type": "integer", "minimum": 0, "maximum": 100},
                        "available_storage_mb": {"type": "integer", "minimum": 0},
                        "is_recording": {"type": "boolean"}
                    }
                }
            },
            "additionalProperties": False
        }

    def validate_session_id(self, session_id: str) -> bool:
        """Validate session ID format"""
        if not session_id or len(session_id) > 64:
            return False
        return bool(self.session_id_pattern.match(session_id))

    def validate_device_id(self, device_id: str) -> bool:
        """Validate device ID format"""
        if not device_id or len(device_id) > 32:
            return False
        return bool(self.device_id_pattern.match(device_id))

    def validate_filename(self, filename: str) -> bool:
        """Validate filename for path traversal protection"""
        if not filename or len(filename) > 255:
            return False
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        return bool(self.filename_pattern.match(filename))

    def validate_heartbeat_message(self, message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate heartbeat message structure"""
        try:
            jsonschema.validate(message, self.heartbeat_schema)
            return True, ""
        except jsonschema.ValidationError as e:
            return False, f"Validation error: {e.message}"

    def sanitize_user_input(self, input_data: str) -> str:
        """Sanitize user input to prevent XSS"""
        if not input_data:
            return ""

        # HTML escape
        sanitized = html.escape(input_data)

        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')

        # Limit length
        return sanitized[:1000]
```

## Infrastructure Setup

### Docker Deployment

#### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  platform-hub:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: sensor-platform-hub
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - PC_TLS_ENABLED=true
      - PC_TLS_CERT_FILE=/etc/ssl/certs/server.crt
      - PC_TLS_KEY_FILE=/etc/ssl/private/server.key
      - PC_TLS_CA_FILE=/etc/ssl/certs/ca.crt
      - DATABASE_URL=postgresql://platform_user:${DB_PASSWORD}@postgres:5432/platform_db
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8080:8080"
      - "8443:8443"
      - "9001:9001"
      - "8765:8765/udp"
    volumes:
      - ./certs:/etc/ssl/certs:ro
      - ./private:/etc/ssl/private:ro
      - platform_data:/app/data
      - platform_logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - platform-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    container_name: platform-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=platform_db
      - POSTGRES_USER=platform_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - platform-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U platform_user -d platform_db"]
      interval: 30s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: platform-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - platform-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 3s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: platform-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./static:/var/www/static:ro
    depends_on:
      - platform-hub
    networks:
      - platform-network

  prometheus:
    image: prom/prometheus:latest
    container_name: platform-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=90d'
      - '--web.enable-lifecycle'
    networks:
      - platform-network

  grafana:
    image: grafana/grafana:latest
    container_name: platform-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_SECURITY_ADMIN_USER=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - platform-network

volumes:
  platform_data:
  platform_logs:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  platform-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### Production Dockerfile

```dockerfile
# Dockerfile.prod
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r platform && useradd -r -g platform -d /app -s /bin/bash platform

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy application code
COPY --chown=platform:platform . .

# Create directories
RUN mkdir -p /app/data /app/logs && chown -R platform:platform /app

# Switch to non-root user
USER platform

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8443 9001 8765

# Start application
CMD ["python", "pc_controller/src/main.py", "--production"]
```

### Kubernetes Deployment

#### Production Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sensor-platform
  labels:
    name: sensor-platform
    environment: production

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: platform-config
  namespace: sensor-platform
data:
  PC_TLS_ENABLED: "true"
  PC_TLS_VERIFY_MODE: "CERT_REQUIRED"
  PC_TLS_CHECK_HOSTNAME: "true"
  PC_TLS_MIN_VERSION: "TLSv1_2"
  PC_HEARTBEAT_INTERVAL_S: "3.0"
  PC_HEARTBEAT_TIMEOUT_MULTIPLIER: "3"
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: platform-secrets
  namespace: sensor-platform
type: Opaque
data:
  database-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  jwt-secret: <base64-encoded-secret>
  grafana-password: <base64-encoded-password>

---
# k8s/tls-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: platform-tls
  namespace: sensor-platform
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-server-cert>
  tls.key: <base64-encoded-server-key>
  ca.crt: <base64-encoded-ca-cert>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: platform-hub
  namespace: sensor-platform
  labels:
    app: platform-hub
    version: v1.0.0
spec:
  replicas: 2
  selector:
    matchLabels:
      app: platform-hub
  template:
    metadata:
      labels:
        app: platform-hub
        version: v1.0.0
    spec:
      serviceAccountName: platform-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: platform-hub
        image: sensor-platform:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8443
          name: https
        - containerPort: 9001
          name: transfer
        - containerPort: 8765
          name: timesync
          protocol: UDP
        env:
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: platform-secrets
              key: database-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: platform-secrets
              key: redis-password
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: platform-secrets
              key: jwt-secret
        envFrom:
        - configMapRef:
            name: platform-config
        volumeMounts:
        - name: tls-certs
          mountPath: /etc/ssl/certs
          readOnly: true
        - name: platform-data
          mountPath: /app/data
        - name: platform-logs
          mountPath: /app/logs
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          capabilities:
            drop:
            - ALL
      volumes:
      - name: tls-certs
        secret:
          secretName: platform-tls
      - name: platform-data
        persistentVolumeClaim:
          claimName: platform-data-pvc
      - name: platform-logs
        persistentVolumeClaim:
          claimName: platform-logs-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: platform-hub-service
  namespace: sensor-platform
  labels:
    app: platform-hub
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
    name: http
  - port: 443
    targetPort: 8443
    name: https
  - port: 9001
    targetPort: 9001
    name: transfer
  - port: 8765
    targetPort: 8765
    protocol: UDP
    name: timesync
  selector:
    app: platform-hub

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: platform-ingress
  namespace: sensor-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTPS"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - sensor-platform.research.local
    secretName: platform-tls-ingress
  rules:
  - host: sensor-platform.research.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: platform-hub-service
            port:
              number: 443
```

## Network Configuration

### Firewall Rules

#### iptables Configuration (Linux)

```bash
#!/bin/bash
# firewall-setup.sh - Production firewall configuration

# Clear existing rules
iptables -F
iptables -X
iptables -Z

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict to management network)
iptables -A INPUT -p tcp --dport 22 -s 192.168.100.0/24 -j ACCEPT

# Allow HTTP/HTTPS (from research network)
iptables -A INPUT -p tcp --dport 80 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -s 192.168.1.0/24 -j ACCEPT

# Allow platform ports (from device network)
iptables -A INPUT -p tcp --dport 8080 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 8443 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 9001 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p udp --dport 8765 -s 192.168.1.0/24 -j ACCEPT

# Allow monitoring (from management network)
iptables -A INPUT -p tcp --dport 9090 -s 192.168.100.0/24 -j ACCEPT  # Prometheus
iptables -A INPUT -p tcp --dport 3000 -s 192.168.100.0/24 -j ACCEPT  # Grafana

# Rate limiting for platform ports
iptables -A INPUT -p tcp --dport 8080 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
iptables -A INPUT -p tcp --dport 8443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# Log dropped packets (sample only)
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables denied: " --log-level 7

# Save rules
iptables-save > /etc/iptables/rules.v4

echo "Firewall configuration applied"
```

#### Windows Defender Firewall

```powershell
# Windows firewall configuration for production
# Run as Administrator

# Remove existing rules
Remove-NetFirewallRule -DisplayName "Sensor Platform*" -ErrorAction SilentlyContinue

# Allow inbound platform ports
New-NetFirewallRule -DisplayName "Sensor Platform HTTP" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Domain,Private
New-NetFirewallRule -DisplayName "Sensor Platform HTTPS" -Direction Inbound -Protocol TCP -LocalPort 8443 -Action Allow -Profile Domain,Private
New-NetFirewallRule -DisplayName "Sensor Platform Transfer" -Direction Inbound -Protocol TCP -LocalPort 9001 -Action Allow -Profile Domain,Private
New-NetFirewallRule -DisplayName "Sensor Platform TimeSync" -Direction Inbound -Protocol UDP -LocalPort 8765 -Action Allow -Profile Domain,Private

# Allow outbound connections
New-NetFirewallRule -DisplayName "Sensor Platform Outbound" -Direction Outbound -Protocol TCP -Action Allow -Profile Domain,Private

# Monitoring ports (restrict to management network)
New-NetFirewallRule -DisplayName "Sensor Platform Prometheus" -Direction Inbound -Protocol TCP -LocalPort 9090 -RemoteAddress 192.168.100.0/24 -Action Allow -Profile Domain,Private
New-NetFirewallRule -DisplayName "Sensor Platform Grafana" -Direction Inbound -Protocol TCP -LocalPort 3000 -RemoteAddress 192.168.100.0/24 -Action Allow -Profile Domain,Private

Write-Host "Windows Firewall configured for Sensor Platform"
```

### Network Segmentation

#### VLAN Configuration

```bash
# VLAN setup for network segmentation
# This should be configured on your managed switches

# VLAN 10: Management Network (192.168.100.0/24)
# - Admin workstations
# - Monitoring systems
# - SSH access

# VLAN 20: Server Network (192.168.1.0/24)
# - PC Hub servers
# - Database servers
# - Application servers

# VLAN 30: Device Network (192.168.2.0/24)
# - Android sensor devices
# - IoT devices
# - Cameras

# VLAN 40: Guest Network (192.168.50.0/24)
# - Visitor devices
# - Internet-only access

# Inter-VLAN routing rules (implement on your router/firewall)
# Allow Management -> All VLANs
# Allow Server -> Device (controlled)
# Allow Device -> Server (specific ports only)
# Deny Device -> Management
# Deny Guest -> All internal VLANs
```

### Load Balancing and High Availability

#### HAProxy Configuration

```cfg
# /etc/haproxy/haproxy.cfg - Load balancer configuration
global
    log stdout local0
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

    # SSL configuration
    ssl-default-bind-ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS
    ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets

defaults
    mode http
    log global
    option httplog
    option dontlognull
    option http-server-close
    option forwardfor except 127.0.0.0/8
    option redispatch
    retries 3
    timeout http-request 10s
    timeout queue 1m
    timeout connect 10s
    timeout client 1m
    timeout server 1m
    timeout http-keep-alive 10s
    timeout check 10s
    maxconn 3000

frontend platform_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/platform.pem

    # Redirect HTTP to HTTPS
    redirect scheme https code 301 if !{ ssl_fc }

    # Security headers
    http-response set-header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
    http-response set-header X-Frame-Options "DENY"
    http-response set-header X-Content-Type-Options "nosniff"
    http-response set-header X-XSS-Protection "1; mode=block"

    # Route to backend
    default_backend platform_backend

backend platform_backend
    balance roundrobin
    option httpchk GET /health

    # Health check configuration
    http-check expect status 200

    # Server definitions
    server hub1 192.168.1.101:8080 check inter 30s fall 3 rise 2
    server hub2 192.168.1.102:8080 check inter 30s fall 3 rise 2 backup

# Statistics page
frontend stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
```

This comprehensive Production Deployment Guide provides detailed procedures for secure, scalable deployment of the Multi-Modal Sensor Platform in enterprise research environments. The guide covers all aspects from infrastructure setup to operational procedures, ensuring successful production deployments with enterprise-grade security and reliability.
