"""Enhanced TLS utilities for secure communication (NFR5)."""

import logging
import os
import socket
import ssl
from pathlib import Path

logger = logging.getLogger(__name__)


class TLSConfig:
    """Configuration for TLS connections."""

    def __init__(
        self,
        enabled: bool = False,
        cert_file: str | None = None,
        key_file: str | None = None,
        ca_file: str | None = None,
        verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED,
        check_hostname: bool = True,
    ):
        """Initialize TLS configuration.

        Args:
            enabled: Whether TLS is enabled
            cert_file: Path to certificate file
            key_file: Path to private key file
            ca_file: Path to CA certificate file for verification
            verify_mode: SSL verification mode
            check_hostname: Whether to verify hostname
        """
        self.enabled = enabled
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file
        self.verify_mode = verify_mode
        self.check_hostname = check_hostname

    @classmethod
    def from_env(cls) -> "TLSConfig":
        """Create TLS configuration from environment variables."""
        return cls(
            enabled=os.getenv("PC_TLS_ENABLED", "false").lower() == "true",
            cert_file=os.getenv("PC_TLS_CERT_FILE"),
            key_file=os.getenv("PC_TLS_KEY_FILE"),
            ca_file=os.getenv("PC_TLS_CA_FILE"),
            verify_mode=(
                ssl.CERT_REQUIRED
                if os.getenv("PC_TLS_VERIFY_CLIENT", "false").lower() == "true"
                else ssl.CERT_NONE
            ),
            check_hostname=os.getenv("PC_TLS_CHECK_HOSTNAME", "true").lower() == "true",
        )

    def validate(self) -> tuple[bool, str]:
        """Validate TLS configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enabled:
            return True, "TLS disabled"

        if not self.cert_file or not self.key_file:
            return False, "TLS enabled but cert_file or key_file not specified"

        cert_path = Path(self.cert_file)
        key_path = Path(self.key_file)

        if not cert_path.exists():
            return False, f"Certificate file not found: {self.cert_file}"

        if not key_path.exists():
            return False, f"Private key file not found: {self.key_file}"

        if self.ca_file:
            ca_path = Path(self.ca_file)
            if not ca_path.exists():
                return False, f"CA file not found: {self.ca_file}"

        return True, "Configuration valid"


class SecureConnectionManager:
    """Manages secure TLS connections for client and server operations."""

    def __init__(self, config: TLSConfig):
        """Initialize with TLS configuration."""
        self.config = config
        self._server_context: ssl.SSLContext | None = None
        self._client_context: ssl.SSLContext | None = None

    def get_server_context(self) -> ssl.SSLContext | None:
        """Get SSL context for server operations."""
        if not self.config.enabled:
            return None

        if self._server_context is None:
            self._server_context = self._create_server_context()

        return self._server_context

    def get_client_context(self) -> ssl.SSLContext | None:
        """Get SSL context for client operations."""
        if not self.config.enabled:
            return None

        if self._client_context is None:
            self._client_context = self._create_client_context()

        return self._client_context

    def wrap_server_socket(
        self, sock: socket.socket, server_hostname: str | None = None
    ) -> socket.socket | ssl.SSLSocket:
        """Wrap a server socket with TLS if enabled.

        Args:
            sock: The socket to wrap
            server_hostname: Server hostname for SNI

        Returns:
            Original socket if TLS disabled, wrapped SSLSocket if enabled
        """
        context = self.get_server_context()
        if context is None:
            return sock

        try:
            wrapped = context.wrap_socket(sock, server_side=True)
            logger.info("Server socket wrapped with TLS")
            return wrapped
        except Exception as e:
            logger.error(f"Failed to wrap server socket with TLS: {e}")
            raise

    def wrap_client_socket(
        self, sock: socket.socket, server_hostname: str | None = None
    ) -> socket.socket | ssl.SSLSocket:
        """Wrap a client socket with TLS if enabled.

        Args:
            sock: The socket to wrap
            server_hostname: Server hostname for verification

        Returns:
            Original socket if TLS disabled, wrapped SSLSocket if enabled
        """
        context = self.get_client_context()
        if context is None:
            return sock

        try:
            wrapped = context.wrap_socket(
                sock,
                server_side=False,
                server_hostname=server_hostname if self.config.check_hostname else None,
            )
            logger.info(f"Client socket wrapped with TLS (hostname: {server_hostname})")
            return wrapped
        except Exception as e:
            logger.error(f"Failed to wrap client socket with TLS: {e}")
            raise

    def connect_secure(
        self, host: str, port: int, timeout: float = 10.0
    ) -> socket.socket | ssl.SSLSocket:
        """Create a secure connection to a server.

        Args:
            host: Server hostname
            port: Server port
            timeout: Connection timeout

        Returns:
            Connected socket (wrapped with TLS if enabled)
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((host, port))
            return self.wrap_client_socket(sock, server_hostname=host)
        except Exception as e:
            sock.close()
            logger.error(f"Failed to connect securely to {host}:{port}: {e}")
            raise

    def _create_server_context(self) -> ssl.SSLContext:
        """Create SSL context for server operations."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        context.load_cert_chain(self.config.cert_file, self.config.key_file)

        # Configure client verification
        if self.config.verify_mode != ssl.CERT_NONE and self.config.ca_file:
            context.load_verify_locations(cafile=self.config.ca_file)
            context.verify_mode = self.config.verify_mode
        else:
            context.verify_mode = ssl.CERT_NONE
            logger.warning("Client certificate verification disabled")

        # Security settings
        context.check_hostname = False
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        context.set_ciphers(
            "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
        )

        logger.info("Server SSL context created successfully")
        return context

    def _create_client_context(self) -> ssl.SSLContext:
        """Create SSL context for client operations."""
        context = ssl.create_default_context()

        # Configure hostname checking
        context.check_hostname = self.config.check_hostname
        context.verify_mode = self.config.verify_mode

        if self.config.ca_file:
            context.load_verify_locations(cafile=self.config.ca_file)

        if self.config.cert_file and self.config.key_file:
            context.load_cert_chain(self.config.cert_file, self.config.key_file)
            logger.info("Client certificate loaded for mutual TLS")

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        logger.info("Client SSL context created successfully")
        return context


def generate_self_signed_cert(
    cert_file: str, key_file: str, hostname: str = "localhost", days: int = 365
) -> None:
    """Generate a self-signed certificate for testing/development.

    Args:
        cert_file: Path for certificate output
        key_file: Path for private key output
        hostname: Hostname for the certificate
        days: Certificate validity in days
    """
    try:
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        msg = "cryptography library required for certificate generation. "
        msg += "Install with: pip install cryptography"
        logger.error(msg)
        raise

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
            x509.NameAttribute(
                NameOID.ORGANIZATION_NAME, "Multi-Modal Sensor Platform"
            ),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Development"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(
            datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=days)
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    with open(key_file, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"Self-signed certificate generated: {cert_file}, {key_file}")


class SecureMessageHandler:
    """Handles secure message transmission over TLS connections."""

    def __init__(self, connection_manager: SecureConnectionManager):
        """Initialize with connection manager."""
        self.connection_manager = connection_manager

    def send_secure_message(
        self, sock: socket.socket | ssl.SSLSocket, message: bytes
    ) -> bool:
        """Send a message securely over the socket.

        Args:
            sock: Socket to send on
            message: Message to send

        Returns:
            True if successful, False otherwise
        """
        try:
            length = len(message)
            length_bytes = length.to_bytes(4, "big")

            sock.sendall(length_bytes)
            sock.sendall(message)

            if isinstance(sock, ssl.SSLSocket):
                cipher = sock.cipher()
                logger.debug(f"Secure message sent using {cipher[0]} ({cipher[1]})")
            else:
                logger.debug("Message sent over plain socket")

            return True
        except Exception as e:
            logger.error(f"Failed to send secure message: {e}")
            return False

    def receive_secure_message(
        self, sock: socket.socket | ssl.SSLSocket, timeout: float = 10.0
    ) -> bytes | None:
        """Receive a message securely from the socket.

        Args:
            sock: Socket to receive from
            timeout: Receive timeout

        Returns:
            Received message or None if failed
        """
        try:
            sock.settimeout(timeout)

            length_bytes = self._receive_exactly(sock, 4)
            if not length_bytes:
                return None

            length = int.from_bytes(length_bytes, "big")
            if length > 10 * 1024 * 1024:
                logger.error(f"Message too large: {length} bytes")
                return None

            message = self._receive_exactly(sock, length)

            if isinstance(sock, ssl.SSLSocket):
                cipher = sock.cipher()
                logger.debug(f"Secure message received using {cipher[0]} ({cipher[1]})")
            else:
                logger.debug("Message received over plain socket")

            return message
        except Exception as e:
            logger.error(f"Failed to receive secure message: {e}")
            return None

    def _receive_exactly(
        self, sock: socket.socket | ssl.SSLSocket, num_bytes: int
    ) -> bytes | None:
        """Receive exactly num_bytes from socket."""
        data = b""
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data
