"""Tests for enhanced TLS functionality."""
import os
import ssl
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pc_controller.src.network.tls_enhanced import (
    SecureConnectionManager,
    SecureMessageHandler,
    TLSConfig,
    generate_self_signed_cert,
)


class TestTLSConfig:
    """Test TLS configuration functionality."""

    def test_tls_config_initialization(self):
        """Test TLS configuration initialization with defaults."""
        config = TLSConfig()
        assert not config.enabled
        assert config.cert_file is None
        assert config.key_file is None
        assert config.ca_file is None
        assert config.verify_mode == ssl.CERT_REQUIRED
        assert config.check_hostname is True

    def test_tls_config_with_parameters(self):
        """Test TLS configuration with specific parameters."""
        config = TLSConfig(
            enabled=True,
            cert_file='/path/to/cert.pem',
            key_file='/path/to/key.pem',
            ca_file='/path/to/ca.pem',
            verify_mode=ssl.CERT_OPTIONAL,
            check_hostname=False
        )

        assert config.enabled
        assert config.cert_file == '/path/to/cert.pem'
        assert config.key_file == '/path/to/key.pem'
        assert config.ca_file == '/path/to/ca.pem'
        assert config.verify_mode == ssl.CERT_OPTIONAL
        assert not config.check_hostname

    def test_from_env_disabled(self):
        """Test creating TLS config from environment - disabled case."""
        with patch.dict(os.environ, {
            'PC_TLS_ENABLED': 'false'
        }, clear=True):
            config = TLSConfig.from_env()
            assert not config.enabled

    def test_from_env_enabled(self):
        """Test creating TLS config from environment - enabled case."""
        with patch.dict(os.environ, {
            'PC_TLS_ENABLED': 'true',
            'PC_TLS_CERT_FILE': '/test/cert.pem',
            'PC_TLS_KEY_FILE': '/test/key.pem',
            'PC_TLS_CA_FILE': '/test/ca.pem',
            'PC_TLS_VERIFY_CLIENT': 'true',
            'PC_TLS_CHECK_HOSTNAME': 'false'
        }, clear=True):
            config = TLSConfig.from_env()

            assert config.enabled
            assert config.cert_file == '/test/cert.pem'
            assert config.key_file == '/test/key.pem'
            assert config.ca_file == '/test/ca.pem'
            assert config.verify_mode == ssl.CERT_REQUIRED
            assert not config.check_hostname

    def test_validate_disabled_config(self):
        """Test validation of disabled TLS configuration."""
        config = TLSConfig(enabled=False)
        valid, message = config.validate()
        assert valid
        assert "disabled" in message.lower()

    def test_validate_enabled_without_files(self):
        """Test validation of enabled TLS config without cert files."""
        config = TLSConfig(enabled=True)
        valid, message = config.validate()
        assert not valid
        assert "cert_file or key_file not specified" in message

    def test_validate_enabled_with_missing_files(self):
        """Test validation with non-existent cert files."""
        config = TLSConfig(
            enabled=True,
            cert_file='/nonexistent/cert.pem',
            key_file='/nonexistent/key.pem'
        )
        valid, message = config.validate()
        assert not valid
        assert "not found" in message

    def test_validate_enabled_with_existing_files(self):
        """Test validation with existing cert files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cert_file = Path(temp_dir) / 'cert.pem'
            key_file = Path(temp_dir) / 'key.pem'

            cert_file.touch()
            key_file.touch()

            config = TLSConfig(
                enabled=True,
                cert_file=str(cert_file),
                key_file=str(key_file)
            )

            valid, message = config.validate()
            assert valid
            assert "valid" in message.lower()


class TestSecureConnectionManager:
    """Test SecureConnectionManager functionality."""

    def test_initialization_disabled(self):
        """Test manager initialization with TLS disabled."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)

        assert manager.config == config
        assert manager.get_server_context() is None
        assert manager.get_client_context() is None

    @pytest.fixture
    def temp_certs(self):
        """Create temporary certificate files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cert_file = Path(temp_dir) / 'cert.pem'
            key_file = Path(temp_dir) / 'key.pem'
            ca_file = Path(temp_dir) / 'ca.pem'

            cert_data = "-----BEGIN CERTIFICATE-----\nDUMMY_CERT\n-----END CERTIFICATE-----\n"
            key_data = "-----BEGIN PRIVATE KEY-----\nDUMMY_KEY\n-----END PRIVATE KEY-----\n"
            ca_data = "-----BEGIN CERTIFICATE-----\nDUMMY_CA\n-----END CERTIFICATE-----\n"

            cert_file.write_text(cert_data)
            key_file.write_text(key_data)
            ca_file.write_text(ca_data)

            yield {
                'cert_file': str(cert_file),
                'key_file': str(key_file),
                'ca_file': str(ca_file)
            }

    def test_wrap_socket_disabled(self, temp_certs):
        """Test socket wrapping with TLS disabled."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)

        mock_socket = Mock()
        wrapped = manager.wrap_server_socket(mock_socket)

        assert wrapped is mock_socket

        wrapped = manager.wrap_client_socket(mock_socket)
        assert wrapped is mock_socket

    @patch('ssl.create_default_context')
    def test_create_server_context(self, mock_create_context, temp_certs):
        """Test server SSL context creation."""
        mock_context = Mock()
        mock_create_context.return_value = mock_context

        config = TLSConfig(
            enabled=True,
            cert_file=temp_certs['cert_file'],
            key_file=temp_certs['key_file'],
            ca_file=temp_certs['ca_file'],
            verify_mode=ssl.CERT_REQUIRED
        )
        manager = SecureConnectionManager(config)

        context = manager.get_server_context()

        assert context is mock_context
        mock_create_context.assert_called_once_with(ssl.Purpose.CLIENT_AUTH)
        mock_context.load_cert_chain.assert_called_once()
        mock_context.load_verify_locations.assert_called_once()

    @patch('ssl.create_default_context')
    def test_create_client_context(self, mock_create_context, temp_certs):
        """Test client SSL context creation."""
        mock_context = Mock()
        mock_create_context.return_value = mock_context

        config = TLSConfig(
            enabled=True,
            cert_file=temp_certs['cert_file'],
            key_file=temp_certs['key_file'],
            ca_file=temp_certs['ca_file']
        )
        manager = SecureConnectionManager(config)

        context = manager.get_client_context()

        assert context is mock_context
        mock_create_context.assert_called_once()
        mock_context.load_verify_locations.assert_called_once()
        mock_context.load_cert_chain.assert_called_once()

    @patch('socket.socket')
    def test_connect_secure_disabled(self, mock_socket_class):
        """Test secure connection with TLS disabled."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)

        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        result = manager.connect_secure('localhost', 8080)

        assert result is mock_socket
        mock_socket.connect.assert_called_once_with(('localhost', 8080))
        mock_socket.settimeout.assert_called_once_with(10.0)

    @patch('socket.socket')
    @patch('ssl.create_default_context')
    def test_connect_secure_enabled(self, mock_create_context, mock_socket_class, temp_certs):
        """Test secure connection with TLS enabled."""
        mock_context = Mock()
        mock_ssl_socket = Mock()
        mock_context.wrap_socket.return_value = mock_ssl_socket
        mock_create_context.return_value = mock_context

        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        config = TLSConfig(
            enabled=True,
            cert_file=temp_certs['cert_file'],
            key_file=temp_certs['key_file']
        )
        manager = SecureConnectionManager(config)

        result = manager.connect_secure('example.com', 443)

        assert result is mock_ssl_socket
        mock_socket.connect.assert_called_once_with(('example.com', 443))
        mock_context.wrap_socket.assert_called_once_with(
            mock_socket,
            server_side=False,
            server_hostname='example.com'
        )

    @patch('socket.socket')
    def test_connect_secure_connection_failure(self, mock_socket_class):
        """Test secure connection with connection failure."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)

        mock_socket = Mock()
        mock_socket.connect.side_effect = ConnectionError("Connection failed")
        mock_socket_class.return_value = mock_socket

        with pytest.raises(ConnectionError):
            manager.connect_secure('localhost', 8080)

        mock_socket.close.assert_called_once()


class TestSecureMessageHandler:
    """Test SecureMessageHandler functionality."""

    def test_initialization(self):
        """Test message handler initialization."""
        config = TLSConfig()
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        assert handler.connection_manager is manager

    def test_send_secure_message_plain_socket(self):
        """Test sending message over plain socket."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        mock_socket = Mock()
        message = b"test message"

        result = handler.send_secure_message(mock_socket, message)

        assert result is True
        expected_length = len(message).to_bytes(4, 'big')
        mock_socket.sendall.assert_any_call(expected_length)
        mock_socket.sendall.assert_any_call(message)

    def test_send_secure_message_ssl_socket(self):
        """Test sending message over SSL socket."""
        config = TLSConfig(enabled=True)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        mock_ssl_socket = Mock(spec=ssl.SSLSocket)
        mock_ssl_socket.cipher.return_value = ('TLS_AES_256_GCM_SHA384', 'TLSv1.3', 256)
        message = b"secure test message"

        result = handler.send_secure_message(mock_ssl_socket, message)

        assert result is True
        expected_length = len(message).to_bytes(4, 'big')
        mock_ssl_socket.sendall.assert_any_call(expected_length)
        mock_ssl_socket.sendall.assert_any_call(message)
        mock_ssl_socket.cipher.assert_called_once()

    def test_send_secure_message_failure(self):
        """Test sending message with socket failure."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        mock_socket = Mock()
        mock_socket.sendall.side_effect = Exception("Send failed")
        message = b"test message"

        result = handler.send_secure_message(mock_socket, message)

        assert result is False

    def test_receive_secure_message_plain_socket(self):
        """Test receiving message from plain socket."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        message = b"test response"
        length_bytes = len(message).to_bytes(4, 'big')

        mock_socket = Mock()
        mock_socket.recv.side_effect = [length_bytes, message]

        result = handler.receive_secure_message(mock_socket)

        assert result == message
        mock_socket.settimeout.assert_called_once_with(10.0)

    def test_receive_secure_message_ssl_socket(self):
        """Test receiving message from SSL socket."""
        config = TLSConfig(enabled=True)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        message = b"secure response"
        length_bytes = len(message).to_bytes(4, 'big')

        mock_ssl_socket = Mock(spec=ssl.SSLSocket)
        mock_ssl_socket.cipher.return_value = ('TLS_AES_256_GCM_SHA384', 'TLSv1.3', 256)
        mock_ssl_socket.recv.side_effect = [length_bytes, message]

        result = handler.receive_secure_message(mock_ssl_socket)

        assert result == message
        mock_ssl_socket.cipher.assert_called_once()

    def test_receive_secure_message_too_large(self):
        """Test receiving message that's too large."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        large_length = (11 * 1024 * 1024).to_bytes(4, 'big')

        mock_socket = Mock()
        mock_socket.recv.return_value = large_length

        result = handler.receive_secure_message(mock_socket)

        assert result is None

    def test_receive_secure_message_connection_closed(self):
        """Test receiving message when connection is closed."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        mock_socket = Mock()
        mock_socket.recv.return_value = b''

        result = handler.receive_secure_message(mock_socket)

        assert result is None

    def test_receive_exactly_partial_data(self):
        """Test receiving exact amount with partial chunks."""
        config = TLSConfig(enabled=False)
        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        message = b"test message that comes in chunks"

        mock_socket = Mock()
        mock_socket.recv.side_effect = [
            len(message).to_bytes(4, 'big'),
            b"test message",
            b" that comes",
            b" in chunks"
        ]

        result = handler.receive_secure_message(mock_socket)

        assert result == message


class TestSelfSignedCertificateGeneration:
    """Test self-signed certificate generation."""

    @pytest.mark.skipif(
        not pytest.importorskip("cryptography", reason="cryptography library not available"),
        reason="cryptography library required"
    )
    def test_generate_self_signed_cert(self):
        """Test generating self-signed certificate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cert_file = Path(temp_dir) / 'test_cert.pem'
            key_file = Path(temp_dir) / 'test_key.pem'

            generate_self_signed_cert(
                str(cert_file),
                str(key_file),
                hostname='test.localhost',
                days=30
            )

            assert cert_file.exists()
            assert key_file.exists()

            cert_content = cert_file.read_text()
            key_content = key_file.read_text()

            assert "-----BEGIN CERTIFICATE-----" in cert_content
            assert "-----END CERTIFICATE-----" in cert_content
            assert "-----BEGIN PRIVATE KEY-----" in key_content
            assert "-----END PRIVATE KEY-----" in key_content


class TestTLSIntegration:
    """Integration tests for TLS functionality."""

    def test_full_tls_config_workflow_disabled(self):
        """Test complete TLS workflow when disabled."""
        config = TLSConfig(enabled=False)
        valid, message = config.validate()
        assert valid

        manager = SecureConnectionManager(config)
        handler = SecureMessageHandler(manager)

        mock_socket = Mock()
        wrapped = manager.wrap_server_socket(mock_socket)
        assert wrapped is mock_socket

        result = handler.send_secure_message(mock_socket, b"test")
        assert result is True

    def test_tls_config_validation_workflow(self):
        """Test TLS configuration validation workflow."""
        config = TLSConfig(
            enabled=True,
            cert_file='/nonexistent/cert.pem',
            key_file='/nonexistent/key.pem'
        )
        valid, message = config.validate()
        assert not valid
        assert "not found" in message

        # Test with temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            cert_file = Path(temp_dir) / 'cert.pem'
            key_file = Path(temp_dir) / 'key.pem'

            cert_file.touch()
            key_file.touch()

            config = TLSConfig(
                enabled=True,
                cert_file=str(cert_file),
                key_file=str(key_file)
            )

            valid, message = config.validate()
            assert valid

    def test_environment_configuration_integration(self):
        """Test integration with environment configuration."""
        test_cases = [
            # Disabled case
            {
                'env': {'PC_TLS_ENABLED': 'false'},
                'expected_enabled': False
            },
            # Enabled with minimal config
            {
                'env': {
                    'PC_TLS_ENABLED': 'true',
                    'PC_TLS_CERT_FILE': '/test/cert.pem',
                    'PC_TLS_KEY_FILE': '/test/key.pem'
                },
                'expected_enabled': True
            },
            # Enabled with full config
            {
                'env': {
                    'PC_TLS_ENABLED': 'true',
                    'PC_TLS_CERT_FILE': '/test/cert.pem',
                    'PC_TLS_KEY_FILE': '/test/key.pem',
                    'PC_TLS_CA_FILE': '/test/ca.pem',
                    'PC_TLS_VERIFY_CLIENT': 'true',
                    'PC_TLS_CHECK_HOSTNAME': 'false'
                },
                'expected_enabled': True
            }
        ]

        for case in test_cases:
            with patch.dict(os.environ, case['env'], clear=True):
                config = TLSConfig.from_env()
                assert config.enabled == case['expected_enabled']

                if config.enabled:
                    assert config.cert_file is not None
                    assert config.key_file is not None
