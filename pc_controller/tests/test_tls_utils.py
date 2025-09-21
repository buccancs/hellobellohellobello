from __future__ import annotations

import ssl
import types

import pytest

from pc_controller.src.network import tls_utils


class _FakeClientContext:
    def __init__(self) -> None:
        self.minimum_version = None
        self.check_hostname = True
        self.verify_mode = None
        self.loaded_chain = None

    def load_cert_chain(self, certfile: str, keyfile: str) -> None:
        self.loaded_chain = (certfile, keyfile)

    def wrap_socket(self, sock, server_hostname=None):
        return types.SimpleNamespace(kind="wrapped", server_hostname=server_hostname, inner=sock)


class _FakeServerContext:
    def __init__(self) -> None:
        self.minimum_version = None
        self.verify_mode = None
        self.loaded_chain = None
        self.ca_loaded = None

    def load_cert_chain(self, certfile: str, keyfile: str) -> None:
        self.loaded_chain = (certfile, keyfile)

    def load_verify_locations(self, cafile: str) -> None:
        self.ca_loaded = cafile


@pytest.fixture
def clear_env(monkeypatch):
    keys = [
        "PC_TLS_ENABLE",
        "PC_TLS_CA_FILE",
        "PC_TLS_CERT_FILE",
        "PC_TLS_KEY_FILE",
        "PC_TLS_CHECK_HOSTNAME",
        "PC_TLS_REQUIRE_CLIENT_CERT",
        "PC_TLS_SERVER_CERT_FILE",
        "PC_TLS_SERVER_KEY_FILE",
        "PC_TLS_SERVER_CA_FILE",
    ]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    yield


def test_client_ctx_disabled_returns_none(clear_env):
    ctx = tls_utils.create_client_ssl_context()
    assert ctx is None


def test_client_ctx_enabled_with_hostname_toggle_and_cert_chain(monkeypatch, clear_env):
    # Enable TLS and provide CA path
    monkeypatch.setenv("PC_TLS_ENABLE", "1")
    monkeypatch.setenv("PC_TLS_CA_FILE", "dummy-ca.pem")
    fake = _FakeClientContext()

    def _fake_create_default_context(purpose, cafile=None):
        assert purpose == ssl.Purpose.SERVER_AUTH
        assert cafile == "dummy-ca.pem"
        return fake

    monkeypatch.setattr(ssl, "create_default_context", _fake_create_default_context)

    ctx = tls_utils.create_client_ssl_context()
    assert ctx is fake
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
    assert ctx.check_hostname is True
    assert ctx.verify_mode == ssl.CERT_REQUIRED

    monkeypatch.setenv("PC_TLS_CHECK_HOSTNAME", "0")
    fake2 = _FakeClientContext()
    monkeypatch.setattr(ssl, "create_default_context", lambda purpose, cafile=None: fake2)
    ctx2 = tls_utils.create_client_ssl_context()
    assert ctx2.check_hostname is False

    monkeypatch.setenv("PC_TLS_CERT_FILE", "client.pem")
    monkeypatch.setenv("PC_TLS_KEY_FILE", "client.key")
    fake3 = _FakeClientContext()
    monkeypatch.setattr(ssl, "create_default_context", lambda purpose, cafile=None: fake3)
    ctx3 = tls_utils.create_client_ssl_context()
    assert ctx3.loaded_chain == ("client.pem", "client.key")


def test_server_ctx_disabled_and_missing_creds(monkeypatch, clear_env):
    # Disabled
    assert tls_utils.create_server_ssl_context() is None
    # Enabled but missing server cert/key -> None
    monkeypatch.setenv("PC_TLS_ENABLE", "1")
    assert tls_utils.create_server_ssl_context() is None


def test_server_ctx_enabled_basic_and_client_verification_paths(monkeypatch, clear_env):
    monkeypatch.setenv("PC_TLS_ENABLE", "1")
    monkeypatch.setenv("PC_TLS_SERVER_CERT_FILE", "srv.pem")
    monkeypatch.setenv("PC_TLS_SERVER_KEY_FILE", "srv.key")

    fake = _FakeServerContext()

    def _fake_ctx(purpose):
        assert purpose == ssl.Purpose.CLIENT_AUTH
        return fake

    monkeypatch.setattr(ssl, "create_default_context", _fake_ctx)

    ctx = tls_utils.create_server_ssl_context()
    assert ctx is fake
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
    assert ctx.loaded_chain == ("srv.pem", "srv.key")
    assert ctx.verify_mode == ssl.CERT_NONE

    monkeypatch.setenv("PC_TLS_REQUIRE_CLIENT_CERT", "1")
    monkeypatch.setenv("PC_TLS_SERVER_CA_FILE", "ca.pem")
    fake2 = _FakeServerContext()
    monkeypatch.setattr(ssl, "create_default_context", lambda purpose: fake2)
    ctx2 = tls_utils.create_server_ssl_context()
    assert ctx2.verify_mode == ssl.CERT_REQUIRED
    assert ctx2.ca_loaded == "ca.pem"

    # Require client but no CA -> verify_mode downgraded to CERT_NONE
    monkeypatch.delenv("PC_TLS_SERVER_CA_FILE", raising=False)
    fake3 = _FakeServerContext()
    monkeypatch.setattr(ssl, "create_default_context", lambda purpose: fake3)
    ctx3 = tls_utils.create_server_ssl_context()
    assert ctx3.verify_mode == ssl.CERT_NONE


def test__connect_wraps_socket_when_tls_enabled(monkeypatch):
    pytest.importorskip("PyQt6")
    from pc_controller.src.network.network_controller import _connect  # type: ignore

    class _FakeSock:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def _fake_create_connection(addr, timeout=None):
        assert isinstance(addr, tuple)
        return _FakeSock()

    class _FakeTLS:
        def __init__(self, check_hostname=True) -> None:
            self.check_hostname = check_hostname

        def wrap_socket(self, s, server_hostname=None):
            return types.SimpleNamespace(kind="wrapped", hn=server_hostname, inner=s)

    socket_path = "pc_controller.src.network.network_controller.socket.create_connection"
    monkeypatch.setattr(socket_path, _fake_create_connection)
    import pc_controller.src.network.network_controller as nc  # type: ignore

    nc.create_client_ssl_context = lambda: _FakeTLS(check_hostname=True)  # type: ignore

    wrapped = _connect("example.com", 1234, timeout=1.0)
    assert getattr(wrapped, "kind", "") == "wrapped"
    assert getattr(wrapped, "hn", None) == "example.com"

    nc.create_client_ssl_context = lambda: _FakeTLS(check_hostname=False)  # type: ignore
    wrapped2 = _connect("example.com", 1234, timeout=1.0)
    assert getattr(wrapped2, "hn", "sentinel") is None


def test__connect_returns_plain_socket_when_tls_disabled(monkeypatch):
    pytest.importorskip("PyQt6")
    from pc_controller.src.network.network_controller import _connect  # type: ignore

    class _FakeSock:
        def __init__(self) -> None:
            self.closed = False

    def _fake_create_connection(addr, timeout=None):
        return _FakeSock()

    socket_path = "pc_controller.src.network.network_controller.socket.create_connection"
    monkeypatch.setattr(socket_path, _fake_create_connection)
    import pc_controller.src.network.network_controller as nc  # type: ignore

    nc.create_client_ssl_context = lambda: None  # type: ignore

    s = _connect("localhost", 1, timeout=0.1)
    assert isinstance(s, _FakeSock)
