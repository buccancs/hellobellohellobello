"""TLS utility helpers for wrapping sockets with TLS 1.2+.

This module centralizes SSL context creation for both client and server sides
based on environment variables. By default, TLS is disabled to keep unit tests
and development flows working without certificate provisioning. Enable it by
setting the corresponding environment variables.

Environment variables (PC Controller / Python):
- PC_TLS_ENABLE=1                            -> enable TLS wrapping
- PC_TLS_CA_FILE=path                        -> CA bundle/PEM for server verification
- PC_TLS_CERT_FILE=path                      -> client certificate (for mutual auth)
- PC_TLS_KEY_FILE=path                       -> client private key
- PC_TLS_CHECK_HOSTNAME=1 (default 1)        -> verify server hostname
- PC_TLS_REQUIRE_CLIENT_CERT=1 (server side) -> require client certificates
- PC_TLS_SERVER_CERT_FILE=path               -> server certificate (PEM)
- PC_TLS_SERVER_KEY_FILE=path                -> server private key
- PC_TLS_SERVER_CA_FILE=path                 -> CA to verify client certs (optional)

Notes:
- Minimum TLS version is enforced to TLS 1.2.
- If PC_TLS_ENABLE is not set to "1", functions return None indicating TLS is disabled.
"""

from __future__ import annotations

import os
import ssl


def _is_enabled() -> bool:
    return os.environ.get("PC_TLS_ENABLE", "0") == "1"


def create_client_ssl_context() -> ssl.SSLContext | None:
    """Create an SSLContext for client connections or return None if disabled.

    Returns:
        ssl.SSLContext | None: Configured context if TLS is enabled; otherwise None.
    """
    if not _is_enabled():
        return None

    ca_file = os.environ.get("PC_TLS_CA_FILE")
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_file)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    check_hostname = os.environ.get("PC_TLS_CHECK_HOSTNAME", "1") == "1"
    ctx.check_hostname = check_hostname
    # CERT_REQUIRED would fail, tests should keep TLS disabled. Here we stick to
    # CERT_REQUIRED assuming CA is present when enabled.
    ctx.verify_mode = ssl.CERT_REQUIRED

    cert_file = os.environ.get("PC_TLS_CERT_FILE")
    key_file = os.environ.get("PC_TLS_KEY_FILE")
    if cert_file and key_file:
        ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)

    return ctx


def create_server_ssl_context() -> ssl.SSLContext | None:
    """Create an SSLContext for server sockets or return None if disabled.

    Expects PC_TLS_SERVER_CERT_FILE and PC_TLS_SERVER_KEY_FILE when enabled.
    Optionally enforces client certificate verification when PC_TLS_REQUIRE_CLIENT_CERT=1.
    """
    if not _is_enabled():
        return None

    server_cert = os.environ.get("PC_TLS_SERVER_CERT_FILE")
    server_key = os.environ.get("PC_TLS_SERVER_KEY_FILE")
    if not server_cert or not server_key:
        # Without server credentials, cannot enable TLS server; fall back to None
        return None

    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=server_cert, keyfile=server_key)

    require_client = os.environ.get("PC_TLS_REQUIRE_CLIENT_CERT", "0") == "1"
    if require_client:
        ctx.verify_mode = ssl.CERT_REQUIRED
        ca_file = os.environ.get("PC_TLS_SERVER_CA_FILE") or os.environ.get(
            "PC_TLS_CA_FILE"
        )
        if ca_file:
            ctx.load_verify_locations(cafile=ca_file)
        else:
            # If requested but no CA configured, relax to CERT_NONE to avoid
            print(
                "[TLS] PC_TLS_REQUIRE_CLIENT_CERT=1 but no CA provided; "
                "disabling client verification"
            )
            ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx.verify_mode = ssl.CERT_NONE

    return ctx
