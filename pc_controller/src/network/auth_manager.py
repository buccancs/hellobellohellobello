"""Authentication and Authorization Manager for PC Controller.

Implements token-based authentication with replay protection and access control
for secure communication between PC Hub and Android Spokes beyond basic TLS.

Features:
- Token-based handshake with configurable expiration
- Replay protection using nonces and timestamps
- Access control for different command types
- Integration with existing NetworkController
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

try:
    from ..config import get as cfg_get
except Exception:

    def cfg_get(key: str, default=None):
        return default


@dataclass
class AuthToken:
    """Authentication token with expiration and permissions."""

    token: str
    device_id: str
    issued_at: float
    expires_at: float
    permissions: set[str]
    nonce_counter: int = 0


@dataclass
class AuthChallenge:
    """Authentication challenge for handshake."""

    challenge: str
    nonce: str
    timestamp: float


class AuthManager:
    """Manages authentication and authorization for device connections."""

    def __init__(self, secret_key: str | None = None):
        """Initialize AuthManager with optional secret key.

        Args:
            secret_key: Base64 encoded secret key. If None, reads from config or generates.
        """
        self._secret_key = secret_key or self._get_or_generate_secret()
        self._active_tokens: dict[str, AuthToken] = {}
        self._pending_challenges: dict[str, AuthChallenge] = {}
        self._used_nonces: set[str] = set()

        # Configuration
        self._token_lifetime = int(
            cfg_get("auth_token_lifetime_seconds", 3600)
        )
        self._challenge_timeout = int(cfg_get("auth_challenge_timeout_seconds", 30))
        self._nonce_window = int(cfg_get("auth_nonce_window_seconds", 300))
        self._max_nonce_cache = int(cfg_get("auth_max_nonce_cache", 10000))

    def _get_or_generate_secret(self) -> str:
        """Get secret key from config or generate a new one."""
        secret = cfg_get("auth_secret_key")
        if not secret:
            secret = secrets.token_urlsafe(32)
        return secret

    def create_challenge(self, device_id: str) -> dict[str, str]:
        """Create authentication challenge for device.

        Args:
            device_id: Unique identifier for the device

        Returns:
            Dictionary containing challenge data for JSON transmission
        """
        challenge = secrets.token_urlsafe(16)
        nonce = secrets.token_urlsafe(16)
        timestamp = time.time()

        self._pending_challenges[device_id] = AuthChallenge(
            challenge=challenge, nonce=nonce, timestamp=timestamp
        )

        return {
            "type": "auth_challenge",
            "device_id": device_id,
            "challenge": challenge,
            "nonce": nonce,
            "timestamp": str(int(timestamp)),
            "algorithm": "HMAC-SHA256",
        }

    def verify_response(self, device_id: str, response_data: dict[str, str]) -> bool:
        """Verify authentication response from device.

        Args:
            device_id: Device identifier
            response_data: Dictionary containing response fields

        Returns:
            True if authentication successful, False otherwise
        """
        if device_id not in self._pending_challenges:
            return False

        challenge_info = self._pending_challenges[device_id]

        if time.time() - challenge_info.timestamp > self._challenge_timeout:
            del self._pending_challenges[device_id]
            return False

        try:
            received_signature = response_data.get("signature", "")
            received_nonce = response_data.get("nonce", "")
            received_timestamp = int(response_data.get("timestamp", "0"))
        except (ValueError, TypeError):
            return False

        if received_nonce != challenge_info.nonce:
            return False

        if received_nonce in self._used_nonces:
            return False

        if abs(time.time() - received_timestamp) > self._nonce_window:
            return False

        payload = f"{challenge_info.challenge}:{received_nonce}:{received_timestamp}"
        expected_signature = hmac.new(
            self._secret_key.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(received_signature, expected_signature):
            return False

        token = self._create_token(device_id)
        self._active_tokens[device_id] = token

        del self._pending_challenges[device_id]
        self._used_nonces.add(received_nonce)
        self._cleanup_nonces()

        return True

    def _create_token(self, device_id: str) -> AuthToken:
        """Create authenticated token for device."""
        now = time.time()
        token_data = {
            "device_id": device_id,
            "issued_at": now,
            "expires_at": now + self._token_lifetime,
            "permissions": ["record", "sync", "preview", "transfer"],
        }

        token_string = hmac.new(
            self._secret_key.encode(),
            json.dumps(token_data, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()

        return AuthToken(
            token=token_string,
            device_id=device_id,
            issued_at=now,
            expires_at=now + self._token_lifetime,
            permissions=set(token_data["permissions"]),
        )

    def is_authorized(
        self, device_id: str, action: str, nonce: str | None = None
    ) -> bool:
        """Check if device is authorized to perform action.

        Args:
            device_id: Device identifier
            action: Action to authorize (e.g., "record", "sync", "preview")
            nonce: Optional nonce for replay protection

        Returns:
            True if authorized, False otherwise
        """
        if device_id not in self._active_tokens:
            return False

        token = self._active_tokens[device_id]

        if time.time() > token.expires_at:
            del self._active_tokens[device_id]
            return False

        if action not in token.permissions:
            return False

        if nonce:
            if nonce in self._used_nonces:
                return False
            self._used_nonces.add(nonce)
            self._cleanup_nonces()

        return True

    def revoke_token(self, device_id: str) -> bool:
        """Revoke authentication token for device.

        Args:
            device_id: Device identifier

        Returns:
            True if token was revoked, False if not found
        """
        if device_id in self._active_tokens:
            del self._active_tokens[device_id]
            return True
        return False

    def get_active_devices(self) -> set[str]:
        """Get set of currently authenticated device IDs."""
        current_time = time.time()
        active = set()
        expired = []

        for device_id, token in self._active_tokens.items():
            if current_time <= token.expires_at:
                active.add(device_id)
            else:
                expired.append(device_id)

        for device_id in expired:
            del self._active_tokens[device_id]

        return active

    def _cleanup_nonces(self):
        """Clean up old nonces to prevent memory growth."""
        if len(self._used_nonces) > self._max_nonce_cache:
            nonces_list = list(self._used_nonces)
            keep_count = self._max_nonce_cache // 2
            self._used_nonces = set(nonces_list[-keep_count:])

    def cleanup_expired(self):
        """Clean up expired tokens and challenges."""
        current_time = time.time()

        expired_tokens = [
            device_id
            for device_id, token in self._active_tokens.items()
            if current_time > token.expires_at
        ]
        for device_id in expired_tokens:
            del self._active_tokens[device_id]

        expired_challenges = [
            device_id
            for device_id, challenge in self._pending_challenges.items()
            if current_time - challenge.timestamp > self._challenge_timeout
        ]
        for device_id in expired_challenges:
            del self._pending_challenges[device_id]
