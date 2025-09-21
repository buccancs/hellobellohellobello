"""Tests for AuthManager functionality."""

import time
from unittest.mock import patch

from pc_controller.src.network.auth_manager import AuthManager


class TestAuthManager:
    """Test AuthManager authentication and authorization."""

    def test_initialization(self):
        """Test AuthManager initializes properly."""
        auth = AuthManager(secret_key="test_secret")
        assert auth._secret_key == "test_secret"
        assert len(auth._active_tokens) == 0

    def test_create_challenge(self):
        """Test challenge creation."""
        auth = AuthManager(secret_key="test_secret")
        challenge = auth.create_challenge("device_001")

        assert challenge["type"] == "auth_challenge"
        assert challenge["device_id"] == "device_001"
        assert "challenge" in challenge
        assert "nonce" in challenge
        assert "timestamp" in challenge
        assert challenge["algorithm"] == "HMAC-SHA256"

    def test_successful_authentication(self):
        """Test successful authentication flow."""
        auth = AuthManager(secret_key="test_secret")
        device_id = "device_001"

        challenge_data = auth.create_challenge(device_id)

        import hashlib
        import hmac

        challenge = challenge_data["challenge"]
        nonce = challenge_data["nonce"]
        timestamp = int(challenge_data["timestamp"])

        payload = f"{challenge}:{nonce}:{timestamp}"
        signature = hmac.new(
            b"test_secret",
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        response = {
            "signature": signature,
            "nonce": nonce,
            "timestamp": str(timestamp)
        }

        assert auth.verify_response(device_id, response) is True
        assert device_id in auth.get_active_devices()
        assert auth.is_authorized(device_id, "record") is True

    def test_failed_authentication_bad_signature(self):
        """Test authentication fails with bad signature."""
        auth = AuthManager(secret_key="test_secret")
        device_id = "device_001"

        challenge_data = auth.create_challenge(device_id)

        response = {
            "signature": "bad_signature",
            "nonce": challenge_data["nonce"],
            "timestamp": challenge_data["timestamp"]
        }

        assert auth.verify_response(device_id, response) is False
        assert device_id not in auth.get_active_devices()

    def test_replay_attack_prevention(self):
        """Test replay attack prevention."""
        auth = AuthManager(secret_key="test_secret")
        device_id = "device_001"

        challenge_data = auth.create_challenge(device_id)

        import hashlib
        import hmac

        challenge = challenge_data["challenge"]
        nonce = challenge_data["nonce"]
        timestamp = int(challenge_data["timestamp"])

        payload = f"{challenge}:{nonce}:{timestamp}"
        signature = hmac.new(
            b"test_secret",
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        response = {
            "signature": signature,
            "nonce": nonce,
            "timestamp": str(timestamp)
        }

        assert auth.verify_response(device_id, response) is True

        device_id_2 = "device_002"
        auth.create_challenge(device_id_2)
        assert auth.verify_response(device_id_2, response) is False

    def test_token_expiration(self):
        """Test token expiration functionality."""
        with patch('pc_controller.src.network.auth_manager.cfg_get') as mock_cfg:
            mock_cfg.side_effect = (
                lambda key, default: 1 if key == "auth_token_lifetime_seconds" else default
            )

            auth = AuthManager(secret_key="test_secret")
            device_id = "device_001"

            challenge_data = auth.create_challenge(device_id)

            import hashlib
            import hmac

            challenge = challenge_data["challenge"]
            nonce = challenge_data["nonce"]
            timestamp = int(challenge_data["timestamp"])

            payload = f"{challenge}:{nonce}:{timestamp}"
            signature = hmac.new(
                b"test_secret",
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

            response = {
                "signature": signature,
                "nonce": nonce,
                "timestamp": str(timestamp)
            }

            assert auth.verify_response(device_id, response) is True
            assert auth.is_authorized(device_id, "record") is True

            time.sleep(2)
            assert auth.is_authorized(device_id, "record") is False
            assert device_id not in auth.get_active_devices()

    def test_permission_checking(self):
        """Test permission-based authorization."""
        auth = AuthManager(secret_key="test_secret")
        device_id = "device_001"

        challenge_data = auth.create_challenge(device_id)

        import hashlib
        import hmac

        challenge = challenge_data["challenge"]
        nonce = challenge_data["nonce"]
        timestamp = int(challenge_data["timestamp"])

        payload = f"{challenge}:{nonce}:{timestamp}"
        signature = hmac.new(
            b"test_secret",
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        response = {
            "signature": signature,
            "nonce": nonce,
            "timestamp": str(timestamp)
        }

        assert auth.verify_response(device_id, response) is True

        assert auth.is_authorized(device_id, "record") is True
        assert auth.is_authorized(device_id, "sync") is True
        assert auth.is_authorized(device_id, "preview") is True
        assert auth.is_authorized(device_id, "transfer") is True

        assert auth.is_authorized(device_id, "admin") is False

    def test_token_revocation(self):
        """Test token revocation."""
        auth = AuthManager(secret_key="test_secret")
        device_id = "device_001"

        challenge_data = auth.create_challenge(device_id)

        import hashlib
        import hmac

        challenge = challenge_data["challenge"]
        nonce = challenge_data["nonce"]
        timestamp = int(challenge_data["timestamp"])

        payload = f"{challenge}:{nonce}:{timestamp}"
        signature = hmac.new(
            b"test_secret",
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        response = {
            "signature": signature,
            "nonce": nonce,
            "timestamp": str(timestamp)
        }

        assert auth.verify_response(device_id, response) is True
        assert auth.is_authorized(device_id, "record") is True

        assert auth.revoke_token(device_id) is True
        assert auth.is_authorized(device_id, "record") is False
        assert device_id not in auth.get_active_devices()

        assert auth.revoke_token("nonexistent") is False
