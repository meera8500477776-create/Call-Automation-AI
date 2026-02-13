"""
Unit tests for Encryption Service.

Tests AES-256-GCM encryption, TLS connection establishment, PII tokenization,
and key management with rotation support.

Requirements: 7.2, 7.3, 7.4
"""

import ssl
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from voice_agent.security import EncryptionKey, EncryptionService, TokenizedText


class TestEncryptionService:
    """Test suite for EncryptionService."""

    def test_initialization(self):
        """Test that EncryptionService initializes correctly."""
        service = EncryptionService()

        assert service is not None
        assert "default" in service._keys
        assert service._keys["default"].is_active

    def test_initialization_with_custom_master_key(self):
        """Test initialization with a custom master key."""
        master_key = b"0" * 32  # 256-bit key
        service = EncryptionService(master_key=master_key)

        assert service._master_key == master_key

    def test_encrypt_decrypt_data(self):
        """Test basic encryption and decryption."""
        service = EncryptionService()
        original_data = b"Hello, World! This is sensitive data."

        # Encrypt
        encrypted = service.encrypt_data(original_data)

        # Verify encrypted data is different from original
        assert encrypted != original_data
        assert len(encrypted) > len(original_data)  # Includes nonce and tag

        # Decrypt
        decrypted = service.decrypt_data(encrypted)

        # Verify decrypted data matches original
        assert decrypted == original_data

    def test_encrypt_decrypt_empty_data(self):
        """Test encryption and decryption of empty data."""
        service = EncryptionService()
        original_data = b""

        encrypted = service.encrypt_data(original_data)
        decrypted = service.decrypt_data(encrypted)

        assert decrypted == original_data

    def test_encrypt_decrypt_large_data(self):
        """Test encryption and decryption of large data."""
        service = EncryptionService()
        original_data = b"X" * 10000  # 10KB of data

        encrypted = service.encrypt_data(original_data)
        decrypted = service.decrypt_data(encrypted)

        assert decrypted == original_data

    def test_encrypt_with_custom_key_id(self):
        """Test encryption with a custom key ID."""
        service = EncryptionService()
        original_data = b"Custom key test"

        encrypted = service.encrypt_data(original_data, key_id="custom_key")
        decrypted = service.decrypt_data(encrypted, key_id="custom_key")

        assert decrypted == original_data
        assert "custom_key" in service._keys

    def test_decrypt_with_wrong_key_fails(self):
        """Test that decryption with wrong key fails."""
        service = EncryptionService()
        original_data = b"Secret data"

        # Encrypt with one key
        encrypted = service.encrypt_data(original_data, key_id="key1")

        # Try to decrypt with different key - should fail
        with pytest.raises(ValueError, match="Decryption failed"):
            service.decrypt_data(encrypted, key_id="key2")

    def test_decrypt_corrupted_data_fails(self):
        """Test that decryption of corrupted data fails."""
        service = EncryptionService()
        original_data = b"Secret data"

        encrypted = service.encrypt_data(original_data)

        # Corrupt the encrypted data
        corrupted = encrypted[:-1] + b"X"

        with pytest.raises(ValueError, match="Decryption failed"):
            service.decrypt_data(corrupted)

    def test_key_rotation(self):
        """Test key rotation functionality."""
        service = EncryptionService()

        # Get original key
        original_key = service._keys["default"]
        assert original_key.is_active

        # Rotate key
        new_key = service.rotate_key("default")

        # Verify old key is inactive
        assert not original_key.is_active

        # Verify new key is active and different
        assert new_key.is_active
        assert new_key.key_material != original_key.key_material

    def test_key_expiration(self):
        """Test that expired keys raise an error."""
        service = EncryptionService(key_rotation_days=0)

        # Create a key that expires immediately
        key = service._create_key("test_key")
        key.expires_at = datetime.utcnow() - timedelta(days=1)
        service._keys["test_key"] = key

        # Try to use expired key
        with pytest.raises(ValueError, match="has expired"):
            service.encrypt_data(b"test", key_id="test_key")

    def test_inactive_key_raises_error(self):
        """Test that inactive keys raise an error."""
        service = EncryptionService()

        # Create and deactivate a key
        service._create_key("inactive_key")
        service._keys["inactive_key"].is_active = False

        # Try to use inactive key
        with pytest.raises(ValueError, match="is inactive"):
            service.encrypt_data(b"test", key_id="inactive_key")

    def test_establish_tls_connection(self):
        """Test TLS connection context creation."""
        service = EncryptionService()

        context = service.establish_tls_connection("example.com")

        # Verify SSL context is configured correctly
        assert isinstance(context, ssl.SSLContext)
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2
        assert context.check_hostname is True
        assert context.verify_mode == ssl.CERT_REQUIRED

    def test_establish_tls_connection_custom_port(self):
        """Test TLS connection with custom port."""
        service = EncryptionService()

        context = service.establish_tls_connection("example.com", port=8443)

        assert isinstance(context, ssl.SSLContext)

    def test_establish_tls_connection_tls13(self):
        """Test TLS connection with TLS 1.3 minimum version."""
        service = EncryptionService()

        # TLS 1.3 may not be available in all Python versions
        # This test verifies the method accepts the parameter
        try:
            context = service.establish_tls_connection(
                "example.com",
                min_tls_version=ssl.TLSVersion.TLSv1_3
            )
            assert context.minimum_version == ssl.TLSVersion.TLSv1_3
        except (ValueError, AttributeError):
            # TLS 1.3 not supported in this Python version, skip test
            pytest.skip("TLS 1.3 not supported in this Python version")

    def test_tokenize_pii_email(self):
        """Test PII tokenization for email addresses."""
        service = EncryptionService()
        text = "Contact me at john.doe@example.com for more info."

        tokenized = service.tokenize_pii(text)

        # Verify email is tokenized
        assert "john.doe@example.com" not in tokenized.text
        assert "[EMAIL_" in tokenized.text
        assert len(tokenized.tokens) == 1

        # Verify we can detokenize
        original = service.detokenize(tokenized)
        assert original == text

    def test_tokenize_pii_phone(self):
        """Test PII tokenization for phone numbers."""
        service = EncryptionService()
        text = "Call me at 555-123-4567 or 555.987.6543"

        tokenized = service.tokenize_pii(text)

        # Verify phone numbers are tokenized
        assert "555-123-4567" not in tokenized.text
        assert "555.987.6543" not in tokenized.text
        assert "[PHONE_" in tokenized.text
        assert len(tokenized.tokens) == 2

    def test_tokenize_pii_ssn(self):
        """Test PII tokenization for SSN."""
        service = EncryptionService()
        text = "My SSN is 123-45-6789"

        tokenized = service.tokenize_pii(text)

        # Verify SSN is tokenized
        assert "123-45-6789" not in tokenized.text
        assert "[SSN_" in tokenized.text
        assert len(tokenized.tokens) == 1

    def test_tokenize_pii_multiple_types(self):
        """Test PII tokenization with multiple PII types."""
        service = EncryptionService()
        text = "Email: user@test.com, Phone: 555-123-4567, SSN: 123-45-6789"

        tokenized = service.tokenize_pii(text)

        # Verify all PII is tokenized
        assert "user@test.com" not in tokenized.text
        assert "555-123-4567" not in tokenized.text
        assert "123-45-6789" not in tokenized.text
        assert len(tokenized.tokens) == 3

        # Verify detokenization
        original = service.detokenize(tokenized)
        assert original == text

    def test_tokenize_pii_no_pii(self):
        """Test tokenization of text without PII."""
        service = EncryptionService()
        text = "This is a normal sentence without any PII."

        tokenized = service.tokenize_pii(text)

        # Verify text is unchanged
        assert tokenized.text == text
        assert len(tokenized.tokens) == 0

    def test_tokenize_pii_deterministic(self):
        """Test that tokenization is deterministic for the same value."""
        service = EncryptionService()
        text1 = "Email: test@example.com"
        text2 = "Another email: test@example.com"

        tokenized1 = service.tokenize_pii(text1)
        tokenized2 = service.tokenize_pii(text2)

        # Extract tokens for the email
        token1 = [t for t in tokenized1.tokens.keys()][0]
        token2 = [t for t in tokenized2.tokens.keys()][0]

        # Verify same email gets same token
        assert token1 == token2

    def test_tokenize_pii_custom_patterns(self):
        """Test tokenization with custom PII patterns."""
        service = EncryptionService()
        text = "Account: ACC-12345"

        custom_patterns = {
            'account': r'ACC-\d{5}'
        }

        tokenized = service.tokenize_pii(text, pii_patterns=custom_patterns)

        # Verify custom pattern is tokenized
        assert "ACC-12345" not in tokenized.text
        assert "[ACCOUNT_" in tokenized.text
        assert len(tokenized.tokens) == 1

    def test_clear_token_cache(self):
        """Test clearing the token cache."""
        service = EncryptionService()
        text = "Email: test@example.com"

        # Tokenize to populate cache
        tokenized = service.tokenize_pii(text)
        assert len(service._token_map) > 0

        # Clear cache
        service.clear_token_cache()

        # Verify cache is empty
        assert len(service._token_map) == 0
        assert len(service._reverse_token_map) == 0

    def test_detokenize_without_tokens(self):
        """Test detokenization of text without tokens."""
        service = EncryptionService()
        text = "Normal text"
        tokenized = TokenizedText(text=text, tokens={})

        result = service.detokenize(tokenized)

        assert result == text


class TestEncryptionKey:
    """Test suite for EncryptionKey dataclass."""

    def test_encryption_key_creation(self):
        """Test EncryptionKey creation."""
        key = EncryptionKey(
            key_id="test_key",
            key_material=b"0" * 32,
            created_at=datetime.utcnow()
        )

        assert key.key_id == "test_key"
        assert key.key_material == b"0" * 32
        assert key.is_active is True
        assert key.expires_at is None


class TestTokenizedText:
    """Test suite for TokenizedText dataclass."""

    def test_tokenized_text_creation(self):
        """Test TokenizedText creation."""
        tokenized = TokenizedText(
            text="Hello [TOKEN_123]",
            tokens={"[TOKEN_123]": "secret"}
        )

        assert tokenized.text == "Hello [TOKEN_123]"
        assert tokenized.tokens == {"[TOKEN_123]": "secret"}

    def test_tokenized_text_default_tokens(self):
        """Test TokenizedText with default empty tokens."""
        tokenized = TokenizedText(text="Hello")

        assert tokenized.text == "Hello"
        assert tokenized.tokens == {}



class TestEncryptionProperties:
    """Property-based tests for EncryptionService."""

    @given(st.binary(min_size=0, max_size=10000))
    @settings(deadline=None)
    def test_encryption_round_trip(self, data: bytes):
        """
        Property 3: Encryption round-trip
        
        For any data, encrypting then decrypting should produce the original data.
        
        **Validates: Requirements 7.3**
        """
        service = EncryptionService()

        # Encrypt the data
        encrypted = service.encrypt_data(data)

        # Decrypt the data
        decrypted = service.decrypt_data(encrypted)

        # Verify round-trip produces original data
        assert decrypted == data

    @given(
        data=st.binary(min_size=0, max_size=10000),
        key_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'))
    )
    @settings(deadline=None)
    def test_encryption_round_trip_with_custom_key(self, data: bytes, key_id: str):
        """
        Property 3 (extended): Encryption round-trip with custom key IDs
        
        For any data and any valid key ID, encrypting then decrypting should produce the original data.
        
        **Validates: Requirements 7.3**
        """
        service = EncryptionService()

        # Encrypt the data with custom key ID
        encrypted = service.encrypt_data(data, key_id=key_id)

        # Decrypt the data with the same key ID
        decrypted = service.decrypt_data(encrypted, key_id=key_id)

        # Verify round-trip produces original data
        assert decrypted == data

    @given(st.binary(min_size=1, max_size=10000))
    @settings(deadline=None)
    def test_encrypted_data_differs_from_original(self, data: bytes):
        """
        Property: Encrypted data should differ from original (except for edge cases).
        
        Verifies that encryption actually transforms the data.
        
        **Validates: Requirements 7.3**
        """
        service = EncryptionService()

        # Encrypt the data
        encrypted = service.encrypt_data(data)

        # Encrypted data should be different from original
        # (includes nonce and authentication tag)
        assert encrypted != data

        # Encrypted data should be longer (nonce + ciphertext + tag)
        assert len(encrypted) > len(data)
