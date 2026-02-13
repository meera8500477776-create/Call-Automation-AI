"""
Encryption Service for Config-Driven Voice Agent Platform.

Provides AES-256-GCM encryption for data at rest, TLS 1.2+ connection establishment,
PII/PHI tokenization, and key management with rotation support.

Requirements: 7.2, 7.3, 7.4
"""

import hashlib
import secrets
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata."""
    key_id: str
    key_material: bytes
    created_at: datetime
    expires_at: datetime | None = None
    is_active: bool = True


@dataclass
class TokenizedText:
    """Represents tokenized text with token mapping."""
    text: str
    tokens: dict[str, str] = field(default_factory=dict)


class EncryptionService:
    """
    Encryption service providing AES-256-GCM encryption, TLS connections,
    and PII/PHI tokenization.
    
    Requirements:
    - 7.2: TLS 1.2+ for all voice stream transmission
    - 7.3: AES-256 encryption for all stored data
    - 7.4: Minimize data collection and tokenize PII and PHI
    """

    def __init__(self, master_key: bytes | None = None, key_rotation_days: int = 90):
        """
        Initialize the encryption service.
        
        Args:
            master_key: Master key for key derivation. If None, generates a new one.
            key_rotation_days: Number of days before key rotation is required.
        """
        self._master_key = master_key or self._generate_master_key()
        self._key_rotation_days = key_rotation_days
        self._keys: dict[str, EncryptionKey] = {}
        self._token_map: dict[str, str] = {}  # token -> original value
        self._reverse_token_map: dict[str, str] = {}  # original value -> token

        # Create default encryption key
        self._create_key("default")

    def _generate_master_key(self) -> bytes:
        """Generate a secure master key."""
        return secrets.token_bytes(32)  # 256 bits

    def _derive_key(self, key_id: str, salt: bytes) -> bytes:
        """
        Derive an encryption key from the master key using PBKDF2.
        
        Args:
            key_id: Identifier for the key
            salt: Salt for key derivation
            
        Returns:
            Derived 256-bit key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self._master_key + key_id.encode())

    def _create_key(self, key_id: str) -> EncryptionKey:
        """
        Create a new encryption key.
        
        Args:
            key_id: Identifier for the key
            
        Returns:
            Created encryption key
        """
        salt = secrets.token_bytes(16)
        key_material = self._derive_key(key_id, salt)

        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(days=self._key_rotation_days)

        key = EncryptionKey(
            key_id=key_id,
            key_material=key_material,
            created_at=created_at,
            expires_at=expires_at,
            is_active=True
        )

        self._keys[key_id] = key
        return key

    def _get_key(self, key_id: str) -> EncryptionKey:
        """
        Get an encryption key by ID, creating it if it doesn't exist.
        
        Args:
            key_id: Identifier for the key
            
        Returns:
            Encryption key
            
        Raises:
            ValueError: If key is expired or inactive
        """
        if key_id not in self._keys:
            return self._create_key(key_id)

        key = self._keys[key_id]

        # Check if key needs rotation
        if key.expires_at and datetime.utcnow() > key.expires_at:
            raise ValueError(f"Key {key_id} has expired and needs rotation")

        if not key.is_active:
            raise ValueError(f"Key {key_id} is inactive")

        return key

    def rotate_key(self, key_id: str) -> EncryptionKey:
        """
        Rotate an encryption key by creating a new one and marking the old one inactive.
        
        Args:
            key_id: Identifier for the key to rotate
            
        Returns:
            New encryption key
        """
        if key_id in self._keys:
            self._keys[key_id].is_active = False

        return self._create_key(key_id)

    def encrypt_data(self, data: bytes, key_id: str = "default") -> bytes:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            data: Data to encrypt
            key_id: Identifier for the encryption key to use
            
        Returns:
            Encrypted data with nonce prepended (nonce + ciphertext + tag)
            
        Raises:
            ValueError: If key is invalid or expired
        """
        key = self._get_key(key_id)

        # Generate a random 96-bit nonce (12 bytes) for GCM
        nonce = secrets.token_bytes(12)

        # Create AESGCM cipher
        aesgcm = AESGCM(key.key_material)

        # Encrypt data (returns ciphertext + authentication tag)
        ciphertext = aesgcm.encrypt(nonce, data, None)

        # Return nonce + ciphertext (which includes the tag)
        return nonce + ciphertext

    def decrypt_data(self, encrypted_data: bytes, key_id: str = "default") -> bytes:
        """
        Decrypt data using AES-256-GCM.
        
        Args:
            encrypted_data: Encrypted data with nonce prepended
            key_id: Identifier for the encryption key to use
            
        Returns:
            Decrypted data
            
        Raises:
            ValueError: If key is invalid, expired, or decryption fails
        """
        key = self._get_key(key_id)

        # Extract nonce (first 12 bytes) and ciphertext
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]

        # Create AESGCM cipher
        aesgcm = AESGCM(key.key_material)

        # Decrypt data
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

    def establish_tls_connection(
        self,
        hostname: str,
        port: int = 443,
        min_tls_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2
    ) -> ssl.SSLContext:
        """
        Create an SSL context for TLS 1.2+ connections.
        
        Args:
            hostname: Hostname to connect to
            port: Port number (default: 443)
            min_tls_version: Minimum TLS version (default: TLS 1.2)
            
        Returns:
            Configured SSL context
        """
        # Create SSL context with secure defaults
        context = ssl.create_default_context()

        # Set minimum TLS version to 1.2 or higher
        context.minimum_version = min_tls_version

        # Disable insecure protocols
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1

        # Enable hostname checking
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED

        return context

    def tokenize_pii(self, text: str, pii_patterns: dict[str, str] | None = None) -> TokenizedText:
        """
        Tokenize PII/PHI in text by replacing sensitive data with tokens.
        
        Args:
            text: Text containing PII/PHI
            pii_patterns: Optional dict mapping pattern names to regex patterns
            
        Returns:
            TokenizedText with PII replaced by tokens
        """
        import re

        # Default PII patterns
        default_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'date_of_birth': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        }

        patterns = pii_patterns or default_patterns
        tokenized_text = text
        tokens = {}

        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, tokenized_text)
            for match in matches:
                original_value = match.group()

                # Check if we already have a token for this value
                if original_value in self._reverse_token_map:
                    token = self._reverse_token_map[original_value]
                else:
                    # Generate a deterministic token based on the value
                    token_hash = hashlib.sha256(original_value.encode()).hexdigest()[:16]
                    token = f"[{pattern_name.upper()}_{token_hash}]"

                    # Store token mappings
                    self._token_map[token] = original_value
                    self._reverse_token_map[original_value] = token

                tokens[token] = original_value
                tokenized_text = tokenized_text.replace(original_value, token, 1)

        return TokenizedText(text=tokenized_text, tokens=tokens)

    def detokenize(self, tokenized_text: TokenizedText) -> str:
        """
        Restore original text from tokenized text.
        
        Args:
            tokenized_text: TokenizedText with tokens
            
        Returns:
            Original text with PII restored
        """
        result = tokenized_text.text

        for token, original_value in tokenized_text.tokens.items():
            result = result.replace(token, original_value)

        return result

    def clear_token_cache(self) -> None:
        """Clear the token mapping cache."""
        self._token_map.clear()
        self._reverse_token_map.clear()
