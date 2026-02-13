"""
Unit tests for Authentication Service.

Tests OAuth 2.0 client credentials flow, token caching, automatic refresh,
and token validation.

Requirements: 3.1, 7.8
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientError, ClientTimeout
from voice_agent.security import AuthenticationService, AccessToken, TokenValidation


class TestAccessToken:
    """Test suite for AccessToken dataclass."""
    
    def test_access_token_creation(self):
        """Test AccessToken creation with required fields."""
        token = AccessToken(
            token="test_token_123",
            token_type="Bearer",
            expires_in=3600,
            scope=["read", "write"]
        )
        
        assert token.token == "test_token_123"
        assert token.token_type == "Bearer"
        assert token.expires_in == 3600
        assert token.scope == ["read", "write"]
        assert token.refresh_token is None
        assert isinstance(token.created_at, datetime)
    
    def test_access_token_with_refresh_token(self):
        """Test AccessToken creation with refresh token."""
        token = AccessToken(
            token="access_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"],
            refresh_token="refresh_token_123"
        )
        
        assert token.refresh_token == "refresh_token_123"
    
    def test_is_expired_not_expired(self):
        """Test is_expired property for non-expired token."""
        token = AccessToken(
            token="test_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"],
            created_at=datetime.utcnow()
        )
        
        assert not token.is_expired
    
    def test_is_expired_expired(self):
        """Test is_expired property for expired token."""
        token = AccessToken(
            token="test_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"],
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        assert token.is_expired
    
    def test_is_expired_near_expiry(self):
        """Test is_expired property for token near expiry (60 second buffer)."""
        # Token expires in 30 seconds - should be considered expired due to 60s buffer
        token = AccessToken(
            token="test_token",
            token_type="Bearer",
            expires_in=30,
            scope=["read"],
            created_at=datetime.utcnow()
        )
        
        assert token.is_expired
    
    def test_is_expired_zero_expires_in(self):
        """Test is_expired property for token with zero expires_in."""
        token = AccessToken(
            token="test_token",
            token_type="Bearer",
            expires_in=0,
            scope=["read"]
        )
        
        assert token.is_expired
    
    def test_expires_at_property(self):
        """Test expires_at property calculation."""
        created_at = datetime.utcnow()
        token = AccessToken(
            token="test_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"],
            created_at=created_at
        )
        
        expected_expiry = created_at + timedelta(seconds=3600)
        assert token.expires_at == expected_expiry


class TestTokenValidation:
    """Test suite for TokenValidation dataclass."""
    
    def test_token_validation_valid(self):
        """Test TokenValidation for valid token."""
        validation = TokenValidation(
            is_valid=True,
            expires_at=datetime.utcnow() + timedelta(hours=1),
            scope=["read", "write"]
        )
        
        assert validation.is_valid
        assert validation.expires_at is not None
        assert validation.scope == ["read", "write"]
        assert validation.error is None
    
    def test_token_validation_invalid(self):
        """Test TokenValidation for invalid token."""
        validation = TokenValidation(
            is_valid=False,
            error="Token is expired"
        )
        
        assert not validation.is_valid
        assert validation.error == "Token is expired"
        assert validation.expires_at is None
        assert validation.scope is None


class TestAuthenticationService:
    """Test suite for AuthenticationService."""
    
    def test_initialization(self):
        """Test that AuthenticationService initializes correctly."""
        service = AuthenticationService()
        
        assert service is not None
        assert service._timeout == 30
        assert len(service._token_cache) == 0
    
    def test_initialization_custom_timeout(self):
        """Test initialization with custom timeout."""
        service = AuthenticationService(timeout=60)
        
        assert service._timeout == 60
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        service = AuthenticationService()
        
        key1 = service._generate_cache_key("client1", ["read", "write"])
        key2 = service._generate_cache_key("client1", ["write", "read"])  # Different order
        key3 = service._generate_cache_key("client2", ["read", "write"])
        
        # Same client and scopes (regardless of order) should produce same key
        assert key1 == key2
        
        # Different client should produce different key
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_get_access_token_success(self):
        """Test successful access token retrieval."""
        service = AuthenticationService()
        
        mock_response = {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            token = await service.get_access_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                scopes=["read", "write"]
            )
            
            assert token.token == "test_access_token"
            assert token.token_type == "Bearer"
            assert token.expires_in == 3600
            assert token.scope == ["read", "write"]
    
    @pytest.mark.asyncio
    async def test_get_access_token_with_refresh_token(self):
        """Test access token retrieval that includes refresh token."""
        service = AuthenticationService()
        
        mock_response = {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write",
            "refresh_token": "test_refresh_token"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            token = await service.get_access_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                scopes=["read", "write"]
            )
            
            assert token.refresh_token == "test_refresh_token"
    
    @pytest.mark.asyncio
    async def test_get_access_token_caching(self):
        """Test that tokens are cached and reused."""
        service = AuthenticationService()
        
        mock_response = {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            # First call should make HTTP request
            token1 = await service.get_access_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                scopes=["read", "write"]
            )
            
            # Second call should use cache (no additional HTTP request)
            token2 = await service.get_access_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                scopes=["read", "write"]
            )
            
            # Should be the same token
            assert token1.token == token2.token
            
            # HTTP request should only be called once
            assert mock_post.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_access_token_expired_cache(self):
        """Test that expired cached tokens trigger new request."""
        service = AuthenticationService()
        
        # Create an expired token in cache
        expired_token = AccessToken(
            token="expired_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read", "write"],
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        cache_key = service._generate_cache_key("client123", ["read", "write"])
        service._token_cache[cache_key] = expired_token
        
        mock_response = {
            "access_token": "new_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            token = await service.get_access_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                scopes=["read", "write"]
            )
            
            # Should get new token
            assert token.token == "new_access_token"
            assert mock_post.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_access_token_http_error(self):
        """Test handling of HTTP errors during token request."""
        service = AuthenticationService()
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 401
            mock_resp.text = AsyncMock(return_value="Unauthorized")
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(ValueError, match="Token request failed with status 401"):
                await service.get_access_token(
                    token_url="https://auth.example.com/token",
                    client_id="client123",
                    client_secret="wrong_secret",
                    scopes=["read"]
                )
    
    @pytest.mark.asyncio
    async def test_get_access_token_missing_access_token(self):
        """Test handling of response missing access_token field."""
        service = AuthenticationService()
        
        mock_response = {
            "token_type": "Bearer",
            "expires_in": 3600
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(ValueError, match="missing 'access_token' field"):
                await service.get_access_token(
                    token_url="https://auth.example.com/token",
                    client_id="client123",
                    client_secret="secret456",
                    scopes=["read"]
                )
    
    @pytest.mark.asyncio
    async def test_get_access_token_missing_token_type(self):
        """Test handling of response missing token_type field."""
        service = AuthenticationService()
        
        mock_response = {
            "access_token": "test_token"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(ValueError, match="missing 'token_type' field"):
                await service.get_access_token(
                    token_url="https://auth.example.com/token",
                    client_id="client123",
                    client_secret="secret456",
                    scopes=["read"]
                )
    
    @pytest.mark.asyncio
    async def test_get_access_token_with_additional_params(self):
        """Test token request with additional parameters."""
        service = AuthenticationService()
        
        mock_response = {
            "access_token": "test_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            await service.get_access_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                scopes=["read"],
                additional_params={"audience": "https://api.example.com"}
            )
            
            # Verify additional params were included in request
            call_args = mock_post.call_args
            assert "audience" in call_args[1]["data"]
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        """Test successful token refresh."""
        service = AuthenticationService()
        
        mock_response = {
            "access_token": "new_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write",
            "refresh_token": "new_refresh_token"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            token = await service.refresh_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                refresh_token="old_refresh_token"
            )
            
            assert token.token == "new_access_token"
            assert token.refresh_token == "new_refresh_token"
    
    @pytest.mark.asyncio
    async def test_refresh_token_preserves_old_refresh_token(self):
        """Test that refresh preserves old refresh token if new one not provided."""
        service = AuthenticationService()
        
        mock_response = {
            "access_token": "new_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            token = await service.refresh_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                refresh_token="old_refresh_token"
            )
            
            # Should preserve old refresh token
            assert token.refresh_token == "old_refresh_token"
    
    @pytest.mark.asyncio
    async def test_refresh_token_http_error(self):
        """Test handling of HTTP errors during token refresh."""
        service = AuthenticationService()
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 400
            mock_resp.text = AsyncMock(return_value="Invalid refresh token")
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(ValueError, match="Token refresh failed with status 400"):
                await service.refresh_token(
                    token_url="https://auth.example.com/token",
                    client_id="client123",
                    client_secret="secret456",
                    refresh_token="invalid_token"
                )
    
    @pytest.mark.asyncio
    async def test_validate_token_local_valid(self):
        """Test local token validation for valid cached token."""
        service = AuthenticationService()
        
        # Add a valid token to cache
        token = AccessToken(
            token="valid_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read", "write"],
            created_at=datetime.utcnow()
        )
        
        cache_key = service._generate_cache_key("client123", ["read", "write"])
        service._token_cache[cache_key] = token
        
        validation = await service.validate_token("valid_token")
        
        assert validation.is_valid
        assert validation.expires_at is not None
        assert validation.scope == ["read", "write"]
        assert validation.error is None
    
    @pytest.mark.asyncio
    async def test_validate_token_local_expired(self):
        """Test local token validation for expired cached token."""
        service = AuthenticationService()
        
        # Add an expired token to cache
        token = AccessToken(
            token="expired_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"],
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        cache_key = service._generate_cache_key("client123", ["read"])
        service._token_cache[cache_key] = token
        
        validation = await service.validate_token("expired_token")
        
        assert not validation.is_valid
        assert validation.error == "Token is expired"
    
    @pytest.mark.asyncio
    async def test_validate_token_local_not_found(self):
        """Test local token validation for token not in cache."""
        service = AuthenticationService()
        
        validation = await service.validate_token("unknown_token")
        
        assert not validation.is_valid
        assert validation.error == "Token not found in cache"
    
    @pytest.mark.asyncio
    async def test_validate_token_introspection_valid(self):
        """Test server-side token validation (introspection) for valid token."""
        service = AuthenticationService()
        
        mock_response = {
            "active": True,
            "scope": "read write",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp())
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            validation = await service.validate_token(
                token="test_token",
                introspection_url="https://auth.example.com/introspect",
                client_id="client123",
                client_secret="secret456"
            )
            
            assert validation.is_valid
            assert validation.scope == ["read", "write"]
            assert validation.expires_at is not None
    
    @pytest.mark.asyncio
    async def test_validate_token_introspection_inactive(self):
        """Test server-side token validation for inactive token."""
        service = AuthenticationService()
        
        mock_response = {
            "active": False
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            validation = await service.validate_token(
                token="test_token",
                introspection_url="https://auth.example.com/introspect",
                client_id="client123",
                client_secret="secret456"
            )
            
            assert not validation.is_valid
            assert validation.error == "Token is not active"
    
    @pytest.mark.asyncio
    async def test_validate_token_introspection_missing_credentials(self):
        """Test introspection validation without required credentials."""
        service = AuthenticationService()
        
        validation = await service.validate_token(
            token="test_token",
            introspection_url="https://auth.example.com/introspect"
        )
        
        assert not validation.is_valid
        assert "client_id and client_secret required" in validation.error
    
    @pytest.mark.asyncio
    async def test_revoke_token_success(self):
        """Test successful token revocation."""
        service = AuthenticationService()
        
        # Add token to cache
        token = AccessToken(
            token="token_to_revoke",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"],
            created_at=datetime.utcnow()
        )
        
        cache_key = service._generate_cache_key("client123", ["read"])
        service._token_cache[cache_key] = token
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await service.revoke_token(
                revocation_url="https://auth.example.com/revoke",
                token="token_to_revoke",
                client_id="client123",
                client_secret="secret456"
            )
            
            assert result is True
            # Token should be removed from cache
            assert cache_key not in service._token_cache
    
    @pytest.mark.asyncio
    async def test_revoke_token_with_type_hint(self):
        """Test token revocation with token type hint."""
        service = AuthenticationService()
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await service.revoke_token(
                revocation_url="https://auth.example.com/revoke",
                token="test_token",
                client_id="client123",
                client_secret="secret456",
                token_type_hint="access_token"
            )
            
            assert result is True
            
            # Verify token_type_hint was included
            call_args = mock_post.call_args
            assert "token_type_hint" in call_args[1]["data"]
    
    @pytest.mark.asyncio
    async def test_revoke_token_http_error(self):
        """Test handling of HTTP errors during token revocation."""
        service = AuthenticationService()
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 500
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await service.revoke_token(
                revocation_url="https://auth.example.com/revoke",
                token="test_token",
                client_id="client123",
                client_secret="secret456"
            )
            
            assert result is False
    
    def test_clear_cache(self):
        """Test clearing the token cache."""
        service = AuthenticationService()
        
        # Add some tokens to cache
        token1 = AccessToken(
            token="token1",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"]
        )
        
        token2 = AccessToken(
            token="token2",
            token_type="Bearer",
            expires_in=3600,
            scope=["write"]
        )
        
        service._token_cache["key1"] = token1
        service._token_cache["key2"] = token2
        
        assert len(service._token_cache) == 2
        
        # Clear cache
        service.clear_cache()
        
        assert len(service._token_cache) == 0
    
    def test_get_cached_token_valid(self):
        """Test getting a valid cached token."""
        service = AuthenticationService()
        
        token = AccessToken(
            token="cached_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read", "write"],
            created_at=datetime.utcnow()
        )
        
        cache_key = service._generate_cache_key("client123", ["read", "write"])
        service._token_cache[cache_key] = token
        
        cached = service.get_cached_token("client123", ["read", "write"])
        
        assert cached is not None
        assert cached.token == "cached_token"
    
    def test_get_cached_token_expired(self):
        """Test getting an expired cached token returns None."""
        service = AuthenticationService()
        
        token = AccessToken(
            token="expired_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read"],
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        cache_key = service._generate_cache_key("client123", ["read"])
        service._token_cache[cache_key] = token
        
        cached = service.get_cached_token("client123", ["read"])
        
        assert cached is None
    
    def test_get_cached_token_not_found(self):
        """Test getting a non-existent cached token returns None."""
        service = AuthenticationService()
        
        cached = service.get_cached_token("client123", ["read"])
        
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_automatic_refresh_on_expired_cache(self):
        """Test that expired tokens with refresh tokens are automatically refreshed."""
        service = AuthenticationService()
        
        # Create an expired token with refresh token in cache
        expired_token = AccessToken(
            token="expired_token",
            token_type="Bearer",
            expires_in=3600,
            scope=["read", "write"],
            refresh_token="refresh_token_123",
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        cache_key = service._generate_cache_key("client123", ["read", "write"])
        service._token_cache[cache_key] = expired_token
        
        mock_refresh_response = {
            "access_token": "refreshed_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write",
            "refresh_token": "new_refresh_token"
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_refresh_response)
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            token = await service.get_access_token(
                token_url="https://auth.example.com/token",
                client_id="client123",
                client_secret="secret456",
                scopes=["read", "write"]
            )
            
            # Should get refreshed token
            assert token.token == "refreshed_token"
            assert token.refresh_token == "new_refresh_token"
            
            # Should have called refresh endpoint (grant_type=refresh_token)
            call_args = mock_post.call_args
            assert call_args[1]["data"]["grant_type"] == "refresh_token"
