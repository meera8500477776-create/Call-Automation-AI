"""
Authentication Service for Config-Driven Voice Agent Platform.

Provides OAuth 2.0 authentication for external APIs with token caching,
automatic refresh, and validation.

Requirements: 3.1, 7.8
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import aiohttp


@dataclass
class AccessToken:
    """Represents an OAuth 2.0 access token with metadata."""
    token: str
    token_type: str
    expires_in: int
    scope: list[str]
    refresh_token: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if self.expires_in <= 0:
            return True
        expiry_time = self.created_at + timedelta(seconds=self.expires_in)
        # Consider token expired 60 seconds before actual expiry for safety
        return datetime.utcnow() >= (expiry_time - timedelta(seconds=60))

    @property
    def expires_at(self) -> datetime:
        """Get the expiration datetime of the token."""
        return self.created_at + timedelta(seconds=self.expires_in)


@dataclass
class TokenValidation:
    """Represents the result of token validation."""
    is_valid: bool
    expires_at: datetime | None = None
    scope: list[str] | None = None
    error: str | None = None


class AuthenticationService:
    """
    Authentication service providing OAuth 2.0 client credentials flow,
    token caching, automatic refresh, and validation.
    
    Requirements:
    - 3.1: OAuth 2.0 authentication with scoped access tokens
    - 7.8: OAuth 2.0 authentication for all external API access
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the authentication service.
        
        Args:
            timeout: HTTP request timeout in seconds (default: 30)
        """
        self._timeout = timeout
        self._token_cache: dict[str, AccessToken] = {}
        self._lock = asyncio.Lock()

    def _generate_cache_key(self, client_id: str, scopes: list[str]) -> str:
        """
        Generate a cache key for a client ID and scopes combination.
        
        Args:
            client_id: OAuth client ID
            scopes: List of OAuth scopes
            
        Returns:
            Cache key string
        """
        scope_str = " ".join(sorted(scopes))
        key_material = f"{client_id}:{scope_str}"
        return hashlib.sha256(key_material.encode()).hexdigest()

    async def get_access_token(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scopes: list[str],
        additional_params: dict[str, str] | None = None
    ) -> AccessToken:
        """
        Get an access token using OAuth 2.0 client credentials flow.
        
        This method implements token caching and will return a cached token
        if available and not expired. Otherwise, it will request a new token.
        
        Args:
            token_url: OAuth token endpoint URL
            client_id: OAuth client ID
            client_secret: OAuth client secret
            scopes: List of OAuth scopes to request
            additional_params: Optional additional parameters for the token request
            
        Returns:
            AccessToken with token and metadata
            
        Raises:
            ValueError: If token request fails or returns invalid response
            aiohttp.ClientError: If HTTP request fails
        """
        cache_key = self._generate_cache_key(client_id, scopes)

        # Check cache first
        async with self._lock:
            if cache_key in self._token_cache:
                cached_token = self._token_cache[cache_key]
                if not cached_token.is_expired:
                    return cached_token

                # Try to refresh if refresh token is available
                if cached_token.refresh_token:
                    try:
                        refreshed_token = await self.refresh_token(
                            token_url=token_url,
                            client_id=client_id,
                            client_secret=client_secret,
                            refresh_token=cached_token.refresh_token
                        )
                        self._token_cache[cache_key] = refreshed_token
                        return refreshed_token
                    except Exception:
                        # If refresh fails, fall through to get new token
                        pass

        # Request new token
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": " ".join(scopes)
        }

        if additional_params:
            token_data.update(additional_params)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(
                            f"Token request failed with status {response.status}: {error_text}"
                        )

                    response_data = await response.json()

                    # Validate required fields
                    if "access_token" not in response_data:
                        raise ValueError("Token response missing 'access_token' field")

                    if "token_type" not in response_data:
                        raise ValueError("Token response missing 'token_type' field")

                    # Create AccessToken
                    access_token = AccessToken(
                        token=response_data["access_token"],
                        token_type=response_data.get("token_type", "Bearer"),
                        expires_in=response_data.get("expires_in", 3600),
                        scope=response_data.get("scope", " ".join(scopes)).split(),
                        refresh_token=response_data.get("refresh_token"),
                        created_at=datetime.utcnow()
                    )

                    # Cache the token
                    async with self._lock:
                        self._token_cache[cache_key] = access_token

                    return access_token

            except aiohttp.ClientError as e:
                raise ValueError(f"HTTP request failed: {str(e)}")

    async def refresh_token(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        refresh_token: str
    ) -> AccessToken:
        """
        Refresh an expired access token using a refresh token.
        
        Args:
            token_url: OAuth token endpoint URL
            client_id: OAuth client ID
            client_secret: OAuth client secret
            refresh_token: Refresh token from previous token response
            
        Returns:
            New AccessToken with refreshed token
            
        Raises:
            ValueError: If refresh request fails or returns invalid response
            aiohttp.ClientError: If HTTP request fails
        """
        token_data = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(
                            f"Token refresh failed with status {response.status}: {error_text}"
                        )

                    response_data = await response.json()

                    # Validate required fields
                    if "access_token" not in response_data:
                        raise ValueError("Token response missing 'access_token' field")

                    # Create new AccessToken
                    return AccessToken(
                        token=response_data["access_token"],
                        token_type=response_data.get("token_type", "Bearer"),
                        expires_in=response_data.get("expires_in", 3600),
                        scope=response_data.get("scope", "").split(),
                        refresh_token=response_data.get("refresh_token", refresh_token),
                        created_at=datetime.utcnow()
                    )

            except aiohttp.ClientError as e:
                raise ValueError(f"HTTP request failed: {str(e)}")

    async def validate_token(
        self,
        token: str,
        introspection_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None
    ) -> TokenValidation:
        """
        Validate an access token.
        
        If introspection_url is provided, performs server-side validation using
        OAuth 2.0 token introspection (RFC 7662). Otherwise, performs basic
        local validation by checking if the token exists in cache and is not expired.
        
        Args:
            token: Access token to validate
            introspection_url: Optional OAuth introspection endpoint URL
            client_id: Optional OAuth client ID (required for introspection)
            client_secret: Optional OAuth client secret (required for introspection)
            
        Returns:
            TokenValidation with validation result
        """
        # If introspection URL is provided, use server-side validation
        if introspection_url:
            if not client_id or not client_secret:
                return TokenValidation(
                    is_valid=False,
                    error="client_id and client_secret required for introspection"
                )

            return await self._introspect_token(
                introspection_url=introspection_url,
                token=token,
                client_id=client_id,
                client_secret=client_secret
            )

        # Otherwise, perform local validation
        async with self._lock:
            for cached_token in self._token_cache.values():
                if cached_token.token == token:
                    if cached_token.is_expired:
                        return TokenValidation(
                            is_valid=False,
                            error="Token is expired"
                        )

                    return TokenValidation(
                        is_valid=True,
                        expires_at=cached_token.expires_at,
                        scope=cached_token.scope
                    )

        # Token not found in cache
        return TokenValidation(
            is_valid=False,
            error="Token not found in cache"
        )

    async def _introspect_token(
        self,
        introspection_url: str,
        token: str,
        client_id: str,
        client_secret: str
    ) -> TokenValidation:
        """
        Perform server-side token introspection (RFC 7662).
        
        Args:
            introspection_url: OAuth introspection endpoint URL
            token: Access token to introspect
            client_id: OAuth client ID
            client_secret: OAuth client secret
            
        Returns:
            TokenValidation with introspection result
        """
        introspection_data = {
            "token": token,
            "client_id": client_id,
            "client_secret": client_secret
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    introspection_url,
                    data=introspection_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return TokenValidation(
                            is_valid=False,
                            error=f"Introspection failed with status {response.status}: {error_text}"
                        )

                    response_data = await response.json()

                    # Check if token is active
                    is_active = response_data.get("active", False)

                    if not is_active:
                        return TokenValidation(
                            is_valid=False,
                            error="Token is not active"
                        )

                    # Extract expiration time
                    expires_at = None
                    if "exp" in response_data:
                        expires_at = datetime.utcfromtimestamp(response_data["exp"])

                    # Extract scope
                    scope = None
                    if "scope" in response_data:
                        scope = response_data["scope"].split()

                    return TokenValidation(
                        is_valid=True,
                        expires_at=expires_at,
                        scope=scope
                    )

            except aiohttp.ClientError as e:
                return TokenValidation(
                    is_valid=False,
                    error=f"HTTP request failed: {str(e)}"
                )

    async def revoke_token(
        self,
        revocation_url: str,
        token: str,
        client_id: str,
        client_secret: str,
        token_type_hint: str | None = None
    ) -> bool:
        """
        Revoke an access or refresh token (RFC 7009).
        
        Args:
            revocation_url: OAuth revocation endpoint URL
            token: Token to revoke
            client_id: OAuth client ID
            client_secret: OAuth client secret
            token_type_hint: Optional hint about token type ("access_token" or "refresh_token")
            
        Returns:
            True if revocation succeeded, False otherwise
        """
        revocation_data = {
            "token": token,
            "client_id": client_id,
            "client_secret": client_secret
        }

        if token_type_hint:
            revocation_data["token_type_hint"] = token_type_hint

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    revocation_url,
                    data=revocation_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    # RFC 7009: The revocation endpoint responds with HTTP 200
                    # for both successful revocations and invalid tokens
                    if response.status == 200:
                        # Remove from cache if present
                        async with self._lock:
                            keys_to_remove = []
                            for key, cached_token in self._token_cache.items():
                                if cached_token.token == token or cached_token.refresh_token == token:
                                    keys_to_remove.append(key)

                            for key in keys_to_remove:
                                del self._token_cache[key]

                        return True

                    return False

            except aiohttp.ClientError:
                return False

    def clear_cache(self) -> None:
        """Clear the token cache."""
        self._token_cache.clear()

    def get_cached_token(self, client_id: str, scopes: list[str]) -> AccessToken | None:
        """
        Get a cached token without making any network requests.
        
        Args:
            client_id: OAuth client ID
            scopes: List of OAuth scopes
            
        Returns:
            Cached AccessToken if available and not expired, None otherwise
        """
        cache_key = self._generate_cache_key(client_id, scopes)

        if cache_key in self._token_cache:
            cached_token = self._token_cache[cache_key]
            if not cached_token.is_expired:
                return cached_token

        return None
