"""Kalshi API Client for BattleBot.

Handles authentication, market data, and order execution for Kalshi.
CFTC-regulated, legal for US residents including California.
"""

import os
import time
import base64
import hashlib
from datetime import datetime
from typing import Optional
import httpx
from loguru import logger

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography not installed - Kalshi auth will be limited")


class KalshiClient:
    """Client for Kalshi prediction market API."""
    
    # API endpoints - elections subdomain serves ALL Kalshi markets
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(self, 
                 api_key_id: Optional[str] = None,
                 private_key_path: Optional[str] = None,
                 private_key_pem: Optional[str] = None,
                 use_demo: bool = False):
        """Initialize Kalshi client.
        
        Args:
            api_key_id: Kalshi API key ID
            private_key_path: Path to RSA private key file
            private_key_pem: RSA private key as PEM string
            use_demo: Use demo environment instead of production
        """
        self.api_key_id = api_key_id or os.getenv('KALSHI_API_KEY_ID')
        self.base_url = self.DEMO_URL if use_demo else self.PROD_URL
        self.use_demo = use_demo
        
        # Load private key for authentication
        self._private_key = None
        if private_key_pem:
            self._load_private_key_from_pem(private_key_pem)
        elif private_key_path:
            self._load_private_key_from_file(private_key_path)
        elif os.getenv('KALSHI_PRIVATE_KEY'):
            self._load_private_key_from_pem(os.getenv('KALSHI_PRIVATE_KEY'))
        elif os.getenv('KALSHI_PRIVATE_KEY_PATH'):
            self._load_private_key_from_file(os.getenv('KALSHI_PRIVATE_KEY_PATH'))
            
        logger.info(f"KalshiClient initialized | Demo: {use_demo} | Auth: {self._private_key is not None}")
    
    def _load_private_key_from_file(self, path: str):
        """Load RSA private key from file."""
        if not HAS_CRYPTO:
            logger.error("cryptography library required for Kalshi authentication")
            return
        try:
            with open(path, 'rb') as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            logger.info(f"Loaded Kalshi private key from {path}")
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
    
    def _load_private_key_from_pem(self, pem_string: str):
        """Load RSA private key from PEM string."""
        if not HAS_CRYPTO:
            logger.error("cryptography library required for Kalshi authentication")
            return
        try:
            # Handle newlines in env var
            pem_string = pem_string.replace('\\n', '\n')
            self._private_key = serialization.load_pem_private_key(
                pem_string.encode(),
                password=None,
                backend=default_backend()
            )
            logger.info("Loaded Kalshi private key from PEM string")
        except Exception as e:
            logger.error(f"Failed to load private key from PEM: {e}")
    
    def _sign_request(self, method: str, path: str, timestamp: int) -> str:
        """Generate RSA-PSS signature for authenticated request."""
        if not self._private_key:
            raise ValueError("Private key not loaded - cannot sign request")
        
        # Message format: timestamp + method + path
        message = f"{timestamp}{method}{path}".encode()
        
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    def _get_auth_headers(self, method: str, path: str) -> dict:
        """Generate authentication headers for request."""
        if not self.api_key_id or not self._private_key:
            return {}
        
        timestamp = int(time.time() * 1000)
        # Signature must include the FULL path (including /trade-api/v2 prefix)
        full_path = f"/trade-api/v2{path}"
        signature = self._sign_request(method, full_path, timestamp)
        
        return {
            'KALSHI-ACCESS-KEY': self.api_key_id,
            'KALSHI-ACCESS-TIMESTAMP': str(timestamp),
            'KALSHI-ACCESS-SIGNATURE': signature,
        }
    
    async def get_markets(self, 
                         status: str = "open",
                         series_ticker: Optional[str] = None,
                         limit: int = 100,
                         cursor: Optional[str] = None) -> dict:
        """Fetch markets from Kalshi.
        
        Args:
            status: Market status filter (open, closed, settled)
            series_ticker: Filter by series/event ticker
            limit: Max results to return
            cursor: Pagination cursor
            
        Returns:
            Dict with 'markets' list and 'cursor' for pagination
        """
        path = "/markets"
        params = {'status': status, 'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker
        if cursor:
            params['cursor'] = cursor
            
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                params=params,
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
    
    async def get_market(self, ticker: str) -> dict:
        """Get single market by ticker."""
        path = f"/markets/{ticker}"
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
    
    async def get_events(self, 
                        status: str = "open",
                        limit: int = 100,
                        series_ticker: Optional[str] = None,
                        cursor: Optional[str] = None) -> dict:
        """Fetch events (series of related markets).
        
        Args:
            status: Event status (open, closed, settled)
            limit: Max results per page
            series_ticker: Filter by series ticker (e.g., 'PRES' for politics)
            cursor: Pagination cursor
        """
        path = "/events"
        params = {'status': status, 'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker
        if cursor:
            params['cursor'] = cursor
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                params=params,
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
    
    async def get_orderbook(self, ticker: str) -> dict:
        """Get market orderbook."""
        path = f"/markets/{ticker}/orderbook"
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
    
    async def get_balance(self) -> dict:
        """Get account balance (requires auth)."""
        path = "/portfolio/balance"
        headers = self._get_auth_headers('GET', path)
        headers['Accept'] = 'application/json'
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    
    async def place_order(self,
                         ticker: str,
                         side: str,
                         count: int,
                         price: int,
                         order_type: str = "limit") -> dict:
        """Place an order (requires auth).
        
        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            count: Number of contracts
            price: Price in cents (1-99)
            order_type: 'limit' or 'market'
            
        Returns:
            Order response dict
        """
        path = "/portfolio/orders"
        headers = self._get_auth_headers('POST', path)
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        
        body = {
            'ticker': ticker,
            'action': 'buy',
            'side': side.lower(),
            'count': count,
            'type': order_type,
        }
        # Only include the price for the side we're trading
        if order_type == 'limit':
            if side.lower() == 'yes':
                body['yes_price'] = price
            else:
                body['no_price'] = price
        
        logger.debug(f"Placing order: {body}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}{path}",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            return response.json()
    
    async def sell_position(self,
                           ticker: str,
                           side: str,
                           count: int,
                           price: int = None,
                           order_type: str = "market") -> dict:
        """Sell/close a position (requires auth).
        
        Args:
            ticker: Market ticker  
            side: 'yes' or 'no' (the side you're selling)
            count: Number of contracts to sell
            price: Price in cents (only used for limit orders)
            order_type: 'limit' or 'market'
            
        Returns:
            Order response dict
        """
        path = "/portfolio/orders"
        headers = self._get_auth_headers('POST', path)
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        
        body = {
            'ticker': ticker,
            'action': 'sell',
            'side': side.lower(),
            'count': count,
            'type': order_type,
        }
        # Only include the price for limit orders
        if order_type == 'limit' and price is not None:
            if side.lower() == 'yes':
                body['yes_price'] = price
            else:
                body['no_price'] = price
        
        logger.debug(f"Selling position: {body}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}{path}",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            return response.json()
    
    async def get_positions(self) -> dict:
        """Get current positions (requires auth)."""
        path = "/portfolio/positions"
        headers = self._get_auth_headers('GET', path)
        headers['Accept'] = 'application/json'
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    
    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order (requires auth)."""
        path = f"/portfolio/orders/{order_id}"
        headers = self._get_auth_headers('DELETE', path)
        headers['Accept'] = 'application/json'
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"{self.base_url}{path}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    
    async def get_order(self, order_id: str) -> dict:
        """Get status of a specific order (requires auth)."""
        path = f"/portfolio/orders/{order_id}"
        headers = self._get_auth_headers('GET', path)
        headers['Accept'] = 'application/json'
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    
    async def get_fills(self, ticker: Optional[str] = None, limit: int = 100) -> dict:
        """Get recent fills/trades for the account (requires auth)."""
        path = "/portfolio/fills"
        headers = self._get_auth_headers('GET', path)
        headers['Accept'] = 'application/json'
        
        params = {'limit': limit}
        if ticker:
            params['ticker'] = ticker
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    async def get_orders(self, status: str = "resting") -> dict:
        """Get orders by status (requires auth).
        
        Args:
            status: 'resting' (open), 'canceled', or 'executed'
        """
        path = "/portfolio/orders"
        headers = self._get_auth_headers('GET', path)
        headers['Accept'] = 'application/json'
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers=headers,
                params={'status': status}
            )
            response.raise_for_status()
            return response.json()


def parse_kalshi_market(market: dict) -> dict:
    """Convert Kalshi market format to BattleBot internal format.
    
    Args:
        market: Raw market dict from Kalshi API
        
    Returns:
        Standardized market dict for BattleBot
    """
    # Kalshi prices are in cents (1-99), convert to decimal (0.01-0.99)
    yes_price = market.get('yes_price', 50) / 100 if market.get('yes_price') else 0.5
    no_price = market.get('no_price', 50) / 100 if market.get('no_price') else 0.5
    
    # Get best bid/ask for spread calculation
    yes_bid = market.get('yes_bid', 0) / 100 if market.get('yes_bid') else yes_price - 0.01
    yes_ask = market.get('yes_ask', 0) / 100 if market.get('yes_ask') else yes_price + 0.01
    spread = yes_ask - yes_bid
    
    # Parse close time
    close_time = market.get('close_time', '')
    end_date = None
    if close_time:
        try:
            end_date = close_time
        except:
            pass
    
    # Volume - Kalshi reports in contracts
    volume = market.get('volume', 0) or 0
    volume_24h = market.get('volume_24h', volume) or volume
    
    # Liquidity estimate from open interest
    liquidity = market.get('open_interest', 0) * yes_price
    
    # Build resolution rules from available fields
    # Kalshi uses different fields depending on market type
    rules = (
        market.get('rules_primary', '') or 
        market.get('rules_secondary', '') or
        market.get('subtitle', '') or
        market.get('settlement_source_url', '') or
        ''
    )
    
    # For sports/event markets without explicit rules, create synthetic rules
    title = market.get('title', '')
    if not rules and title:
        # Generate rules from the market structure
        rules = f"Market resolves YES if '{title}' occurs. Market resolves NO otherwise. Settlement based on official results."
    
    description = market.get('subtitle', '') or market.get('rules_primary', '') or title
    
    return {
        'id': market.get('ticker', ''),
        'token_id': market.get('ticker', ''),
        'question': title,
        'description': description,
        'rules': rules,
        'price': yes_price,
        'yes_price': yes_price,
        'no_price': no_price,
        'price_pct': int(yes_price * 100),
        'spread': spread,
        'spread_display': f"{spread*100:.1f}¢",
        'liquidity': liquidity,
        'volume_24h': volume_24h,
        'volume_display': f"${volume_24h:,.0f}" if volume_24h >= 1000 else f"${volume_24h:.0f}",
        'end_date': end_date,
        'category': market.get('category', 'other'),
        'url': f"https://kalshi.com/markets/{market.get('ticker', '')}",
        'image': market.get('image_url', ''),
        'event_ticker': market.get('event_ticker', ''),
        'series_ticker': market.get('series_ticker', ''),
        'platform': 'kalshi',
    }
