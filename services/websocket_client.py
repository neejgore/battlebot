"""WebSocket Client for Polymarket Battle-Bot V2.0.

Real-time listener for price updates from the Polymarket CLOB WebSocket feed.
Implements reconnection logic and event distribution to subscribers.
"""

import asyncio
import json
from datetime import datetime
from typing import Callable, Optional, Any
from collections.abc import Awaitable

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from loguru import logger

from data.models import PriceUpdate


# Type alias for async callback functions
PriceCallback = Callable[[PriceUpdate], Awaitable[None]]


class WebSocketClient:
    """WebSocket client for Polymarket real-time data.
    
    Connects to the Polymarket CLOB WebSocket endpoint and distributes
    price updates to registered callbacks.
    
    Attributes:
        url: WebSocket endpoint URL
        subscribed_markets: Set of market token IDs to subscribe to
        reconnect_delay: Delay between reconnection attempts
        max_reconnect_attempts: Maximum number of reconnection attempts
    """
    
    DEFAULT_URL = "wss://clob.polymarket.com/ws"
    RECONNECT_DELAY = 5.0  # seconds
    MAX_RECONNECT_ATTEMPTS = 10
    HEARTBEAT_INTERVAL = 30.0  # seconds
    
    def __init__(
        self,
        url: str = DEFAULT_URL,
        reconnect_delay: float = RECONNECT_DELAY,
        max_reconnect_attempts: int = MAX_RECONNECT_ATTEMPTS,
    ):
        """Initialize the WebSocket client.
        
        Args:
            url: WebSocket endpoint URL
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._callbacks: list[PriceCallback] = []
        self._subscribed_markets: set[str] = set()
        self._running = False
        self._reconnect_count = 0
        self._message_count = 0
        self._last_message_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        
        logger.info(f"WebSocketClient initialized | URL: {url}")
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._ws.open
    
    @property
    def subscribed_markets(self) -> set[str]:
        """Get set of subscribed market token IDs."""
        return self._subscribed_markets.copy()
    
    def register_callback(self, callback: PriceCallback) -> None:
        """Register a callback for price updates.
        
        Args:
            callback: Async function to call with PriceUpdate objects
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Callback registered | Total callbacks: {len(self._callbacks)}")
    
    def unregister_callback(self, callback: PriceCallback) -> None:
        """Unregister a callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Callback unregistered | Total callbacks: {len(self._callbacks)}")
    
    async def subscribe(self, token_ids: list[str]) -> None:
        """Subscribe to price updates for specific markets.
        
        Args:
            token_ids: List of token IDs to subscribe to
        """
        async with self._lock:
            new_tokens = set(token_ids) - self._subscribed_markets
            if not new_tokens:
                return
            
            self._subscribed_markets.update(new_tokens)
            
            if self.is_connected:
                await self._send_subscription(list(new_tokens))
            
            logger.info(f"Subscribed to {len(new_tokens)} markets | Total: {len(self._subscribed_markets)}")
    
    async def unsubscribe(self, token_ids: list[str]) -> None:
        """Unsubscribe from specific markets.
        
        Args:
            token_ids: List of token IDs to unsubscribe from
        """
        async with self._lock:
            tokens_to_remove = set(token_ids) & self._subscribed_markets
            if not tokens_to_remove:
                return
            
            self._subscribed_markets -= tokens_to_remove
            
            if self.is_connected:
                await self._send_unsubscription(list(tokens_to_remove))
            
            logger.info(f"Unsubscribed from {len(tokens_to_remove)} markets")
    
    async def _send_subscription(self, token_ids: list[str]) -> None:
        """Send subscription message to WebSocket.
        
        Args:
            token_ids: Token IDs to subscribe to
        """
        if not self._ws:
            return
        
        message = {
            "type": "subscribe",
            "channel": "price",
            "assets": token_ids,
        }
        
        try:
            await self._ws.send(json.dumps(message))
            logger.debug(f"Subscription sent for {len(token_ids)} assets")
        except WebSocketException as e:
            logger.error(f"Failed to send subscription: {e}")
    
    async def _send_unsubscription(self, token_ids: list[str]) -> None:
        """Send unsubscription message to WebSocket.
        
        Args:
            token_ids: Token IDs to unsubscribe from
        """
        if not self._ws:
            return
        
        message = {
            "type": "unsubscribe",
            "channel": "price",
            "assets": token_ids,
        }
        
        try:
            await self._ws.send(json.dumps(message))
            logger.debug(f"Unsubscription sent for {len(token_ids)} assets")
        except WebSocketException as e:
            logger.error(f"Failed to send unsubscription: {e}")
    
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(
                self.url,
                ping_interval=self.HEARTBEAT_INTERVAL,
                ping_timeout=10.0,
            )
            self._reconnect_count = 0
            logger.info(f"WebSocket connected to {self.url}")
            
            # Resubscribe to markets after reconnection
            if self._subscribed_markets:
                await self._send_subscription(list(self._subscribed_markets))
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._ws = None
        
        logger.info("WebSocket disconnected")
    
    async def _handle_message(self, raw_message: str) -> None:
        """Process incoming WebSocket message.
        
        Args:
            raw_message: Raw JSON message string
        """
        try:
            data = json.loads(raw_message)
            
            # Update stats
            self._message_count += 1
            self._last_message_time = datetime.utcnow()
            
            # Handle different message types
            msg_type = data.get("type", "")
            
            if msg_type == "price_update" or "price" in data:
                await self._process_price_update(data)
            elif msg_type == "subscribed":
                logger.debug(f"Subscription confirmed: {data}")
            elif msg_type == "error":
                logger.error(f"WebSocket error message: {data}")
            else:
                logger.debug(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _process_price_update(self, data: dict[str, Any]) -> None:
        """Process a price update message and notify callbacks.
        
        Args:
            data: Parsed price update data
        """
        try:
            # Extract price update fields
            price_update = PriceUpdate(
                token_id=data.get("token_id", data.get("asset_id", "")),
                asset_id=data.get("asset_id"),
                price=float(data.get("price", 0)),
                bid=float(data["bid"]) if "bid" in data else None,
                ask=float(data["ask"]) if "ask" in data else None,
                spread=float(data["spread"]) if "spread" in data else None,
                volume=float(data["volume"]) if "volume" in data else None,
            )
            
            # Notify all registered callbacks
            for callback in self._callbacks:
                try:
                    await callback(price_update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing price update: {e}")
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect to WebSocket.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self._reconnect_count += 1
        
        if self._reconnect_count > self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) exceeded")
            return False
        
        logger.warning(
            f"Attempting reconnection ({self._reconnect_count}/{self.max_reconnect_attempts}) "
            f"in {self.reconnect_delay}s..."
        )
        
        await asyncio.sleep(self.reconnect_delay)
        
        try:
            await self.connect()
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    async def run(self) -> None:
        """Main loop for receiving WebSocket messages.
        
        Runs continuously, handling reconnection on connection loss.
        """
        self._running = True
        
        while self._running:
            try:
                if not self.is_connected:
                    await self.connect()
                
                async for message in self._ws:
                    if not self._running:
                        break
                    await self._handle_message(message)
                    
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                if self._running:
                    if not await self._reconnect():
                        break
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    if not await self._reconnect():
                        break
        
        await self.disconnect()
    
    def get_stats(self) -> dict:
        """Get WebSocket client statistics.
        
        Returns:
            Dictionary containing client stats
        """
        return {
            "connected": self.is_connected,
            "url": self.url,
            "subscribed_markets": len(self._subscribed_markets),
            "message_count": self._message_count,
            "last_message_time": self._last_message_time.isoformat() if self._last_message_time else None,
            "reconnect_count": self._reconnect_count,
            "callbacks_registered": len(self._callbacks),
        }
