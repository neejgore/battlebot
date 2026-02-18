"""Execution Engine for Polymarket Battle-Bot V2.0.

Handles order placement with EIP-712 authentication using the ClobClient.
Enforces 500ms delay between orders and manages order lifecycle.
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, Any
from loguru import logger

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType as ClobOrderType
from py_clob_client.constants import POLYGON

from data.models import Order, OrderSide, OrderType, OrderStatus


class ExecutionEngine:
    """Handles order execution on Polymarket.
    
    Manages authenticated order placement using the ClobClient with EIP-712
    signatures. Enforces rate limiting (500ms between orders) and provides
    order tracking.
    
    Attributes:
        clob_client: Authenticated ClobClient instance
        order_delay_ms: Minimum delay between orders in milliseconds
    """
    
    ORDER_DELAY_MS = 500  # 500ms between orders
    
    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        chain_id: int = POLYGON,
        private_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
    ):
        """Initialize the execution engine.
        
        Args:
            host: Polymarket CLOB API host
            chain_id: Blockchain chain ID (default: Polygon)
            private_key: Wallet private key (from env if not provided)
            api_key: API key for authentication
            api_secret: API secret for authentication
            api_passphrase: API passphrase for authentication
        """
        # Load credentials from environment if not provided
        self._private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY")
        self._api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self._api_secret = api_secret or os.getenv("POLYMARKET_API_SECRET")
        self._api_passphrase = api_passphrase or os.getenv("POLYMARKET_API_PASSPHRASE")
        
        if not self._private_key:
            raise ValueError("Private key required - set POLYMARKET_PRIVATE_KEY env var")
        
        # Initialize ClobClient
        self.clob_client = ClobClient(
            host=host,
            chain_id=chain_id,
            key=self._private_key,
            creds={
                "api_key": self._api_key,
                "api_secret": self._api_secret,
                "api_passphrase": self._api_passphrase,
            } if self._api_key else None,
        )
        
        self._last_order_time: Optional[datetime] = None
        self._pending_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []
        self._lock = asyncio.Lock()
        
        logger.info(f"ExecutionEngine initialized | Host: {host}")
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce minimum delay between orders."""
        if self._last_order_time is None:
            return
        
        elapsed_ms = (datetime.utcnow() - self._last_order_time).total_seconds() * 1000
        
        if elapsed_ms < self.ORDER_DELAY_MS:
            delay_needed = (self.ORDER_DELAY_MS - elapsed_ms) / 1000
            logger.debug(f"Rate limiting: waiting {delay_needed:.3f}s")
            await asyncio.sleep(delay_needed)
    
    async def create_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """Create and submit an order.
        
        Args:
            token_id: Token ID to trade
            side: BUY or SELL
            price: Order price (0-1)
            size: Order size in shares
            order_type: Order type (LIMIT, GTC, FOK)
            
        Returns:
            Order object with status and order_id (if successful)
        """
        async with self._lock:
            # Enforce rate limit
            await self._enforce_rate_limit()
            
            # Create order model
            order = Order(
                token_id=token_id,
                side=side,
                order_type=order_type,
                price=price,
                size=size,
                status=OrderStatus.PENDING,
            )
            
            try:
                # Map to ClobClient order type
                clob_order_type = self._map_order_type(order_type)
                
                # Build order args
                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=size,
                    side="BUY" if side == OrderSide.BUY else "SELL",
                )
                
                # Create signed order
                signed_order = self.clob_client.create_order(order_args)
                
                # Submit order
                response = await asyncio.to_thread(
                    self.clob_client.post_order,
                    signed_order,
                    clob_order_type,
                )
                
                # Update order with response
                if response and "orderID" in response:
                    order.order_id = response["orderID"]
                    order.status = OrderStatus.OPEN
                    self._pending_orders[order.order_id] = order
                    
                    logger.info(
                        f"Order submitted | ID: {order.order_id} | "
                        f"{side.value} {size} @ {price} | Token: {token_id[:16]}..."
                    )
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Order rejected: {response}")
                
            except Exception as e:
                order.status = OrderStatus.REJECTED
                logger.error(f"Order submission failed: {e}")
            
            finally:
                self._last_order_time = datetime.utcnow()
                order.updated_at = self._last_order_time
                self._order_history.append(order)
            
            return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        async with self._lock:
            await self._enforce_rate_limit()
            
            try:
                response = await asyncio.to_thread(
                    self.clob_client.cancel,
                    order_id,
                )
                
                if response:
                    if order_id in self._pending_orders:
                        self._pending_orders[order_id].status = OrderStatus.CANCELLED
                        self._pending_orders[order_id].updated_at = datetime.utcnow()
                        del self._pending_orders[order_id]
                    
                    logger.info(f"Order cancelled: {order_id}")
                    return True
                    
            except Exception as e:
                logger.error(f"Cancel order failed: {e}")
            
            finally:
                self._last_order_time = datetime.utcnow()
            
            return False
    
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        order_ids = list(self._pending_orders.keys())
        
        for order_id in order_ids:
            if await self.cancel_order(order_id):
                cancelled += 1
        
        logger.info(f"Cancelled {cancelled}/{len(order_ids)} orders")
        return cancelled
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Updated Order object or None if not found
        """
        try:
            response = await asyncio.to_thread(
                self.clob_client.get_order,
                order_id,
            )
            
            if response:
                # Update local order if we're tracking it
                if order_id in self._pending_orders:
                    order = self._pending_orders[order_id]
                    order.filled_size = float(response.get("size_matched", 0))
                    order.status = self._map_status(response.get("status", ""))
                    order.updated_at = datetime.utcnow()
                    
                    # Remove from pending if no longer open
                    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
                        del self._pending_orders[order_id]
                    
                    return order
            
        except Exception as e:
            logger.error(f"Get order status failed: {e}")
        
        return None
    
    async def get_open_orders(self, token_id: Optional[str] = None) -> list[Order]:
        """Get all open orders, optionally filtered by token.
        
        Args:
            token_id: Optional token ID to filter by
            
        Returns:
            List of open orders
        """
        try:
            response = await asyncio.to_thread(
                self.clob_client.get_orders,
            )
            
            orders = []
            for order_data in response or []:
                if token_id and order_data.get("token_id") != token_id:
                    continue
                
                order = Order(
                    order_id=order_data.get("id"),
                    token_id=order_data.get("token_id", ""),
                    side=OrderSide(order_data.get("side", "BUY")),
                    price=float(order_data.get("price", 0)),
                    size=float(order_data.get("original_size", 0)),
                    filled_size=float(order_data.get("size_matched", 0)),
                    status=self._map_status(order_data.get("status", "")),
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Get open orders failed: {e}")
            return []
    
    async def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            response = await asyncio.to_thread(
                self.clob_client.get_balances,
            )
            return response or []
            
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            return []
    
    def _map_order_type(self, order_type: OrderType) -> ClobOrderType:
        """Map internal order type to ClobClient order type.
        
        Args:
            order_type: Internal OrderType enum
            
        Returns:
            ClobClient OrderType
        """
        mapping = {
            OrderType.LIMIT: ClobOrderType.GTC,
            OrderType.GTC: ClobOrderType.GTC,
            OrderType.FOK: ClobOrderType.FOK,
            OrderType.MARKET: ClobOrderType.FOK,  # Market orders as FOK
        }
        return mapping.get(order_type, ClobOrderType.GTC)
    
    def _map_status(self, status_str: str) -> OrderStatus:
        """Map API status string to OrderStatus enum.
        
        Args:
            status_str: Status string from API
            
        Returns:
            OrderStatus enum value
        """
        status_str = status_str.upper()
        mapping = {
            "OPEN": OrderStatus.OPEN,
            "LIVE": OrderStatus.OPEN,
            "FILLED": OrderStatus.FILLED,
            "MATCHED": OrderStatus.FILLED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "CANCELED": OrderStatus.CANCELLED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(status_str, OrderStatus.PENDING)
    
    def get_stats(self) -> dict:
        """Get execution engine statistics.
        
        Returns:
            Dictionary containing execution stats
        """
        return {
            "pending_orders": len(self._pending_orders),
            "total_orders": len(self._order_history),
            "last_order_time": self._last_order_time.isoformat() if self._last_order_time else None,
        }
