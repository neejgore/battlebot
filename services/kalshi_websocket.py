"""Kalshi WebSocket Client for BattleBot.

Maintains a persistent, authenticated WebSocket connection to Kalshi's
real-time API (wss://api.elections.kalshi.com/trade-api/ws/v2).

Two channels are used:
  ticker   — bid/ask/last/volume updates for every subscribed market.
             Replaces the per-market GET /markets/{ticker} polling loop
             (price_refresh_loop) so prices update in real time rather
             than on a 30-second REST poll cycle.
  fill     — immediate notification when any of our orders execute.
             Replaces the 10-second GET /orders/{id} polling loop so
             entry and exit fills are processed without delay.

The bot falls back gracefully to REST polling whenever the WebSocket is
disconnected. The polling intervals are lengthened (30 s → 5 min for
prices, 10 s → 60 s for order checks) when the WS is live so we stay
well inside Kalshi's rate limits.

Authentication:
  Same RSA-PSS signature scheme as the REST client, passed as
  HTTP headers on the WebSocket upgrade request.
  Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
  Signing message: f"{timestamp_ms}GET/trade-api/ws/v2"
"""

import asyncio
import base64
import json
import time
from datetime import datetime
from typing import Callable, Optional

from loguru import logger

try:
    from websockets.asyncio.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("websockets package not available — KalshiWebSocketClient disabled")

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

# Callback for ticker updates.
# Args: ticker (str), yes_bid (int|None), yes_ask (int|None),
#       last_price (int|None), volume (int|None), open_interest (int|None)
TickerCallback = Callable[..., None]

# Callback for fill events.
# Args: fill_msg (dict) — raw "msg" sub-dict from the Kalshi fill event.
FillCallback = Callable[[dict], None]


# ---------------------------------------------------------------------------
# KalshiWebSocketClient
# ---------------------------------------------------------------------------

class KalshiWebSocketClient:
    """Persistent, authenticated WebSocket connection to Kalshi's real-time API.

    Usage::

        ws = KalshiWebSocketClient(
            api_key_id=os.getenv('KALSHI_API_KEY_ID'),
            private_key=loaded_rsa_key,
            use_demo=False,
        )
        ws.add_ticker_callback(my_on_ticker)
        ws.add_fill_callback(my_on_fill)

        # Start background task (non-blocking):
        asyncio.create_task(ws.run())

        # Subscribe to markets (safe to call before/after connection):
        await ws.subscribe_tickers(['KXETHD-24DEC01-T2000', 'KXSPY-24DEC20-P450'])

        # Later, shut down cleanly:
        await ws.stop()

    The ``run()`` coroutine never returns under normal operation; it loops
    forever reconnecting on transient failures.  Call ``stop()`` to
    terminate it.
    """

    PROD_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    # Seconds between reconnect attempts (caps at _MAX_RECONNECT_DELAY).
    _INITIAL_RECONNECT_DELAY = 2.0
    _MAX_RECONNECT_DELAY = 60.0

    # Maximum age (seconds) of the last received message before we consider
    # the connection stale.  We rely on the WS ping-pong mechanism for
    # keep-alive, but track message age for dashboard display.
    _STALE_THRESHOLD = 120.0

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key=None,
        use_demo: bool = False,
    ):
        """
        Args:
            api_key_id:  Kalshi API key ID (KALSHI-ACCESS-KEY).
            private_key: Loaded RSA private key object (from cryptography lib).
                         If None, connection works but only unauthenticated
                         public channels are available (no fill channel).
            use_demo:    Connect to demo environment instead of production.
        """
        self._api_key_id = api_key_id
        self._private_key = private_key
        self._ws_url = self.DEMO_WS_URL if use_demo else self.PROD_WS_URL

        self._ticker_callbacks: list[TickerCallback] = []
        self._fill_callbacks: list[FillCallback] = []

        # Tickers we want to receive ticker-channel updates for.
        self._subscribed_tickers: set[str] = set()

        # Subscription commands waiting to be sent after (re-)connect.
        # Keyed by sequential command id (int).
        self._pending_cmd_id: int = 0

        # Runtime state
        self._running = False
        self._ws = None  # active websockets.ClientConnection
        self._connected = False

        # Stats
        self._msg_count: int = 0
        self._ticker_count: int = 0
        self._fill_count: int = 0
        self._last_msg_time: Optional[datetime] = None
        self._reconnect_count: int = 0
        self._connect_time: Optional[datetime] = None

        # Fill event — the position sync loop waits on this so it can
        # process fills immediately instead of on the next 10-second poll.
        self.fill_event: asyncio.Event = asyncio.Event()

        # Lock protecting subscription state during concurrent subscribe calls.
        self._sub_lock: asyncio.Lock = asyncio.Lock()

        logger.info(f"KalshiWebSocketClient initialised | URL: {self._ws_url} | "
                    f"Auth: {bool(api_key_id and private_key)}")

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def add_ticker_callback(self, cb: TickerCallback) -> None:
        """Register a callback for real-time ticker updates.

        The callback is called synchronously in the receive loop; it should
        be fast (no I/O, no blocking calls).  Signature::

            def cb(ticker: str, yes_bid: Optional[int], yes_ask: Optional[int],
                   last_price: Optional[int], volume: Optional[int],
                   open_interest: Optional[int]) -> None
        """
        if cb not in self._ticker_callbacks:
            self._ticker_callbacks.append(cb)

    def remove_ticker_callback(self, cb: TickerCallback) -> None:
        """Remove a previously registered ticker callback."""
        self._ticker_callbacks.discard(cb) if hasattr(
            self._ticker_callbacks, 'discard') else None
        try:
            self._ticker_callbacks.remove(cb)
        except ValueError:
            pass

    def add_fill_callback(self, cb: FillCallback) -> None:
        """Register a callback for fill events.

        The callback is called synchronously with the raw fill message dict.
        Signature::

            def cb(fill: dict) -> None
        """
        if cb not in self._fill_callbacks:
            self._fill_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def subscribe_tickers(self, tickers: list[str]) -> None:
        """Subscribe to ticker-channel updates for the given market tickers.

        Safe to call at any time (before or after connect).  New tickers are
        sent immediately if the connection is live; otherwise queued for the
        next (re-)connect.

        Args:
            tickers: Kalshi market ticker strings, e.g. ``["KXSPY-24DEC20-P450"]``
        """
        async with self._sub_lock:
            new = set(tickers) - self._subscribed_tickers
            if not new:
                return
            self._subscribed_tickers.update(new)
            if self._connected and self._ws:
                await self._send_subscribe_cmd(list(new), ['ticker'])
                logger.debug(f"[WS] Subscribed to {len(new)} new tickers")

    async def unsubscribe_tickers(self, tickers: list[str]) -> None:
        """Unsubscribe from ticker-channel updates.

        Args:
            tickers: Market tickers to stop receiving updates for.
        """
        async with self._sub_lock:
            remove = set(tickers) & self._subscribed_tickers
            if not remove:
                return
            self._subscribed_tickers -= remove
            if self._connected and self._ws:
                await self._send_unsubscribe_cmd(list(remove), ['ticker'])
                logger.debug(f"[WS] Unsubscribed from {len(remove)} tickers")

    async def sync_subscriptions(self, tickers: list[str]) -> None:
        """Align WebSocket subscriptions with the provided ticker list.

        Adds tickers missing from current subscriptions and removes those
        no longer in the list.  Called by ``_price_refresh_loop`` whenever
        the set of monitored markets changes.

        Args:
            tickers: Desired complete list of tickers to track.
        """
        desired = set(tickers)
        current = set(self._subscribed_tickers)
        to_add = desired - current
        to_remove = current - desired

        if to_add:
            await self.subscribe_tickers(list(to_add))
        if to_remove:
            await self.unsubscribe_tickers(list(to_remove))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Signal the run loop to terminate and close the connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("[WS] KalshiWebSocketClient stopped")

    @property
    def is_connected(self) -> bool:
        """True if the WebSocket connection is currently open."""
        return self._connected

    @property
    def last_message_age_seconds(self) -> Optional[float]:
        """Seconds since the last message was received, or None if never."""
        if self._last_msg_time is None:
            return None
        return (datetime.utcnow() - self._last_msg_time).total_seconds()

    @property
    def is_stale(self) -> bool:
        """True if no message has been received in _STALE_THRESHOLD seconds."""
        age = self.last_message_age_seconds
        return age is None or age > self._STALE_THRESHOLD

    def get_stats(self) -> dict:
        """Return a snapshot of connection statistics for dashboard display."""
        return {
            'connected': self._connected,
            'url': self._ws_url,
            'msg_count': self._msg_count,
            'ticker_count': self._ticker_count,
            'fill_count': self._fill_count,
            'last_msg_time': self._last_msg_time.isoformat() if self._last_msg_time else None,
            'last_msg_age_s': self.last_message_age_seconds,
            'reconnect_count': self._reconnect_count,
            'connect_time': self._connect_time.isoformat() if self._connect_time else None,
            'subscribed_tickers': len(self._subscribed_tickers),
        }

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main connection loop.

        Connects to Kalshi's WS endpoint and receives messages indefinitely.
        On disconnection, reconnects with exponential backoff.  Returns only
        when ``stop()`` has been called.
        """
        if not HAS_WEBSOCKETS:
            logger.error("[WS] websockets package not installed — skipping WS run")
            return

        self._running = True
        delay = self._INITIAL_RECONNECT_DELAY

        while self._running:
            try:
                headers = self._build_auth_headers()
                logger.info(f"[WS] Connecting to {self._ws_url}...")

                async with ws_connect(
                    self._ws_url,
                    additional_headers=headers,
                    ping_interval=30,
                    ping_timeout=15,
                    open_timeout=15,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._connect_time = datetime.utcnow()
                    self._reconnect_count += 1 if self._reconnect_count > 0 else 0
                    delay = self._INITIAL_RECONNECT_DELAY  # reset backoff on success
                    logger.info("[WS] Connected to Kalshi WebSocket")

                    # Always subscribe fill channel (private, requires auth)
                    if self._api_key_id and self._private_key:
                        await self._send_subscribe_cmd([], ['fill'])

                    # Subscribe to all tickers we already know about
                    async with self._sub_lock:
                        if self._subscribed_tickers:
                            await self._send_subscribe_cmd(
                                list(self._subscribed_tickers), ['ticker']
                            )

                    # Receive messages until disconnected
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle_message(raw)

            except ConnectionClosed as exc:
                self._connected = False
                self._ws = None
                if not self._running:
                    break
                logger.warning(f"[WS] Connection closed ({exc.rcvd}) — "
                               f"reconnecting in {delay:.0f}s")

            except OSError as exc:
                self._connected = False
                self._ws = None
                if not self._running:
                    break
                logger.warning(f"[WS] Network error: {exc} — "
                               f"reconnecting in {delay:.0f}s")

            except Exception as exc:
                self._connected = False
                self._ws = None
                if not self._running:
                    break
                logger.error(f"[WS] Unexpected error: {exc} — "
                             f"reconnecting in {delay:.0f}s")

            if self._running:
                self._reconnect_count += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._MAX_RECONNECT_DELAY)

        self._connected = False
        self._ws = None
        logger.info("[WS] Run loop exited")

    # ------------------------------------------------------------------
    # Internal: message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch a raw WebSocket message.

        Args:
            raw: JSON string received from Kalshi.
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(f"[WS] JSON decode error: {exc}")
            return

        self._msg_count += 1
        self._last_msg_time = datetime.utcnow()

        msg_type = msg.get('type', '')

        if msg_type == 'subscribed':
            _ch = msg.get('msg', {}).get('channel', '?')
            _cnt = len(msg.get('msg', {}).get('market_tickers') or [])
            logger.info(f"[WS] Subscribed | channel={_ch} | markets={_cnt}")

        elif msg_type == 'ticker':
            self._handle_ticker(msg.get('msg', {}))

        elif msg_type == 'fill':
            self._handle_fill(msg.get('msg', {}))

        elif msg_type == 'error':
            logger.error(f"[WS] Server error: {msg.get('msg', msg)}")

        # 'subscribed_ack', heartbeat, other types — silently ignored

    def _handle_ticker(self, payload: dict) -> None:
        """Dispatch a ticker update to all registered callbacks.

        Args:
            payload: The ``msg`` sub-dict from a ``ticker`` WS message.
                     Fields are in cents (int) or None if absent.
        """
        ticker = payload.get('market_ticker', '')
        if not ticker:
            return

        yes_bid = payload.get('yes_bid')         # cents, e.g. 34
        yes_ask = payload.get('yes_ask')         # cents, e.g. 36
        last_price = payload.get('price')        # cents, last trade
        volume = payload.get('volume')
        open_interest = payload.get('open_interest')

        self._ticker_count += 1

        for cb in self._ticker_callbacks:
            try:
                cb(ticker, yes_bid, yes_ask, last_price, volume, open_interest)
            except Exception as exc:
                logger.error(f"[WS] Ticker callback error: {exc}")

    def _handle_fill(self, payload: dict) -> None:
        """Dispatch a fill event to all registered callbacks and set fill_event.

        Args:
            payload: The ``msg`` sub-dict from a ``fill`` WS message.
        """
        order_id = payload.get('order_id', '?')
        ticker = payload.get('market_ticker', '?')
        count = payload.get('count', '?')
        price = payload.get('price', '?')
        action = payload.get('action', '?')

        self._fill_count += 1
        logger.info(f"[WS] Fill | order={order_id} | {ticker} | "
                    f"{action} {count} @ {price}¢")

        for cb in self._fill_callbacks:
            try:
                cb(payload)
            except Exception as exc:
                logger.error(f"[WS] Fill callback error: {exc}")

        # Wake up the position sync loop immediately so the fill is processed
        # without waiting for the next 10-second REST poll.
        self.fill_event.set()

    # ------------------------------------------------------------------
    # Internal: command helpers
    # ------------------------------------------------------------------

    def _next_cmd_id(self) -> int:
        """Return the next sequential command ID."""
        self._pending_cmd_id += 1
        return self._pending_cmd_id

    async def _send_subscribe_cmd(
        self, market_tickers: list[str], channels: list[str]
    ) -> None:
        """Send a ``subscribe`` command to the server.

        Args:
            market_tickers: Tickers to include in the subscription
                            (empty for channels with no market filter, e.g. fill).
            channels:       Channel names, e.g. ``["ticker"]``.
        """
        if self._ws is None:
            return
        params: dict = {'channels': channels}
        if market_tickers:
            params['market_tickers'] = market_tickers
        cmd = {'id': self._next_cmd_id(), 'cmd': 'subscribe', 'params': params}
        try:
            await self._ws.send(json.dumps(cmd))
        except Exception as exc:
            logger.error(f"[WS] Failed to send subscribe: {exc}")

    async def _send_unsubscribe_cmd(
        self, market_tickers: list[str], channels: list[str]
    ) -> None:
        """Send an ``unsubscribe`` command to the server.

        Args:
            market_tickers: Tickers to remove from the subscription.
            channels:       Channel names.
        """
        if self._ws is None:
            return
        params: dict = {'channels': channels}
        if market_tickers:
            params['market_tickers'] = market_tickers
        cmd = {'id': self._next_cmd_id(), 'cmd': 'unsubscribe', 'params': params}
        try:
            await self._ws.send(json.dumps(cmd))
        except Exception as exc:
            logger.error(f"[WS] Failed to send unsubscribe: {exc}")

    # ------------------------------------------------------------------
    # Internal: authentication
    # ------------------------------------------------------------------

    def _build_auth_headers(self) -> dict:
        """Build RSA-PSS authentication headers for the WebSocket upgrade.

        Uses the same signing scheme as the REST client:
          message = f"{timestamp_ms}GET/trade-api/ws/v2"

        Returns an empty dict if credentials are not configured, allowing
        unauthenticated connections (public channels only).

        Returns:
            Dict of header name → value, or {} if no credentials.
        """
        if not (self._api_key_id and self._private_key and HAS_CRYPTO):
            return {}

        try:
            timestamp_ms = int(time.time() * 1000)
            message = f"{timestamp_ms}GET/trade-api/ws/v2".encode()
            signature_bytes = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            signature = base64.b64encode(signature_bytes).decode()
            return {
                'KALSHI-ACCESS-KEY': self._api_key_id,
                'KALSHI-ACCESS-TIMESTAMP': str(timestamp_ms),
                'KALSHI-ACCESS-SIGNATURE': signature,
            }
        except Exception as exc:
            logger.error(f"[WS] Failed to build auth headers: {exc}")
            return {}
