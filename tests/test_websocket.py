"""Tests for KalshiWebSocketClient.

Tests cover:
  - Authentication header generation
  - Ticker message parsing and callback dispatch
  - Fill message parsing, callback dispatch, and fill_event signalling
  - Subscription state management (subscribe / unsubscribe / sync)
  - Graceful handling of malformed messages
  - Connection stats
  - Stale-connection detection
  - run_kalshi integration callbacks (_on_ws_ticker / _on_ws_fill)
"""

import asyncio
import json
import time
import unittest
from datetime import datetime, timedelta
from functools import wraps
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.kalshi_websocket import KalshiWebSocketClient


def async_test(coro):
    """Decorator: run an async test function synchronously via asyncio.run()."""
    @wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client(with_auth: bool = False) -> KalshiWebSocketClient:
    """Return a client with or without auth credentials."""
    if with_auth:
        # Build a minimal real RSA key so _build_auth_headers works
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.backends import default_backend
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend(),
            )
        except ImportError:
            key = None
        return KalshiWebSocketClient(api_key_id='test-key-id', private_key=key)
    return KalshiWebSocketClient()


def make_ticker_msg(
    ticker: str = 'KXSPY-24DEC20-P450',
    yes_bid: int = 34,
    yes_ask: int = 36,
    last_price: int = 35,
    volume: int = 1000,
    open_interest: int = 500,
) -> str:
    return json.dumps({
        'type': 'ticker',
        'msg': {
            'market_ticker': ticker,
            'yes_bid': yes_bid,
            'yes_ask': yes_ask,
            'price': last_price,
            'volume': volume,
            'open_interest': open_interest,
        },
    })


def make_fill_msg(
    order_id: str = 'order-abc',
    ticker: str = 'KXSPY-24DEC20-P450',
    count: int = 10,
    price: int = 35,
    action: str = 'buy',
) -> str:
    return json.dumps({
        'type': 'fill',
        'msg': {
            'order_id': order_id,
            'trade_id': 'trade-xyz',
            'market_ticker': ticker,
            'side': 'yes',
            'count': count,
            'price': price,
            'action': action,
            'is_taker': True,
            'created_time': datetime.utcnow().isoformat() + 'Z',
        },
    })


# ---------------------------------------------------------------------------
# Unit tests — no network I/O
# ---------------------------------------------------------------------------

class TestClientInitialState:
    def test_defaults(self):
        c = make_client()
        assert not c.is_connected
        assert c._subscribed_tickers == set()
        assert c._ticker_callbacks == []
        assert c._fill_callbacks == []
        assert c._msg_count == 0
        assert c._fill_count == 0
        assert c._ticker_count == 0
        assert c.last_message_age_seconds is None

    def test_is_stale_when_never_received(self):
        c = make_client()
        assert c.is_stale is True

    def test_is_stale_when_recent_message(self):
        c = make_client()
        c._last_msg_time = datetime.utcnow()
        assert c.is_stale is False

    def test_is_stale_when_old_message(self):
        c = make_client()
        c._last_msg_time = datetime.utcnow() - timedelta(seconds=200)
        assert c.is_stale is True

    def test_get_stats_shape(self):
        c = make_client()
        s = c.get_stats()
        for key in ('connected', 'url', 'msg_count', 'ticker_count',
                    'fill_count', 'last_msg_time', 'reconnect_count',
                    'subscribed_tickers'):
            assert key in s, f"Missing key: {key}"


class TestCallbackRegistration:
    def test_add_ticker_callback(self):
        c = make_client()
        cb = MagicMock()
        c.add_ticker_callback(cb)
        assert cb in c._ticker_callbacks

    def test_add_ticker_callback_idempotent(self):
        c = make_client()
        cb = MagicMock()
        c.add_ticker_callback(cb)
        c.add_ticker_callback(cb)
        assert c._ticker_callbacks.count(cb) == 1

    def test_remove_ticker_callback(self):
        c = make_client()
        cb = MagicMock()
        c.add_ticker_callback(cb)
        c.remove_ticker_callback(cb)
        assert cb not in c._ticker_callbacks

    def test_add_fill_callback(self):
        c = make_client()
        cb = MagicMock()
        c.add_fill_callback(cb)
        assert cb in c._fill_callbacks


class TestTickerHandling:
    def test_ticker_updates_stats(self):
        c = make_client()
        c._handle_ticker({'market_ticker': 'KXSPY', 'yes_bid': 34, 'yes_ask': 36, 'price': 35})
        assert c._ticker_count == 1

    def test_ticker_ignores_empty_ticker(self):
        c = make_client()
        cb = MagicMock()
        c.add_ticker_callback(cb)
        c._handle_ticker({})
        cb.assert_not_called()
        assert c._ticker_count == 0

    def test_ticker_callback_receives_correct_args(self):
        c = make_client()
        received = {}
        def cb(ticker, yes_bid, yes_ask, last_price, volume, open_interest):
            received.update(locals())
        c.add_ticker_callback(cb)
        c._handle_ticker({
            'market_ticker': 'KXSPY',
            'yes_bid': 30,
            'yes_ask': 40,
            'price': 35,
            'volume': 999,
            'open_interest': 123,
        })
        assert received['ticker'] == 'KXSPY'
        assert received['yes_bid'] == 30
        assert received['yes_ask'] == 40
        assert received['last_price'] == 35
        assert received['volume'] == 999
        assert received['open_interest'] == 123

    def test_ticker_callback_exception_is_caught(self):
        """A crashing callback must not kill the receive loop."""
        c = make_client()
        def bad_cb(*a): raise RuntimeError("boom")
        good_called = []
        def good_cb(*a): good_called.append(True)
        c.add_ticker_callback(bad_cb)
        c.add_ticker_callback(good_cb)
        # Should not raise
        c._handle_ticker({'market_ticker': 'X', 'yes_bid': 1, 'yes_ask': 2, 'price': 1})
        assert good_called  # Second callback still ran


class TestFillHandling:
    def test_fill_updates_stats(self):
        c = make_client()
        c._handle_fill({'order_id': 'o1', 'market_ticker': 'X', 'count': 5, 'price': 50, 'action': 'buy'})
        assert c._fill_count == 1

    def test_fill_sets_fill_event(self):
        c = make_client()
        assert not c.fill_event.is_set()
        c._handle_fill({'order_id': 'o1', 'market_ticker': 'X', 'count': 5, 'price': 50, 'action': 'buy'})
        assert c.fill_event.is_set()

    def test_fill_callback_receives_payload(self):
        c = make_client()
        payloads = []
        c.add_fill_callback(lambda p: payloads.append(p))
        payload = {'order_id': 'o1', 'market_ticker': 'X', 'count': 5, 'price': 50, 'action': 'buy'}
        c._handle_fill(payload)
        assert len(payloads) == 1
        assert payloads[0]['order_id'] == 'o1'

    def test_fill_callback_exception_is_caught(self):
        c = make_client()
        def bad_cb(p): raise ValueError("nope")
        c.add_fill_callback(bad_cb)
        # Should not raise
        c._handle_fill({'order_id': 'o1', 'market_ticker': 'X', 'count': 1, 'price': 10, 'action': 'sell'})


class TestMessageDispatch:
    @async_test
    async def test_valid_ticker_json(self):
        c = make_client()
        received = []
        c.add_ticker_callback(lambda *a: received.append(a))
        await c._handle_message(make_ticker_msg())
        assert c._msg_count == 1
        assert len(received) == 1

    @async_test
    async def test_valid_fill_json(self):
        c = make_client()
        fills = []
        c.add_fill_callback(lambda f: fills.append(f))
        await c._handle_message(make_fill_msg())
        assert c._msg_count == 1
        assert len(fills) == 1

    @async_test
    async def test_invalid_json_does_not_raise(self):
        c = make_client()
        await c._handle_message("not-json{{")
        assert c._msg_count == 0

    @async_test
    async def test_subscribed_message_is_ignored_gracefully(self):
        c = make_client()
        msg = json.dumps({'type': 'subscribed', 'msg': {'channel': 'ticker', 'market_tickers': []}})
        await c._handle_message(msg)
        assert c._msg_count == 1

    @async_test
    async def test_error_message_is_logged_not_raised(self):
        c = make_client()
        msg = json.dumps({'type': 'error', 'msg': {'code': 'UNAUTHORIZED', 'message': 'bad key'}})
        await c._handle_message(msg)  # Must not raise
        assert c._msg_count == 1

    @async_test
    async def test_last_msg_time_updated(self):
        c = make_client()
        before = datetime.utcnow()
        await c._handle_message(make_ticker_msg())
        assert c._last_msg_time >= before


class TestSubscriptionManagement:
    @async_test
    async def test_subscribe_adds_to_set(self):
        c = make_client()
        await c.subscribe_tickers(['KXSPY', 'KXBTC'])
        assert 'KXSPY' in c._subscribed_tickers
        assert 'KXBTC' in c._subscribed_tickers

    @async_test
    async def test_subscribe_idempotent(self):
        c = make_client()
        await c.subscribe_tickers(['KXSPY'])
        await c.subscribe_tickers(['KXSPY'])
        assert len(c._subscribed_tickers) == 1

    @async_test
    async def test_unsubscribe_removes_from_set(self):
        c = make_client()
        await c.subscribe_tickers(['KXSPY', 'KXBTC'])
        await c.unsubscribe_tickers(['KXSPY'])
        assert 'KXSPY' not in c._subscribed_tickers
        assert 'KXBTC' in c._subscribed_tickers

    @async_test
    async def test_unsubscribe_noop_when_not_subscribed(self):
        c = make_client()
        await c.unsubscribe_tickers(['NONEXISTENT'])  # Must not raise

    @async_test
    async def test_sync_subscriptions_adds_and_removes(self):
        c = make_client()
        c._subscribed_tickers = {'OLD1', 'OLD2', 'KEEP'}
        await c.sync_subscriptions(['KEEP', 'NEW1', 'NEW2'])
        assert 'KEEP' in c._subscribed_tickers
        assert 'NEW1' in c._subscribed_tickers
        assert 'NEW2' in c._subscribed_tickers
        assert 'OLD1' not in c._subscribed_tickers
        assert 'OLD2' not in c._subscribed_tickers

    @async_test
    async def test_subscribe_does_not_send_when_disconnected(self):
        """subscribe_tickers should NOT attempt to send if _ws is None."""
        c = make_client()
        c._ws = None
        c._connected = False
        # Should complete without error even though there's no connection
        await c.subscribe_tickers(['KXSPY'])
        assert 'KXSPY' in c._subscribed_tickers

    @async_test
    async def test_subscribe_sends_when_connected(self):
        """subscribe_tickers should call _send_subscribe_cmd when connected."""
        c = make_client()
        c._connected = True
        mock_ws = AsyncMock()
        c._ws = mock_ws
        c._send_subscribe_cmd = AsyncMock()
        await c.subscribe_tickers(['KXSPY'])
        c._send_subscribe_cmd.assert_awaited_once()
        args = c._send_subscribe_cmd.call_args[0]
        assert 'KXSPY' in args[0]
        assert 'ticker' in args[1]


class TestAuthHeaders:
    def test_no_credentials_returns_empty_dict(self):
        c = make_client(with_auth=False)
        assert c._build_auth_headers() == {}

    def test_with_credentials_returns_required_keys(self):
        c = make_client(with_auth=True)
        if c._private_key is None:
            pytest.skip("cryptography library not available")
        headers = c._build_auth_headers()
        assert 'KALSHI-ACCESS-KEY' in headers
        assert 'KALSHI-ACCESS-TIMESTAMP' in headers
        assert 'KALSHI-ACCESS-SIGNATURE' in headers

    def test_timestamp_is_milliseconds(self):
        c = make_client(with_auth=True)
        if c._private_key is None:
            pytest.skip("cryptography library not available")
        headers = c._build_auth_headers()
        if not headers:
            pytest.skip("No headers returned")
        ts = int(headers['KALSHI-ACCESS-TIMESTAMP'])
        # Millisecond epoch should be 13 digits in 2024
        assert 1_000_000_000_000 < ts < 9_999_999_999_999

    def test_signature_is_base64(self):
        import base64
        c = make_client(with_auth=True)
        if c._private_key is None:
            pytest.skip("cryptography library not available")
        headers = c._build_auth_headers()
        if not headers:
            pytest.skip("No headers returned")
        sig = headers['KALSHI-ACCESS-SIGNATURE']
        # Should be valid base64 (no exception)
        decoded = base64.b64decode(sig)
        assert len(decoded) > 0


class TestCommandHelpers:
    def test_next_cmd_id_increments(self):
        c = make_client()
        ids = [c._next_cmd_id() for _ in range(5)]
        assert ids == list(range(1, 6))

    @async_test
    async def test_send_subscribe_cmd_noop_when_no_ws(self):
        c = make_client()
        c._ws = None
        await c._send_subscribe_cmd(['T1'], ['ticker'])  # Must not raise

    @async_test
    async def test_send_subscribe_cmd_sends_json(self):
        c = make_client()
        mock_ws = AsyncMock()
        c._ws = mock_ws
        await c._send_subscribe_cmd(['T1', 'T2'], ['ticker'])
        mock_ws.send.assert_awaited_once()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload['cmd'] == 'subscribe'
        assert 'ticker' in payload['params']['channels']
        assert set(payload['params']['market_tickers']) == {'T1', 'T2'}

    @async_test
    async def test_send_unsubscribe_cmd_sends_json(self):
        c = make_client()
        mock_ws = AsyncMock()
        c._ws = mock_ws
        await c._send_unsubscribe_cmd(['T1'], ['ticker'])
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload['cmd'] == 'unsubscribe'

    @async_test
    async def test_send_subscribe_omits_market_tickers_for_fill(self):
        """fill channel subscription has no market_tickers key."""
        c = make_client()
        mock_ws = AsyncMock()
        c._ws = mock_ws
        await c._send_subscribe_cmd([], ['fill'])
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert 'market_tickers' not in payload['params']
        assert 'fill' in payload['params']['channels']


class TestStopMethod:
    @async_test
    async def test_stop_clears_running(self):
        c = make_client()
        c._running = True
        await c.stop()
        assert not c._running

    @async_test
    async def test_stop_closes_ws(self):
        c = make_client()
        mock_ws = AsyncMock()
        c._ws = mock_ws
        c._running = True
        await c.stop()
        mock_ws.close.assert_awaited_once()

    @async_test
    async def test_stop_handles_ws_close_error(self):
        c = make_client()
        mock_ws = AsyncMock()
        mock_ws.close.side_effect = RuntimeError("already closed")
        c._ws = mock_ws
        c._running = True
        await c.stop()  # Must not raise


# ---------------------------------------------------------------------------
# Integration: run_kalshi _on_ws_ticker / _on_ws_fill callbacks
# ---------------------------------------------------------------------------

class TestRunKalshiCallbacks:
    """Test that the bot callbacks correctly update internal state."""

    def _make_bot(self):
        """Build a minimal KalshiBattleBot without file I/O or API calls."""
        import os
        os.environ.setdefault('DRY_RUN', 'true')
        os.environ.setdefault('KALSHI_API_KEY_ID', '')
        # Patch KalshiClient to avoid network/file access during construction
        with patch('services.kalshi_client.KalshiClient.__init__', return_value=None), \
             patch('services.kalshi_websocket.KalshiWebSocketClient.__init__', return_value=None), \
             patch('run_kalshi.KalshiBattleBot._load_state', return_value=None), \
             patch('run_kalshi.TelemetryDB'), \
             patch('run_kalshi.AISignalGenerator'), \
             patch('run_kalshi.CalibrationEngine'), \
             patch('run_kalshi.get_intelligence_service'), \
             patch('run_kalshi.CryptoEdgeService'):
            from run_kalshi import KalshiBattleBot
            bot = object.__new__(KalshiBattleBot)
            # Minimal required attributes
            bot._markets = {}
            bot._monitored = {}
            bot._pending_orders = {}
            bot._positions = {}
            bot._price_update_count = 0
            # Stub _ws_client so fill_event exists
            bot._ws_client = MagicMock()
            bot._ws_client.fill_event = asyncio.Event()
        return bot

    def test_on_ws_ticker_updates_markets(self):
        bot = self._make_bot()
        bot._markets['KXSPY'] = {'yes_price': 0.0}
        bot._monitored['KXSPY'] = {'yes_price': 0.0}
        bot._on_ws_ticker('KXSPY', yes_bid=34, yes_ask=36, last_price=35,
                          volume=1000, open_interest=500)
        assert abs(bot._markets['KXSPY']['yes_price'] - 0.35) < 0.001
        assert abs(bot._markets['KXSPY']['no_price'] - 0.65) < 0.001
        assert bot._markets['KXSPY']['volume'] == 1000
        assert bot._price_update_count == 1

    def test_on_ws_ticker_updates_monitored(self):
        bot = self._make_bot()
        bot._markets['KXSPY'] = {'yes_price': 0.0}
        bot._monitored['KXSPY'] = {'yes_price': 0.0}
        bot._on_ws_ticker('KXSPY', yes_bid=60, yes_ask=70, last_price=65,
                          volume=500, open_interest=200)
        assert abs(bot._monitored['KXSPY']['yes_price'] - 0.65) < 0.001

    def test_on_ws_ticker_skips_unknown_market(self):
        """Ticker for an un-monitored market should NOT be inserted."""
        bot = self._make_bot()
        bot._on_ws_ticker('UNKNOWN', yes_bid=30, yes_ask=40, last_price=35,
                          volume=0, open_interest=0)
        assert 'UNKNOWN' not in bot._markets
        assert 'UNKNOWN' not in bot._monitored

    def test_on_ws_ticker_falls_back_to_last_price(self):
        """When yes_bid/yes_ask are absent, use last_price."""
        bot = self._make_bot()
        bot._markets['KXSPY'] = {}
        bot._monitored['KXSPY'] = {}
        bot._on_ws_ticker('KXSPY', yes_bid=None, yes_ask=None,
                          last_price=50, volume=None, open_interest=None)
        assert abs(bot._markets['KXSPY']['yes_price'] - 0.50) < 0.001

    def test_on_ws_ticker_returns_when_no_prices(self):
        """When both bid/ask and last_price are None, nothing is updated."""
        bot = self._make_bot()
        bot._markets['KXSPY'] = {'yes_price': 0.42}
        bot._on_ws_ticker('KXSPY', yes_bid=None, yes_ask=None,
                          last_price=None, volume=None, open_interest=None)
        # Should remain unchanged
        assert abs(bot._markets['KXSPY']['yes_price'] - 0.42) < 0.001
        assert bot._price_update_count == 0

    def test_on_ws_fill_logs_known_order(self, capsys):
        bot = self._make_bot()
        bot._pending_orders['order-123'] = {'ticker': 'KXSPY', 'contracts': 5}
        bot._on_ws_fill({'order_id': 'order-123', 'market_ticker': 'KXSPY',
                         'count': 5, 'price': 35, 'action': 'buy'})
        captured = capsys.readouterr()
        assert 'order-123' in captured.out

    def test_on_ws_fill_silent_for_unknown_order(self):
        bot = self._make_bot()
        # Should not raise for an order the bot doesn't know about
        bot._on_ws_fill({'order_id': 'unknown-999', 'market_ticker': 'KXSPY',
                         'count': 1, 'price': 50, 'action': 'buy'})

    def test_on_ws_ticker_price_count_increments(self):
        bot = self._make_bot()
        bot._markets['T1'] = {}
        bot._markets['T2'] = {}
        bot._on_ws_ticker('T1', 30, 40, 35, 100, 50)
        bot._on_ws_ticker('T2', 60, 70, 65, 200, 100)
        assert bot._price_update_count == 2


