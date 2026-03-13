"""CryptoEdgeService: Quantitative edge calculator for crypto range bucket markets.

Instead of asking Claude to guess which $3,000 BTC price bucket wins,
this service uses actual derivatives market data to compute the true
log-normal probability that price lands in the Kalshi range.

Data sources (both free, no API key required):
  - Deribit DVOL index (BTC/ETH implied vol): https://www.deribit.com/api/v2/public
  - Binance spot price + realized vol fallback: https://api.binance.com/api/v3

Edge formula:
  edge = lognormal_range_prob(spot, low, high, iv, T) - kalshi_price
"""

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class CryptoEdgeResult:
    """Quantitative analysis result for a crypto range market."""
    asset: str             # 'BTC', 'ETH', 'SOL', 'DOGE'
    spot_price: float      # Current spot price
    range_low: float       # Range lower bound
    range_high: float      # Range upper bound
    implied_vol: float     # Annualized implied vol (e.g. 0.65 = 65%)
    hours_to_expiry: float # Time to resolution in hours
    quant_prob: float      # Log-normal P(price lands in range)
    kalshi_price: float    # Current Kalshi market price for this side
    edge: float            # quant_prob - kalshi_price
    context_summary: str   # Human-readable block for Claude prompt injection
    vol_source: str        # 'deribit_dvol', 'realized_7d', or 'historical_default'


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class CryptoEdgeService:
    """Quantitative probability engine for crypto range bucket markets.

    Usage:
        service = CryptoEdgeService()
        result = await service.evaluate_range_market(ticker, question, price, hours)
        if result and result.edge >= MIN_EDGE:
            # pass result.context_summary to Claude
    """

    DERIBIT_BASE   = "https://www.deribit.com/api/v2/public"
    BINANCE_BASE   = "https://api.binance.com/api/v3"
    COINBASE_BASE  = "https://api.coinbase.com/v2/prices"
    KRAKEN_BASE    = "https://api.kraken.com/0/public"
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    # Kraken symbol map (Kraken uses different symbols)
    _KRAKEN_SYMBOLS = {
        'BTC': 'XBTUSD',
        'ETH': 'ETHUSD',
        'SOL': 'SOLUSD',
        'DOGE': 'XDGUSD',
        'XRP': 'XXRPZUSD',
        'BCH': 'BCHUSD',
    }
    # CoinGecko ID map
    _COINGECKO_IDS = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'DOGE': 'dogecoin',
        'XRP': 'ripple',
        'BCH': 'bitcoin-cash',
    }

    # Default annualized vol by asset when all APIs fail
    _VOL_DEFAULTS = {
        'BTC':    0.70,
        'ETH':    0.90,
        'SOL':    1.20,
        'DOGE':   1.60,
        'XRP':    1.10,   # similar volatility tier to SOL
        'BCH':    0.95,   # Bitcoin Cash: higher than BTC, lower than altcoins
    }

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        # {asset: (fetched_at_ts, vol, vol_source)}
        self._vol_cache: dict[str, tuple[float, float, str]] = {}
        # {asset: (fetched_at_ts, price)}
        self._price_cache: dict[str, tuple[float, float]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=8.0)
        return self._client

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Spot price
    # ------------------------------------------------------------------

    async def get_spot_price(self, asset: str) -> Optional[float]:
        """Fetch current spot price — tries Binance, Coinbase, Kraken, CoinGecko in order.

        Binance returns 451 (geo-blocked) on some hosting providers (e.g. Railway US).
        Multiple fallbacks ensure the quant model always has a live price.
        30-second cache shared across all sources.
        """
        now = time.time()
        cached = self._price_cache.get(asset)
        if cached and now - cached[0] < 30:
            return cached[1]

        client = await self._get_client()

        # 1. Binance (fastest, most liquid — may be geo-blocked)
        try:
            r = await client.get(
                f"{self.BINANCE_BASE}/ticker/price",
                params={"symbol": f"{asset}USDT"},
            )
            r.raise_for_status()
            price = float(r.json()["price"])
            self._price_cache[asset] = (now, price)
            return price
        except Exception:
            pass

        # 2. Coinbase (widely available, no auth required)
        try:
            r = await client.get(f"{self.COINBASE_BASE}/{asset}-USD/spot")
            r.raise_for_status()
            price = float(r.json()["data"]["amount"])
            self._price_cache[asset] = (now, price)
            logger.debug(f"[CryptoEdge] Coinbase spot for {asset}: ${price:,.2f}")
            return price
        except Exception:
            pass

        # 3. Kraken
        try:
            kraken_sym = self._KRAKEN_SYMBOLS.get(asset)
            if kraken_sym:
                r = await client.get(
                    f"{self.KRAKEN_BASE}/Ticker",
                    params={"pair": kraken_sym},
                )
                r.raise_for_status()
                result = r.json().get("result", {})
                for v in result.values():
                    price = float(v["c"][0])  # last trade close price
                    self._price_cache[asset] = (now, price)
                    logger.debug(f"[CryptoEdge] Kraken spot for {asset}: ${price:,.2f}")
                    return price
        except Exception:
            pass

        # 4. CoinGecko (rate-limited but free fallback)
        try:
            cg_id = self._COINGECKO_IDS.get(asset)
            if cg_id:
                r = await client.get(
                    f"{self.COINGECKO_BASE}/simple/price",
                    params={"ids": cg_id, "vs_currencies": "usd"},
                )
                r.raise_for_status()
                price = float(r.json()[cg_id]["usd"])
                self._price_cache[asset] = (now, price)
                logger.debug(f"[CryptoEdge] CoinGecko spot for {asset}: ${price:,.2f}")
                return price
        except Exception:
            pass

        logger.warning(f"[CryptoEdge] All spot price sources failed for {asset}")
        return None

    # ------------------------------------------------------------------
    # Implied volatility
    # ------------------------------------------------------------------

    async def get_implied_vol(self, asset: str) -> tuple[float, str]:
        """Get annualized implied vol (5-minute cache).

        Priority: Deribit DVOL → 7-day realized vol from Binance → hard default.
        Returns (vol, source_label).
        """
        now = time.time()
        cached = self._vol_cache.get(asset)
        if cached and now - cached[0] < 300:
            return cached[1], cached[2]

        # 1. Deribit DVOL (only available for BTC and ETH)
        if asset in ('BTC', 'ETH'):
            try:
                vol, src = await self._fetch_deribit_dvol(asset)
                if vol:
                    self._vol_cache[asset] = (now, vol, src)
                    return vol, src
            except Exception as exc:
                logger.warning(f"[CryptoEdge] Deribit DVOL failed for {asset}: {exc}")

        # 2. Realized vol from Binance hourly klines
        try:
            vol, src = await self._fetch_realized_vol(asset)
            if vol:
                self._vol_cache[asset] = (now, vol, src)
                return vol, src
        except Exception as exc:
            logger.warning(f"[CryptoEdge] Realized vol fallback failed for {asset}: {exc}")

        # 3. Hard default
        vol = self._VOL_DEFAULTS.get(asset, 0.80)
        src = 'historical_default'
        self._vol_cache[asset] = (now, vol, src)
        return vol, src

    async def _fetch_deribit_dvol(self, asset: str) -> tuple[Optional[float], str]:
        """Fetch the Deribit DVOL implied-vol index (annualized %)."""
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - 3_600_000  # 1 hour back

        client = await self._get_client()
        r = await client.get(
            f"{self.DERIBIT_BASE}/get_volatility_index_data",
            params={
                "currency": asset,
                "resolution": 3600,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
            },
        )
        r.raise_for_status()
        ticks = r.json().get("result", {}).get("data", [])
        if not ticks:
            return None, ""

        # Each tick: [timestamp_ms, open, high, low, close]
        # DVOL is in percentage points (e.g. 65.3 = 65.3%), convert to decimal
        dvol_close = ticks[-1][4]
        vol = dvol_close / 100.0
        return vol, "deribit_dvol"

    async def _fetch_realized_vol(self, asset: str) -> tuple[Optional[float], str]:
        """Compute 7-day realized vol from Binance hourly klines (annualized)."""
        symbol = f"{asset}USDT"
        client = await self._get_client()
        r = await client.get(
            f"{self.BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": "1h", "limit": 168},  # 7d × 24h
        )
        r.raise_for_status()
        klines = r.json()
        if len(klines) < 24:
            return None, ""

        closes = [float(k[4]) for k in klines]
        log_returns = [
            math.log(closes[i + 1] / closes[i])
            for i in range(len(closes) - 1)
        ]
        hourly_std = math.sqrt(sum(r ** 2 for r in log_returns) / len(log_returns))
        annualized_vol = hourly_std * math.sqrt(8760)
        return annualized_vol, "realized_7d"

    # ------------------------------------------------------------------
    # Range parsing
    # ------------------------------------------------------------------

    def parse_range_from_ticker(self, ticker: str) -> Optional[tuple[float, float]]:
        """Extract (low, high) from Kalshi ticker like KXBTC-26MAR06-B95000T96000.

        Handles both integer prices (BTC: B95000T96000) and decimal prices
        (XRP: B2T2.50, ETH: B2100T2200).
        """
        up = ticker.upper()
        m = re.search(r"-B([\d.]+)T([\d.]+)", up)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo >= hi:  # degenerate parse (e.g. T missing decimal part)
                return None
            return lo, hi
        # Reverse order guard
        m = re.search(r"-T([\d.]+)B([\d.]+)", up)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            lo, hi = min(a, b), max(a, b)
            if lo >= hi:
                return None
            return lo, hi
        return None

    def parse_threshold_from_ticker(self, ticker: str) -> Optional[tuple[str, float]]:
        """Extract (direction, threshold) from Kalshi threshold market tickers.

        Kalshi switched from range markets (BxTy) to threshold markets:
          KXBTC-26MAR1301-B78875     → ('below', 78875.0)   YES = P(BTC < 78875)
          KXBTC-26MAR1301-T78999.99  → ('above', 78999.99)  YES = P(BTC > 78999.99)

        Returns None if not a threshold market (keeps range market path unaffected).
        """
        up = ticker.upper()
        # Must NOT have both B and T in the price suffix (that's a range market)
        # Threshold suffix is the LAST segment after the date part
        parts = up.rsplit('-', 1)
        if len(parts) != 2:
            return None
        suffix = parts[1]
        # Below threshold: starts with B, no T following
        if suffix.startswith('B') and 'T' not in suffix:
            try:
                return ('below', float(suffix[1:]))
            except ValueError:
                return None
        # Above threshold: starts with T, no B following
        if suffix.startswith('T') and 'B' not in suffix:
            try:
                return ('above', float(suffix[1:]))
            except ValueError:
                return None
        return None

    def parse_range_from_question(self, question: str) -> Optional[tuple[float, float]]:
        """Extract (low, high) from text like 'between $95,000 and $96,000'."""
        nums_raw = re.findall(r"\$[\d,]+(?:\.\d+)?", question)
        parsed: list[float] = []
        for n in nums_raw:
            try:
                parsed.append(float(n.replace("$", "").replace(",", "")))
            except ValueError:
                pass
        if len(parsed) >= 2:
            a, b = parsed[0], parsed[1]
            return min(a, b), max(a, b)
        return None

    def detect_asset(self, ticker: str, question: str) -> Optional[str]:
        """Detect the underlying crypto asset. Returns None for NASDAQ (different model)."""
        up = ticker.upper()
        ql = question.lower()
        if up.startswith("KXNASDAQ") or "nasdaq" in ql:
            return None  # NASDAQ range: equity index, skip quant model for now
        if up.startswith("KXBTC") or "bitcoin" in ql or " btc " in ql:
            return "BTC"
        if up.startswith("KXETH") or "ethereum" in ql or " eth " in ql:
            return "ETH"
        if up.startswith("KXSOL") or "solana" in ql or " sol " in ql:
            return "SOL"
        if up.startswith("KXDOGE") or "dogecoin" in ql or " doge " in ql:
            return "DOGE"
        if up.startswith("KXXRP") or "ripple" in ql or " xrp " in ql:
            return "XRP"   # uses XRPUSDT spot + realized vol (Deribit DVOL not available for XRP)
        if up.startswith("KXBCH") or "bitcoin cash" in ql or " bch " in ql:
            return "BCH"   # uses BCHUSDT spot + realized vol
        return "BTC"  # safe default for unknown crypto range

    # ------------------------------------------------------------------
    # Core math: log-normal range + threshold probability
    # ------------------------------------------------------------------

    @staticmethod
    def _ncdf(x: float) -> float:
        """Standard normal CDF via math.erf (no scipy required)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def lognormal_threshold_prob(
        self,
        spot: float,
        threshold: float,
        direction: str,
        vol: float,
        hours_to_expiry: float,
    ) -> float:
        """P(S_T < threshold) or P(S_T > threshold) under log-normal model.

        Black-Scholes digital formula:
            d = [ln(K/S₀) - σ²T/2] / (σ√T)
            P(S_T < K) = N(d)
            P(S_T > K) = 1 - N(d)

        Args:
            spot: Current price
            threshold: Strike price K
            direction: 'below' or 'above'
            vol: Annualized implied vol
            hours_to_expiry: Time to resolution in hours
        """
        if hours_to_expiry <= 0 or vol <= 0 or spot <= 0 or threshold <= 0:
            return 0.5
        T = max(hours_to_expiry, 0.5) / 8760.0
        sigma_sqrt_T = vol * math.sqrt(T)
        drift = -0.5 * vol ** 2 * T  # risk-neutral log drift (Itô correction)
        d = (math.log(threshold / spot) + drift) / sigma_sqrt_T
        prob_below = self._ncdf(d)
        if direction == 'below':
            return max(0.0, min(1.0, prob_below))
        else:
            return max(0.0, min(1.0, 1.0 - prob_below))

    def lognormal_range_prob(
        self,
        spot: float,
        low: float,
        high: float,
        vol: float,
        hours_to_expiry: float,
    ) -> float:
        """P(low ≤ S_T ≤ high) under log-normal price model.

        Uses the Black-Scholes digital range formula:
            P = N(d_high) - N(d_low)
        where d = [ln(K/S0) + (σ²/2)*T] / (σ*√T)
        and N is the standard normal CDF.
        (The +σ²/2 comes from subtracting the negative Itô drift: μ = -σ²/2)
        """
        if hours_to_expiry <= 0 or vol <= 0 or spot <= 0 or low >= high:
            return 0.5

        T = max(hours_to_expiry, 0.5) / 8760.0  # annualize; floor at 30min
        sigma_sqrt_T = vol * math.sqrt(T)
        drift = -0.5 * vol ** 2 * T  # risk-neutral log drift

        def d(K: float) -> float:
            return (math.log(K / spot) - drift) / sigma_sqrt_T

        # P(low ≤ S_T ≤ high) = N(d(high)) - N(d(low))
        # where d(K) = (ln(K/S_0) + σ²/2*T) / (σ√T) = standardised log-distance
        # = P(S_T ≤ high) - P(S_T ≤ low)  [CDF evaluated at boundary strikes]
        prob = self._ncdf(d(high)) - self._ncdf(d(low))
        return max(0.0, min(1.0, prob))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def evaluate_range_market(
        self,
        ticker: str,
        question: str,
        kalshi_price: float,
        hours_to_expiry: float,
    ) -> Optional[CryptoEdgeResult]:
        """Full pipeline: parse range → fetch spot + vol → compute edge.

        Returns None if the market cannot be evaluated (unknown asset,
        unparseable range, or data fetch failure).
        """
        # 1. Detect asset
        asset = self.detect_asset(ticker, question)
        if asset is None:
            return None  # NASDAQ: skip quant model

        # 2. Parse price range
        bounds = self.parse_range_from_ticker(ticker) or self.parse_range_from_question(question)
        if not bounds:
            logger.warning(
                f"[CryptoEdge] Cannot parse range from ticker={ticker!r} "
                f"question={question[:50]!r}"
            )
            return None
        range_low, range_high = bounds

        # 3. Fetch spot price and implied vol concurrently
        # asyncio.gather ensures both coroutines run in parallel and neither
        # is orphaned if the other raises an exception.
        spot, vol_result = await asyncio.gather(
            self.get_spot_price(asset),
            self.get_implied_vol(asset),
            return_exceptions=True,
        )
        if isinstance(spot, BaseException):
            logger.warning(f"[CryptoEdge] Spot price fetch raised: {spot}")
            spot = None
        if isinstance(vol_result, BaseException):
            logger.warning(f"[CryptoEdge] Vol fetch raised: {vol_result}")
            vol_result = (self._VOL_DEFAULTS.get(asset, 0.80), 'historical_default')
        vol, vol_src = vol_result

        if spot is None:
            logger.warning(f"[CryptoEdge] Could not fetch {asset} spot price — skipping")
            return None

        # 4. Compute log-normal probability and edge
        prob = self.lognormal_range_prob(spot, range_low, range_high, vol, hours_to_expiry)
        edge = prob - kalshi_price

        # 5. Build human-readable context block for Claude
        range_width_pct = (range_high - range_low) / spot * 100
        spot_in_range   = range_low <= spot <= range_high
        if range_high > range_low:
            pct_from_low = (spot - range_low) / (range_high - range_low) * 100
        else:
            pct_from_low = 50.0

        lines = [
            f"QUANTITATIVE RANGE ANALYSIS ({asset}):",
            f"  Current {asset} spot price : ${spot:,.0f}",
            f"  Kalshi range               : ${range_low:,.0f} – ${range_high:,.0f}  "
            f"(width {range_width_pct:.1f}% of spot)",
        ]
        if spot_in_range:
            lines.append(
                f"  Spot position              : INSIDE range "
                f"({pct_from_low:.0f}% from lower bound)"
            )
        else:
            dist = min(abs(spot - range_low), abs(spot - range_high))
            lines.append(
                f"  Spot position              : OUTSIDE range "
                f"(${dist:,.0f} from nearest boundary)"
            )
        lines += [
            f"  Implied vol (annualized)   : {vol * 100:.0f}%  [{vol_src}]",
            f"  Hours to expiry            : {hours_to_expiry:.1f}h",
            f"  Log-normal P(in range)     : {prob:.1%}",
            f"  Kalshi market price        : {kalshi_price:.1%}",
            f"  Quant edge                 : {edge:+.1%}",
        ]

        if edge >= 0.10:
            lines.append("  Signal: QUANT CONFIRMS EDGE — log-normal probability exceeds market price.")
        elif edge >= 0:
            lines.append("  Signal: SLIGHT QUANT EDGE — proceed with caution.")
        else:
            lines.append("  Signal: QUANT SEES NO EDGE — market price exceeds log-normal probability.")

        context_summary = "\n".join(lines)

        return CryptoEdgeResult(
            asset=asset,
            spot_price=spot,
            range_low=range_low,
            range_high=range_high,
            implied_vol=vol,
            hours_to_expiry=hours_to_expiry,
            quant_prob=prob,
            kalshi_price=kalshi_price,
            edge=edge,
            context_summary=context_summary,
            vol_source=vol_src,
        )

    async def evaluate_threshold_market(
        self,
        ticker: str,
        question: str,
        kalshi_price: float,
        hours_to_expiry: float,
    ) -> Optional['CryptoEdgeResult']:
        """Evaluate a BTC/ETH threshold market using Black-Scholes digital formula.

        Kalshi threshold markets (new format as of 2026):
          KXBTC-26MAR1301-B78875    → YES = P(BTC < 78875)
          KXBTC-26MAR1301-T78999.99 → YES = P(BTC > 78999.99)

        Edge = model_prob - kalshi_yes_price. No AI involved.
        """
        asset = self.detect_asset(ticker, question)
        if asset is None:
            return None

        parsed = self.parse_threshold_from_ticker(ticker)
        if not parsed:
            return None
        direction, threshold = parsed

        spot, vol_result = await asyncio.gather(
            self.get_spot_price(asset),
            self.get_implied_vol(asset),
            return_exceptions=True,
        )
        if isinstance(spot, BaseException) or spot is None:
            logger.warning(f"[CryptoEdge] Spot fetch failed for threshold market {ticker}")
            return None
        if isinstance(vol_result, BaseException):
            vol_result = (self._VOL_DEFAULTS.get(asset, 0.80), 'historical_default')
        vol, vol_src = vol_result

        prob = self.lognormal_threshold_prob(spot, threshold, direction, vol, hours_to_expiry)
        edge = prob - kalshi_price

        dist_pct = (threshold - spot) / spot * 100
        dist_label = f"{abs(dist_pct):.1f}% {'above' if threshold > spot else 'below'} spot"

        lines = [
            f"QUANTITATIVE THRESHOLD ANALYSIS ({asset}):",
            f"  Current {asset} spot        : ${spot:,.2f}",
            f"  Threshold                   : ${threshold:,.2f}  ({dist_label})",
            f"  Market question             : Will {asset} be {'below' if direction == 'below' else 'above'} ${threshold:,.2f}?",
            f"  Implied vol (annualized)    : {vol * 100:.0f}%  [{vol_src}]",
            f"  Hours to expiry             : {hours_to_expiry:.1f}h",
            f"  Log-normal P(YES)           : {prob:.1%}",
            f"  Kalshi YES price            : {kalshi_price:.1%}",
            f"  Quant edge                  : {edge:+.1%}",
        ]
        if edge >= 0.10:
            lines.append("  Signal: STRONG QUANT EDGE — model probability exceeds market price.")
        elif edge >= 0.05:
            lines.append("  Signal: MODERATE QUANT EDGE.")
        elif edge <= -0.10:
            lines.append("  Signal: STRONG QUANT EDGE FOR NO SIDE.")
        else:
            lines.append("  Signal: NO SIGNIFICANT EDGE — model close to market price.")

        return CryptoEdgeResult(
            asset=asset,
            spot_price=spot,
            range_low=min(threshold, spot),
            range_high=max(threshold, spot),
            implied_vol=vol,
            hours_to_expiry=hours_to_expiry,
            quant_prob=prob,
            kalshi_price=kalshi_price,
            edge=edge,
            context_summary="\n".join(lines),
            vol_source=vol_src,
        )
