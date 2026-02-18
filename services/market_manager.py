"""Automated Market Manager for Polymarket Battle-Bot.

Periodically discovers and refreshes markets from Polymarket's API,
automatically adding eligible markets and removing ineligible ones.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
import httpx
from loguru import logger


@dataclass
class MarketManagerConfig:
    """Configuration for market manager."""
    
    # Refresh interval
    refresh_interval_minutes: int = 15  # Every 15 minutes by default
    
    # API settings
    api_timeout_seconds: int = 30
    max_markets_to_fetch: int = 200
    
    # Eligibility criteria
    min_volume_24h: float = 5000.0
    max_volume_24h: float = 500000.0  # Avoid headline markets
    max_days_to_resolution: int = 30
    min_liquidity: float = 1000.0
    min_price: float = 0.05
    max_price: float = 0.95
    max_spread: float = 0.04
    
    # Category filters (empty = all categories)
    allowed_categories: list[str] = field(default_factory=list)
    blocked_categories: list[str] = field(default_factory=list)
    
    # Market limits
    max_active_markets: int = 20  # Don't monitor too many at once


@dataclass
class DiscoveredMarket:
    """A market discovered from the API."""
    condition_id: str
    question: str
    token_id: str
    current_price: float
    volume_24h: float
    liquidity: float
    end_date: Optional[datetime]
    resolution_rules: str
    category: str
    url: str
    spread: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'condition_id': self.condition_id,
            'question': self.question,
            'token_id': self.token_id,
            'current_price': self.current_price,
            'volume_24h': self.volume_24h,
            'liquidity': self.liquidity,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'resolution_rules': self.resolution_rules,
            'category': self.category,
            'url': self.url,
        }


# Type for callback when markets change
MarketCallback = Callable[[DiscoveredMarket, str], Awaitable[None]]  # (market, action: 'add'|'remove')


class MarketManager:
    """Manages automatic market discovery and refresh.
    
    Periodically fetches markets from Polymarket, filters by eligibility,
    and notifies the strategy of markets to add/remove.
    """
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    
    def __init__(
        self,
        config: Optional[MarketManagerConfig] = None,
        on_market_change: Optional[MarketCallback] = None,
    ):
        """Initialize the market manager.
        
        Args:
            config: Configuration options
            on_market_change: Callback when markets are added/removed
        """
        self.config = config or MarketManagerConfig()
        self._on_market_change = on_market_change
        
        self._active_markets: dict[str, DiscoveredMarket] = {}
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Stats
        self._last_refresh: Optional[datetime] = None
        self._total_refreshes = 0
        self._markets_added = 0
        self._markets_removed = 0
        
        logger.info(
            f"MarketManager initialized | Refresh: {self.config.refresh_interval_minutes}min | "
            f"Volume: ${self.config.min_volume_24h:,.0f}-${self.config.max_volume_24h:,.0f}"
        )
    
    @property
    def active_markets(self) -> list[DiscoveredMarket]:
        """Get list of currently active markets."""
        return list(self._active_markets.values())
    
    @property
    def active_market_ids(self) -> set[str]:
        """Get set of active market condition IDs."""
        return set(self._active_markets.keys())
    
    async def start(self) -> None:
        """Start the market manager."""
        self._running = True
        
        # Do initial refresh immediately
        await self.refresh_markets()
        
        # Start periodic refresh loop
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        
        logger.info("MarketManager started")
    
    async def stop(self) -> None:
        """Stop the market manager."""
        self._running = False
        
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MarketManager stopped")
    
    async def _refresh_loop(self) -> None:
        """Periodic refresh loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.refresh_interval_minutes * 60)
                
                if self._running:
                    await self.refresh_markets()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market refresh error: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
    
    async def refresh_markets(self) -> dict:
        """Refresh markets from API and update active set.
        
        Returns:
            Dict with refresh results
        """
        logger.info("Refreshing markets from Polymarket...")
        
        try:
            # Fetch from API
            raw_markets = await self._fetch_markets()
            logger.debug(f"Fetched {len(raw_markets)} raw markets")
            
            # Filter eligible markets
            eligible = self._filter_markets(raw_markets)
            logger.info(f"Found {len(eligible)} eligible markets")
            
            # Determine adds/removes
            async with self._lock:
                new_ids = {m.condition_id for m in eligible}
                old_ids = set(self._active_markets.keys())
                
                to_add = new_ids - old_ids
                to_remove = old_ids - new_ids
                
                # Respect max_active_markets limit
                if len(new_ids) > self.config.max_active_markets:
                    # Sort by volume and take top N
                    eligible.sort(key=lambda m: m.volume_24h, reverse=True)
                    eligible = eligible[:self.config.max_active_markets]
                    new_ids = {m.condition_id for m in eligible}
                    to_add = new_ids - old_ids
                    to_remove = old_ids - new_ids
                
                # Process removals
                for cid in to_remove:
                    market = self._active_markets.pop(cid)
                    self._markets_removed += 1
                    
                    if self._on_market_change:
                        try:
                            await self._on_market_change(market, 'remove')
                        except Exception as e:
                            logger.error(f"Error in remove callback: {e}")
                    
                    logger.info(f"Market removed: {market.question[:50]}...")
                
                # Process additions
                eligible_by_id = {m.condition_id: m for m in eligible}
                for cid in to_add:
                    market = eligible_by_id[cid]
                    self._active_markets[cid] = market
                    self._markets_added += 1
                    
                    if self._on_market_change:
                        try:
                            await self._on_market_change(market, 'add')
                        except Exception as e:
                            logger.error(f"Error in add callback: {e}")
                    
                    logger.info(f"Market added: {market.question[:50]}... | Vol: ${market.volume_24h:,.0f}")
                
                # Update existing markets (prices may have changed)
                for market in eligible:
                    if market.condition_id in self._active_markets:
                        self._active_markets[market.condition_id] = market
            
            self._last_refresh = datetime.utcnow()
            self._total_refreshes += 1
            
            result = {
                'total_fetched': len(raw_markets),
                'eligible': len(eligible),
                'added': len(to_add),
                'removed': len(to_remove),
                'active': len(self._active_markets),
            }
            
            logger.info(
                f"Market refresh complete | Active: {result['active']} | "
                f"Added: {result['added']} | Removed: {result['removed']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Market refresh failed: {e}")
            return {'error': str(e)}
    
    async def _fetch_markets(self) -> list[dict]:
        """Fetch markets from Polymarket API."""
        async with httpx.AsyncClient(timeout=self.config.api_timeout_seconds) as client:
            params = {
                'limit': self.config.max_markets_to_fetch,
                'active': 'true',
                'closed': 'false',
            }
            
            response = await client.get(f"{self.GAMMA_API}/markets", params=params)
            response.raise_for_status()
            
            return response.json()
    
    def _filter_markets(self, raw_markets: list[dict]) -> list[DiscoveredMarket]:
        """Filter markets by eligibility criteria."""
        eligible = []
        
        for market in raw_markets:
            result = self._check_eligibility(market)
            if result:
                eligible.append(result)
        
        return eligible
    
    def _check_eligibility(self, market: dict) -> Optional[DiscoveredMarket]:
        """Check if a market is eligible and return DiscoveredMarket if so."""
        
        # Extract fields
        question = market.get('question', '')
        condition_id = market.get('conditionId', market.get('condition_id', ''))
        
        if not condition_id:
            return None
        
        # Volume
        volume = float(market.get('volume', 0) or market.get('volume24hr', 0) or 0)
        if volume < self.config.min_volume_24h:
            return None
        if volume > self.config.max_volume_24h:
            return None
        
        # Liquidity
        liquidity = float(market.get('liquidity', 0) or 0)
        if liquidity < self.config.min_liquidity:
            return None
        
        # Resolution rules
        rules = market.get('description', '') or market.get('rules', '') or ''
        if len(rules) < 20:
            return None
        
        # Price
        price = self._extract_price(market)
        if price < self.config.min_price or price > self.config.max_price:
            return None
        
        # Resolution date
        end_date = self._parse_date(market)
        if end_date:
            days_remaining = (end_date - datetime.utcnow()).days
            if days_remaining < 0:
                return None
            if days_remaining > self.config.max_days_to_resolution:
                return None
        
        # Category
        category = self._categorize(question)
        
        if self.config.allowed_categories and category not in self.config.allowed_categories:
            return None
        if category in self.config.blocked_categories:
            return None
        
        # Extract token ID
        token_id = self._extract_token_id(market)
        if not token_id:
            return None
        
        # Build URL
        slug = market.get('slug', condition_id[:20])
        url = f"https://polymarket.com/event/{slug}"
        
        return DiscoveredMarket(
            condition_id=condition_id,
            question=question,
            token_id=token_id,
            current_price=price,
            volume_24h=volume,
            liquidity=liquidity,
            end_date=end_date,
            resolution_rules=rules[:1000],
            category=category,
            url=url,
        )
    
    def _extract_price(self, market: dict) -> float:
        """Extract price from market data."""
        for field in ['outcomePrices', 'outcome_prices', 'bestBid', 'lastTradePrice']:
            if field in market:
                val = market[field]
                if isinstance(val, list) and len(val) > 0:
                    try:
                        return float(val[0])
                    except:
                        pass
                elif isinstance(val, (int, float)):
                    return float(val)
        return 0.5
    
    def _extract_token_id(self, market: dict) -> Optional[str]:
        """Extract token ID from market data."""
        if 'clobTokenIds' in market:
            tokens = market['clobTokenIds']
            if isinstance(tokens, list) and len(tokens) > 0:
                first = tokens[0]
                if isinstance(first, str):
                    return first.strip('"')
                return str(first)
            elif isinstance(tokens, str):
                try:
                    parsed = json.loads(tokens)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return str(parsed[0])
                except:
                    return tokens
        elif 'tokenId' in market:
            return str(market['tokenId'])
        return None
    
    def _parse_date(self, market: dict) -> Optional[datetime]:
        """Parse resolution date."""
        for field in ['endDate', 'end_date', 'resolutionDate']:
            if field in market and market[field]:
                try:
                    date_str = market[field]
                    if 'T' in date_str:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        return dt.replace(tzinfo=None)
                    else:
                        return datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    pass
        return None
    
    def _categorize(self, question: str) -> str:
        """Categorize market by question text."""
        q = question.lower()
        
        if any(w in q for w in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana']):
            return 'crypto'
        elif any(w in q for w in ['trump', 'biden', 'election', 'senate', 'congress', 'president', 'democrat', 'republican']):
            return 'politics'
        elif any(w in q for w in ['nfl', 'nba', 'mlb', 'super bowl', 'championship', 'game']):
            return 'sports'
        elif any(w in q for w in ['fed', 'rate', 'inflation', 'gdp', 'jobs', 'cpi']):
            return 'economics'
        elif any(w in q for w in ['ai', 'openai', 'google', 'apple', 'microsoft', 'meta']):
            return 'tech'
        else:
            return 'other'
    
    def get_stats(self) -> dict:
        """Get manager statistics."""
        return {
            'running': self._running,
            'active_markets': len(self._active_markets),
            'total_refreshes': self._total_refreshes,
            'markets_added': self._markets_added,
            'markets_removed': self._markets_removed,
            'last_refresh': self._last_refresh.isoformat() if self._last_refresh else None,
            'refresh_interval_minutes': self.config.refresh_interval_minutes,
        }
