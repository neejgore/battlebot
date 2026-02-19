"""Market Intelligence Service for Polymarket Battle-Bot.

Provides real-time information advantage through:
1. News aggregation for market topics
2. Domain-specific data (sports, weather, economics)
3. Market inefficiency detection
4. Contrarian timing signals

This is the key to making the strategy profitable - giving the AI
information that the market might not have fully priced in.
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import httpx
from loguru import logger


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NewsItem:
    """A single news item."""
    title: str
    snippet: str
    source: str
    url: str
    published: Optional[datetime] = None
    relevance_score: float = 0.0


@dataclass
class DomainData:
    """Domain-specific data for a market."""
    category: str
    data_type: str
    data: dict
    source: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketIntelligence:
    """Complete intelligence package for a market."""
    market_id: str
    market_question: str
    
    # News
    news_items: list[NewsItem] = field(default_factory=list)
    news_summary: str = ""
    
    # Domain data
    domain_data: list[DomainData] = field(default_factory=list)
    domain_summary: str = ""
    
    # Market dynamics
    price_history: list[tuple[datetime, float]] = field(default_factory=list)
    recent_price_change: float = 0.0
    volatility_24h: float = 0.0
    
    # Inefficiency signals
    inefficiency_score: float = 0.0
    inefficiency_reasons: list[str] = field(default_factory=list)
    
    # Contrarian signals
    overreaction_detected: bool = False
    overreaction_direction: str = ""  # "up" or "down"
    overreaction_magnitude: float = 0.0
    
    # Meta
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    fetch_latency_ms: int = 0


# ============================================================================
# News Service
# ============================================================================

class NewsService:
    """Fetches relevant news for market topics using web search."""
    
    def __init__(self):
        self._cache: dict[str, tuple[datetime, list[NewsItem]]] = {}
        self._cache_ttl = timedelta(minutes=15)
        
    def _extract_search_terms(self, question: str) -> list[str]:
        """Extract key search terms from a market question."""
        # Remove common prediction market phrases
        cleaned = question.lower()
        for phrase in ['will', 'be', 'the', 'by', 'on', 'in', 'at', 'to', 'of', 
                       'yes', 'no', 'this', 'that', 'before', 'after', 'during',
                       'more than', 'less than', 'over', 'under', 'above', 'below',
                       'reach', 'exceed', 'fall', 'rise', 'drop', 'increase', 'decrease']:
            cleaned = cleaned.replace(phrase, ' ')
        
        # Extract key entities (capitalized words, numbers with context)
        words = cleaned.split()
        
        # Get the main subject (usually first few significant words)
        significant = [w for w in words if len(w) > 2 and w.isalpha()][:5]
        
        return significant
    
    def _build_search_query(self, question: str, category: Optional[str] = None) -> str:
        """Build an effective search query for news."""
        terms = self._extract_search_terms(question)
        
        # Add category context
        if category:
            category_terms = {
                'politics': 'politics news',
                'sports': 'sports news today',
                'crypto': 'cryptocurrency news',
                'economics': 'economic news today',
                'entertainment': 'entertainment news',
                'tech': 'technology news',
                'weather': 'weather forecast',
            }
            if category.lower() in category_terms:
                terms.append(category_terms[category.lower()])
        
        # Build query
        query = ' '.join(terms[:6])  # Limit to avoid overly specific queries
        
        # Add "latest" or "news" if not present
        if 'news' not in query.lower():
            query = f"latest news {query}"
        
        return query
    
    async def fetch_news(
        self,
        question: str,
        category: Optional[str] = None,
        max_results: int = 5,
    ) -> list[NewsItem]:
        """Fetch relevant news for a market question.
        
        Uses DuckDuckGo instant answers API (no API key needed).
        """
        # Check cache
        cache_key = f"{question[:50]}:{category}"
        if cache_key in self._cache:
            cached_time, cached_items = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.debug(f"News cache hit for: {question[:30]}...")
                return cached_items
        
        query = self._build_search_query(question, category)
        news_items = []
        
        try:
            # Use DuckDuckGo instant answers (free, no API key)
            async with httpx.AsyncClient(timeout=10.0) as client:
                # DDG instant answer API
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract from abstract
                    if data.get("Abstract"):
                        news_items.append(NewsItem(
                            title=data.get("Heading", "Summary"),
                            snippet=data["Abstract"][:500],
                            source=data.get("AbstractSource", "DuckDuckGo"),
                            url=data.get("AbstractURL", ""),
                        ))
                    
                    # Extract from related topics
                    for topic in data.get("RelatedTopics", [])[:max_results]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            news_items.append(NewsItem(
                                title=topic.get("Text", "")[:100],
                                snippet=topic.get("Text", "")[:300],
                                source="DuckDuckGo",
                                url=topic.get("FirstURL", ""),
                            ))
                    
                    # Extract from news results if available
                    for result in data.get("Results", [])[:max_results]:
                        if isinstance(result, dict):
                            news_items.append(NewsItem(
                                title=result.get("Text", "")[:100],
                                snippet=result.get("Text", "")[:300],
                                source="DuckDuckGo",
                                url=result.get("FirstURL", ""),
                            ))
                
        except Exception as e:
            logger.warning(f"News fetch failed for '{query[:30]}...': {e}")
        
        # Dedupe and limit
        seen_urls = set()
        unique_items = []
        for item in news_items:
            if item.url not in seen_urls and item.snippet:
                seen_urls.add(item.url)
                unique_items.append(item)
        
        news_items = unique_items[:max_results]
        
        # Cache results
        self._cache[cache_key] = (datetime.utcnow(), news_items)
        
        logger.debug(f"Fetched {len(news_items)} news items for: {question[:30]}...")
        return news_items


# ============================================================================
# Domain Data Service
# ============================================================================

class DomainDataService:
    """Fetches domain-specific data based on market category."""
    
    def __init__(self):
        self._cache: dict[str, tuple[datetime, list[DomainData]]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def _detect_domain(self, question: str, category: Optional[str] = None) -> str:
        """Detect the domain/category of a market question."""
        question_lower = question.lower()
        
        # Explicit category mapping
        if category:
            return category.lower()
        
        # Pattern-based detection
        patterns = {
            'sports': ['nfl', 'nba', 'mlb', 'nhl', 'game', 'score', 'win', 'championship', 
                      'playoff', 'super bowl', 'world series', 'finals', 'match'],
            'crypto': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol',
                      'dogecoin', 'token', 'blockchain'],
            'economics': ['gdp', 'inflation', 'cpi', 'jobs', 'unemployment', 'fed', 
                         'interest rate', 'recession', 'growth'],
            'weather': ['temperature', 'weather', 'rain', 'snow', 'hurricane', 'storm',
                       'celsius', 'fahrenheit', 'forecast'],
            'politics': ['election', 'vote', 'president', 'congress', 'senate', 'democrat',
                        'republican', 'biden', 'trump', 'primary', 'poll'],
            'tech': ['ai', 'openai', 'google', 'apple', 'microsoft', 'tesla', 'launch',
                    'release', 'iphone', 'chatgpt'],
        }
        
        for domain, keywords in patterns.items():
            if any(kw in question_lower for kw in keywords):
                return domain
        
        return 'general'
    
    async def fetch_domain_data(
        self,
        question: str,
        category: Optional[str] = None,
    ) -> list[DomainData]:
        """Fetch domain-specific data relevant to the market."""
        domain = self._detect_domain(question, category)
        
        # Check cache
        cache_key = f"{domain}:{question[:30]}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                return cached_data
        
        data_items = []
        
        try:
            if domain == 'crypto':
                data_items.extend(await self._fetch_crypto_data(question))
            elif domain == 'economics':
                data_items.extend(await self._fetch_economics_data(question))
            elif domain == 'sports':
                data_items.extend(await self._fetch_sports_context(question))
            elif domain == 'weather':
                data_items.extend(await self._fetch_weather_context(question))
            # Politics, tech, etc. rely more on news than structured data
            
        except Exception as e:
            logger.warning(f"Domain data fetch failed for {domain}: {e}")
        
        # Cache results
        self._cache[cache_key] = (datetime.utcnow(), data_items)
        
        return data_items
    
    async def _fetch_crypto_data(self, question: str) -> list[DomainData]:
        """Fetch cryptocurrency price data."""
        data_items = []
        
        # Detect which crypto is mentioned
        cryptos = {
            'bitcoin': 'bitcoin',
            'btc': 'bitcoin',
            'ethereum': 'ethereum',
            'eth': 'ethereum',
            'solana': 'solana',
            'sol': 'solana',
        }
        
        question_lower = question.lower()
        crypto_id = None
        for keyword, cid in cryptos.items():
            if keyword in question_lower:
                crypto_id = cid
                break
        
        if not crypto_id:
            return data_items
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # CoinGecko free API
                response = await client.get(
                    f"https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": crypto_id,
                        "vs_currencies": "usd",
                        "include_24hr_change": "true",
                        "include_24hr_vol": "true",
                    }
                )
                
                if response.status_code == 200:
                    price_data = response.json().get(crypto_id, {})
                    if price_data:
                        data_items.append(DomainData(
                            category="crypto",
                            data_type="price",
                            data={
                                "asset": crypto_id,
                                "price_usd": price_data.get("usd"),
                                "change_24h_pct": price_data.get("usd_24h_change"),
                                "volume_24h": price_data.get("usd_24h_vol"),
                            },
                            source="CoinGecko",
                        ))
                        
        except Exception as e:
            logger.warning(f"Crypto data fetch failed: {e}")
        
        return data_items
    
    async def _fetch_economics_data(self, question: str) -> list[DomainData]:
        """Fetch economic indicator context."""
        # For economics, we mainly rely on news but can add structured data
        # from FRED API if available in the future
        data_items = []
        
        # Detect economic indicators mentioned
        indicators = {
            'inflation': 'Consumer Price Index (CPI)',
            'cpi': 'Consumer Price Index (CPI)',
            'unemployment': 'Unemployment Rate',
            'jobs': 'Nonfarm Payrolls',
            'gdp': 'Gross Domestic Product',
            'interest rate': 'Federal Funds Rate',
            'fed': 'Federal Reserve Policy',
        }
        
        question_lower = question.lower()
        for keyword, indicator_name in indicators.items():
            if keyword in question_lower:
                data_items.append(DomainData(
                    category="economics",
                    data_type="indicator_context",
                    data={
                        "indicator": indicator_name,
                        "note": f"Market relates to {indicator_name}. Check latest economic releases.",
                    },
                    source="Internal",
                ))
                break
        
        return data_items
    
    async def _fetch_sports_context(self, question: str) -> list[DomainData]:
        """Fetch sports context (teams, standings, etc.)."""
        data_items = []
        
        # Detect league
        leagues = {
            'nfl': 'NFL (American Football)',
            'nba': 'NBA (Basketball)',
            'mlb': 'MLB (Baseball)',
            'nhl': 'NHL (Hockey)',
            'super bowl': 'NFL Super Bowl',
            'world series': 'MLB World Series',
        }
        
        question_lower = question.lower()
        for keyword, league_name in leagues.items():
            if keyword in question_lower:
                data_items.append(DomainData(
                    category="sports",
                    data_type="league_context",
                    data={
                        "league": league_name,
                        "note": f"Market relates to {league_name}. Check current standings and recent results.",
                    },
                    source="Internal",
                ))
                break
        
        return data_items
    
    async def _fetch_weather_context(self, question: str) -> list[DomainData]:
        """Fetch weather context."""
        data_items = []
        
        # Weather markets often reference specific locations or thresholds
        data_items.append(DomainData(
            category="weather",
            data_type="weather_context",
            data={
                "note": "Weather market - check official forecasts from NOAA/NWS.",
            },
            source="Internal",
        ))
        
        return data_items


# ============================================================================
# Market Inefficiency Detector
# ============================================================================

class InefficiencyDetector:
    """Detects markets that may be inefficiently priced."""
    
    # Thresholds for inefficiency signals
    LOW_VOLUME_THRESHOLD = 1000  # Markets with < $1000 volume are less efficient
    HIGH_SPREAD_THRESHOLD = 0.05  # 5% spread indicates low liquidity
    LOW_OI_THRESHOLD = 50  # Low open interest = less attention
    
    def calculate_inefficiency_score(
        self,
        volume: float,
        spread: float,
        open_interest: float,
        price: float,
        recent_price_change: float = 0.0,
    ) -> tuple[float, list[str]]:
        """Calculate an inefficiency score for a market.
        
        Higher score = more likely to be inefficient (opportunity).
        
        Returns:
            Tuple of (score 0-1, list of reasons)
        """
        score = 0.0
        reasons = []
        
        # Low volume = less efficient pricing
        if volume < self.LOW_VOLUME_THRESHOLD:
            vol_score = min(0.3, (self.LOW_VOLUME_THRESHOLD - volume) / self.LOW_VOLUME_THRESHOLD * 0.3)
            score += vol_score
            reasons.append(f"Low volume (${volume:,.0f})")
        
        # Wide spread = less liquidity, more opportunity
        if spread > self.HIGH_SPREAD_THRESHOLD:
            spread_score = min(0.2, (spread - self.HIGH_SPREAD_THRESHOLD) / 0.10 * 0.2)
            score += spread_score
            reasons.append(f"Wide spread ({spread:.1%})")
        
        # Low open interest = less attention
        if open_interest < self.LOW_OI_THRESHOLD:
            oi_score = min(0.2, (self.LOW_OI_THRESHOLD - open_interest) / self.LOW_OI_THRESHOLD * 0.2)
            score += oi_score
            reasons.append(f"Low OI ({open_interest})")
        
        # Extreme prices can be mispriced (near 0 or 100)
        if price < 0.10 or price > 0.90:
            score += 0.15
            reasons.append(f"Extreme price ({price:.0%})")
        
        # Recent large price move = potential overreaction
        if abs(recent_price_change) > 0.10:
            score += 0.15
            reasons.append(f"Recent move ({recent_price_change:+.0%})")
        
        return min(1.0, score), reasons


# ============================================================================
# Contrarian Timing Detector
# ============================================================================

class ContrarianDetector:
    """Detects potential overreactions to bet against."""
    
    # Thresholds
    OVERREACTION_THRESHOLD = 0.15  # 15% move in short period
    EXTREME_OVERREACTION = 0.25  # 25% move is extreme
    
    def detect_overreaction(
        self,
        current_price: float,
        price_1h_ago: Optional[float] = None,
        price_24h_ago: Optional[float] = None,
        news_sentiment: Optional[str] = None,  # 'positive', 'negative', 'neutral'
    ) -> tuple[bool, str, float]:
        """Detect if market has overreacted.
        
        Returns:
            Tuple of (is_overreaction, direction, magnitude)
        """
        if price_24h_ago is None:
            return False, "", 0.0
        
        # Calculate 24h change
        change_24h = current_price - price_24h_ago
        
        # Calculate 1h change if available
        change_1h = 0.0
        if price_1h_ago is not None:
            change_1h = current_price - price_1h_ago
        
        # Detect overreaction
        is_overreaction = False
        direction = ""
        magnitude = abs(change_24h)
        
        # Large 24h move
        if abs(change_24h) >= self.OVERREACTION_THRESHOLD:
            is_overreaction = True
            direction = "up" if change_24h > 0 else "down"
        
        # Rapid 1h move (even smaller threshold)
        elif price_1h_ago and abs(change_1h) >= 0.08:
            is_overreaction = True
            direction = "up" if change_1h > 0 else "down"
            magnitude = abs(change_1h)
        
        # Extreme price with recent move suggests overreaction
        if (current_price > 0.90 or current_price < 0.10) and magnitude > 0.05:
            is_overreaction = True
        
        return is_overreaction, direction, magnitude
    
    def get_contrarian_edge_boost(
        self,
        is_overreaction: bool,
        overreaction_direction: str,
        ai_predicted_direction: str,  # "yes" or "no" - which side AI favors
        magnitude: float,
    ) -> float:
        """Calculate edge boost for contrarian positions.
        
        If AI agrees with contrarian (betting against overreaction), boost edge.
        If AI agrees with overreaction, reduce edge (likely chasing).
        
        Returns:
            Multiplier for edge (0.5-1.5)
        """
        if not is_overreaction:
            return 1.0
        
        # Determine if AI is contrarian
        # If market moved UP and AI says NO (bearish), AI is contrarian
        # If market moved DOWN and AI says YES (bullish), AI is contrarian
        ai_is_contrarian = (
            (overreaction_direction == "up" and ai_predicted_direction == "no") or
            (overreaction_direction == "down" and ai_predicted_direction == "yes")
        )
        
        if ai_is_contrarian:
            # Boost edge for contrarian positions
            # Larger overreaction = larger boost (mean reversion more likely)
            boost = 1.0 + min(0.5, magnitude)  # Max 1.5x
            return boost
        else:
            # Reduce edge if AI is chasing the move
            # Could be valid breakout, but more likely to be late
            penalty = max(0.5, 1.0 - magnitude * 0.5)  # Min 0.5x
            return penalty


# ============================================================================
# Main Intelligence Service
# ============================================================================

class MarketIntelligenceService:
    """Main service that orchestrates all intelligence gathering."""
    
    def __init__(self):
        self.news_service = NewsService()
        self.domain_service = DomainDataService()
        self.inefficiency_detector = InefficiencyDetector()
        self.contrarian_detector = ContrarianDetector()
        
        # Price history for contrarian detection
        self._price_history: dict[str, list[tuple[datetime, float]]] = {}
        
        logger.info("MarketIntelligenceService initialized")
    
    def record_price(self, market_id: str, price: float):
        """Record a price observation for a market."""
        if market_id not in self._price_history:
            self._price_history[market_id] = []
        
        history = self._price_history[market_id]
        history.append((datetime.utcnow(), price))
        
        # Keep only last 48 hours
        cutoff = datetime.utcnow() - timedelta(hours=48)
        self._price_history[market_id] = [(t, p) for t, p in history if t > cutoff]
    
    def _get_historical_price(
        self,
        market_id: str,
        hours_ago: float,
    ) -> Optional[float]:
        """Get the price from approximately X hours ago."""
        if market_id not in self._price_history:
            return None
        
        target_time = datetime.utcnow() - timedelta(hours=hours_ago)
        history = self._price_history[market_id]
        
        # Find closest price to target time
        closest_price = None
        closest_diff = float('inf')
        
        for timestamp, price in history:
            diff = abs((timestamp - target_time).total_seconds())
            if diff < closest_diff:
                closest_diff = diff
                closest_price = price
        
        # Only return if within reasonable window (±30 minutes for 1h, ±2h for 24h)
        max_diff = hours_ago * 1800  # Allow 30 min per hour of lookback
        if closest_diff < max_diff:
            return closest_price
        
        return None
    
    async def gather_intelligence(
        self,
        market_id: str,
        market_question: str,
        current_price: float,
        spread: float,
        volume: float,
        open_interest: float,
        category: Optional[str] = None,
    ) -> MarketIntelligence:
        """Gather complete intelligence for a market.
        
        This is the main entry point - call this before AI analysis.
        """
        start_time = datetime.utcnow()
        
        # Record current price
        self.record_price(market_id, current_price)
        
        # Fetch news and domain data in parallel
        news_task = self.news_service.fetch_news(market_question, category)
        domain_task = self.domain_service.fetch_domain_data(market_question, category)
        
        news_items, domain_data = await asyncio.gather(news_task, domain_task)
        
        # Build news summary
        news_summary = ""
        if news_items:
            summaries = [f"- {item.title}: {item.snippet[:150]}..." for item in news_items[:3]]
            news_summary = "\n".join(summaries)
        
        # Build domain summary
        domain_summary = ""
        if domain_data:
            for item in domain_data[:2]:
                if item.data_type == "price":
                    d = item.data
                    price_usd = d.get('price_usd')
                    change_24h = d.get('change_24h_pct', 0) or 0
                    if price_usd is not None:
                        domain_summary += f"- {d.get('asset', 'Asset')} price: ${price_usd:,.2f} ({change_24h:+.1f}% 24h)\n"
                    else:
                        domain_summary += f"- {d.get('asset', 'Asset')} price: N/A\n"
                else:
                    note = item.data.get('note', '')
                    if note:
                        domain_summary += f"- {note}\n"
        
        # Get historical prices for contrarian detection
        price_1h = self._get_historical_price(market_id, 1.0)
        price_24h = self._get_historical_price(market_id, 24.0)
        
        # Calculate recent price change
        recent_change = 0.0
        if price_24h is not None:
            recent_change = current_price - price_24h
        
        # Calculate inefficiency score
        ineff_score, ineff_reasons = self.inefficiency_detector.calculate_inefficiency_score(
            volume=volume,
            spread=spread,
            open_interest=open_interest,
            price=current_price,
            recent_price_change=recent_change,
        )
        
        # Detect overreaction
        is_overreaction, overreaction_dir, overreaction_mag = self.contrarian_detector.detect_overreaction(
            current_price=current_price,
            price_1h_ago=price_1h,
            price_24h_ago=price_24h,
        )
        
        # Build price history for response
        history = self._price_history.get(market_id, [])
        
        latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return MarketIntelligence(
            market_id=market_id,
            market_question=market_question,
            news_items=news_items,
            news_summary=news_summary,
            domain_data=domain_data,
            domain_summary=domain_summary,
            price_history=history[-20:],  # Last 20 observations
            recent_price_change=recent_change,
            volatility_24h=abs(recent_change),  # Simplified volatility
            inefficiency_score=ineff_score,
            inefficiency_reasons=ineff_reasons,
            overreaction_detected=is_overreaction,
            overreaction_direction=overreaction_dir,
            overreaction_magnitude=overreaction_mag,
            fetched_at=datetime.utcnow(),
            fetch_latency_ms=latency_ms,
        )


# ============================================================================
# Singleton Instance
# ============================================================================

_intelligence_service: Optional[MarketIntelligenceService] = None


def get_intelligence_service() -> MarketIntelligenceService:
    """Get the singleton intelligence service instance."""
    global _intelligence_service
    if _intelligence_service is None:
        _intelligence_service = MarketIntelligenceService()
    return _intelligence_service
