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
    """Fetches relevant news for market topics.
    
    Uses Brave Search API (primary) with Google News RSS as fallback.
    """
    
    def __init__(self):
        self._cache: dict[str, tuple[datetime, list[NewsItem]]] = {}
        self._cache_ttl = timedelta(minutes=30)  # 30 min cache - balance freshness vs API costs
        self._brave_api_key = os.getenv('BRAVE_API_KEY')
        self._brave_searches = 0  # Track usage
        
        if self._brave_api_key:
            logger.info("[News] Brave Search API configured (primary source)")
        else:
            logger.info("[News] No Brave API key - using Google News RSS only")
        
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
        question_lower = question.lower()
        
        # SPORTS-SPECIFIC QUERY BUILDING
        # Detect sports betting markets and build targeted queries
        # Exclude financial/economics terms that contain sports words ("basis points", "rate spread")
        is_economics = any(k in question_lower for k in [
            'basis point', 'interest rate', 'federal reserve', 'central bank', 'pboc',
            'inflation', 'gdp', 'cpi', 'unemployment', 'rate cut', 'rate hike',
            'yield', 'treasury', 'bond', 'stock market', 's&p', 'nasdaq',
        ])
        is_sports = not is_economics and (
            category and category.lower() == 'sports' or
            any(term in question_lower for term in [
                'wins', 'points', 'spread', 'total', 'over', 'under',
                'nba', 'nfl', 'mlb', 'nhl', 'ncaa', 'basketball', 'football',
                'hockey', 'baseball', 'soccer', 'tennis', 'golf'
            ])
        )
        
        if is_sports:
            return self._build_sports_query(question)
        
        # Non-sports: build SPECIFIC queries based on category
        terms = self._extract_search_terms(question)
        question_lower = question.lower()
        
        # DOGE / federal spending cuts — highly specific to current events
        if any(k in question_lower for k in ['doge', 'elon', 'federal spending', 'budget cut',
                                              'cut between', 'cut less', 'cut more', 'cut the budget']):
            import re as _re
            amounts = _re.findall(r'\$[\d,]+[bBmM]?', question)
            if amounts:
                return f'DOGE federal spending cuts {" ".join(amounts[:2])} update 2025'
            return 'DOGE federal spending cuts total amount 2025 latest'

        # Deportation / immigration enforcement
        if any(k in question_lower for k in ['deport', 'immigr', 'border', 'ice ', 'removed']):
            import re as _re
            nums = _re.findall(r'[\d,]+,\d{3}', question)
            if nums:
                return f'Trump deportation total numbers {nums[0]} 2025'
            return 'Trump deportation total numbers 2025 latest update'

        # Politics: Focus on specific names and actions
        if category and category.lower() == 'politics':
            # Extract person names (capitalized words)
            names = [t for t in question.split() if t[0].isupper() and len(t) > 2][:2]
            if 'resign' in question_lower or 'out' in question_lower:
                return f'"{" ".join(names)}" resign resignation latest'
            if 'confirm' in question_lower or 'nominat' in question_lower:
                return f'"{" ".join(names)}" confirmation vote senate'
            if 'attend' in question_lower:
                return f'"{" ".join(names)}" attending will attend'
            if names:
                return f'"{" ".join(names)}" news today latest'
        
        # Crypto: Focus on specific coin and price action
        if category and category.lower() == 'crypto':
            coins = []
            for coin in ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'xrp']:
                if coin in question_lower:
                    coins.append(coin.upper() if len(coin) <= 4 else coin.capitalize())
            if coins:
                return f'{coins[0]} price prediction news today'
        
        # Economics: Focus on specific indicators/events
        if category and category.lower() == 'economics':
            if 'fed' in question_lower or 'rate' in question_lower:
                return 'Federal Reserve interest rate decision latest'
            if 'inflation' in question_lower or 'cpi' in question_lower:
                return 'inflation CPI report latest'
            if 'jobs' in question_lower or 'employment' in question_lower:
                return 'jobs report unemployment latest'
        
        # World events: Focus on specific countries/leaders
        if category and category.lower() == 'world':
            # Extract country/leader names
            for leader in ['Putin', 'Xi', 'Zelensky', 'Netanyahu', 'Khamenei', 'Kim']:
                if leader.lower() in question_lower:
                    return f'"{leader}" news today latest'
            for country in ['Ukraine', 'Russia', 'China', 'Israel', 'Iran', 'Taiwan', 'Gaza']:
                if country.lower() in question_lower:
                    return f'"{country}" breaking news latest'
        
        # Default: Use quotes around key terms for exact matching
        if terms:
            # Put the most specific term in quotes
            return f'"{terms[0]}" {" ".join(terms[1:4])} news'
        
        return question[:50]
    
    def _build_sports_query(self, question: str) -> str:
        """Build SPECIFIC sports search queries that return actionable intel.
        
        Focus on FACTS that move lines, not generic predictions:
        - Injuries: Who's out/questionable?
        - Lineups: Who's starting?
        - Recent form: How are they playing?
        """
        question_lower = question.lower()
        
        # Extract team/player names
        teams = self._extract_sports_entities(question)
        
        # Detect sport type for better queries
        sport = self._detect_sport(question_lower)
        
        # Detect bet type to focus the query
        if 'spread' in question_lower or 'by over' in question_lower or 'by under' in question_lower:
            # Spread bets: injuries and scoring trends matter most
            if teams:
                return f'"{teams[0]}" injury report out questionable'
        
        elif 'total' in question_lower:
            # Over/under: pace and scoring trends
            if len(teams) >= 2:
                return f'"{teams[0]}" "{teams[1]}" game total points scoring'
            elif teams:
                return f'"{teams[0]}" scoring average points per game'
        
        elif 'wins' in question_lower and 'season' not in question_lower:
            # Single game winner: recent form and injuries
            if teams:
                return f'"{teams[0]}" injury report starting lineup'
        
        # Default: injury report is always the most valuable
        if teams:
            primary_team = teams[0]
            # Make it SPECIFIC with quotes and focused terms
            return f'"{primary_team}" injury report ruled out questionable'
        
        # Fallback for unrecognized format
        terms = self._extract_search_terms(question)
        if terms:
            return f'"{terms[0]}" {sport} injury news'
        
        return question[:50]
    
    def _extract_sports_entities(self, question: str) -> list[str]:
        """Extract team and player names from a sports question."""
        import re
        words = question.split()
        
        # Words that are definitely NOT team/player names
        skip_words = {
            'wins', 'win', 'points', 'point', 'spread', 'total', 'over', 'under',
            'the', 'by', 'at', 'vs', 'and', 'or', 'will', 'what', 'how', 'who',
            'game', 'match', 'tonight', 'today', 'this', 'season', 'announcers',
            'say', 'during', 'mentions', 'mention', 'double', 'triple', 'pro',
            'basketball', 'football', 'hockey', 'baseball', 'soccer', 'nba',
            'nfl', 'mlb', 'nhl', 'ncaa', 'college', 'mens', "men's", 'womens',
        }
        
        entities = []
        current_entity = []
        
        for word in words:
            clean = re.sub(r'[^\w\s-]', '', word).strip()
            if not clean or len(clean) < 2:
                # End of potential entity
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                continue
            
            if clean.lower() in skip_words:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                continue
            
            # Check if it's a proper noun (capitalized)
            if word[0].isupper():
                current_entity.append(clean)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        # Don't forget the last entity
        if current_entity:
            entities.append(' '.join(current_entity))
        
        # Filter out single-letter entities and duplicates
        seen = set()
        result = []
        for e in entities:
            if len(e) > 2 and e.lower() not in seen:
                seen.add(e.lower())
                result.append(e)
        
        return result[:3]  # Max 3 entities
    
    def _detect_sport(self, question_lower: str) -> str:
        """Detect the sport type from question."""
        if any(term in question_lower for term in ['nba', 'basketball', 'lakers', 'celtics', 'warriors']):
            return 'NBA'
        if any(term in question_lower for term in ['nfl', 'football', 'chiefs', 'eagles', 'cowboys']):
            return 'NFL'
        if any(term in question_lower for term in ['mlb', 'baseball', 'yankees', 'dodgers']):
            return 'MLB'
        if any(term in question_lower for term in ['nhl', 'hockey', 'bruins', 'rangers']):
            return 'NHL'
        if any(term in question_lower for term in ['ncaa', 'college', 'march madness']):
            return 'college'
        return 'sports'
    
    async def _fetch_brave(self, query: str, max_results: int = 5) -> list[NewsItem]:
        """Fetch news using Brave Search API."""
        news_items = []
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/news/search",
                    params={
                        "q": query,
                        "count": max_results,
                        "freshness": "pd",  # Past day
                    },
                    headers={
                        "X-Subscription-Token": self._brave_api_key,
                        "Accept": "application/json",
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    for r in results[:max_results]:
                        news_items.append(NewsItem(
                            title=r.get('title', '')[:200],
                            snippet=r.get('description', r.get('title', ''))[:1000],  # Much more context
                            source=r.get('meta_url', {}).get('hostname', 'News'),
                            url=r.get('url', ''),
                        ))
                    
                    self._brave_searches += 1
                    if news_items:
                        logger.info(f"[Brave] Fetched {len(news_items)} items (total searches: {self._brave_searches})")
                elif response.status_code == 429:
                    logger.warning("[Brave] Rate limited - falling back to Google News")
                else:
                    logger.warning(f"[Brave] Status {response.status_code}: {response.text[:100]}")
                    
        except Exception as e:
            logger.warning(f"[Brave] Error: {e}")
        
        return news_items
    
    async def _fetch_reddit_rss(self, question: str, max_results: int = 5) -> list[NewsItem]:
        """Fetch top posts from relevant subreddits via free RSS — no API key needed."""
        question_lower = question.lower()

        # Pick subreddits based on market topic
        subreddits = []
        if any(k in question_lower for k in ['doge', 'elon', 'federal spending', 'budget cut']):
            subreddits = ['DOGE', 'elonmusk', 'politics']
        elif any(k in question_lower for k in ['deport', 'immigr', 'border', 'ice ']):
            subreddits = ['immigration', 'politics', 'worldnews']
        elif any(k in question_lower for k in ['trump', 'maga', 'executive order', 'white house']):
            subreddits = ['politics', 'Conservative', 'worldnews']
        elif any(k in question_lower for k in ['fed', 'rate', 'cpi', 'inflation', 'gdp', 'recession']):
            subreddits = ['Economics', 'investing', 'wallstreetbets']
        elif any(k in question_lower for k in ['bitcoin', 'btc', 'crypto', 'eth']):
            subreddits = ['Bitcoin', 'CryptoCurrency', 'investing']
        elif any(k in question_lower for k in ['ukraine', 'russia', 'china', 'taiwan', 'israel', 'iran']):
            subreddits = ['worldnews', 'geopolitics']
        else:
            subreddits = ['news', 'worldnews']

        news_items = []
        seen_titles: set[str] = set()

        for sub in subreddits[:2]:  # Max 2 subreddits
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(
                        f"https://www.reddit.com/r/{sub}/hot.rss",
                        params={"limit": 10},
                        headers={"User-Agent": "battlebot/1.0 (prediction market research)"},
                        follow_redirects=True,
                    )
                    if resp.status_code == 200:
                        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', resp.text)
                        titles = [t for t in titles if len(t) > 10 and t not in seen_titles]
                        for title in titles[:3]:
                            seen_titles.add(title)
                            news_items.append(NewsItem(
                                title=title[:200],
                                snippet=title[:500],
                                source=f"r/{sub}",
                                url=f"https://reddit.com/r/{sub}",
                            ))
            except Exception as e:
                logger.debug(f"[Reddit RSS] r/{sub} failed: {e}")

        return news_items[:max_results]

    async def _fetch_google_rss(self, query: str, max_results: int = 5) -> list[NewsItem]:
        """Fetch news using Google News RSS (free fallback)."""
        from urllib.parse import quote
        news_items = []
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                encoded_query = quote(query)
                url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                
                response = await client.get(url, follow_redirects=True)
                
                if response.status_code == 200:
                    rss_text = response.text
                    
                    # Parse RSS items
                    item_pattern = r'<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>.*?<source[^>]*>(.*?)</source>.*?</item>'
                    matches = re.findall(item_pattern, rss_text, re.DOTALL)
                    
                    for title, link, source in matches[:max_results]:
                        title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\\1', title).strip()
                        source = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\\1', source).strip()
                        
                        if title:
                            news_items.append(NewsItem(
                                title=title[:200],
                                snippet=title[:500],  # Google RSS only has titles, but keep more
                                source=source or "News",
                                url=link.strip(),
                            ))
                    
                    # Fallback if source tag not found
                    if not news_items:
                        simple_pattern = r'<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>.*?</item>'
                        matches = re.findall(simple_pattern, rss_text, re.DOTALL)
                        for title, link in matches[:max_results]:
                            title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\\1', title).strip()
                            if title:
                                news_items.append(NewsItem(
                                    title=title[:200],
                                    snippet=title[:500],
                                    source="News",
                                    url=link.strip(),
                                ))
                    
                    if news_items:
                        logger.debug(f"[Google RSS] Fetched {len(news_items)} items")
                        
        except Exception as e:
            logger.warning(f"[Google RSS] Error: {e}")
        
        return news_items
    
    async def fetch_news(
        self,
        question: str,
        category: Optional[str] = None,
        max_results: int = 5,
    ) -> list[NewsItem]:
        """Fetch relevant news for a market question.
        
        Uses Brave Search API (primary) with Google News RSS fallback.
        Results are cached for 60 minutes to minimize API usage.
        """
        # Check cache
        cache_key = f"{question[:50]}:{category}"
        if cache_key in self._cache:
            cached_time, cached_items = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.debug(f"[News] Cache hit for: {question[:30]}...")
                return cached_items
        
        query = self._build_search_query(question, category)
        logger.info(f"[News] Query: '{query}' | Category: {category or 'none'}")
        news_items = []

        # Determine if Reddit RSS would add signal for this market type
        reddit_categories = {'politics', 'world', 'economics', 'crypto'}
        use_reddit = (
            (category and category.lower() in reddit_categories) or
            any(k in question.lower() for k in [
                'doge', 'elon', 'trump', 'deport', 'fed', 'rate', 'bitcoin',
                'ukraine', 'china', 'taiwan', 'israel', 'congress', 'senate',
            ])
        )

        # Run Brave/Google and Reddit in parallel where useful
        if self._brave_api_key and use_reddit:
            brave_task = self._fetch_brave(query, max_results)
            reddit_task = self._fetch_reddit_rss(question, 3)
            brave_results, reddit_results = await asyncio.gather(brave_task, reddit_task)
            news_items = brave_results + reddit_results
        elif self._brave_api_key:
            news_items = await self._fetch_brave(query, max_results)
        elif use_reddit:
            google_task = self._fetch_google_rss(query, max_results)
            reddit_task = self._fetch_reddit_rss(question, 3)
            google_results, reddit_results = await asyncio.gather(google_task, reddit_task)
            news_items = google_results + reddit_results
        else:
            news_items = await self._fetch_google_rss(query, max_results)
        
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
        
        return news_items
    
    def get_usage_stats(self) -> dict:
        """Get API usage statistics."""
        return {
            'brave_searches': self._brave_searches,
            'cache_size': len(self._cache),
            'cache_ttl_minutes': self._cache_ttl.total_seconds() / 60,
        }
    
    @staticmethod
    def score_news_relevance(question: str, category: Optional[str] = None) -> float:
        """Score how much news intelligence should matter for this market (0-1).
        
        Higher scores = news is more likely to provide edge.
        Used to prioritize which markets to analyze with full intelligence.
        """
        question_lower = question.lower()
        score = 0.5  # Base score
        
        # HIGH NEWS VALUE: Events that break suddenly
        high_value_patterns = [
            ('resign', 0.4), ('fired', 0.4), ('indicted', 0.4), ('arrested', 0.4),
            ('announced', 0.3), ('breaking', 0.3), ('confirms', 0.3), ('denies', 0.3),
            ('injury', 0.3), ('ruled out', 0.35), ('questionable', 0.25), ('starting', 0.2),
            ('cabinet', 0.3), ('nomination', 0.3), ('confirmed', 0.25),
            ('peace', 0.3), ('war', 0.3), ('ceasefire', 0.35), ('attack', 0.3),
        ]
        
        for pattern, bonus in high_value_patterns:
            if pattern in question_lower:
                score += bonus
        
        # CATEGORY BONUSES
        if category:
            cat_lower = category.lower()
            category_scores = {
                'politics': 0.25,  # News breaks fast in politics
                'world': 0.25,    # International events
                'sports': 0.15,   # Injuries, lineups matter
                'crypto': 0.1,    # Market news
                'economics': 0.15, # Fed announcements, data
                'entertainment': 0.1,
                'weather': 0.05,  # Weather forecasts are known
            }
            score += category_scores.get(cat_lower, 0)
        
        # TIME-SENSITIVE BONUSES (detected from question)
        if any(term in question_lower for term in ['tonight', 'today', 'tomorrow', 'this week']):
            score += 0.1  # Imminent events = news matters more
        
        # SPORTS-SPECIFIC: Certain bet types benefit more from news
        if 'spread' in question_lower or 'total' in question_lower:
            score += 0.1  # Injury news directly impacts spreads
        if 'wins' in question_lower and 'season' not in question_lower:
            score += 0.1  # Single game outcomes
        
        # MENTION/SAY markets have LOW news value (entertainment speculation)
        if 'mention' in question_lower or 'say' in question_lower or 'announcer' in question_lower:
            score -= 0.3
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]


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
        """Fetch real economic data from FRED API (free, no key required)."""
        data_items = []
        question_lower = question.lower()

        # Map market keywords to FRED series IDs
        # FRED API is completely free and requires no API key for these public series
        fred_series = []
        if any(k in question_lower for k in ['fed', 'interest rate', 'rate hike', 'rate cut', 'fomc']):
            fred_series.append(('FEDFUNDS', 'Federal Funds Rate'))
        if any(k in question_lower for k in ['inflation', 'cpi', 'consumer price']):
            fred_series.append(('CPIAUCSL', 'CPI (All Urban Consumers)'))
        if any(k in question_lower for k in ['unemployment', 'jobs', 'payroll', 'employment']):
            fred_series.append(('UNRATE', 'Unemployment Rate'))
        if any(k in question_lower for k in ['gdp', 'recession', 'growth']):
            fred_series.append(('GDP', 'US GDP'))
        if any(k in question_lower for k in ['10 year', '10-year', 'treasury', 'yield']):
            fred_series.append(('DGS10', '10-Year Treasury Yield'))

        for series_id, series_name in fred_series[:2]:  # Max 2 series per market
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(
                        "https://fred.stlouisfed.org/graph/fredgraph.csv",
                        params={"id": series_id, "vintage_date": datetime.utcnow().strftime('%Y-%m-%d')},
                        headers={"User-Agent": "battlebot/1.0"},
                    )
                    if resp.status_code == 200:
                        lines = [l for l in resp.text.strip().splitlines() if l and not l.startswith('DATE')]
                        if lines:
                            # Last two non-empty data rows
                            rows = [r for r in lines[-5:] if '.' in r]
                            if rows:
                                last = rows[-1].split(',')
                                prev = rows[-2].split(',') if len(rows) >= 2 else last
                                last_date, last_val = last[0], last[1]
                                prev_val = prev[1] if len(prev) > 1 else last_val
                                try:
                                    change = float(last_val) - float(prev_val)
                                    data_items.append(DomainData(
                                        category="economics",
                                        data_type="fred_indicator",
                                        data={
                                            "series": series_name,
                                            "latest_value": last_val,
                                            "as_of": last_date,
                                            "change_from_prior": f"{change:+.3f}",
                                        },
                                        source="FRED (St. Louis Fed)",
                                    ))
                                except ValueError:
                                    pass
            except Exception as e:
                logger.debug(f"[FRED] {series_id} fetch failed: {e}")

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
        
        # Build news summary - include full snippets for better AI context
        news_summary = ""
        if news_items:
            summaries = [f"- [{item.source}] {item.title}\n  {item.snippet}" for item in news_items[:5]]
            news_summary = "\n\n".join(summaries)
        
        # Build domain summary
        domain_summary = ""
        if domain_data:
            for item in domain_data[:3]:
                if item.data_type == "price":
                    d = item.data
                    price_usd = d.get('price_usd')
                    change_24h = d.get('change_24h_pct', 0) or 0
                    if price_usd is not None:
                        domain_summary += f"- {d.get('asset', 'Asset')} price: ${price_usd:,.2f} ({change_24h:+.1f}% 24h)\n"
                    else:
                        domain_summary += f"- {d.get('asset', 'Asset')} price: N/A\n"
                elif item.data_type == "fred_indicator":
                    d = item.data
                    domain_summary += (
                        f"- {d['series']}: {d['latest_value']} (as of {d['as_of']},"
                        f" change: {d['change_from_prior']}) [FRED]\n"
                    )
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
