#!/usr/bin/env python3
"""Market Discovery Tool for Polymarket Battle-Bot.

Fetches active markets from Polymarket's API and filters them
based on strategy eligibility criteria.

Usage:
    python scripts/discover_markets.py
    python scripts/discover_markets.py --category crypto --min-volume 10000
    python scripts/discover_markets.py --output markets.json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import httpx
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Polymarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


async def fetch_markets(
    limit: int = 100,
    active_only: bool = True,
) -> list[dict]:
    """Fetch markets from Polymarket Gamma API.
    
    Args:
        limit: Maximum markets to fetch
        active_only: Only fetch active markets
        
    Returns:
        List of market dictionaries
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {
            "limit": limit,
            "active": str(active_only).lower(),
            "closed": "false",
        }
        
        response = await client.get(f"{GAMMA_API}/markets", params=params)
        response.raise_for_status()
        
        return response.json()


async def fetch_market_details(condition_id: str) -> Optional[dict]:
    """Fetch detailed market info including orderbook.
    
    Args:
        condition_id: Market condition ID
        
    Returns:
        Market details or None
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{GAMMA_API}/markets/{condition_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch details for {condition_id}: {e}")
            return None


def parse_resolution_date(market: dict) -> Optional[datetime]:
    """Parse resolution date from market data."""
    # Try different fields
    for field in ['endDate', 'end_date', 'resolutionDate', 'resolution_date']:
        if field in market and market[field]:
            try:
                date_str = market[field]
                # Handle various formats
                if 'T' in date_str:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    # Make naive for comparison
                    return dt.replace(tzinfo=None)
                else:
                    return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                pass
    return None


def categorize_market(question: str) -> str:
    """Auto-categorize market based on question text."""
    question_lower = question.lower()
    
    if any(w in question_lower for w in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol']):
        return 'crypto'
    elif any(w in question_lower for w in ['trump', 'biden', 'election', 'senate', 'congress', 'president', 'democrat', 'republican', 'vote']):
        return 'politics'
    elif any(w in question_lower for w in ['nfl', 'nba', 'mlb', 'nhl', 'super bowl', 'championship', 'game', 'win', 'score']):
        return 'sports'
    elif any(w in question_lower for w in ['oscar', 'grammy', 'emmy', 'movie', 'album', 'release']):
        return 'entertainment'
    elif any(w in question_lower for w in ['fed', 'rate', 'inflation', 'gdp', 'jobs', 'unemployment', 'cpi']):
        return 'economics'
    elif any(w in question_lower for w in ['ai', 'openai', 'google', 'apple', 'microsoft', 'tech']):
        return 'tech'
    else:
        return 'other'


def filter_markets(
    markets: list[dict],
    min_volume: float = 5000,
    max_volume: float = 500000,  # Avoid headline markets
    max_days_to_resolution: int = 30,
    min_price: float = 0.05,
    max_price: float = 0.95,
    category_filter: Optional[str] = None,
) -> list[dict]:
    """Filter markets based on strategy eligibility criteria.
    
    Args:
        markets: Raw market list
        min_volume: Minimum 24h volume
        max_volume: Maximum volume (avoid headline markets)
        max_days_to_resolution: Maximum days until resolution
        min_price: Minimum tradeable price
        max_price: Maximum tradeable price
        category_filter: Optional category to filter by
        
    Returns:
        Filtered list of eligible markets
    """
    eligible = []
    
    for market in markets:
        reasons = []
        
        # Get basic info
        question = market.get('question', '')
        volume = float(market.get('volume', 0) or market.get('volume24hr', 0) or 0)
        liquidity = float(market.get('liquidity', 0) or 0)
        
        # Get price (try different fields)
        price = None
        for field in ['outcomePrices', 'outcome_prices', 'bestBid', 'lastTradePrice']:
            if field in market:
                val = market[field]
                if isinstance(val, list) and len(val) > 0:
                    price = float(val[0])
                elif isinstance(val, (int, float)):
                    price = float(val)
                break
        
        if price is None:
            price = 0.5  # Default
        
        # Check resolution rules
        rules = market.get('description', '') or market.get('rules', '') or market.get('resolutionSource', '')
        if not rules or len(rules) < 20:
            reasons.append('NO_RESOLUTION_RULES')
        
        # Check volume
        if volume < min_volume:
            reasons.append(f'LOW_VOLUME_{volume:.0f}')
        if volume > max_volume:
            reasons.append(f'HEADLINE_MARKET_{volume:.0f}')
        
        # Check resolution date
        end_date = parse_resolution_date(market)
        if end_date:
            days_remaining = (end_date - datetime.utcnow()).days
            if days_remaining > max_days_to_resolution:
                reasons.append(f'TOO_FAR_OUT_{days_remaining}d')
            if days_remaining < 0:
                reasons.append('ALREADY_ENDED')
        
        # Check price
        if price < min_price:
            reasons.append(f'PRICE_TOO_LOW_{price:.2f}')
        if price > max_price:
            reasons.append(f'PRICE_TOO_HIGH_{price:.2f}')
        
        # Categorize
        category = categorize_market(question)
        
        # Category filter
        if category_filter and category != category_filter:
            reasons.append(f'WRONG_CATEGORY_{category}')
        
        # Extract token ID properly (first token is YES outcome)
        token_id = ''
        if 'clobTokenIds' in market:
            tokens = market['clobTokenIds']
            if isinstance(tokens, list) and len(tokens) > 0:
                # Get first token (YES outcome)
                first_token = tokens[0]
                if isinstance(first_token, str):
                    token_id = first_token.strip('"')
                else:
                    token_id = str(first_token)
            elif isinstance(tokens, str):
                # Might be JSON string
                try:
                    parsed = json.loads(tokens)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        token_id = str(parsed[0])
                except:
                    token_id = tokens
        elif 'tokenId' in market:
            token_id = str(market['tokenId'])
        
        # Build result
        result = {
            'condition_id': market.get('conditionId', market.get('condition_id', '')),
            'question': question,
            'token_id': token_id,
            'current_price': round(price, 4),
            'volume_24h': volume,
            'liquidity': liquidity,
            'end_date': end_date.isoformat() if end_date else None,
            'days_remaining': (end_date - datetime.utcnow()).days if end_date else None,
            'category': category,
            'resolution_rules': rules[:500] if rules else None,  # Truncate
            'eligible': len(reasons) == 0,
            'rejection_reasons': reasons,
            'url': f"https://polymarket.com/event/{market.get('slug', '')}",
        }
        
        if len(reasons) == 0:
            eligible.append(result)
        else:
            # Log rejections at debug level
            logger.debug(f"Rejected: {question[:50]}... | {reasons}")
    
    return eligible


def generate_code_snippet(markets: list[dict]) -> str:
    """Generate Python code to add markets to the bot."""
    lines = ["# Add these markets to your bot in main.py:\n"]
    
    for m in markets:
        lines.append(f'''
await bot.add_market(
    condition_id="{m['condition_id']}",
    question="""{m['question']}""",
    token_id="{m['token_id']}",
    outcome=MarketOutcome.YES,
    current_price={m['current_price']},
    resolution_rules="""{m['resolution_rules'][:300] if m['resolution_rules'] else 'See Polymarket for rules'}...""",
    category="{m['category']}",
    volume_24h={m['volume_24h']:.0f},
    liquidity={m['liquidity']:.0f},
    end_date=datetime.fromisoformat("{m['end_date']}") if m['end_date'] else None,
)
''')
    
    return '\n'.join(lines)


async def main():
    parser = argparse.ArgumentParser(description='Discover eligible Polymarket markets')
    parser.add_argument('--limit', type=int, default=100, help='Max markets to fetch')
    parser.add_argument('--min-volume', type=float, default=5000, help='Minimum 24h volume')
    parser.add_argument('--max-volume', type=float, default=500000, help='Maximum volume (avoid headlines)')
    parser.add_argument('--max-days', type=int, default=30, help='Max days to resolution')
    parser.add_argument('--category', type=str, help='Filter by category')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--code', action='store_true', help='Generate Python code snippet')
    
    args = parser.parse_args()
    
    logger.info("Fetching markets from Polymarket...")
    
    try:
        # Fetch markets
        raw_markets = await fetch_markets(limit=args.limit)
        logger.info(f"Fetched {len(raw_markets)} markets")
        
        # Filter
        eligible = filter_markets(
            raw_markets,
            min_volume=args.min_volume,
            max_volume=args.max_volume,
            max_days_to_resolution=args.max_days,
            category_filter=args.category,
        )
        
        logger.info(f"Found {len(eligible)} eligible markets")
        
        if not eligible:
            print("\nNo eligible markets found. Try adjusting filters:")
            print("  --min-volume 1000    (lower volume threshold)")
            print("  --max-days 60        (longer resolution window)")
            return
        
        # Sort by volume
        eligible.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        # Display results
        print("\n" + "=" * 80)
        print("ELIGIBLE MARKETS")
        print("=" * 80)
        
        for i, m in enumerate(eligible[:20], 1):  # Show top 20
            print(f"\n{i}. {m['question'][:70]}...")
            print(f"   Category: {m['category']} | Price: {m['current_price']:.0%} | Volume: ${m['volume_24h']:,.0f}")
            print(f"   Days remaining: {m['days_remaining']} | Token: {m['token_id'][:20]}...")
            print(f"   URL: {m['url']}")
        
        print("\n" + "=" * 80)
        print(f"Total eligible: {len(eligible)}")
        print("=" * 80)
        
        # Output options
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(eligible, f, indent=2)
            print(f"\nSaved to {args.output}")
        
        if args.code:
            code = generate_code_snippet(eligible[:5])  # Top 5
            print("\n" + "=" * 80)
            print("PYTHON CODE SNIPPET (top 5 markets)")
            print("=" * 80)
            print(code)
        
        # Always show code for top 3
        if len(eligible) > 0:
            print("\n" + "=" * 80)
            print("QUICK START - Add these to main.py:")
            print("=" * 80)
            print(generate_code_snippet(eligible[:3]))
            
    except httpx.HTTPError as e:
        logger.error(f"API error: {e}")
        print("\nFailed to fetch markets. Polymarket API may be down or rate-limited.")
        print("Try again in a few minutes.")


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    asyncio.run(main())
