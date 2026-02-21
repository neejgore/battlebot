#!/usr/bin/env python3
"""Analyze settled markets - fetch via events endpoint."""

import asyncio
import os
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from services.kalshi_client import KalshiClient

async def analyze_settled_markets():
    """Fetch settled events and their markets."""
    
    client = KalshiClient(use_demo=False)
    
    print("=" * 80)
    print("KALSHI SETTLED EVENTS & MARKETS ANALYSIS")
    print("=" * 80)
    
    # First, let's look at the ticker patterns
    print("\n[Checking market ticker patterns...]")
    
    # Get a sample of settled markets to understand the ticker structure
    result = await client.get_markets(status='settled', limit=100)
    markets = result.get('markets', [])
    
    # Analyze ticker patterns
    ticker_patterns = defaultdict(int)
    for m in markets:
        ticker = m.get('ticker', '')
        # Extract prefix pattern (first part before any numbers/hyphens)
        if '-' in ticker:
            prefix = ticker.split('-')[0]
        else:
            prefix = ticker[:10]
        ticker_patterns[prefix] += 1
    
    print("\nTicker patterns in recent settled markets:")
    for pattern, count in sorted(ticker_patterns.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pattern}: {count}")
    
    # Now let's try fetching events to get better data
    print("\n[Fetching settled events...]")
    
    all_events = []
    cursor = None
    pages = 0
    max_pages = 10
    
    while pages < max_pages:
        try:
            await asyncio.sleep(0.3)
            result = await client.get_events(status='settled', limit=100, cursor=cursor)
            events = result.get('events', [])
            
            if not events:
                break
            
            all_events.extend(events)
            pages += 1
            print(f"  Page {pages}: {len(all_events)} events")
            
            cursor = result.get('cursor')
            if not cursor:
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    print(f"\n[Total settled events: {len(all_events)}]")
    
    # Analyze events
    if all_events:
        print("\n" + "=" * 80)
        print("SAMPLE SETTLED EVENTS")
        print("=" * 80)
        
        for i, e in enumerate(all_events[:10]):
            print(f"\n{i+1}. {e.get('ticker', 'N/A')}")
            print(f"   Title: {e.get('title', '')[:60]}")
            print(f"   Category: {e.get('category', 'N/A')}")
            print(f"   Series: {e.get('series_ticker', 'N/A')}")
            print(f"   Markets count: {e.get('markets_count', 'N/A')}")
            # Print all keys
            print(f"   Keys: {list(e.keys())}")
    
    # Let's also try getting markets with specific series tickers
    print("\n" + "=" * 80)
    print("ANALYZING MARKETS BY SERIES")
    print("=" * 80)
    
    # Common series tickers for different categories
    series_to_check = [
        ('KXNBAMLWIN', 'NBA Winner'),
        ('KXNFLMLWIN', 'NFL Winner'),
        ('KXMLBMLWIN', 'MLB Winner'),
        ('KXNHLMLWIN', 'NHL Winner'),
        ('KXNCAABMW', 'NCAA Basketball'),
        ('KXWEATHER', 'Weather'),
        ('KXBTC', 'Bitcoin'),
        ('KXETH', 'Ethereum'),
        ('KXFED', 'Fed/FOMC'),
        ('KXCPI', 'CPI'),
        ('KXGDP', 'GDP'),
        ('KXTRUMP', 'Trump'),
        ('KXPRES', 'Presidential'),
    ]
    
    for series_ticker, name in series_to_check:
        try:
            await asyncio.sleep(0.3)
            result = await client.get_markets(
                status='settled',
                series_ticker=series_ticker,
                limit=100
            )
            markets = result.get('markets', [])
            
            if markets:
                yes_count = len([m for m in markets if m.get('result') == 'yes'])
                no_count = len([m for m in markets if m.get('result') == 'no'])
                total = yes_count + no_count
                
                if total > 0:
                    print(f"\n{name} ({series_ticker}): {total} markets")
                    print(f"  YES won: {yes_count} ({yes_count/total*100:.1f}%)")
                    print(f"  NO won:  {no_count} ({no_count/total*100:.1f}%)")
                    
                    # Show samples
                    for m in markets[:3]:
                        title = m.get('title', '')[:50]
                        result = m.get('result', '')
                        last_price = m.get('last_price', 0)
                        print(f"    - {title}... | result={result} | price={last_price}¢")
                        
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    # Now let's look at what's actually in the bot's trades
    print("\n" + "=" * 80)
    print("ANALYZING YOUR BOT'S HISTORICAL TRADES")
    print("=" * 80)
    
    # Load bot state to get trade history
    try:
        import json
        state_path = '/Users/neej.gore/Documents/battlebot/battlebot_state.json'
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        trades = state.get('trades', [])
        print(f"\nTotal trades in bot history: {len(trades)}")
        
        if trades:
            # Analyze by market type
            market_types = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
            price_buckets = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
            side_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
            
            for t in trades:
                if t.get('action') != 'EXIT':
                    continue
                
                question = (t.get('question', '') or '').lower()
                pnl = t.get('pnl', 0) or 0
                side = t.get('side', '').upper()
                entry_price = t.get('entry_price', 0.5) or 0.5
                exit_reason = t.get('exit_reason', 'UNKNOWN')
                
                # Determine if win (positive P&L)
                is_win = pnl > 0
                
                # Categorize
                if any(s in question for s in ['nba', 'nfl', 'mlb', 'nhl', 'ncaa']):
                    if 'point' in question or 'total' in question or 'spread' in question:
                        cat = 'SPORTS_SPREAD'
                    else:
                        cat = 'SPORTS_WINNER'
                elif any(s in question for s in ['weather', 'rain', 'temperature']):
                    cat = 'WEATHER'
                elif any(s in question for s in ['bitcoin', 'crypto', 'ethereum']):
                    cat = 'CRYPTO'
                else:
                    cat = 'OTHER'
                
                # Price bucket
                if entry_price < 0.15:
                    bucket = '<15%'
                elif entry_price < 0.30:
                    bucket = '15-30%'
                elif entry_price < 0.50:
                    bucket = '30-50%'
                else:
                    bucket = '50%+'
                
                # Record stats
                if is_win:
                    market_types[cat]['wins'] += 1
                    price_buckets[bucket]['wins'] += 1
                    side_stats[side]['wins'] += 1
                else:
                    market_types[cat]['losses'] += 1
                    price_buckets[bucket]['losses'] += 1
                    side_stats[side]['losses'] += 1
                
                market_types[cat]['pnl'] += pnl
                price_buckets[bucket]['pnl'] += pnl
                side_stats[side]['pnl'] += pnl
            
            print("\nBy Market Type:")
            for cat, stats in sorted(market_types.items()):
                total = stats['wins'] + stats['losses']
                if total > 0:
                    win_rate = stats['wins'] / total * 100
                    print(f"  {cat}: {win_rate:.0f}% win ({stats['wins']}/{total}) | P&L: ${stats['pnl']:.2f}")
            
            print("\nBy Entry Price:")
            for bucket, stats in sorted(price_buckets.items()):
                total = stats['wins'] + stats['losses']
                if total > 0:
                    win_rate = stats['wins'] / total * 100
                    print(f"  {bucket}: {win_rate:.0f}% win ({stats['wins']}/{total}) | P&L: ${stats['pnl']:.2f}")
            
            print("\nBy Side:")
            for side, stats in sorted(side_stats.items()):
                total = stats['wins'] + stats['losses']
                if total > 0:
                    win_rate = stats['wins'] / total * 100
                    print(f"  {side}: {win_rate:.0f}% win ({stats['wins']}/{total}) | P&L: ${stats['pnl']:.2f}")
            
            # Summary
            total_trades = sum(s['wins'] + s['losses'] for s in market_types.values())
            total_wins = sum(s['wins'] for s in market_types.values())
            total_pnl = sum(s['pnl'] for s in market_types.values())
            
            print(f"\nOVERALL:")
            print(f"  Total EXIT trades: {total_trades}")
            print(f"  Win rate: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "  No trades")
            print(f"  Total P&L: ${total_pnl:.2f}")
            
    except Exception as e:
        print(f"Error loading bot state: {e}")
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("""
Based on the analysis:

1. KALSHI API STRUCTURE:
   - Most recent settled markets are MVE (combo/parlay) markets
   - Single markets settle faster and are further back in the API
   - Use series_ticker to find specific market types

2. YOUR BOT'S HISTORICAL PERFORMANCE:
   - Check the stats above from battlebot_state.json
   
3. RECOMMENDED STRATEGY ADJUSTMENTS:
   - NO-only betting: Your YES bets have 0% win rate historically
   - Price floor 15%: Cheap contracts lose disproportionately  
   - Avoid sports spreads: Coin flips with no edge
   - Let bets settle: Stop losses are cutting winners
""")

if __name__ == '__main__':
    asyncio.run(analyze_settled_markets())
