#!/usr/bin/env python3
"""Analyze settled markets across Kalshi to identify winning patterns - V2."""

import asyncio
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from services.kalshi_client import KalshiClient

async def analyze_settled_markets():
    """Fetch and analyze recently settled markets across all of Kalshi."""
    
    client = KalshiClient(use_demo=False)
    
    print("=" * 80)
    print("KALSHI SETTLED MARKETS ANALYSIS - DETAILED")
    print("=" * 80)
    
    all_settled = []
    cursor = None
    pages = 0
    max_pages = 15  # Get more data
    
    print("\n[Fetching settled markets...]")
    
    while pages < max_pages:
        try:
            await asyncio.sleep(0.3)  # Rate limit
            result = await client.get_markets(
                status='settled',
                limit=1000,
                cursor=cursor
            )
            
            markets = result.get('markets', [])
            if not markets:
                break
                
            all_settled.extend(markets)
            pages += 1
            print(f"  Page {pages}: {len(all_settled)} total markets")
            
            cursor = result.get('cursor')
            if not cursor:
                break
                
        except Exception as e:
            print(f"  Error fetching: {e}")
            break
    
    print(f"\n[Total settled markets fetched: {len(all_settled)}]")
    
    if not all_settled:
        print("No settled markets found!")
        return
    
    # Debug: Look at actual market structure
    print("\n" + "=" * 80)
    print("SAMPLE MARKET DATA (first 3 markets)")
    print("=" * 80)
    for i, m in enumerate(all_settled[:3]):
        print(f"\nMarket {i+1}:")
        print(f"  ticker: {m.get('ticker')}")
        print(f"  title: {m.get('title', '')[:60]}")
        print(f"  subtitle: {m.get('subtitle', '')[:60]}")
        print(f"  result: {m.get('result')}")
        print(f"  yes_price: {m.get('yes_price')}")
        print(f"  no_price: {m.get('no_price')}")
        print(f"  last_price: {m.get('last_price')}")
        print(f"  volume: {m.get('volume')}")
        print(f"  open_interest: {m.get('open_interest')}")
        print(f"  close_time: {m.get('close_time')}")
        # Print all keys to see what's available
        print(f"  ALL KEYS: {list(m.keys())}")
    
    # Calculate actual statistics
    results_yes = [m for m in all_settled if m.get('result') == 'yes']
    results_no = [m for m in all_settled if m.get('result') == 'no']
    
    print("\n" + "=" * 80)
    print("OVERALL SETTLEMENT DISTRIBUTION")
    print("=" * 80)
    total = len(results_yes) + len(results_no)
    if total > 0:
        print(f"  YES outcomes: {len(results_yes)} ({len(results_yes)/total*100:.1f}%)")
        print(f"  NO outcomes:  {len(results_no)} ({len(results_no)/total*100:.1f}%)")
        print(f"  Unknown/Other: {len(all_settled) - total}")
    
    # Categorize markets more carefully
    categories = defaultdict(list)
    
    for m in all_settled:
        result = m.get('result', '')
        title = (m.get('title', '') or '').lower()
        subtitle = (m.get('subtitle', '') or '').lower()
        ticker = (m.get('ticker', '') or '').lower()
        question = f"{title} {subtitle}"
        
        # Better categorization using ticker patterns
        if 'nba' in ticker or 'nfl' in ticker or 'mlb' in ticker or 'nhl' in ticker or 'ncaa' in ticker:
            if 'ou' in ticker or 'spread' in ticker or 'pts' in ticker:
                cat = 'SPORTS_SPREAD'
            else:
                cat = 'SPORTS_WINNER'
        elif 'weather' in ticker or 'temp' in ticker or 'precip' in ticker:
            cat = 'WEATHER'
        elif 'btc' in ticker or 'eth' in ticker or 'crypto' in ticker:
            cat = 'CRYPTO'
        elif 'pres' in ticker or 'elect' in ticker or 'congress' in ticker or 'senate' in ticker:
            cat = 'POLITICS'
        elif 'fed' in ticker or 'fomc' in ticker or 'cpi' in ticker or 'gdp' in ticker:
            cat = 'ECONOMICS'
        elif any(s in question for s in ['nba', 'nfl', 'mlb', 'nhl', 'ncaa', 'game', 'match', 'win']):
            if 'point' in question or 'total' in question or 'spread' in question or 'over' in question:
                cat = 'SPORTS_SPREAD'
            else:
                cat = 'SPORTS_WINNER'
        elif any(s in question for s in ['weather', 'rain', 'snow', 'temperature', 'precipitation']):
            cat = 'WEATHER'
        elif any(s in question for s in ['bitcoin', 'ethereum', 'crypto']):
            cat = 'CRYPTO'
        elif any(s in question for s in ['trump', 'biden', 'congress', 'senate', 'election']):
            cat = 'POLITICS'
        elif any(s in question for s in ['fed', 'interest rate', 'inflation', 'cpi']):
            cat = 'ECONOMICS'
        else:
            cat = 'OTHER'
        
        # Store with last_price if available
        last_price = m.get('last_price') or m.get('yes_price') or 0.5
        categories[cat].append({
            'ticker': m.get('ticker', ''),
            'question': question[:80],
            'result': result,
            'last_price': last_price,
            'volume': m.get('volume', 0) or 0,
        })
    
    print("\n" + "=" * 80)
    print("BY CATEGORY - DETAILED")
    print("=" * 80)
    
    for cat in sorted(categories.keys()):
        markets = categories[cat]
        yes_count = len([m for m in markets if m['result'] == 'yes'])
        no_count = len([m for m in markets if m['result'] == 'no'])
        total_cat = yes_count + no_count
        
        if total_cat == 0:
            continue
            
        print(f"\n{'='*40}")
        print(f"{cat}: {total_cat} markets")
        print(f"{'='*40}")
        print(f"  YES won: {yes_count} ({yes_count/total_cat*100:.1f}%)")
        print(f"  NO won:  {no_count} ({no_count/total_cat*100:.1f}%)")
        
        # Show sample markets
        print(f"\n  Sample YES outcomes:")
        yes_samples = [m for m in markets if m['result'] == 'yes'][:3]
        for m in yes_samples:
            print(f"    - {m['ticker']}: {m['question'][:50]}...")
        
        print(f"\n  Sample NO outcomes:")
        no_samples = [m for m in markets if m['result'] == 'no'][:3]
        for m in no_samples:
            print(f"    - {m['ticker']}: {m['question'][:50]}...")
        
        # Price analysis using last_price
        price_buckets = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0})
        
        for m in markets:
            price = m.get('last_price', 0.5)
            result = m.get('result', '')
            
            if price < 0.10:
                bucket = '<10%'
            elif price < 0.20:
                bucket = '10-20%'
            elif price < 0.30:
                bucket = '20-30%'
            elif price < 0.40:
                bucket = '30-40%'
            elif price < 0.50:
                bucket = '40-50%'
            elif price < 0.60:
                bucket = '50-60%'
            elif price < 0.70:
                bucket = '60-70%'
            elif price < 0.80:
                bucket = '70-80%'
            elif price < 0.90:
                bucket = '80-90%'
            else:
                bucket = '90-100%'
            
            price_buckets[bucket]['total'] += 1
            if result == 'yes':
                price_buckets[bucket]['yes'] += 1
            elif result == 'no':
                price_buckets[bucket]['no'] += 1
        
        print(f"\n  By Last Price (YES win rate):")
        for bucket in ['<10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']:
            data = price_buckets.get(bucket, {'yes': 0, 'no': 0, 'total': 0})
            if data['total'] > 5:
                win_rate = data['yes'] / data['total'] * 100
                print(f"    {bucket}: {win_rate:.1f}% YES won ({data['yes']}/{data['total']})")
    
    # CRITICAL: Calculate actual edge per category
    print("\n" + "=" * 80)
    print("EDGE ANALYSIS BY CATEGORY")
    print("=" * 80)
    print("\nComparing actual YES win rate vs market-implied probability")
    print("Positive edge = market underprices YES, Negative = overprices YES")
    
    for cat in sorted(categories.keys()):
        markets = categories[cat]
        if len(markets) < 20:
            continue
            
        # Calculate implied vs actual
        total_implied_yes = 0
        total_actual_yes = 0
        count = 0
        
        for m in markets:
            price = m.get('last_price', 0.5)
            result = m.get('result', '')
            if result in ['yes', 'no']:
                total_implied_yes += price  # Market implies this chance of YES
                total_actual_yes += 1 if result == 'yes' else 0
                count += 1
        
        if count > 0:
            avg_implied = total_implied_yes / count * 100
            avg_actual = total_actual_yes / count * 100
            edge = avg_actual - avg_implied
            
            print(f"\n{cat} ({count} markets):")
            print(f"  Market implied YES probability: {avg_implied:.1f}%")
            print(f"  Actual YES win rate: {avg_actual:.1f}%")
            print(f"  Edge for YES bets: {edge:+.1f}%")
            print(f"  Edge for NO bets: {-edge:+.1f}%")
            
            if edge > 5:
                print(f"  >>> OPPORTUNITY: Bet YES in {cat} markets")
            elif edge < -5:
                print(f"  >>> OPPORTUNITY: Bet NO in {cat} markets")
    
    # Final recommendations
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    # Calculate overall
    total_implied = 0
    total_actual = 0
    count = 0
    
    for m in all_settled:
        price = m.get('last_price') or m.get('yes_price') or 0.5
        result = m.get('result', '')
        if result in ['yes', 'no']:
            total_implied += price
            total_actual += 1 if result == 'yes' else 0
            count += 1
    
    if count > 0:
        avg_implied = total_implied / count * 100
        avg_actual = total_actual / count * 100
        overall_edge = avg_actual - avg_implied
        
        print(f"\nOVERALL ({count} markets):")
        print(f"  Average implied YES probability: {avg_implied:.1f}%")
        print(f"  Actual YES win rate: {avg_actual:.1f}%")
        print(f"  Systematic edge for NO bets: {-overall_edge:+.1f}%")
        
        if overall_edge < -5:
            print(f"\n  ✓ CONFIRMED: Market systematically overprices YES")
            print(f"    - Betting NO has a {-overall_edge:.1f}% edge on average")
            print(f"    - This aligns with your historical 0% YES win rate vs 54% NO win rate")

if __name__ == '__main__':
    asyncio.run(analyze_settled_markets())
