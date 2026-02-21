#!/usr/bin/env python3
"""Analyze settled markets across Kalshi - V3 (fixed pricing, exclude combos)."""

import asyncio
import os
from collections import defaultdict
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from services.kalshi_client import KalshiClient

async def analyze_settled_markets():
    """Fetch and analyze recently settled SINGLE markets (exclude combos)."""
    
    client = KalshiClient(use_demo=False)
    
    print("=" * 80)
    print("KALSHI SETTLED MARKETS - SINGLE BETS ONLY (EXCLUDING COMBOS/PARLAYS)")
    print("=" * 80)
    
    all_settled = []
    cursor = None
    pages = 0
    max_pages = 20
    
    print("\n[Fetching settled markets...]")
    
    while pages < max_pages:
        try:
            await asyncio.sleep(0.3)
            result = await client.get_markets(
                status='settled',
                limit=1000,
                cursor=cursor
            )
            
            markets = result.get('markets', [])
            if not markets:
                break
            
            # Filter OUT MVE (multi-variate/combo) markets
            single_markets = [m for m in markets if 'KXMVE' not in m.get('ticker', '')]
            all_settled.extend(single_markets)
            pages += 1
            
            excluded = len(markets) - len(single_markets)
            print(f"  Page {pages}: {len(all_settled)} single markets (excluded {excluded} combos)")
            
            cursor = result.get('cursor')
            if not cursor:
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    print(f"\n[Total single settled markets: {len(all_settled)}]")
    
    if not all_settled:
        print("No settled markets found!")
        return
    
    # Debug: Sample market structure
    print("\n" + "=" * 80)
    print("SAMPLE SINGLE MARKET DATA")
    print("=" * 80)
    for i, m in enumerate(all_settled[:5]):
        ticker = m.get('ticker', '')
        title = m.get('title', '')[:60]
        result = m.get('result', '')
        last_price = m.get('last_price', 0)  # In cents!
        volume = m.get('volume', 0)
        print(f"\n{i+1}. {ticker}")
        print(f"   Title: {title}")
        print(f"   Result: {result}")
        print(f"   Last Price: {last_price}¢")  
        print(f"   Volume: {volume}")
    
    # Calculate statistics with proper price conversion
    results_yes = [m for m in all_settled if m.get('result') == 'yes']
    results_no = [m for m in all_settled if m.get('result') == 'no']
    
    print("\n" + "=" * 80)
    print("OVERALL SETTLEMENT DISTRIBUTION")
    print("=" * 80)
    total = len(results_yes) + len(results_no)
    if total > 0:
        print(f"  YES outcomes: {len(results_yes)} ({len(results_yes)/total*100:.1f}%)")
        print(f"  NO outcomes:  {len(results_no)} ({len(results_no)/total*100:.1f}%)")
    
    # Categorize markets
    categories = defaultdict(list)
    
    for m in all_settled:
        result = m.get('result', '')
        title = (m.get('title', '') or '').lower()
        subtitle = (m.get('subtitle', '') or '').lower()
        ticker = (m.get('ticker', '') or '').lower()
        question = f"{title} {subtitle}"
        
        # Convert last_price from cents to decimal
        last_price_cents = m.get('last_price', 50) or 50
        last_price = last_price_cents / 100.0
        
        # Categorize
        if any(s in ticker for s in ['nba', 'nfl', 'mlb', 'nhl', 'ncaa', 'soccer', 'tennis', 'golf', 'pga']):
            if any(s in ticker for s in ['-ou-', '-spread-', '-pts-', 'total', 'over', 'under']):
                cat = 'SPORTS_SPREAD'
            elif 'win' in ticker or 'ml' in ticker:
                cat = 'SPORTS_WINNER'
            else:
                cat = 'SPORTS_OTHER'
        elif any(s in ticker for s in ['temp', 'weather', 'precip', 'rain', 'snow']):
            cat = 'WEATHER'
        elif any(s in ticker for s in ['btc', 'eth', 'crypto', 'bitcoin']):
            cat = 'CRYPTO'
        elif any(s in ticker for s in ['trump', 'biden', 'elect', 'pres', 'senate', 'house', 'congress']):
            cat = 'POLITICS'
        elif any(s in ticker for s in ['fed', 'fomc', 'cpi', 'gdp', 'jobs', 'unemployment']):
            cat = 'ECONOMICS'
        elif any(s in question for s in ['nba', 'nfl', 'mlb', 'nhl', 'game', 'match']):
            if 'point' in question or 'total' in question or 'spread' in question:
                cat = 'SPORTS_SPREAD'
            else:
                cat = 'SPORTS_WINNER'
        elif any(s in question for s in ['weather', 'rain', 'snow', 'temperature']):
            cat = 'WEATHER'
        elif any(s in question for s in ['bitcoin', 'ethereum', 'crypto']):
            cat = 'CRYPTO'
        elif any(s in question for s in ['trump', 'biden', 'election']):
            cat = 'POLITICS'
        else:
            cat = 'OTHER'
        
        categories[cat].append({
            'ticker': m.get('ticker', ''),
            'question': question[:80],
            'result': result,
            'last_price': last_price,  # Now in decimal
            'volume': m.get('volume', 0) or 0,
        })
    
    print("\n" + "=" * 80)
    print("ANALYSIS BY CATEGORY")
    print("=" * 80)
    
    category_summary = []
    
    for cat in sorted(categories.keys()):
        markets = categories[cat]
        yes_count = len([m for m in markets if m['result'] == 'yes'])
        no_count = len([m for m in markets if m['result'] == 'no'])
        total_cat = yes_count + no_count
        
        if total_cat < 10:
            continue
        
        # Calculate implied vs actual (proper edge calculation)
        total_implied_yes = 0
        total_actual_yes = 0
        count = 0
        
        for m in markets:
            price = m.get('last_price', 0.5)
            result = m.get('result', '')
            if result in ['yes', 'no'] and 0 < price < 1:
                total_implied_yes += price
                total_actual_yes += 1 if result == 'yes' else 0
                count += 1
        
        if count > 0:
            avg_implied = total_implied_yes / count
            avg_actual = total_actual_yes / count
            edge_yes = (avg_actual - avg_implied) * 100
            edge_no = -edge_yes
            
            category_summary.append({
                'category': cat,
                'count': count,
                'yes_rate': avg_actual * 100,
                'implied_yes': avg_implied * 100,
                'edge_yes': edge_yes,
                'edge_no': edge_no,
            })
            
            print(f"\n{'='*50}")
            print(f"{cat}: {count} markets")
            print(f"{'='*50}")
            print(f"  YES won: {yes_count} ({avg_actual*100:.1f}%)")
            print(f"  NO won:  {no_count} ({(1-avg_actual)*100:.1f}%)")
            print(f"  Avg implied YES probability: {avg_implied*100:.1f}%")
            print(f"  Edge for YES bets: {edge_yes:+.1f}%")
            print(f"  Edge for NO bets: {edge_no:+.1f}%")
            
            if edge_no > 5:
                print(f"  >>> RECOMMEND: Bet NO on {cat}")
            elif edge_yes > 5:
                print(f"  >>> RECOMMEND: Bet YES on {cat}")
            
            # Price bucket analysis
            print(f"\n  By Last Price:")
            price_buckets = defaultdict(lambda: {'yes': 0, 'total': 0})
            
            for m in markets:
                price = m.get('last_price', 0.5)
                result = m.get('result', '')
                
                if price < 0.15:
                    bucket = '<15%'
                elif price < 0.30:
                    bucket = '15-30%'
                elif price < 0.50:
                    bucket = '30-50%'
                elif price < 0.70:
                    bucket = '50-70%'
                elif price < 0.85:
                    bucket = '70-85%'
                else:
                    bucket = '85%+'
                
                price_buckets[bucket]['total'] += 1
                if result == 'yes':
                    price_buckets[bucket]['yes'] += 1
            
            for bucket in ['<15%', '15-30%', '30-50%', '50-70%', '70-85%', '85%+']:
                data = price_buckets.get(bucket)
                if data and data['total'] >= 5:
                    win_rate = data['yes'] / data['total'] * 100
                    # Expected based on bucket midpoint
                    if bucket == '<15%':
                        expected = 7.5
                    elif bucket == '15-30%':
                        expected = 22.5
                    elif bucket == '30-50%':
                        expected = 40
                    elif bucket == '50-70%':
                        expected = 60
                    elif bucket == '70-85%':
                        expected = 77.5
                    else:
                        expected = 92.5
                    
                    edge = win_rate - expected
                    marker = "✓" if edge > 5 else ("✗" if edge < -5 else " ")
                    print(f"    {bucket:>8}: YES won {win_rate:5.1f}% (expected {expected:.0f}%) | edge {edge:+5.1f}% {marker} n={data['total']}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("STRATEGY SUMMARY")
    print("=" * 80)
    
    # Sort by edge
    category_summary.sort(key=lambda x: x['edge_no'], reverse=True)
    
    print("\nBest NO opportunities (categories where YES loses more than expected):")
    for c in category_summary[:5]:
        if c['edge_no'] > 0:
            print(f"  {c['category']}: {c['edge_no']:+.1f}% edge on NO ({c['count']} markets)")
    
    print("\nBest YES opportunities (categories where YES wins more than expected):")
    category_summary.sort(key=lambda x: x['edge_yes'], reverse=True)
    for c in category_summary[:5]:
        if c['edge_yes'] > 0:
            print(f"  {c['category']}: {c['edge_yes']:+.1f}% edge on YES ({c['count']} markets)")
    
    # Overall
    total_implied = 0
    total_actual = 0
    count = 0
    
    for m in all_settled:
        price = (m.get('last_price', 50) or 50) / 100.0
        result = m.get('result', '')
        if result in ['yes', 'no'] and 0 < price < 1:
            total_implied += price
            total_actual += 1 if result == 'yes' else 0
            count += 1
    
    if count > 0:
        avg_implied = total_implied / count * 100
        avg_actual = total_actual / count * 100
        
        print(f"\n\nOVERALL MARKET EFFICIENCY ({count} markets):")
        print(f"  Average implied YES probability: {avg_implied:.1f}%")
        print(f"  Actual YES win rate: {avg_actual:.1f}%")
        print(f"  Market bias: {avg_actual - avg_implied:+.1f}% (negative = YES overpriced)")
        
        if avg_actual - avg_implied < -5:
            print(f"\n  ✓ CONFIRMED: Markets OVERPRICE YES by {avg_implied - avg_actual:.1f}%")
            print(f"    Your NO-only strategy is mathematically sound!")
        elif avg_actual - avg_implied > 5:
            print(f"\n  Markets UNDERPRICE YES by {avg_actual - avg_implied:.1f}%")

if __name__ == '__main__':
    asyncio.run(analyze_settled_markets())
