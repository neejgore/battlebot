#!/usr/bin/env python3
"""Analyze settled markets across Kalshi to identify winning patterns."""

import asyncio
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

# Load environment
from dotenv import load_dotenv
load_dotenv()

from services.kalshi_client import KalshiClient

async def analyze_settled_markets():
    """Fetch and analyze recently settled markets across all of Kalshi."""
    
    client = KalshiClient(use_demo=False)
    
    print("=" * 80)
    print("KALSHI SETTLED MARKETS ANALYSIS")
    print("=" * 80)
    
    all_settled = []
    cursor = None
    pages = 0
    max_pages = 10  # Limit to avoid too many API calls
    
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
    
    # Categorize markets
    categories = defaultdict(list)
    results_yes = []
    results_no = []
    
    for m in all_settled:
        # Get result
        result = m.get('result', '')
        question = m.get('title', m.get('subtitle', ''))[:100]
        ticker = m.get('ticker', '')
        yes_price = m.get('yes_price', 0.5)
        no_price = m.get('no_price', 0.5)
        volume = m.get('volume', 0) or 0
        
        # Categorize by keywords
        q_lower = question.lower()
        if any(s in q_lower for s in ['nba', 'nfl', 'mlb', 'nhl', 'ncaa', 'soccer', 'tennis', 'golf', 'game', 'match', 'score', 'win by']):
            if 'point' in q_lower or 'total' in q_lower or 'spread' in q_lower or 'over' in q_lower:
                cat = 'SPORTS_SPREAD'
            else:
                cat = 'SPORTS_WINNER'
        elif any(s in q_lower for s in ['weather', 'rain', 'snow', 'temperature', 'precipitation']):
            cat = 'WEATHER'
        elif any(s in q_lower for s in ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth']):
            cat = 'CRYPTO'
        elif any(s in q_lower for s in ['trump', 'biden', 'congress', 'senate', 'election', 'president']):
            cat = 'POLITICS'
        elif any(s in q_lower for s in ['fed', 'interest rate', 'inflation', 'cpi', 'gdp']):
            cat = 'ECONOMICS'
        else:
            cat = 'OTHER'
        
        categories[cat].append({
            'question': question,
            'result': result,
            'yes_price': yes_price,
            'no_price': no_price,
            'volume': volume,
        })
        
        if result == 'yes':
            results_yes.append(m)
        elif result == 'no':
            results_no.append(m)
    
    # Print analysis
    print("\n" + "=" * 80)
    print("OVERALL SETTLEMENT DISTRIBUTION")
    print("=" * 80)
    total = len(results_yes) + len(results_no)
    if total > 0:
        print(f"  YES outcomes: {len(results_yes)} ({len(results_yes)/total*100:.1f}%)")
        print(f"  NO outcomes:  {len(results_no)} ({len(results_no)/total*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("BY CATEGORY")
    print("=" * 80)
    
    for cat in sorted(categories.keys()):
        markets = categories[cat]
        yes_count = len([m for m in markets if m['result'] == 'yes'])
        no_count = len([m for m in markets if m['result'] == 'no'])
        total_cat = yes_count + no_count
        
        # Calculate "cheap contract" win rate (contracts priced < 20%)
        cheap_yes_win = len([m for m in markets if m['result'] == 'yes' and m['yes_price'] < 0.20])
        cheap_yes_total = len([m for m in markets if m['yes_price'] < 0.20])
        
        cheap_no_win = len([m for m in markets if m['result'] == 'no' and m['no_price'] < 0.20])
        cheap_no_total = len([m for m in markets if m['no_price'] < 0.20])
        
        print(f"\n{cat}:")
        if total_cat > 0:
            print(f"  Total: {total_cat} markets")
            print(f"  YES won: {yes_count} ({yes_count/total_cat*100:.1f}%)")
            print(f"  NO won:  {no_count} ({no_count/total_cat*100:.1f}%)")
        
        if cheap_yes_total > 0:
            cheap_yes_rate = cheap_yes_win / cheap_yes_total * 100
            print(f"  Cheap YES (<20¢) win rate: {cheap_yes_win}/{cheap_yes_total} = {cheap_yes_rate:.1f}%")
        if cheap_no_total > 0:
            cheap_no_rate = cheap_no_win / cheap_no_total * 100
            print(f"  Cheap NO (<20¢) win rate: {cheap_no_win}/{cheap_no_total} = {cheap_no_rate:.1f}%")
    
    # Price bucket analysis
    print("\n" + "=" * 80)
    print("PRICE BUCKET ANALYSIS (System-wide)")
    print("=" * 80)
    
    price_buckets = {
        '0-10%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '10-20%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '20-30%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '30-40%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '40-50%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '50-60%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '60-70%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '70-80%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '80-90%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
        '90-100%': {'yes_win': 0, 'yes_total': 0, 'no_win': 0, 'no_total': 0},
    }
    
    for m in all_settled:
        yes_price = m.get('yes_price', 0.5)
        no_price = m.get('no_price', 0.5)
        result = m.get('result', '')
        
        # Determine bucket for YES price
        if yes_price < 0.10:
            bucket = '0-10%'
        elif yes_price < 0.20:
            bucket = '10-20%'
        elif yes_price < 0.30:
            bucket = '20-30%'
        elif yes_price < 0.40:
            bucket = '30-40%'
        elif yes_price < 0.50:
            bucket = '40-50%'
        elif yes_price < 0.60:
            bucket = '50-60%'
        elif yes_price < 0.70:
            bucket = '60-70%'
        elif yes_price < 0.80:
            bucket = '70-80%'
        elif yes_price < 0.90:
            bucket = '80-90%'
        else:
            bucket = '90-100%'
        
        price_buckets[bucket]['yes_total'] += 1
        if result == 'yes':
            price_buckets[bucket]['yes_win'] += 1
        
        # Also track NO side
        if no_price < 0.10:
            no_bucket = '0-10%'
        elif no_price < 0.20:
            no_bucket = '10-20%'
        elif no_price < 0.30:
            no_bucket = '20-30%'
        elif no_price < 0.40:
            no_bucket = '30-40%'
        elif no_price < 0.50:
            no_bucket = '40-50%'
        elif no_price < 0.60:
            no_bucket = '50-60%'
        elif no_price < 0.70:
            no_bucket = '60-70%'
        elif no_price < 0.80:
            no_bucket = '70-80%'
        elif no_price < 0.90:
            no_bucket = '80-90%'
        else:
            no_bucket = '90-100%'
        
        price_buckets[no_bucket]['no_total'] += 1
        if result == 'no':
            price_buckets[no_bucket]['no_win'] += 1
    
    print("\nYES side (buying YES contracts at price X):")
    print(f"{'Price Range':<12} {'Win Rate':<12} {'Count':<10} {'Expected':<12} {'Edge'}")
    print("-" * 60)
    for bucket, data in price_buckets.items():
        if data['yes_total'] > 10:  # Only show meaningful buckets
            win_rate = data['yes_win'] / data['yes_total']
            parts = bucket.replace('%', '').split('-')
            midpoint = (int(parts[0]) + int(parts[1])) / 2 / 100
            expected = midpoint
            edge = win_rate - expected
            print(f"{bucket:<12} {win_rate*100:.1f}%{'':<6} {data['yes_total']:<10} {expected*100:.1f}%{'':<7} {edge*100:+.1f}%")
    
    print("\nNO side (buying NO contracts at price X):")
    print(f"{'Price Range':<12} {'Win Rate':<12} {'Count':<10} {'Expected':<12} {'Edge'}")
    print("-" * 60)
    for bucket, data in price_buckets.items():
        if data['no_total'] > 10:  # Only show meaningful buckets
            win_rate = data['no_win'] / data['no_total']
            parts = bucket.replace('%', '').split('-')
            midpoint = (int(parts[0]) + int(parts[1])) / 2 / 100
            expected = midpoint
            edge = win_rate - expected
            print(f"{bucket:<12} {win_rate*100:.1f}%{'':<6} {data['no_total']:<10} {expected*100:.1f}%{'':<7} {edge*100:+.1f}%")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR STRATEGY")
    print("=" * 80)
    
    # Find best edge opportunities
    best_yes_edge = None
    best_yes_bucket = None
    best_no_edge = None
    best_no_bucket = None
    
    for bucket, data in price_buckets.items():
        if data['yes_total'] > 20:
            win_rate = data['yes_win'] / data['yes_total']
            parts = bucket.replace('%', '').split('-')
            midpoint = (int(parts[0]) + int(parts[1])) / 2 / 100
            edge = win_rate - midpoint
            if best_yes_edge is None or edge > best_yes_edge:
                best_yes_edge = edge
                best_yes_bucket = bucket
        
        if data['no_total'] > 20:
            win_rate = data['no_win'] / data['no_total']
            parts = bucket.replace('%', '').split('-')
            midpoint = (int(parts[0]) + int(parts[1])) / 2 / 100
            edge = win_rate - midpoint
            if best_no_edge is None or edge > best_no_edge:
                best_no_edge = edge
                best_no_bucket = bucket
    
    if best_yes_edge:
        print(f"\n  Best YES opportunity: {best_yes_bucket} (edge: {best_yes_edge*100:+.1f}%)")
    if best_no_edge:
        print(f"  Best NO opportunity: {best_no_bucket} (edge: {best_no_edge*100:+.1f}%)")
    
    # Overall recommendation
    print("\n  STRATEGY RECOMMENDATIONS:")
    print("  -------------------------")
    
    # Check if cheap contracts are good or bad
    cheap_yes = price_buckets['0-10%']
    cheap_no = price_buckets['0-10%']
    
    if cheap_yes['yes_total'] > 10:
        cheap_yes_rate = cheap_yes['yes_win'] / cheap_yes['yes_total']
        if cheap_yes_rate < 0.10:
            print(f"  ✗ AVOID cheap YES (<10%): Win rate {cheap_yes_rate*100:.1f}% vs expected 5%")
        else:
            print(f"  ✓ Cheap YES (<10%): Win rate {cheap_yes_rate*100:.1f}% vs expected 5% = {(cheap_yes_rate - 0.05)*100:+.1f}% edge")
    
    if cheap_no['no_total'] > 10:
        cheap_no_rate = cheap_no['no_win'] / cheap_no['no_total']
        if cheap_no_rate < 0.10:
            print(f"  ✗ AVOID cheap NO (<10%): Win rate {cheap_no_rate*100:.1f}% vs expected 5%")
        else:
            print(f"  ✓ Cheap NO (<10%): Win rate {cheap_no_rate*100:.1f}% vs expected 5% = {(cheap_no_rate - 0.05)*100:+.1f}% edge")
    
    # Compare YES vs NO overall
    total_yes_wins = sum(d['yes_win'] for d in price_buckets.values())
    total_yes = sum(d['yes_total'] for d in price_buckets.values())
    total_no_wins = sum(d['no_win'] for d in price_buckets.values())
    total_no = sum(d['no_total'] for d in price_buckets.values())
    
    if total_yes > 0 and total_no > 0:
        print(f"\n  Overall YES win rate: {total_yes_wins/total_yes*100:.1f}% ({total_yes_wins}/{total_yes})")
        print(f"  Overall NO win rate: {total_no_wins/total_no*100:.1f}% ({total_no_wins}/{total_no})")

if __name__ == '__main__':
    asyncio.run(analyze_settled_markets())
