#!/usr/bin/env python3
"""Deeper analysis of NO-only strategy with price-based edge calculation."""

import asyncio
import os
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from services.kalshi_client import KalshiClient

async def analyze():
    client = KalshiClient(use_demo=False)
    
    print("="*80)
    print("NO-ONLY STRATEGY VALIDATION WITH PRICE ANALYSIS")
    print("="*80)
    
    # Key series to analyze (non-bucket markets)
    series = [
        ('KXNCAAMBTOTAL', 'College Basketball Totals'),
        ('KXNCAAMBGAME', 'College Basketball Games'),
        ('KXNBATOTAL', 'NBA Totals'),
        ('KXNBAGAME', 'NBA Games'),
        ('KXHIGH', 'Weather High Temp'),
        ('KXLOW', 'Weather Low Temp'),
        ('KXPGATOUR', 'PGA Golf'),
        ('KXLPGATOUR', 'LPGA Golf'),
        ('KXWOMHOCKEY', 'Women Hockey'),
        ('KXHOCKEY', 'Hockey'),
    ]
    
    all_results = []
    
    for series_ticker, name in series:
        print(f"\n[Fetching {name} ({series_ticker})...]")
        
        try:
            result = await client.get_markets(
                series_ticker=series_ticker,
                status='settled',
                limit=200
            )
            markets = result.get('markets', []) if isinstance(result, dict) else []
            
            if not markets:
                print(f"  No settled markets found")
                continue
            
            print(f"  Found {len(markets)} settled markets")
            
            for m in markets:
                result_val = m.get('result', '')
                if not result_val:
                    continue
                
                # Get last traded price (what you could have bought at)
                last_price = m.get('last_price', 0) or m.get('yes_price', 0) or 0
                if last_price > 1:
                    last_price = last_price / 100  # cents to decimal
                
                ticker = m.get('ticker', '')
                
                # Skip MVE markets
                if 'MVE' in ticker:
                    continue
                
                # For bucket markets, last_price is the YES price of that bucket
                # NO price = 1 - YES price
                no_price = 1 - last_price if last_price > 0 else 0
                
                all_results.append({
                    'series': series_ticker,
                    'name': name,
                    'result': result_val.upper() if isinstance(result_val, str) else str(result_val),
                    'yes_price': last_price,
                    'no_price': no_price,
                    'ticker': ticker,
                    'question': m.get('title', m.get('question', ''))[:50],
                })
            
            await asyncio.sleep(0.3)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    if not all_results:
        print("\nNo results to analyze!")
        return
    
    print(f"\n{'='*80}")
    print(f"TOTAL MARKETS ANALYZED: {len(all_results)}")
    print("="*80)
    
    # Overall YES vs NO wins
    yes_wins = sum(1 for r in all_results if r['result'] == 'YES')
    no_wins = sum(1 for r in all_results if r['result'] == 'NO')
    total = yes_wins + no_wins
    
    print(f"\nRAW SETTLEMENT RESULTS:")
    print(f"  YES won: {yes_wins} ({yes_wins/total*100:.1f}%)")
    print(f"  NO won: {no_wins} ({no_wins/total*100:.1f}%)")
    
    # Analyze by price buckets - simulating NO-only strategy
    print(f"\n{'='*80}")
    print("NO-ONLY STRATEGY: P&L BY ENTRY PRICE RANGE")
    print("(Assuming you buy 1 NO contract at the last traded NO price)")
    print("="*80)
    
    price_ranges = [
        (0.05, 0.20, "Cheap NO (5-20¢)"),
        (0.15, 0.35, "Low NO (15-35¢)"),
        (0.35, 0.65, "Mid NO (35-65¢)"),
        (0.65, 0.85, "High NO (65-85¢)"),
        (0.85, 0.98, "Expensive NO (85-98¢)"),
        (0.15, 0.95, "Full Range (15-95¢)"),
    ]
    
    for min_p, max_p, label in price_ranges:
        bucket = [r for r in all_results if min_p <= r['no_price'] <= max_p]
        if len(bucket) < 3:
            continue
        
        total_pnl = 0
        wins = 0
        losses = 0
        
        for r in bucket:
            entry = r['no_price']
            if r['result'] == 'NO':
                # NO wins: we get $1, paid entry
                pnl = 1.0 - entry
                total_pnl += pnl
                wins += 1
            else:
                # YES wins: we lose our entry
                pnl = -entry
                total_pnl += pnl
                losses += 1
        
        n = wins + losses
        if n == 0:
            continue
        
        win_rate = wins / n * 100
        avg_entry = sum(r['no_price'] for r in bucket) / len(bucket)
        breakeven = avg_entry * 100  # Need this win rate to break even
        edge = win_rate - breakeven
        
        total_invested = sum(r['no_price'] for r in bucket)
        roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        status = "✅ PROFIT" if total_pnl > 0 else "❌ LOSS"
        
        print(f"\n{label} ({n} markets):")
        print(f"  Win Rate: {win_rate:.1f}% | Need {breakeven:.1f}% to break even")
        print(f"  Edge: {edge:+.1f}pp | ROI: {roi:+.1f}%")
        print(f"  P&L: ${total_pnl:+.2f} on ${total_invested:.2f} wagered {status}")
    
    # Compare with YES-only at same price points
    print(f"\n{'='*80}")
    print("YES-ONLY STRATEGY (comparison)")
    print("="*80)
    
    for min_p, max_p, label in [(0.05, 0.20, "Cheap YES (5-20¢)"), (0.15, 0.35, "Low YES (15-35¢)"), (0.15, 0.95, "Full Range (15-95¢)")]:
        bucket = [r for r in all_results if min_p <= r['yes_price'] <= max_p]
        if len(bucket) < 3:
            continue
        
        total_pnl = 0
        wins = 0
        
        for r in bucket:
            entry = r['yes_price']
            if r['result'] == 'YES':
                total_pnl += (1.0 - entry)
                wins += 1
            else:
                total_pnl += -entry
        
        n = len(bucket)
        win_rate = wins / n * 100
        total_invested = sum(r['yes_price'] for r in bucket)
        roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        status = "✅" if total_pnl > 0 else "❌"
        
        print(f"  {label}: {win_rate:.1f}% WR | ROI: {roi:+.1f}% | P&L: ${total_pnl:+.2f} {status}")
    
    # By category
    print(f"\n{'='*80}")
    print("BY CATEGORY - NO @ 15-95¢")
    print("="*80)
    
    for series_ticker, name in series:
        bucket = [r for r in all_results if r['series'] == series_ticker and 0.15 <= r['no_price'] <= 0.95]
        if len(bucket) < 3:
            continue
        
        wins = sum(1 for r in bucket if r['result'] == 'NO')
        n = len(bucket)
        win_rate = wins / n * 100
        
        total_pnl = sum(
            (1.0 - r['no_price']) if r['result'] == 'NO' else -r['no_price']
            for r in bucket
        )
        total_invested = sum(r['no_price'] for r in bucket)
        roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        status = "✅" if total_pnl > 0 else "❌"
        print(f"  {name}: {wins}/{n} ({win_rate:.0f}%) | ROI: {roi:+.1f}% | P&L: ${total_pnl:+.2f} {status}")
    
    # Final recommendation
    print(f"\n{'='*80}")
    print("CONCLUSION & RECOMMENDATION")
    print("="*80)
    
    # Find the best price range
    best_range = None
    best_roi = -999
    
    for min_p, max_p, label in price_ranges:
        bucket = [r for r in all_results if min_p <= r['no_price'] <= max_p]
        if len(bucket) < 10:
            continue
        
        total_pnl = sum(
            (1.0 - r['no_price']) if r['result'] == 'NO' else -r['no_price']
            for r in bucket
        )
        total_invested = sum(r['no_price'] for r in bucket)
        roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        if roi > best_roi:
            best_roi = roi
            best_range = label
    
    print(f"\n  Best NO price range: {best_range} (ROI: {best_roi:+.1f}%)")

if __name__ == '__main__':
    asyncio.run(analyze())
