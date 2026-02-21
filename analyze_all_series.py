#!/usr/bin/env python3
"""Comprehensive analysis of all Kalshi settled markets by series."""

import asyncio
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from services.kalshi_client import KalshiClient

async def main():
    client = KalshiClient(use_demo=False)
    
    print("=" * 80)
    print("COMPREHENSIVE KALSHI SETTLED MARKETS ANALYSIS")
    print("=" * 80)
    
    # Get events to find all series
    print("\n[Discovering series from settled events...]")
    
    all_series = set()
    cursor = None
    pages = 0
    
    while pages < 20:
        try:
            await asyncio.sleep(0.2)
            result = await client.get_events(status='settled', limit=100, cursor=cursor)
            events = result.get('events', [])
            
            if not events:
                break
            
            for e in events:
                series = e.get('series_ticker')
                if series:
                    all_series.add(series)
            
            pages += 1
            cursor = result.get('cursor')
            if not cursor:
                break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"\nFound {len(all_series)} unique series tickers")
    
    # Analyze each series
    series_stats = []
    
    print("\n[Analyzing settled markets by series...]")
    
    for series in sorted(all_series):
        try:
            await asyncio.sleep(0.3)
            result = await client.get_markets(
                status='settled',
                series_ticker=series,
                limit=200
            )
            markets = result.get('markets', [])
            
            if not markets:
                continue
            
            # Filter out MVE (combo) markets
            single_markets = [m for m in markets if 'KXMVE' not in m.get('ticker', '')]
            
            if len(single_markets) < 5:
                continue
            
            yes_count = len([m for m in single_markets if m.get('result') == 'yes'])
            no_count = len([m for m in single_markets if m.get('result') == 'no'])
            total = yes_count + no_count
            
            if total < 5:
                continue
            
            # Calculate implied probability vs actual
            implied_sum = 0
            actual_sum = 0
            valid_count = 0
            
            for m in single_markets:
                price = (m.get('last_price', 50) or 50) / 100.0
                result = m.get('result', '')
                if result in ['yes', 'no'] and 0 < price < 1:
                    implied_sum += price
                    actual_sum += 1 if result == 'yes' else 0
                    valid_count += 1
            
            if valid_count > 0:
                avg_implied = implied_sum / valid_count
                avg_actual = actual_sum / valid_count
                edge_no = (avg_implied - avg_actual) * 100
                
                series_stats.append({
                    'series': series,
                    'total': total,
                    'yes_rate': yes_count / total * 100,
                    'implied': avg_implied * 100,
                    'edge_no': edge_no,
                })
                
                # Print as we go
                print(f"  {series}: {total} mkts | YES {yes_count/total*100:.0f}% | implied {avg_implied*100:.0f}% | NO edge {edge_no:+.1f}%")
                
        except Exception as e:
            pass  # Skip errors silently
    
    # Summary
    print("\n" + "=" * 80)
    print("TOP NO OPPORTUNITIES (Series where NO has edge)")
    print("=" * 80)
    
    # Sort by NO edge
    series_stats.sort(key=lambda x: x['edge_no'], reverse=True)
    
    for s in series_stats[:20]:
        if s['edge_no'] > 0:
            print(f"\n{s['series']}:")
            print(f"  Markets: {s['total']}")
            print(f"  YES win rate: {s['yes_rate']:.1f}%")
            print(f"  Market implied YES: {s['implied']:.1f}%")
            print(f"  NO EDGE: {s['edge_no']:+.1f}%")
    
    print("\n" + "=" * 80)
    print("TOP YES OPPORTUNITIES (Series where YES has edge)")
    print("=" * 80)
    
    series_stats.sort(key=lambda x: -x['edge_no'])
    
    for s in series_stats[:20]:
        if s['edge_no'] < 0:
            print(f"\n{s['series']}:")
            print(f"  Markets: {s['total']}")
            print(f"  YES win rate: {s['yes_rate']:.1f}%")
            print(f"  Market implied YES: {s['implied']:.1f}%")
            print(f"  YES EDGE: {-s['edge_no']:+.1f}%")
    
    # Overall market efficiency
    print("\n" + "=" * 80)
    print("OVERALL MARKET EFFICIENCY")
    print("=" * 80)
    
    if series_stats:
        total_markets = sum(s['total'] for s in series_stats)
        weighted_edge = sum(s['edge_no'] * s['total'] for s in series_stats) / total_markets
        
        print(f"\nAcross {len(series_stats)} series with {total_markets} total markets:")
        print(f"  Average NO edge: {weighted_edge:+.1f}%")
        
        if weighted_edge > 2:
            print(f"\n  >>> CONCLUSION: Markets systematically OVERPRICE YES")
            print(f"      Betting NO has a {weighted_edge:.1f}% edge on average")
        elif weighted_edge < -2:
            print(f"\n  >>> CONCLUSION: Markets systematically UNDERPRICE YES")
            print(f"      Betting YES has a {-weighted_edge:.1f}% edge on average")
        else:
            print(f"\n  >>> CONCLUSION: Markets are fairly efficient")

if __name__ == '__main__':
    asyncio.run(main())
