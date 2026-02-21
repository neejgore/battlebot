#!/usr/bin/env python3
"""Quick analysis of key Kalshi series."""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from services.kalshi_client import KalshiClient

async def main():
    client = KalshiClient(use_demo=False)
    
    print("=" * 80)
    print("KALSHI SETTLED MARKETS - KEY SERIES ANALYSIS")
    print("=" * 80)
    
    # Key series to analyze
    series_list = [
        # Sports
        ('KXNBAML', 'NBA Moneyline'),
        ('KXNFLML', 'NFL Moneyline'),  
        ('KXMLBML', 'MLB Moneyline'),
        ('KXNHLML', 'NHL Moneyline'),
        ('KXNCAAB', 'NCAA Basketball'),
        ('KXNCAAF', 'NCAA Football'),
        ('KXSOCCER', 'Soccer'),
        ('KXTENNIS', 'Tennis'),
        ('KXGOLF', 'Golf'),
        # Economics
        ('KXCPI', 'CPI Inflation'),
        ('KXGDP', 'GDP'),
        ('KXFED', 'Fed/FOMC'),
        ('KXJOBLESS', 'Jobless Claims'),
        ('KXUNEMPLOYMENT', 'Unemployment'),
        # Crypto
        ('KXBTC', 'Bitcoin Price'),
        ('KXETH', 'Ethereum Price'),
        # Weather
        ('KXWEATHER', 'Weather'),
        ('KXHIGHTEMP', 'High Temperature'),
        # Politics
        ('KXTRUMP', 'Trump'),
        ('KXBIDEN', 'Biden'),
        ('KXHOUSE', 'House'),
        ('KXSENATE', 'Senate'),
        # Entertainment
        ('KXOSCARS', 'Oscars'),
        ('KXEMMYS', 'Emmys'),
    ]
    
    results = []
    
    print("\n[Fetching data for each series...]\n")
    
    for series_ticker, name in series_list:
        try:
            await asyncio.sleep(0.25)
            result = await client.get_markets(
                status='settled',
                series_ticker=series_ticker,
                limit=500
            )
            markets = result.get('markets', [])
            
            # Filter out MVE combos
            single = [m for m in markets if 'MVE' not in m.get('ticker', '')]
            
            if len(single) < 3:
                continue
            
            yes_ct = len([m for m in single if m.get('result') == 'yes'])
            no_ct = len([m for m in single if m.get('result') == 'no'])
            total = yes_ct + no_ct
            
            if total < 3:
                continue
            
            yes_rate = yes_ct / total * 100
            
            # Get price distribution
            cheap_yes_win = len([m for m in single if m.get('result') == 'yes' and (m.get('last_price', 50) or 50) < 20])
            cheap_yes_total = len([m for m in single if (m.get('last_price', 50) or 50) < 20])
            
            results.append({
                'series': series_ticker,
                'name': name,
                'total': total,
                'yes_rate': yes_rate,
                'cheap_yes_win': cheap_yes_win,
                'cheap_yes_total': cheap_yes_total,
            })
            
            # Print as we go
            cheap_info = f" | cheap(<20¢): {cheap_yes_win}/{cheap_yes_total}" if cheap_yes_total > 0 else ""
            print(f"{name:20} ({series_ticker:12}): {total:4} markets | YES wins {yes_rate:5.1f}%{cheap_info}")
            
        except Exception as e:
            pass  # Skip errors
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    if results:
        # Sort by YES win rate
        results.sort(key=lambda x: x['yes_rate'])
        
        print("\n📉 LOWEST YES WIN RATE (best for NO bets):")
        for r in results[:5]:
            print(f"  {r['name']}: {r['yes_rate']:.1f}% YES wins ({r['total']} markets)")
        
        print("\n📈 HIGHEST YES WIN RATE (worst for NO bets):")
        for r in sorted(results, key=lambda x: -x['yes_rate'])[:5]:
            print(f"  {r['name']}: {r['yes_rate']:.1f}% YES wins ({r['total']} markets)")
        
        # Cheap contract analysis
        total_cheap_win = sum(r['cheap_yes_win'] for r in results)
        total_cheap = sum(r['cheap_yes_total'] for r in results)
        
        if total_cheap > 0:
            print(f"\n💰 CHEAP CONTRACTS (<20¢):")
            print(f"  Across all series: {total_cheap_win}/{total_cheap} = {total_cheap_win/total_cheap*100:.1f}% YES win rate")
            print(f"  Expected if fair: ~10%")
            if total_cheap_win / total_cheap < 0.10:
                print(f"  >>> Markets OVERPRICE cheap YES contracts")
            else:
                print(f"  >>> Cheap contracts may have edge")
        
        # Overall
        total_markets = sum(r['total'] for r in results)
        weighted_yes = sum(r['yes_rate'] * r['total'] for r in results) / total_markets
        
        print(f"\n🎯 OVERALL ({total_markets} markets):")
        print(f"  Average YES win rate: {weighted_yes:.1f}%")
        print(f"  Average NO win rate: {100 - weighted_yes:.1f}%")
        
        if weighted_yes < 40:
            print(f"\n  ✓ CONFIRMED: NO bets win more often ({100-weighted_yes:.1f}% vs {weighted_yes:.1f}%)")

if __name__ == '__main__':
    asyncio.run(main())
