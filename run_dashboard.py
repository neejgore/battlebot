#!/usr/bin/env python3
"""Standalone dashboard demo - runs without full bot dependencies."""

import asyncio
import random
import sys
from datetime import datetime

# Direct import to avoid pulling in other services
import importlib.util
spec = importlib.util.spec_from_file_location("dashboard", "services/dashboard.py")
dashboard_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard_module)
Dashboard = dashboard_module.Dashboard

# Sample markets for demo
DEMO_MARKETS = [
    {"token_id": "1", "question": "Will Bitcoin reach $150k by end of 2026?", "current_price": 0.42, "volume_24h": 85000, "category": "crypto"},
    {"token_id": "2", "question": "Will the Fed cut rates in March 2026?", "current_price": 0.68, "volume_24h": 120000, "category": "economics"},
    {"token_id": "3", "question": "Will OpenAI release GPT-5 by Q2 2026?", "current_price": 0.55, "volume_24h": 45000, "category": "tech"},
    {"token_id": "4", "question": "Will Ethereum flip Bitcoin market cap in 2026?", "current_price": 0.12, "volume_24h": 32000, "category": "crypto"},
    {"token_id": "5", "question": "Will there be a US government shutdown in 2026?", "current_price": 0.35, "volume_24h": 28000, "category": "politics"},
]


async def simulate_activity(dashboard: Dashboard):
    """Simulate bot activity for demo."""
    
    # Add initial markets
    await asyncio.sleep(2)
    for market in DEMO_MARKETS:
        await dashboard.market_added(market)
        await asyncio.sleep(0.5)
    
    trade_count = 0
    
    while True:
        await asyncio.sleep(random.uniform(5, 15))
        
        # Simulate AI analysis
        market = random.choice(DEMO_MARKETS)
        raw_prob = random.uniform(0.3, 0.7)
        market_price = market['current_price']
        edge = raw_prob - market_price
        confidence = random.uniform(0.5, 0.9)
        
        decision = 'TRADE' if abs(edge) > 0.05 and confidence > 0.6 else 'NO_TRADE'
        reason = ''
        if decision == 'NO_TRADE':
            reasons = ['Edge too small', 'Low confidence', 'Spread too wide', 'Cooldown active']
            reason = random.choice(reasons)
        
        await dashboard.analysis_complete(
            market_id=market['token_id'],
            question=market['question'],
            result={
                'raw_prob': raw_prob,
                'market_price': market_price,
                'edge': edge,
                'confidence': confidence,
                'decision': decision,
                'reason': reason,
            }
        )
        
        # Occasionally enter a trade
        if decision == 'TRADE' and random.random() > 0.5:
            trade_count += 1
            side = 'YES' if edge > 0 else 'NO'
            size = random.uniform(10, 50)
            
            await dashboard.trade_entry({
                'token_id': market['token_id'],
                'question': market['question'],
                'side': side,
                'price': market_price,
                'size': size,
                'edge': edge,
            })
            
            # Update stats
            await dashboard.update_stats({
                'total_trades': trade_count,
                'active_positions': random.randint(0, 3),
                'daily_pnl': random.uniform(-20, 50),
            })
            
            # Sometimes exit after a delay
            if random.random() > 0.6:
                await asyncio.sleep(random.uniform(10, 30))
                pnl = random.uniform(-15, 25)
                await dashboard.trade_exit({
                    'token_id': market['token_id'],
                    'question': market['question'],
                    'side': side,
                    'price': market_price + (0.05 if pnl > 0 else -0.03),
                    'size': size,
                    'pnl': pnl,
                })


async def main():
    dashboard = Dashboard(port=8080)
    
    print("=" * 50)
    print("BATTLE-BOT DASHBOARD DEMO")
    print("=" * 50)
    print()
    
    await dashboard.start()
    await dashboard.update_stats({
        'dry_run': True,
        'running': True,
        'bankroll': 1000,
        'daily_pnl': 0,
        'total_trades': 0,
        'active_positions': 0,
        'markets_monitored': 0,
    })
    
    print()
    print("Open http://localhost:8080 in your browser")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        await simulate_activity(dashboard)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await dashboard.stop()


if __name__ == "__main__":
    asyncio.run(main())
