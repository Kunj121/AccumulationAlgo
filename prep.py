import pandas as pd

import gzip
import pandas as pd
import os



def get_trade_book(YEAR='2023', CURRENCY_1='BTC', CURRENCY_2='USD', TYPE='trades'):
    notebook_dir = os.getcwd()

    file_path = os.path.join(
        notebook_dir,
        f"crypto_{YEAR}",
        f"{TYPE}_narrow_{CURRENCY_1}-{CURRENCY_2}_{YEAR}.delim.gz"

    )

    tradebook = read_gzip(file_path)

    return tradebook

def get_trade_book_2024(CURRENCY_1, CURRENCY_2):
    notebook_dir = os.getcwd()

    file_path = os.path.join(
        notebook_dir,
        f"crypto_2024",
        f"2024-57__{CURRENCY_1}-{CURRENCY_2}_trades.h5")

    df = pd.read_hdf(file_path)

    return df


def get_order_book(YEAR='2023', CURRENCY_1='BTC', CURRENCY_2='USD', TYPE='book'):
    notebook_dir = os.getcwd()

    file_path = os.path.join(
        notebook_dir,
        f"crypto_{YEAR}",
        f"{TYPE}_narrow_{CURRENCY_1}-{CURRENCY_2}_{YEAR}.delim.gz"

    )

    orderbook = read_gzip(file_path)

    return orderbook


def read_gzip(filepath):

    with gzip.open(filepath, 'rt') as f:
        for i, line in enumerate(f):
            #print(line.strip())
            if i == 5:
                break
    name = pd.read_csv(filepath, compression='gzip', delimiter='\t')
    # if name.columns ==['Ask1PriceMillionths']:
    #     name.drop(name.index, inplace=True)
    #     name = name.T
    # else:
    #     pass
    return name


import pandas as pd
import numpy as np


def analyze_orderbook(ob_df: pd.DataFrame) -> dict:

    # Convert prices from millionths
    ob_df = ob_df.copy()
    for col in ['Ask1PriceMillionths', 'Bid1PriceMillionths', 'Ask2PriceMillionths', 'Bid2PriceMillionths']:
        ob_df[col] = ob_df[col] / 1_000_000

    # Convert sizes from billionths
    for col in ['Ask1SizeBillionths', 'Bid1SizeBillionths', 'Ask2SizeBillionths', 'Bid2SizeBillionths']:
        ob_df[col] = ob_df[col] / 1_000_000_000

    # Calculate spreads
    ob_df['spread_L1'] = ob_df['Ask1PriceMillionths'] - ob_df['Bid1PriceMillionths']
    ob_df['spread_L2'] = ob_df['Ask2PriceMillionths'] - ob_df['Bid2PriceMillionths']

    # Basic statistics
    stats = {
        'spread': {
            'min': ob_df['spread_L1'].min(),
            'max': ob_df['spread_L1'].max(),
            'mean': ob_df['spread_L1'].mean(),
            'median': ob_df['spread_L1'].median(),
            'std': ob_df['spread_L1'].std()
        },
        'top_of_book': {
            'avg_ask_size': ob_df['Ask1SizeBillionths'].mean(),
            'avg_bid_size': ob_df['Bid1SizeBillionths'].mean(),
            'median_ask_size': ob_df['Ask1SizeBillionths'].median(),
            'median_bid_size': ob_df['Bid1SizeBillionths'].median()
        },
        'depth': {
            'avg_ask_depth': ob_df[['Ask1SizeBillionths', 'Ask2SizeBillionths']].sum(axis=1).mean(),
            'avg_bid_depth': ob_df[['Bid1SizeBillionths', 'Bid2SizeBillionths']].sum(axis=1).mean()
        },
        'price_levels': {
            'avg_ask_tick': (ob_df['Ask2PriceMillionths'] - ob_df['Ask1PriceMillionths']).mean(),
            'avg_bid_tick': (ob_df['Bid1PriceMillionths'] - ob_df['Bid2PriceMillionths']).mean()
        }
    }

    return stats


def analyze_trades(trades_df: pd.DataFrame, window_minutes: int = 5) -> dict:
    """
    Analyze trade data
    """
    # Convert sizes from billionths
    trades_df = trades_df.copy()
    trades_df['size'] = trades_df['SizeBillionths'] / 1_000_000_000

    # Basic size statistics
    size_stats = {
        'min': trades_df['size'].min(),
        'max': trades_df['size'].max(),
        'mean': trades_df['size'].mean(),
        'median': trades_df['size'].median(),
        'std': trades_df['size'].std(),
        'p5': trades_df['size'].quantile(0.05),  # Potential g value
        'p95': trades_df['size'].quantile(0.95)
    }

    # Time analysis
    trades_df = trades_df.sort_values('timestamp_utc_nanoseconds')
    time_gaps = trades_df['timestamp_utc_nanoseconds'].diff() / 1e9  # Convert to seconds

    time_stats = {
        'avg_gap': time_gaps.mean(),
        'median_gap': time_gaps.median(),
        'min_gap': time_gaps.min(),
        'max_gap': time_gaps.max()
    }

    # Volume analysis
    trades_df['datetime'] = pd.to_datetime(trades_df['timestamp_utc_nanoseconds'])
    volume_5min = trades_df.set_index('datetime')['size'].resample(f'{window_minutes}T').sum()

    volume_stats = {
        'avg_5min_volume': volume_5min.mean(),
        'median_5min_volume': volume_5min.median(),
        'p65_5min_volume': volume_5min.quantile(0.65),  # For setting target quantity
        'max_5min_volume': volume_5min.max()
    }

    return {
        'size_stats': size_stats,
        'time_stats': time_stats,
        'volume_stats': volume_stats
    }


def suggest_vwap_parameters(ob_stats: dict, trade_stats: dict) -> dict:
    """
    Suggest parameters for VWAP algorithm based on market analysis
    """
    # Get 65th percentile of 5-min volume for Q calculation
    Q_base = trade_stats['volume_stats']['p65_5min_volume']

    suggestions = {
        'target_quantity': Q_base * 0.1,  # Target 10% of 65th percentile 5-min volume
        'min_size_threshold': trade_stats['size_stats']['p5'],  # 5th percentile of trade sizes
        'target_participation': 0.02,  # Start with 2% participation
        'pause_seconds': max(0.05, min(5, trade_stats['time_stats']['median_gap'])),  # Based on typical trade gaps
        'expected_spread_cost': ob_stats['spread']['median']  # Typical spread cost
    }

    return suggestions


# Example usage:
def run_market_analysis(ob_df: pd.DataFrame, trades_df: pd.DataFrame):
    """
    Run complete market analysis and print results
    """
    print("Analyzing market data...\n")

    # Analyze orderbook
    ob_stats = analyze_orderbook(ob_df)
    print("Order Book Statistics:")
    print(f"Average spread: {ob_stats['spread']['mean']:.8f}")
    print(f"Median spread: {ob_stats['spread']['median']:.8f}")
    print(
        f"Average top of book size - Ask: {ob_stats['top_of_book']['avg_ask_size']:.4f}, Bid: {ob_stats['top_of_book']['avg_bid_size']:.4f}")
    print(
        f"Average market depth - Ask: {ob_stats['depth']['avg_ask_depth']:.4f}, Bid: {ob_stats['depth']['avg_bid_depth']:.4f}")
    print()

    # Analyze trades
    trade_stats = analyze_trades(trades_df)
    print("Trade Statistics:")
    print(
        f"Trade size - Mean: {trade_stats['size_stats']['mean']:.4f}, Median: {trade_stats['size_stats']['median']:.4f}")
    print(f"5th percentile size (suggested g): {trade_stats['size_stats']['p5']:.4f}")
    print(f"Average time between trades: {trade_stats['time_stats']['avg_gap']:.3f} seconds")
    print(f"Average 5-min volume: {trade_stats['volume_stats']['avg_5min_volume']:.4f}")
    print()

    # Get suggestions
    suggestions = suggest_vwap_parameters(ob_stats, trade_stats)
    print("Suggested VWAP Parameters:")
    print(f"Target quantity (Q): {suggestions['target_quantity']:.4f}")
    print(f"Minimum size threshold (g): {suggestions['min_size_threshold']:.4f}")
    print(f"Target participation rate (p): {suggestions['target_participation']:.1%}")
    print(f"Pause after multi-level trades: {suggestions['pause_seconds']:.2f} seconds")

    return ob_stats, trade_stats, suggestions

# To use this code:
# ob_stats, trade_stats, suggestions = run_market_analysis(orderbook_df, tradebook_df)


def analyze_trade_levels(trade, orderbook_df, timestamp):
    """
    Analyze if a trade broke through multiple price levels
    """
    # Get closest orderbook snapshot before trade
    ob_snapshot = orderbook_df[orderbook_df['timestamp_utc_nanoseconds'] <= timestamp].iloc[-1]

    # For a sell order (hitting bids)
    if trade['Side'] < 0:
        bid1 = ob_snapshot['Bid1PriceMillionths'] / 1_000_000
        bid2 = ob_snapshot['Bid2PriceMillionths'] / 1_000_000
        bid1_size = ob_snapshot['Bid1SizeBillionths'] / 1_000_000_000
        bid2_size = ob_snapshot['Bid2SizeBillionths'] / 1_000_000_000

        # Check if trade size was bigger than level 1
        if trade['SizeBillionths'] / 1_000_000_000 > bid1_size:
            return True

    # For a buy order (hitting asks)
    else:
        ask1 = ob_snapshot['Ask1PriceMillionths'] / 1_000_000
        ask2 = ob_snapshot['Ask2PriceMillionths'] / 1_000_000
        ask1_size = ob_snapshot['Ask1SizeBillionths'] / 1_000_000_000
        ask2_size = ob_snapshot['Ask2SizeBillionths'] / 1_000_000_000

        # Check if trade size was bigger than level 1
        if trade['SizeBillionths'] / 1_000_000_000 > ask1_size:
            return True

    return False


def run_vwap_simulation_with_ob(algo, tradebook_df, orderbook_df):
    pause_until = None

    for _, trade in tradebook_df.iterrows():
        timestamp = trade['timestamp_utc_nanoseconds']

        # Skip if we're in pause period
        if pause_until and timestamp < pause_until:
            continue

        # Convert units
        price = trade['PriceMillionths'] / 1_000_000
        size = trade['SizeBillionths'] / 1_000_000_000

        # Check if trade broke through levels
        broke_levels = analyze_trade_levels(trade, orderbook_df, timestamp)

        # If broke levels, implement pause
        if broke_levels:
            pause_until = timestamp + int(algo.pause * 1e9)  # Convert pause seconds to nanos
            continue

        # Process trade if not paused
        qty, cost = algo.process_trade(
            timestamp=timestamp,
            price=price,
            size=size,
            side=trade['Side']
        )

        # Update market VWAP
        algo.update_market_vwap(
            timestamp=timestamp,
            price=price,
            size=size
        )
