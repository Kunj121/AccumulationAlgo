# VWAP Participation Algorithm for Cryptocurrency Markets

## Overview
This project implements a simple VWAP (Volume-Weighted Average Price) participation algorithm for cryptocurrency markets. The algorithm simulates position accumulation while attempting to minimize trading costs in electronic markets, using high-frequency cryptocurrency trading data from 2024. Nanosecond level data was pulled from our professor at the University of Chicago. If you have any questions please email me at kunjs@uchicago.edu

## Data Structure
The project utilizes high-frequency trade and order book data with the following formats:

### Trade Data
Each record contains:
- Received UTC nanoseconds
- Timestamp UTC nanoseconds
- Price (in millionths)
- Size (in billionths)
- Side (-1 or +1, representing the trade direction)

### Order Book Data
Each record contains:
- Ask/Bid prices at multiple levels (in millionths)
- Ask/Bid sizes at multiple levels (in billionths)
- Received UTC nanoseconds
- Timestamp UTC nanoseconds
- Mid price

## Algorithm Specifications
The VWAP participation algorithm:

- **Inputs**: 
  - Target quantity (Q, positive for buying, negative for selling)
  - Start time (τs)
  - Target participation rate (p)

- **Key Parameters**:
  - Minimum size threshold (g) - representing unlikelihood of being first in queue
  - Quoting participation rate k(p) - necessarily larger than p (max 5%)
  - Pause duration (P) - ranges from 0.05 to 5 seconds

- **Conservative Assumptions**:
  - Only trades with opposite Side are available for participation
  - For each price level, accumulation size = max(0, (total_level_quantity - g) × k)
  - Algorithm pauses for P seconds after participating in a flurry of trades

## Transaction Fees
- 50 basis points (0.5%) for trades between crypto-tokens and traditional currencies
- 10 basis points (0.1%) for trades between crypto-tokens

## Implementation Notes
- The algorithm is post-only (passive)
- Target quantity (Q) selection can be based on quantiles of 5-minute volumes
- Minimum size threshold (g) can be derived from the 5th percentile of trade sizes
- The algorithm accommodates both buying and selling scenarios
