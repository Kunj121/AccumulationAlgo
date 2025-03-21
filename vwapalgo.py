import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# Below functions were utilized GPT if green code blocks

def clean_tradebook(tradebook: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the tradebook:
      - Create 'ts_ns': nanosecond integer timestamp.
      - Convert 'PriceMillionths' to 'price' in dollars.
      - Convert 'SizeBillionths' to 'size' in standard units.
    """
    tradebook = tradebook.copy()
    tradebook["ts_ns"] = tradebook["timestamp_utc_nanoseconds"].view("int64")
    tradebook["price"] = tradebook["PriceMillionths"] / 1e6
    tradebook["size"] = tradebook["SizeBillionths"] / 1e9
    return tradebook

def compute_target_quantity(tradebook: pd.DataFrame,
                            volume_interval: str = '5min',
                            fraction: float = 0.01,
                            quantile: float = 0.65) -> float:
    """
    Compute target quantity Q as a small fraction of the quantile (e.g. 65th percentile)
    of the volume over a given interval.
    For example, if the 65th percentile of the 5‐minute volume is V, then Q = fraction * V.
    """
    df = tradebook.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp_utc_nanoseconds"])
    volume_by_interval = df.resample(volume_interval, on="timestamp")["size"].sum()
    vol_quantile = volume_by_interval.quantile(quantile)
    return fraction * vol_quantile

def compute_min_size_threshold(tradebook: pd.DataFrame) -> float:
    """
    Compute the minimum size threshold g as the 5th percentile of trade sizes.
    """
    return tradebook["size"].quantile(0.05)

# ==============================================================================
# 2. Create a Sample 15-Minute Window
# ==============================================================================

def get_sample_window(tradebook: pd.DataFrame, window_length_min: float = 15, start_time: int = None):
    """
    Extract a sample window of length 'window_length_min' minutes from the tradebook.
    If start_time (in nanoseconds) is provided, use it; otherwise, choose a random start time
    such that the window [start_time, start_time + window_length] is contained within the tradebook.

    Returns:
        window: DataFrame with trades in the window.
        start_time: window start time (ns).
        end_time: window end time (ns).
    """
    min_trade_ns = tradebook["ts_ns"].min()
    max_trade_ns = tradebook["ts_ns"].max()
    window_length_ns = int(window_length_min * 60 * 1e9)
    if start_time is None:
        max_start_time = max_trade_ns - window_length_ns
        start_time = int(random.uniform(min_trade_ns, max_start_time))
    end_time = start_time + window_length_ns
    window = tradebook[(tradebook["ts_ns"] >= start_time) & (tradebook["ts_ns"] <= end_time)].copy()
    return window, start_time, end_time

# ==============================================================================
# 3. Sequential Simulation with Pause Mechanism
# ==============================================================================

def sequential_run_simulation_with_pause(tradebook_window: pd.DataFrame,
                                         Q: float,
                                         g: float,
                                         k: float,
                                         fee_rate: float,
                                         pause: float) -> dict:
    """
    Run the VWAP participation simulation sequentially (processing groups one by one)
    with a pause mechanism.

    For each group (grouped by ts_ns and price) that is eligible (i.e. whose timestamp is
    not in a pause period), compute:
         fill = max(0, (group_size - g)) * k.
    Then, add the fill to the cumulative fill. Immediately after processing a group (if a fill is taken),
    set resume_time = current_group_timestamp + (pause in ns). Continue until cumulative fill ≥ |Q|.

    Parameters:
       tradebook_window: DataFrame of trades in the simulation window.
       Q: target quantity (positive for buy, negative for sell).
       g: minimum size threshold.
       k: fixed participation factor.
       fee_rate: fee rate (e.g., 0.005 for 0.5% fee).
       pause: pause duration P in seconds (0.05 to 5).

    Returns a dictionary of summary metrics.
    """
    # Select trades on the aggressive side.
    if Q > 0:
        trades = tradebook_window[tradebook_window["Side"] < 0].copy()
    else:
        trades = tradebook_window[tradebook_window["Side"] > 0].copy()

    if trades.empty:
        window_length_sec = (tradebook_window["ts_ns"].max() - tradebook_window["ts_ns"].min()) / 1e9
        return {
            "executed_quantity": 0.0,
            "completion_rate": 0.0,
            "execution_vwap": 0.0,
            "market_vwap": 0.0,
            "vwap_performance": 0.0,
            "total_value": 0.0,
            "total_fees": 0.0,
            "num_trades": 0,
            "time_to_completion_sec": window_length_sec
        }

    # Group trades by timestamp and price.
    grouped = trades.groupby(["ts_ns", "price"]).agg({"size": "sum", "Side": "first"}).reset_index()
    grouped = grouped.sort_values("ts_ns").reset_index(drop=True)

    pause_ns = int(pause * 1e9)
    window_start = tradebook_window["ts_ns"].min()
    resume_time = window_start  # initial resume_time
    cumulative_fill = 0.0
    effective_fills = np.zeros(len(grouped))
    used_groups = []
    cutoff_idx = None

    for idx, row in grouped.iterrows():
        group_time = row["ts_ns"]
        if group_time < resume_time:
            continue  # skip groups within the pause period
        fill = max(0, row["size"] - g) * k
        # If adding this fill would exceed the target, adjust the fill.
        if cumulative_fill + fill >= abs(Q):
            fill = abs(Q) - cumulative_fill
            effective_fills[idx] = fill
            cumulative_fill += fill
            used_groups.append(idx)
            cutoff_idx = idx
            # Once the target is reached, break out.
            break
        else:
            effective_fills[idx] = fill
            cumulative_fill += fill
            used_groups.append(idx)
            # Set pause: skip any group until current time + pause_ns.
            resume_time = group_time + pause_ns

    if cutoff_idx is None:
        cutoff_idx = len(grouped) - 1

    # Compute execution VWAP using only groups with nonzero effective fill.
    if cumulative_fill > 0:
        used = grouped.loc[used_groups].copy()
        used["effective_fill"] = effective_fills[used_groups]
        total_notional = (used["effective_fill"] * used["price"]).sum()
        execution_vwap = total_notional / cumulative_fill
    else:
        execution_vwap = 0

    num_trades = len(used_groups)
    time_to_completion_sec = (grouped.loc[cutoff_idx, "ts_ns"] - window_start) / 1e9

    # Compute market VWAP over the entire window.
    total_market_value = (tradebook_window["price"] * tradebook_window["size"]).sum()
    total_market_volume = tradebook_window["size"].sum()
    market_vwap = total_market_value / total_market_volume if total_market_volume > 0 else 0

    total_fees = 0.0
    for idx in used_groups:
        fee = effective_fills[idx] * grouped.loc[idx, "price"] * fee_rate
        total_fees += fee
    total_execution_cost = (cumulative_fill * execution_vwap) + total_fees

    completion_rate = cumulative_fill / abs(Q)

    summary = {
        "target_quantity": Q,
        "executed_quantity": cumulative_fill if Q > 0 else -cumulative_fill,
        "completion_rate": completion_rate,
        "execution_vwap": execution_vwap,
        "market_vwap": market_vwap,
        "vwap_spread": market_vwap - execution_vwap,
        "vwap_performance": (execution_vwap - market_vwap) / market_vwap if market_vwap != 0 else 0,
        "total_value": total_execution_cost,
        "total_fees": total_fees,
        "num_trades": num_trades,
        "time_to_completion_sec": time_to_completion_sec
    }
    df_summary = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
    df_summary.set_index("Metric", inplace=True)
    return df_summary

# ==============================================================================
# 4. Monte Carlo Simulation (Sequential with Pause)
# ==============================================================================

def monte_carlo_sequential_simulation_with_pause(tradebook: pd.DataFrame,
                                                 target_quantity: float,
                                                 min_size_threshold: float,
                                                 fixed_k: float,
                                                 fee_rate: float,
                                                 pause: float,
                                                 num_simulations: int = 10,
                                                 window_length_min: float = 15) -> pd.DataFrame:
    """
    Runs the simulation over multiple windows and collects key metrics.
    Adapts to the fact that sequential_run_simulation_with_pause returns a DataFrame.
    """
    results = []
    min_trade_ns = tradebook["ts_ns"].min()
    max_trade_ns = tradebook["ts_ns"].max()
    window_length_ns = int(window_length_min * 60 * 1e9)

    for _ in range(num_simulations):
        max_start_time = max_trade_ns - window_length_ns
        if max_start_time <= min_trade_ns:
            continue
        sim_start_time = int(random.uniform(min_trade_ns, max_start_time))
        sim_end_time = sim_start_time + window_length_ns
        window = tradebook[(tradebook["ts_ns"] >= sim_start_time) & (tradebook["ts_ns"] <= sim_end_time)].copy()
        if window.empty:
            continue
        arrival_price = window.sort_values("ts_ns").iloc[0]["price"]
        summary_df = sequential_run_simulation_with_pause(window, target_quantity, min_size_threshold, fixed_k,
                                                          fee_rate, pause)

        # Extract key metrics from the returned DataFrame:
        exec_vwap = summary_df.loc["execution_vwap", "Value"]
        exec_vs_arrival_pct = ((exec_vwap - arrival_price) / arrival_price) if arrival_price != 0 else 0

        simulation_summary = {
            "target_quantity": summary_df.loc["target_quantity", "Value"],
            "executed_quantity": summary_df.loc["executed_quantity", "Value"],
            "completion_rate": summary_df.loc["completion_rate", "Value"],
            "execution_vwap": exec_vwap,
            "market_vwap": summary_df.loc["market_vwap", "Value"],
            "vwap_performance": summary_df.loc["vwap_performance", "Value"],
            "total_value": summary_df.loc["total_value", "Value"],
            "total_fees": summary_df.loc["total_fees", "Value"],
            "num_trades": summary_df.loc["num_trades", "Value"],
            "time_to_completion_sec": summary_df.loc["time_to_completion_sec", "Value"],
            "arrival_price": arrival_price,
            "exec_vs_arrival_pct": exec_vs_arrival_pct,
            "window_length_sec": (sim_end_time - sim_start_time) / 1e9
        }
        results.append(simulation_summary)

    return pd.DataFrame(results)

# ==============================================================================
# 5. Plotting Function for Monte Carlo Results
# ==============================================================================

def plot_monte_carlo_results(results_df: pd.DataFrame):
    plt.figure(figsize=(14, 10))

    # Histogram: Completion Rate
    plt.subplot(2, 3, 1)
    plt.hist(results_df['completion_rate'], bins=40, edgecolor='k', alpha=0.7)
    plt.title("Completion Rate")
    plt.xlabel("Completion Rate")
    plt.ylabel("Frequency")

    # Histogram: Execution VWAP vs. Arrival Price (% deviation)
    plt.subplot(2, 3, 2)
    plt.hist(results_df['exec_vs_arrival_pct'], bins=40, edgecolor='k', alpha=0.7)
    plt.title("Exec VWAP vs. Arrival Price (%)")
    plt.xlabel("Deviation (%)")
    plt.ylabel("Frequency")

    # Histogram: Execution VWAP vs. Market VWAP (% deviation)
    plt.subplot(2, 3, 3)
    plt.hist(results_df['exec_vs_market_pct'], bins=40, edgecolor='k', alpha=0.7)
    plt.title("Exec VWAP vs. Market VWAP (%)")
    plt.xlabel("Deviation (%)")
    plt.ylabel("Frequency")


    # Histogram: Time to Completion (sec)
    plt.subplot(2, 3, 5)
    plt.hist(results_df['time_to_completion_sec'], bins=40, edgecolor='k', alpha=0.7)
    plt.title("Time to Completion (sec)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency")


def summary_results(results):
    results = results[['target_quantity', 'executed_quantity', 'completion_rate', 'total_fees', 'num_trades', 'time_to_completion_sec']]
    mean_results = results.mean()
    std_results = results.std()

    summary = pd.concat([mean_results, std_results], axis=1)
    summary.rename(columns = {0:'Average', 1:'Standard Deviation'}, inplace=True)

    return summary





import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparison_time(comparison):


    x = [1, 2, 3]  # Representing Trial 1, 2, 3

    # Define x values (Trials)

    # Define y values for each metric
    y1 = list(comparison.loc['time_to_completion_sec'])  # Values for time_to_completion_sec
    y2 = list(comparison.loc['num_trades'])  # Values for num_trades

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    # Plot time_to_completion_sec
    sns.scatterplot(ax=axes[0], x=x, y=y1, color='blue', s=100, label='Time to Completion')
    axes[0].set_title("Time to Completion Across Trials")
    axes[0].set_xlabel("Trials")
    axes[0].set_ylabel("Time to Completion (Sec)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Trial 1", "Trial 2", "Trial 3"])
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot num_trades
    sns.scatterplot(ax=axes[1], x=x, y=y2, color='green', s=100, label='Number of Trades')
    axes[1].set_title("Number of Trades Across Trials")
    axes[1].set_xlabel("Trials")
    axes[1].set_ylabel("Number of Trades")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Trial 1", "Trial 2", "Trial 3"])
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def analytic_metrics(df):

    notional_per_time = df.loc['total_value']/ df.loc['time_to_completion_sec']
    costs_per_notional = df.loc['total_fees']/ df.loc['total_value']

    notional_per_time = pd.DataFrame(notional_per_time, columns=['notional_per_time'])
    costs_per_notional = pd.DataFrame(costs_per_notional, columns=['costs_per_notional'])

    final = pd.merge(notional_per_time, costs_per_notional, left_index=True, right_index=True)



    return final.T




def test_quantity(tradebook, fee_rate: int = 0.005, fraction=[0.01, 0.025, 0.05], window: int = 60):

    min_size_threshold = compute_min_size_threshold(tradebook)

    window, sim_start, sim_end = get_sample_window(tradebook, window_length_min=window)

    target_quantity_1 = compute_target_quantity(tradebook, volume_interval='5min', fraction=fraction[0], quantile=0.65)
    target_quantity_2 = compute_target_quantity(tradebook, volume_interval='5min', fraction=fraction[1], quantile=0.65)
    target_quantity_3 = compute_target_quantity(tradebook, volume_interval='5min', fraction=fraction[2], quantile=0.65)

    trial_1 = sequential_run_simulation_with_pause(window, Q=target_quantity_1, g=min_size_threshold, k=0.03,
                                                   fee_rate=fee_rate, pause=0.1)
    trial_2 = sequential_run_simulation_with_pause(window, Q=target_quantity_2, g=min_size_threshold, k=0.03,
                                                   fee_rate=fee_rate, pause=0.1)
    trial_3 = sequential_run_simulation_with_pause(window, Q=target_quantity_3, g=min_size_threshold, k=0.03,
                                                   fee_rate=fee_rate, pause=0.1)

    comparison = pd.concat([trial_1, trial_2, trial_3], axis=1)
    comparison.columns = [1, 2, 3]

    print(f"Quantity levels: {fraction}")
    return comparison



def test_quoting_rate_k(tradebook, window_length_min: int = 60, fee_rate = 0.005):


    min_size_threshold = compute_min_size_threshold(tradebook)

    window, sim_start, sim_end = get_sample_window(tradebook, window_length_min=window_length_min)
    # %%
    target_quantity = compute_target_quantity(tradebook, volume_interval='15min', fraction=0.02, quantile=0.65)

    trial_1 = sequential_run_simulation_with_pause(window, Q=target_quantity, g=min_size_threshold, k=.01,
                                                      fee_rate=fee_rate, pause=0.1)
    trial_2 = sequential_run_simulation_with_pause(window, Q=target_quantity, g=min_size_threshold, k=.025,
                                                      fee_rate=fee_rate, pause=0.1)
    trial_3 = sequential_run_simulation_with_pause(window, Q=target_quantity, g=min_size_threshold, k=.05,
                                                      fee_rate=fee_rate, pause=0.1)

    comparison = pd.concat([trial_1, trial_2, trial_3], axis=1)
    comparison.columns = [1, 2, 3]
    print("Quoted Participation levels: [0.01,0.025,0.5]")

    return comparison


# ==============================================================================
# 3. Create a Parameter Test Function
# ==============================================================================
def parameter_test(tradebook: pd.DataFrame,
                   parameter_type: str = "Q",
                   parameter_values=None,
                   fraction_for_Q: float = 0.01,
                   fixed_k: float = 0.03,
                   num_simulations: int = 50,
                   window_length_min: float = 15,
                   fee_rate: float = 0.005,
                   pause: float = 0.1):
    """
    Runs Monte Carlo simulations while varying either Q or k.

    For "Q": parameter_values are fractions used to compute Q.
    For "K": parameter_values are the participation factor k values (with Q computed using a fixed fraction).

    Returns a combined DataFrame with an extra 'parameter' column.
    """
    results_list = []
    min_size_threshold = compute_min_size_threshold(tradebook)

    if parameter_type == "Q":
        for frac in parameter_values:
            target_quantity = compute_target_quantity(tradebook, volume_interval='5min', fraction=frac, quantile=0.65)
            sim_results = monte_carlo_sequential_simulation_with_pause(
                tradebook,
                target_quantity,
                min_size_threshold,
                fixed_k,
                fee_rate,
                pause,
                num_simulations,
                window_length_min
            )
            sim_results['parameter'] = frac
            results_list.append(sim_results)
    elif parameter_type == "K":
        target_quantity = compute_target_quantity(tradebook, volume_interval='5min', fraction=fraction_for_Q,
                                                  quantile=0.65)
        for k_val in parameter_values:
            sim_results = monte_carlo_sequential_simulation_with_pause(
                tradebook,
                target_quantity,
                min_size_threshold,
                k_val,
                fee_rate,
                pause,
                num_simulations,
                window_length_min
            )
            sim_results['parameter'] = k_val
            results_list.append(sim_results)
    else:
        raise ValueError("parameter_type must be either 'Q' or 'K'")

    combined_results = pd.concat(results_list, ignore_index=True)
    return combined_results




# Plotting helper function
def plot_parameter_results(df: pd.DataFrame, param_label: str):
    plt.figure(figsize=(14, 4))

    # Histogram for Completion Rate
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='completion_rate', hue='parameter',
                 bins=20, palette="Set2", element='step', stat="density")
    plt.title(f"Histogram of Completion Rate by {param_label}")
    plt.xlabel("Completion Rate")
    plt.ylabel("Density")

    # Histogram for Time to Completion
    plt.subplot(1, 3, 2)
    sns.histplot(data=df, x='time_to_completion_sec', hue='parameter',
                 bins=20, palette="Set3", element='step', stat="density")
    plt.title(f"Histogram of Time to Completion by {param_label}")
    plt.xlabel("Time to Completion (sec)")
    plt.ylabel("Density")

    # Histogram for Exec VWAP vs. Arrival (% deviation)
    plt.subplot(1, 3, 3)
    sns.histplot(data=df, x='exec_vs_arrival_pct', hue='parameter',
                 bins=20, palette="Set1", element='step', stat="density")
    plt.title(f"Histogram of Exec VWAP vs. Arrival (%) by {param_label}")
    plt.xlabel("Exec vs. Arrival (%)")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()


