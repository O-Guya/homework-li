"""
Results Analysis and Visualization Script
Aggregates multiple seeds and generates comparison plots
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import interp1d


def load_and_aggregate_data(log_dir, algo, prob, seeds):
    """
    Load data from multiple seeds and aggregate

    Returns:
        episodes: Common episode array
        mean_rewards: Mean reward across seeds
        std_rewards: Std deviation of rewards
        all_data: List of individual dataframes
    """
    all_data = []

    for seed in seeds:
        filename = f"{algo}_prob{prob}_seed{seed}.csv"
        filepath = os.path.join(log_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue

        df = pd.read_csv(filepath)
        all_data.append(df)

    if len(all_data) == 0:
        print(f"Error: No data found for {algo}, prob={prob}")
        return None, None, None, None

    # Find the minimum length across all seeds
    min_length = min(len(df) for df in all_data)

    # Truncate all dataframes to the same length
    all_data_truncated = [df.iloc[:min_length].copy() for df in all_data]

    # Stack rewards
    rewards_array = np.array([df['Reward'].values for df in all_data_truncated])

    # Compute mean and std
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    episodes = all_data_truncated[0]['Episode'].values

    return episodes, mean_rewards, std_rewards, all_data_truncated


def smooth_curve(data, window=50):
    """Apply moving average smoothing"""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def plot_training_curves(log_dir, prob, seeds, output_dir='plots'):
    """
    Plot training curves for Q-Learning vs MAPPO at a specific deception probability

    Figure 1: Training Curves (full)
    Figure 2: Late-stage Performance (last 2000 episodes)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    algos = ['q_learning', 'mappo']
    colors = {'q_learning': 'blue', 'mappo': 'red'}
    labels = {'q_learning': 'Q-Learning', 'mappo': 'MAPPO'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Subplot 1: Full training curves ---
    ax1 = axes[0]

    for algo in algos:
        episodes, mean_rewards, std_rewards, _ = load_and_aggregate_data(log_dir, algo, prob, seeds)

        if episodes is None:
            continue

        # Smooth curves
        mean_smooth = smooth_curve(mean_rewards, window=50)

        # Plot raw data (very transparent)
        ax1.plot(episodes, mean_rewards, alpha=0.15, color=colors[algo])

        # Plot smoothed mean
        ax1.plot(episodes, mean_smooth, color=colors[algo], linewidth=2.5, label=labels[algo])

        # Plot std deviation band
        std_smooth = smooth_curve(std_rewards, window=50)
        ax1.fill_between(
            episodes,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=colors[algo],
            alpha=0.2
        )

    ax1.set_title(f'Training Curves (Deception Prob = {prob})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Late-stage performance (zoom in) ---
    ax2 = axes[1]

    for algo in algos:
        episodes, mean_rewards, std_rewards, _ = load_and_aggregate_data(log_dir, algo, prob, seeds)

        if episodes is None:
            continue

        # Take last 2000 episodes
        cutoff = max(0, len(episodes) - 2000)
        episodes_late = episodes[cutoff:]
        mean_late = mean_rewards[cutoff:]
        std_late = std_rewards[cutoff:]

        # Smooth
        mean_smooth = smooth_curve(mean_late, window=50)
        std_smooth = smooth_curve(std_late, window=50)

        # Plot
        ax2.plot(episodes_late, mean_smooth, color=colors[algo], linewidth=2.5, label=labels[algo])
        ax2.fill_between(
            episodes_late,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=colors[algo],
            alpha=0.2
        )

    ax2.set_title(f'Late-Stage Performance (Last 2000 Episodes)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/training_curves_prob{prob}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_robustness_comparison(log_dir, probs, seeds, output_dir='plots'):
    """
    Plot robustness comparison across different deception probabilities

    Bar chart: Final performance (avg reward of last 500 episodes) vs deception prob
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    algos = ['q_learning', 'mappo']
    labels = {'q_learning': 'Q-Learning', 'mappo': 'MAPPO'}
    colors = {'q_learning': 'skyblue', 'mappo': 'salmon'}

    results = {algo: {'means': [], 'stds': []} for algo in algos}

    for prob in probs:
        for algo in algos:
            episodes, mean_rewards, std_rewards, all_data = load_and_aggregate_data(log_dir, algo, prob, seeds)

            if episodes is None:
                results[algo]['means'].append(np.nan)
                results[algo]['stds'].append(np.nan)
                continue

            # Compute final performance (last 500 episodes)
            last_500 = mean_rewards[-500:]
            final_mean = np.mean(last_500)
            final_std = np.std(last_500)

            results[algo]['means'].append(final_mean)
            results[algo]['stds'].append(final_std)

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(probs))
    width = 0.35

    for i, algo in enumerate(algos):
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            results[algo]['means'],
            width,
            label=labels[algo],
            color=colors[algo],
            yerr=results[algo]['stds'],
            capsize=5,
            alpha=0.8
        )

    ax.set_xlabel('Deception Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Avg Reward (Last 500 Episodes)', fontsize=12, fontweight='bold')
    ax.set_title('Robustness Comparison: Performance vs Deception Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}' for p in probs])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = f"{output_dir}/robustness_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def compute_statistics(log_dir, algos, probs, seeds, output_dir='plots'):
    """
    Compute and save statistical summary

    Metrics:
    - Final performance (last 500 episodes mean)
    - Convergence speed (first episode to reach threshold)
    - Stability (std of last 1000 episodes)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats = []

    for algo in algos:
        for prob in probs:
            episodes, mean_rewards, std_rewards, all_data = load_and_aggregate_data(log_dir, algo, prob, seeds)

            if episodes is None:
                continue

            # Final performance
            final_perf = np.mean(mean_rewards[-500:])

            # Stability (variance in last 1000 episodes)
            stability = np.var(mean_rewards[-1000:])

            # Convergence speed (first episode to reach -10 reward)
            threshold = -10
            convergence_ep = None
            smoothed = smooth_curve(mean_rewards, window=100)
            for i, r in enumerate(smoothed):
                if r >= threshold:
                    convergence_ep = episodes[i]
                    break

            stats.append({
                'Algorithm': algo.upper(),
                'Deception Prob': prob,
                'Final Performance': f'{final_perf:.2f}',
                'Stability (Var)': f'{stability:.2f}',
                'Convergence Episode': convergence_ep if convergence_ep else 'N/A'
            })

    # Create DataFrame
    df = pd.DataFrame(stats)

    # Save to CSV
    csv_path = f"{output_dir}/statistics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")

    # Print to console
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("MARL EXPERIMENT RESULTS ANALYSIS")
    print("="*70 + "\n")

    # Configuration
    LOG_DIR = './logs'
    OUTPUT_DIR = './plots'
    SEEDS = [42, 100, 2024]
    PROBS = [0.0, 0.5, 0.8]
    ALGOS = ['q_learning', 'mappo']

    # Check if log directory exists
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' not found!")
        print("Please run experiments first using: bash run_experiments.sh")
        return

    # Count available log files
    log_files = list(Path(LOG_DIR).glob('*.csv'))
    print(f"Found {len(log_files)} log files in '{LOG_DIR}'\n")

    if len(log_files) == 0:
        print("Error: No CSV log files found!")
        print("Please run experiments first using: bash run_experiments.sh")
        return

    # Generate plots
    print("Generating plots...\n")

    # Plot 1: Training curves for each deception probability
    for prob in PROBS:
        print(f"Processing Prob={prob}...")
        plot_training_curves(LOG_DIR, prob, SEEDS, OUTPUT_DIR)

    # Plot 2: Robustness comparison
    print("Generating robustness comparison...")
    plot_robustness_comparison(LOG_DIR, PROBS, SEEDS, OUTPUT_DIR)

    # Statistics summary
    print("Computing statistics...")
    compute_statistics(LOG_DIR, ALGOS, PROBS, SEEDS, OUTPUT_DIR)

    print("\n" + "="*70)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*70)
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(Path(OUTPUT_DIR).glob('*')):
        print(f"  - {f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
