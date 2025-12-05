#!/bin/bash

#######################################################################
# Batch Experiment Script for MARL Robustness Comparison
#######################################################################
# This script runs Q-Learning and MAPPO with:
# - 3 random seeds: 42, 100, 2024
# - 3 deception probabilities: 0.0, 0.5, 0.8
# - 5000 training episodes each
# Total: 18 experiments (2 algos × 3 probs × 3 seeds)
#######################################################################

echo "======================================================================="
echo "Starting Batch Experiments: Q-Learning vs MAPPO"
echo "======================================================================="
echo "Configuration:"
echo "  - Algorithms: Q-Learning, MAPPO"
echo "  - Deception Probabilities: 0.0, 0.5, 0.8"
echo "  - Random Seeds: 42, 100, 2024"
echo "  - Total Episodes per run: 5000"
echo "  - Evaluation Frequency: 200 episodes"
echo "======================================================================="

# Configuration
SEEDS=(42 100 2024)
PROBS=(0.0 0.5 0.8)
ALGOS=(q_learning mappo)
TOTAL_EPISODES=5000
EVAL_FREQ=200
SAVE_GIF_FLAG="--save_gif"  # Set to "" to disable GIF generation

# Create output directories
mkdir -p logs checkpoints gifs experiment_logs

# Log file for batch experiment
BATCH_LOG="experiment_logs/batch_run_$(date +%Y%m%d_%H%M%S).log"
echo "Batch experiment log: $BATCH_LOG"

# Counter
TOTAL_RUNS=$((${#ALGOS[@]} * ${#PROBS[@]} * ${#SEEDS[@]}))
CURRENT_RUN=0

# Function to run a single experiment
run_experiment() {
    local algo=$1
    local prob=$2
    local seed=$3

    CURRENT_RUN=$((CURRENT_RUN + 1))

    echo ""
    echo "======================================================================="
    echo "[$CURRENT_RUN / $TOTAL_RUNS] Running: $algo | Prob=$prob | Seed=$seed"
    echo "======================================================================="

    # Run training
    python main.py \
        --algo $algo \
        --prob $prob \
        --seed $seed \
        --total_episodes $TOTAL_EPISODES \
        --eval_freq $EVAL_FREQ \
        $SAVE_GIF_FLAG \
        2>&1 | tee -a $BATCH_LOG

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $algo | Prob=$prob | Seed=$seed"
    else
        echo "❌ FAILED: $algo | Prob=$prob | Seed=$seed"
        echo "Check log for details: $BATCH_LOG"
    fi

    # Clear GPU memory (if using CUDA)
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

    # Short pause between runs
    sleep 5
}

# Main experiment loop
START_TIME=$(date +%s)

for algo in "${ALGOS[@]}"; do
    for prob in "${PROBS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment $algo $prob $seed
        done
    done
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================================="
echo "🎉 ALL EXPERIMENTS COMPLETED! 🎉"
echo "======================================================================="
echo "Total runs: $TOTAL_RUNS"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "======================================================================="

# Summary of generated files
echo ""
echo "Generated Files:"
echo "----------------"
echo "Logs:"
ls -lh logs/*.csv | tail -n 10
echo ""
echo "Checkpoints:"
ls -lh checkpoints/*.pth | tail -n 10
echo ""
if [ -d "gifs" ] && [ "$(ls -A gifs)" ]; then
    echo "GIFs:"
    ls -lh gifs/*.gif | tail -n 10
fi

echo ""
echo "Next steps:"
echo "1. Run: python plot_results.py  (to generate comparison plots)"
echo "2. Check experiment log: $BATCH_LOG"
echo ""
