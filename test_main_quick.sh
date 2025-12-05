#!/bin/bash
# Quick test script for main.py
# Tests both algorithms with minimal episodes

echo "=================================="
echo "Quick Test: main.py"
echo "=================================="

echo ""
echo "Test 1: Q-Learning (100 episodes, no GIF)"
python main.py --algo q_learning --prob 0.5 --seed 42 --total_episodes 100 --eval_freq 50

echo ""
echo "Test 2: MAPPO (100 episodes, no GIF)"
python main.py --algo mappo --prob 0.5 --seed 42 --total_episodes 100 --eval_freq 50

echo ""
echo "=================================="
echo "Checking outputs..."
echo "=================================="

echo ""
echo "Log files created:"
ls -lh logs/

echo ""
echo "✅ Quick test completed!"
echo "If you see CSV files in logs/, the test passed."
