"""
Unit tests for Q-Learning implementation
Run: python test_qlearning.py
"""
import torch
import numpy as np
from algorithm.q_learning import QLearningAgent


def test_save_load():
    """Test save and load functionality"""
    print("=" * 50)
    print("Test: Q-Learning Save and Load")
    print("=" * 50)

    state_dim = 11
    action_dim = 4
    agent1 = QLearningAgent(state_dim, action_dim, epsilon=0.5)

    # Save model
    agent1.save_model('/tmp/test_qlearning.pth')
    print("✅ Model saved to /tmp/test_qlearning.pth")

    # Get original output
    test_state = torch.randn(1, state_dim)
    with torch.no_grad():
        original_q = agent1.q_net(test_state)
    original_epsilon = agent1.epsilon

    # Load into new agent
    agent2 = QLearningAgent(state_dim, action_dim)
    agent2.load_model('/tmp/test_qlearning.pth')
    print("✅ Model loaded into new agent")

    # Check outputs match
    with torch.no_grad():
        loaded_q = agent2.q_net(test_state)

    assert torch.allclose(original_q, loaded_q), "Q-values don't match!"
    assert agent2.epsilon == original_epsilon, f"Epsilon mismatch: {agent2.epsilon} vs {original_epsilon}"
    print(f"✅ Q-values match")
    print(f"✅ Epsilon preserved: {agent2.epsilon}")
    print("✅ Save/load test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Q-LEARNING SAVE/LOAD TEST")
    print("=" * 50 + "\n")

    try:
        test_save_load()

        print("=" * 50)
        print("🎉 TEST PASSED! 🎉")
        print("=" * 50)

    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ TEST FAILED!")
        print("=" * 50)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
