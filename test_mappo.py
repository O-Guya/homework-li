"""
Unit tests for MAPPO implementation
Run: python test_mappo.py
"""
import torch
import numpy as np
from algorithm.mappo import MAPPOAgent, ActorNetwork, CriticNetwork


def test_network_initialization():
    """Test 1: Network initialization"""
    print("=" * 50)
    print("Test 1: Network Initialization")
    print("=" * 50)

    state_dim = 11
    action_dim = 4

    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)

    # Test actor output shape
    dummy_state = torch.randn(1, state_dim)
    logits = actor(dummy_state)
    assert logits.shape == (1, action_dim), f"Expected shape (1, {action_dim}), got {logits.shape}"
    print(f"✅ Actor output shape: {logits.shape}")

    # Test critic output shape
    value = critic(dummy_state)
    assert value.shape == (1, 1), f"Expected shape (1, 1), got {value.shape}"
    print(f"✅ Critic output shape: {value.shape}")

    # Test action sampling
    action, log_prob = actor.get_action_and_log_prob(dummy_state)
    assert 0 <= action.item() < action_dim, f"Action {action.item()} out of range [0, {action_dim})"
    assert log_prob.shape == (1,), f"Log prob shape mismatch: {log_prob.shape}"
    print(f"✅ Sampled action: {action.item()}, log_prob: {log_prob.item():.4f}")

    print("✅ Network initialization test passed!\n")


def test_agent_initialization():
    """Test 2: Agent initialization"""
    print("=" * 50)
    print("Test 2: Agent Initialization")
    print("=" * 50)

    state_dim = 11
    action_dim = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = MAPPOAgent(state_dim, action_dim, device=device)

    print(f"✅ Device: {device}")
    print(f"✅ Actor parameters: {sum(p.numel() for p in agent.actor.parameters())}")
    print(f"✅ Critic parameters: {sum(p.numel() for p in agent.critic.parameters())}")
    print(f"✅ Agent initialized successfully!\n")

    return agent


def test_action_selection(agent):
    """Test 3: Action selection"""
    print("=" * 50)
    print("Test 3: Action Selection")
    print("=" * 50)

    state = np.random.randn(11)

    # Test training mode (stochastic)
    actions_train = [agent.select_action(state, is_training=True) for _ in range(10)]
    print(f"✅ Training mode actions (should vary): {actions_train}")

    # Test eval mode (deterministic)
    actions_eval = [agent.select_action(state, is_training=False) for _ in range(10)]
    print(f"✅ Eval mode actions (should be same): {actions_eval}")
    assert len(set(actions_eval)) == 1, "Eval mode should be deterministic!"

    print("✅ Action selection test passed!\n")


def test_store_and_buffer():
    """Test 4: Transition storage"""
    print("=" * 50)
    print("Test 4: Transition Storage")
    print("=" * 50)

    state_dim = 11
    action_dim = 4
    agent = MAPPOAgent(state_dim, action_dim)

    # Store some transitions
    for i in range(5):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = (i == 4)  # Last transition is terminal

        agent.store_transition(state, action, reward, next_state, done)

    assert len(agent.states) == 5, "Should have 5 transitions"
    assert len(agent.rewards) == 5
    assert agent.dones[-1] == True, "Last transition should be terminal"

    print(f"✅ Stored {len(agent.states)} transitions")
    print(f"✅ Rewards: {[f'{r:.2f}' for r in agent.rewards]}")
    print(f"✅ Dones: {agent.dones}")
    print("✅ Transition storage test passed!\n")

    return agent


def test_gae_computation(agent):
    """Test 5: GAE computation"""
    print("=" * 50)
    print("Test 5: GAE Computation")
    print("=" * 50)

    advantages, returns = agent.compute_gae(next_value=0.0)

    assert len(advantages) == len(agent.rewards), "Advantages length mismatch"
    assert len(returns) == len(agent.rewards), "Returns length mismatch"

    print(f"✅ Advantages shape: {advantages.shape}")
    print(f"✅ Returns shape: {returns.shape}")
    print(f"✅ Sample advantages: {advantages[:3]}")
    print(f"✅ Sample returns: {returns[:3]}")
    print("✅ GAE computation test passed!\n")


def test_update():
    """Test 6: Policy update"""
    print("=" * 50)
    print("Test 6: Policy Update")
    print("=" * 50)

    state_dim = 11
    action_dim = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = MAPPOAgent(state_dim, action_dim, device=device)

    # Collect a short episode
    for i in range(10):
        state = np.random.randn(state_dim)
        action = agent.select_action(state, is_training=True)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = (i == 9)

        agent.store_transition(state, action, reward, next_state, done)

    # Update
    losses = agent.update()

    assert losses is not None, "Update should return loss dict"
    assert 'actor_loss' in losses
    assert 'critic_loss' in losses
    assert 'entropy' in losses

    print(f"✅ Actor loss: {losses['actor_loss']:.4f}")
    print(f"✅ Critic loss: {losses['critic_loss']:.4f}")
    print(f"✅ Entropy: {losses['entropy']:.4f}")

    # Buffer should be cleared after update
    assert len(agent.states) == 0, "Buffer should be cleared after update"
    print("✅ Buffer cleared after update")
    print("✅ Policy update test passed!\n")


def test_save_load():
    """Test 7: Save and load model"""
    print("=" * 50)
    print("Test 7: Save and Load Model")
    print("=" * 50)

    state_dim = 11
    action_dim = 4
    agent1 = MAPPOAgent(state_dim, action_dim)

    # Save model
    agent1.save_model('/tmp/test_mappo.pth')
    print("✅ Model saved to /tmp/test_mappo.pth")

    # Get original output
    test_state = torch.randn(1, state_dim)
    with torch.no_grad():
        original_logits = agent1.actor(test_state)
        original_value = agent1.critic(test_state)

    # Load into new agent
    agent2 = MAPPOAgent(state_dim, action_dim)
    agent2.load_model('/tmp/test_mappo.pth')
    print("✅ Model loaded into new agent")

    # Check outputs match
    with torch.no_grad():
        loaded_logits = agent2.actor(test_state)
        loaded_value = agent2.critic(test_state)

    assert torch.allclose(original_logits, loaded_logits), "Actor outputs don't match!"
    assert torch.allclose(original_value, loaded_value), "Critic outputs don't match!"
    print("✅ Loaded model produces identical outputs")
    print("✅ Save/load test passed!\n")


def test_full_episode_simulation():
    """Test 8: Full episode simulation"""
    print("=" * 50)
    print("Test 8: Full Episode Simulation")
    print("=" * 50)

    from envs.deceptive_wrapper import DeceptiveSpeakerEnv

    # Initialize environment
    env = DeceptiveSpeakerEnv(render_mode=None, continuous_actions=False, deception_prob=0.5)
    agents_list = env.possible_agents

    # Initialize MAPPO agents
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agents = {}
    for agent_id in agents_list:
        obs_dim = env.observation_space(agent_id).shape[0]
        action_dim = env.action_space(agent_id).n
        agents[agent_id] = MAPPOAgent(obs_dim, action_dim, device=device)

    print(f"✅ Environment: simple_speaker_listener_v4")
    print(f"✅ Agents: {agents_list}")

    # Run one episode
    observations, infos = env.reset()
    episode_reward = 0

    for step in range(25):
        actions = {}
        for agent_id in agents_list:
            action = agents[agent_id].select_action(observations[agent_id], is_training=True)
            actions[agent_id] = action

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store transitions
        for agent_id in agents_list:
            agents[agent_id].store_transition(
                observations[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_observations[agent_id],
                terminations[agent_id] or truncations[agent_id]
            )

        episode_reward += rewards['listener_0']
        observations = next_observations

        if any(terminations.values()) or any(truncations.values()):
            break

    print(f"✅ Episode completed in {step + 1} steps")
    print(f"✅ Total reward: {episode_reward:.2f}")

    # Update agents
    for agent_id in agents_list:
        losses = agents[agent_id].update()
        if losses:
            print(f"✅ {agent_id} updated - Actor loss: {losses['actor_loss']:.4f}")

    env.close()
    print("✅ Full episode simulation test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("MAPPO IMPLEMENTATION UNIT TESTS")
    print("=" * 50 + "\n")

    try:
        # Run all tests
        test_network_initialization()
        agent = test_agent_initialization()
        test_action_selection(agent)
        agent_with_data = test_store_and_buffer()
        test_gae_computation(agent_with_data)
        test_update()
        test_save_load()
        test_full_episode_simulation()

        print("=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 50)

    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ TEST FAILED!")
        print("=" * 50)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
