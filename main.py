"""
Main training script for MARL experiments
Supports both Q-Learning and MAPPO algorithms
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import numpy as np
import torch
import random
from pathlib import Path

from pettingzoo.mpe import simple_speaker_listener_v4
from envs.deceptive_wrapper import DeceptiveSpeakerEnv
from algorithm.q_learning import QLearningAgent
from algorithm.mappo import MAPPOAgent
from utils.logger import Logger

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not found. GIF generation will be disabled.")


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_episode_as_gif(env, agents, filepath, max_steps=25, deception_prob=0.5):
    """
    Run one episode and save as GIF

    Args:
        env: Base environment (not wrapped)
        agents: Dictionary of trained agents
        filepath: Path to save GIF
        max_steps: Maximum steps per episode
        deception_prob: Deception probability for the wrapper
    """
    if not HAS_IMAGEIO:
        print("Warning: Cannot save GIF, imageio not installed")
        return

    # Create a new env with render_mode='rgb_array'
    gif_env = simple_speaker_listener_v4.parallel_env(
        render_mode='rgb_array',
        continuous_actions=False
    )

    # Wrap with deceptive wrapper
    from envs.deceptive_wrapper import DeceptiveSpeakerEnv
    gif_env = DeceptiveSpeakerEnv(
        render_mode='rgb_array',
        continuous_actions=False,
        deception_prob=deception_prob
    )

    observations, infos = gif_env.reset()
    frames = []

    for step in range(max_steps):
        # Render current frame
        try:
            frame = gif_env.env.render()
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"Warning: Failed to render frame: {e}")
            break

        # Select actions (deterministic, no exploration)
        actions = {}
        for agent_id in gif_env.possible_agents:
            obs = observations[agent_id]
            action = agents[agent_id].select_action(obs, is_training=False)
            actions[agent_id] = action

        # Step environment
        observations, rewards, terminations, truncations, infos = gif_env.step(actions)

        if any(terminations.values()) or any(truncations.values()):
            break

    # Save GIF
    if len(frames) > 0:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            imageio.mimsave(filepath, frames, fps=5)
            print(f"  ✅ GIF saved: {filepath} ({len(frames)} frames)")
        except Exception as e:
            print(f"Warning: Failed to save GIF: {e}")
    else:
        print("Warning: No frames captured for GIF")

    gif_env.close()


def evaluate(env, agents, num_episodes=5, max_steps=25):
    """
    Evaluate agents without exploration

    Returns:
        avg_reward: Average reward over evaluation episodes
    """
    total_rewards = []

    for ep in range(num_episodes):
        observations, infos = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            actions = {}
            for agent_id in env.possible_agents:
                # Deterministic action selection (no exploration)
                action = agents[agent_id].select_action(observations[agent_id], is_training=False)
                actions[agent_id] = action

            observations, rewards, terminations, truncations, infos = env.step(actions)
            episode_reward += rewards['listener_0']

            if any(terminations.values()) or any(truncations.values()):
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def train_q_learning(args):
    """Training loop for Q-Learning"""
    print(f"\n{'='*60}")
    print(f"Training Q-Learning")
    print(f"Deception Prob: {args.prob}, Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Set seed
    set_seed(args.seed)

    # Initialize environment
    env = DeceptiveSpeakerEnv(
        render_mode=None,
        continuous_actions=False,
        deception_prob=args.prob
    )

    # Initialize agents
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    agents = {}
    for agent_id in env.possible_agents:
        obs_dim = env.observation_space(agent_id).shape[0]
        action_dim = env.action_space(agent_id).n
        agents[agent_id] = QLearningAgent(
            state_dim=obs_dim,
            action_dim=action_dim,
            epsilon_decay=0.9995,
            device=device
        )

    # Initialize logger
    log_filename = f"q_learning_prob{args.prob}_seed{args.seed}.csv"
    logger = Logger(save_dir=args.log_dir, filename=log_filename)

    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for episode in range(args.total_episodes):
        observations, infos = env.reset()
        episode_reward = 0

        for step in range(args.max_steps):
            # Select actions
            actions = {}
            for agent_id in env.possible_agents:
                action = agents[agent_id].select_action(observations[agent_id], is_training=True)
                actions[agent_id] = action

            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Store and update
            for agent_id in env.possible_agents:
                agents[agent_id].store_transition(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    terminations[agent_id] or truncations[agent_id]
                )
                agents[agent_id].update(batch_size=args.batch_size)

            episode_reward += rewards['listener_0']
            observations = next_observations

            if any(terminations.values()) or any(truncations.values()):
                break

        # Log training progress
        current_epsilon = agents['listener_0'].epsilon
        logger.log(episode, episode_reward, current_epsilon)

        # Evaluation and GIF generation
        if (episode + 1) % args.eval_freq == 0:
            avg_reward = evaluate(env, agents, num_episodes=5, max_steps=args.max_steps)
            print(f"[Episode {episode + 1}] Eval Reward: {avg_reward:.2f}")

            # Save GIF at key checkpoints
            if args.save_gif and (episode + 1) in [args.eval_freq, args.total_episodes // 2, args.total_episodes]:
                gif_path = f"{args.gif_dir}/q_learning_prob{args.prob}_seed{args.seed}_ep{episode+1}.gif"
                save_episode_as_gif(env.env, agents, gif_path, args.max_steps, args.prob)

            # Save model checkpoint
            checkpoint_path = checkpoint_dir / f"q_learning_prob{args.prob}_seed{args.seed}_ep{episode+1}.pth"
            for agent_id, agent in agents.items():
                agent.save_model(str(checkpoint_path).replace('.pth', f'_{agent_id}.pth'))

    env.close()
    print("\n✅ Q-Learning training finished!\n")


def train_mappo(args):
    """Training loop for MAPPO"""
    print(f"\n{'='*60}")
    print(f"Training MAPPO")
    print(f"Deception Prob: {args.prob}, Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Set seed
    set_seed(args.seed)

    # Initialize environment
    env = DeceptiveSpeakerEnv(
        render_mode=None,
        continuous_actions=False,
        deception_prob=args.prob
    )

    # Initialize agents
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    agents = {}
    for agent_id in env.possible_agents:
        obs_dim = env.observation_space(agent_id).shape[0]
        action_dim = env.action_space(agent_id).n
        agents[agent_id] = MAPPOAgent(
            state_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )

    # Initialize logger (MAPPO doesn't have epsilon, use 0.0)
    log_filename = f"mappo_prob{args.prob}_seed{args.seed}.csv"
    logger = Logger(save_dir=args.log_dir, filename=log_filename)

    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for episode in range(args.total_episodes):
        observations, infos = env.reset()
        episode_reward = 0

        # Collect one episode of experience
        for step in range(args.max_steps):
            actions = {}
            for agent_id in env.possible_agents:
                action = agents[agent_id].select_action(observations[agent_id], is_training=True)
                actions[agent_id] = action

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Store transitions
            for agent_id in env.possible_agents:
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

        # Update agents after episode
        for agent_id in env.possible_agents:
            agents[agent_id].update()

        # Log (use 0.0 for epsilon since MAPPO doesn't have it)
        logger.log(episode, episode_reward, 0.0)

        # Evaluation and GIF generation
        if (episode + 1) % args.eval_freq == 0:
            avg_reward = evaluate(env, agents, num_episodes=5, max_steps=args.max_steps)
            print(f"[Episode {episode + 1}] Eval Reward: {avg_reward:.2f}")

            # Save GIF at key checkpoints
            if args.save_gif and (episode + 1) in [args.eval_freq, args.total_episodes // 2, args.total_episodes]:
                gif_path = f"{args.gif_dir}/mappo_prob{args.prob}_seed{args.seed}_ep{episode+1}.gif"
                save_episode_as_gif(env.env, agents, gif_path, args.max_steps, args.prob)

            # Save model checkpoint
            checkpoint_path = checkpoint_dir / f"mappo_prob{args.prob}_seed{args.seed}_ep{episode+1}.pth"
            for agent_id, agent in agents.items():
                agent.save_model(str(checkpoint_path).replace('.pth', f'_{agent_id}.pth'))

    env.close()
    print("\n✅ MAPPO training finished!\n")


def main():
    parser = argparse.ArgumentParser(description='MARL Training with Q-Learning or MAPPO')

    # Algorithm and environment
    parser.add_argument('--algo', type=str, required=True, choices=['q_learning', 'mappo'],
                        help='Algorithm to use')
    parser.add_argument('--prob', type=float, required=True,
                        help='Deception probability (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Training parameters
    parser.add_argument('--total_episodes', type=int, default=5000,
                        help='Total training episodes')
    parser.add_argument('--max_steps', type=int, default=25,
                        help='Max steps per episode')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for Q-Learning')

    # Evaluation and logging
    parser.add_argument('--eval_freq', type=int, default=200,
                        help='Evaluation frequency (episodes)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for model checkpoints')

    # GIF generation
    parser.add_argument('--save_gif', action='store_true',
                        help='Save GIFs during evaluation')
    parser.add_argument('--gif_dir', type=str, default='./gifs',
                        help='Directory for GIFs')

    args = parser.parse_args()

    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.save_gif:
        Path(args.gif_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    if args.algo == 'q_learning':
        train_q_learning(args)
    elif args.algo == 'mappo':
        train_mappo(args)


if __name__ == "__main__":
    main()
