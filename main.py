import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
from envs.deceptive_wrapper import DeceptiveSpeakerEnv
from algorithm.q_learning import QLearningAgent
from utils.logger import Logger 

def train():
    # training hyperparameters
    MAX_EPISODES = 1000
    MAX_STEPS = 25
    BATCH_SIZE = 64
    DECEPTION_PROB = 0.5  # 50% deception probability

    # 2. initialize the deceptive environment
    env = DeceptiveSpeakerEnv(render_mode=None, continuous_actions=False, deception_prob=DECEPTION_PROB)
    
    # get agents list
    # simple_speaker_listener has two agents: 'speaker_0', 'listener_0'
    agents_list = env.possible_agents
    print(f"Agents: {agents_list}")

    # 3. initialize the logger
    logger = Logger(save_dir="./logs", filename="q_learning_trust.csv")

    # 4. initialize Q-learning agents for each agent in the environment
    agents = {}
    for agent_id in agents_list:
        obs_dim = env.observation_space(agent_id).shape[0]
        action_dim = env.action_space(agent_id).n
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agents[agent_id] = QLearningAgent(state_dim=obs_dim, action_dim=action_dim, device=device)

    print(f"Start Training on {device}...")
    
    # 5. start training loop
    for episode in range(MAX_EPISODES):
        # reset environment at the start of each episode
        observations, infos = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            # Select Action
            actions = {}
            for agent_id in agents_list:
                obs = observations[agent_id]
                action = agents[agent_id].select_action(obs, is_training=True)
                actions[agent_id] = action

            # environment step
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # store transitions and train
            for agent_id in agents_list:
                # store the transition
                agents[agent_id].store_transition(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    terminations[agent_id] or truncations[agent_id]
                )
                
                # train the agent
                agents[agent_id].update(batch_size=BATCH_SIZE)
                
                # sum up rewards
                if agent_id == 'listener_0':
                    episode_reward += rewards[agent_id]

            # update observations
            observations = next_observations

            # check for termination
            if any(terminations.values()) or any(truncations.values()):
                break

        # use logger to log episode results
        current_epsilon = agents['listener_0'].epsilon
        logger.log(episode, episode_reward, current_epsilon)

    # 5. close the environment
    env.close()
    print("Training Finished!")

if __name__ == "__main__":
    train()