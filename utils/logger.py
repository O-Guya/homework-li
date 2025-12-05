import os
import csv
import numpy as np

class Logger:
    def __init__(self, save_dir="./logs", filename="training_log.csv"):
        """
        initialize the Logger
        :param save_dir: log directory
        :param filename: CSV filename to save logs
        """
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        
        # if the log directory does not exist, create it
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # initialize the CSV file
        # if the file does not exist, create it and write the header
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                # write the episode, reward, epsilon header
                writer.writerow(['Episode', 'Reward', 'Epsilon'])
        
        self.rewards = []

    def log(self, episode, reward, epsilon):
        """
        log the training information
        :param episode: current episode number
        :param reward: reward obtained in the episode
        :param epsilon: current epsilon value
        """
        self.rewards.append(reward)
        
        # 1. output to console
        if (episode + 1) % 100 == 0:
            # calculate average reward over last 100 episodes
            avg_reward = np.mean(self.rewards[-100:])
            print(f"Episode {episode + 1} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {epsilon:.4f}")

        # 2. write to CSV file
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, reward, epsilon])

    def get_avg_reward(self):
        """get average reward over all logged episodes"""
        return np.mean(self.rewards) if self.rewards else 0