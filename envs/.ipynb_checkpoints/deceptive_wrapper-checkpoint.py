import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4

class DeceptiveSpeakerEnv:
    """
    This is a Wrapper for the PettingZoo simple_speaker_listener environment 
    that makes the Speaker lies with a certain probability
    generating a Deceptive Environment
    """
    def __init__(self, render_mode=None, continuous_actions=False, deception_prob=1.0):
        """
        Initialize the Deceptive Environment
        :param render_mode: 'human' or None
        :param continuous_actions: Using continuous action space if True, or discrete if False
        :param deception_prob: probability of lying (between 0 and 1)
        """
        # load the original environment
        # using parallel_env for multiagent training
        self.env = simple_speaker_listener_v4.parallel_env(
            render_mode = render_mode,
            continuous_actions = continuous_actions
        )

        self.render_mode = render_mode
        self.deception_prob = deception_prob
        
        # save the properties of the original environment
        self.reset()
        self.agents = self.env.agents
        self.possible_agents = self.env.possible_agents
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


    def reset(self, seed=None, options=None):
        """
        reset the environment, returns initial state
        """
        observations, infos = self.env.reset(seed=seed,options=options)
        return observations, infos


    def step(self, actions):
        """
        get the actions from the speaker and change it randomly with a certain probability
        """
        modified_actions = actions.copy()

        speaker_id = 'speaker_0'

        all_actions = [0, 1, 2]  # assuming 3 possible actions for the speaker

        if speaker_id in modified_actions:
            if np.random().rand() < self.deceeption_prob:
                action_before = modified_actions[speaker_id]
                if isinstance(action_before,(int, np.integer)):
                    wrong_actions = [a for a in all_actions if a != action_before]
                    modified_actions[speaker_id] = np.random.choice(wrong_actions)
        
        obseravtions, rewards, terminations, truncations, infos = self.env.step(modified_actions)

        return obseravtions, rewards, terminations, truncations, infos

    def close(self):
        self.env.close()

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)