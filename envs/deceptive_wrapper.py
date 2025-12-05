import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4

class DeceptiveSpeakerEnv:
    """
    This is a Wrapper for the PettingZoo simple_speaker_listener environment 
    that makes the Speaker lies with a certain probability
    generating a Deceptive Environment
    """
    def __init__(self, render_mode=None, continuous_actions=False, deception_prob=0.5):
        """
        Initialize the Deceptive Environment
        :param render_mode: 'human' or None
        :param continuous_actions: Using continuous action space if True, or discrete if False
        :param deception_prob: probability of lying (between 0 and 1)
        """
        # load the original environment
        # using parallel_env for multiagent training
        assert continuous_actions is False, "DeceptiveWrapper currently only supports discrete actions."

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

        if speaker_id in modified_actions:
            if np.random.rand() < self.deception_prob:
                action_before = modified_actions[speaker_id]

                n_actions = self.action_space(speaker_id).n

                if isinstance(action_before,(int, np.integer)):
                    wrong_actions = [a for a in range(n_actions) if a != action_before]
                    modified_actions[speaker_id] = np.random.choice(wrong_actions)
        
        observations, rewards, terminations, truncations, infos = self.env.step(modified_actions)

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)