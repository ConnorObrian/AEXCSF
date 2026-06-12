from gymnasium import ActionWrapper
from gymnasium.spaces import Discrete
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv
from itertools import product

class VizdoomIntActionWrapper(ActionWrapper):
    def __init__(self, env, discrete_button_presses=True):
        super().__init__(env)
        #assert isinstance(env, VizdoomEnv), "VizdoomIntActionWrapper has to be used on top of a VizDoom environment"
        self.n_buttons = env.unwrapped.game.get_available_buttons_size()
        self.discrete_buttons = discrete_button_presses

        if discrete_button_presses:
            zeros = [0] * (self.n_buttons - 1)
            self.actual_actions = [zeros[:i] + [1] + zeros[i:] for i in range(self.n_buttons)]
            self.action_space = Discrete(self.n_buttons)
        else:
            self.actual_actions = [list(a) for a in product([0,1], repeat=self.n_buttons)][1:]
            self.action_space = Discrete(len(self.actual_actions))


    def action(self, act:int):
        return self.actual_actions[act]