import vizdoom as vzd
import gymnasium
import numpy as np
import os

ENV_NAME = "CustomVizdoomBasic-v0"

"""
VizdoomBasic-v0 environment: 

actions: 0-Left, 1-Right, 3-Attack

observation: RGB image of the screen

reward: -1 for each step, 100 for each kill
"""

# renaming the functions to fit the gym interface
class VizdoomBasicEnv(gymnasium.Env):
    metadata = {'render_modes': ["rgb_array"]}
    def __init__(self, map : str = "map01", frame_skip : int = 4, max_episode_steps : int = 500, hide_ui : bool = True,
                 hide_gun : bool = False, **kwargs):
        self.frame_skip = frame_skip
        game = vzd.DoomGame()
        # Configurations
        game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
        game.set_doom_map(map)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_depth_buffer_enabled(False)
        game.set_labels_buffer_enabled(False)
        game.set_automap_buffer_enabled(False)
        game.set_objects_info_enabled(False)
        game.set_sectors_info_enabled(False)
        game.set_render_hud(not hide_ui)
        game.set_render_minimal_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(not hide_gun)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)
        game.set_render_messages(False)
        game.set_render_corpses(False)
        game.set_render_screen_flashes(False)
        game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
        game.set_available_game_variables([vzd.GameVariable.AMMO2])
        game.set_episode_timeout(max_episode_steps * frame_skip)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_living_reward(-1)
        game.set_mode(vzd.Mode.PLAYER)
        game.init()
        self.env = game
        # actions should look like this: [0, 0, 1] for attack, [1, 0, 0] for left, [0, 1, 0] for right
        self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.action_space = gymnasium.spaces.Discrete(len(self.actions))
        # observation space should be the shape of the screen buffer
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(self.env.get_screen_height(), self.env.get_screen_width(), 3), dtype=np.uint8)
    
    def reset(self, **kwargs):
        self.env.new_episode()
        state = self.env.get_state()
        return np.array(state.screen_buffer), {}
    
    def step(self, action):
        reward = 0
        state = self.env.get_state()
        for _ in range(self.frame_skip):
            if not self.env.is_episode_finished():
                reward += self.env.make_action(self.actions[action])
        done = self.env.is_episode_finished()
        state = self.env.get_state() if not done else state
        return np.array(state.screen_buffer), reward, done, False, {}
    
    def render(self, **kwargs):
        return self.env.get_state().screen_buffer, {}
    
    def max_payoff(self):
        return 100
    
    def close(self):
        self.env.close()

# registering the environment
gymnasium.register(ENV_NAME, VizdoomBasicEnv)