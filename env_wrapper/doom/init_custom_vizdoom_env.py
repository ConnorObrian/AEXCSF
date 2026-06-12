import vizdoom as vzd
from vizdoom import gymnasium_wrapper
import gymnasium
from PIL import Image
import os
from gymnasium.envs.registration import register

register(
    id="VizdoomBasicCustom-v0",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"config_file": os.path.realpath("./env_wrapper/doom/custombasic.cfg")},
)
