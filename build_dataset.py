import ale_py
import argparse
import ast
import gymnasium

import matplotlib.pyplot as plt
import os
import pandas as pd

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

from DatasetMaker import DatasetGenerator as dm
from env_wrapper.doom.vizdoom_wrap import VizdoomIntActionWrapper
from env_wrapper.doom import init_custom_vizdoom_env
from vizdoom import gymnasium_wrapper

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            try:
                my_dict[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

ENV_NAME = None
ENV_CONTINUOUS = False

# image parameter (changing these also changes the behavior of every image generated)
X_DIMS = 64
Y_DIMS = 64
N_CHANNELS = 1

# Dataset Params
DATASET_MODEL = "ppo" # ("random", "ppo")
DATASET_EPISODES = 2000
DATASET_PATH = "./Dataset"
MAX_STEPS = 500 # maximum steps a agent can take in one episode

ATARI_PREPROCESSING = False

state_function = None


# Argument parser
parser = argparse.ArgumentParser(description="Generate a dataset from a gym environment.")
parser.add_argument("--env_name", type=str, required=True, help="Name of the gym environment.")
parser.add_argument("--env_continuous", type=bool, default=ENV_CONTINUOUS, help="Whether the environment is continuous.")
parser.add_argument("--x_dims", type=int, default=X_DIMS, help="Width of the generated images.")
parser.add_argument("--y_dims", type=int, default=Y_DIMS, help="Height of the generated images.")
parser.add_argument("--n_channels", type=int, default=N_CHANNELS, help="Number of channels in the generated images.")
parser.add_argument("--dataset_model", type=str, default=DATASET_MODEL, help="Model used for dataset generation (random, ppo).")
parser.add_argument("--dataset_episodes", type=int, default=DATASET_EPISODES, help="Number of episodes to generate.")
parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to save the generated dataset.")
parser.add_argument("--dataset_name", type=str, default="", required=False, help="Name of the dataset for its own directory.")
parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Maximum steps an agent can take in one episode.")
parser.add_argument("--atari_preprocessing", action="store_true", default=False, help="Whether to use the Atari preprocessing for the environment.")
parser.add_argument("--use_render_as_state", action="store_true", default=False, help="Whether to use the render as state function.")
parser.add_argument("--legacy_preprocessing", action="store_true", default=False, help="Use the legacy dataset preprocessing pipeline for reproducibility.")
parser.add_argument("--env_flags", dest="env_flags", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help="Environment flags to pass to the environment.")
args = parser.parse_args()

# Override variables with parsed arguments
ENV_NAME = args.env_name
ENV_CONTINUOUS = args.env_continuous
X_DIMS = args.x_dims
Y_DIMS = args.y_dims
N_CHANNELS = args.n_channels
DATASET_MODEL = args.dataset_model
DATASET_EPISODES = args.dataset_episodes
DATASET_PATH = args.dataset_path
DATASET_NAME = args.dataset_name
MAX_STEPS = args.max_steps
ENV_ARGS = {"render_mode" : "rgb_array", "max_episode_steps" if not ENV_NAME.upper().startswith("MINIGRID") else "max_steps" : MAX_STEPS}
# Add environment flags to the environment arguments
if args.env_flags is not None:
    ENV_ARGS.update(args.env_flags)
    
print(f"Environment Flags: {ENV_ARGS}")

if args.use_render_as_state:
    def state_function(env, state):
        return env.render()

# Create the environment
env = gymnasium.make(ENV_NAME, **ENV_ARGS)
if ENV_NAME.upper().startswith("MINIGRID"):
    print("Wrapping MiniGrid environment")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
if args.atari_preprocessing:
    print("Wrapping Atari environment")
    env = gymnasium.wrappers.AtariPreprocessing(env)
if ENV_NAME.upper().startswith("VIZDOOM"):
    print("Wrapping VizDoom environment")
    env = VizdoomIntActionWrapper(env)
    def state_function(env, state):
        return state["screen"]


imageGenerator = dm.DatasetGenerator(env=env, method=DATASET_MODEL, image_size=(X_DIMS, Y_DIMS), image_channel=N_CHANNELS, is_continous=ENV_CONTINUOUS, state_function=state_function, legacy_preprocessing=args.legacy_preprocessing) # creates an ppo agent that uses the env to generate images

# generate a dataset of n episodes
dataset_path = imageGenerator.generate_dataset(DATASET_EPISODES, max_steps=MAX_STEPS, save_path=DATASET_PATH, dataset_name=DATASET_NAME)


# reads csv and generate a diagram of mean rewards
parser = pd.read_csv(os.path.join(dataset_path, "training_metrics_" + DATASET_MODEL + ".csv"))
#calc the mean reward over 100 episodes
mean_rewards = parser["episode_reward"].rolling(window=100).mean()
plt.plot(mean_rewards)
plt.xlabel("Episodes")
plt.ylabel("Mean Reward")
plt.savefig(os.path.join(dataset_path, "mean_rewards.pdf"))
plt.clf()

print(f"Dataset generated at {dataset_path}")
print("Dataset generation finished.")
env.close()
