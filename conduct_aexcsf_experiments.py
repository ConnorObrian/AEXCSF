import argparse
import ast
import logging
import os
import subprocess
import time

import ale_py
import dill
import evaluation_utils
import gymnasium
from gymnasium import wrappers
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from vizdoom import gymnasium_wrapper
from Autoencoder.autoencoder import ConvVariationalAutoEncoder
from env_wrapper.doom import init_custom_vizdoom_env

import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import pandas as pd
import torch
from DatasetMaker.DatasetParser import ScalerKind

from AXNex.Axn import Axnet, AEConfig, VAEConfig, VQVAEConfig
from env_wrapper.doom.vizdoom_wrap import VizdoomIntActionWrapper

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

def save_python_env_info(save_path):
    result = subprocess.run(["uv", "pip", "list"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return_val = result.stdout.decode("utf-8")
    with open(save_path + "/python_env_info.txt", "w") as fp:
        fp.write(str(return_val))

def setup_logging(save_path):
    logging.basicConfig(filename=save_path + "/experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.INFO)


def build_ae_config_from_args(autoencoder_type: str, args) -> AEConfig:
    scaler = ScalerKind(args.latent_scaler) if args.latent_scaler is not None else None
    scaler_kwargs = args.latent_scaler_kwargs or {}

    if autoencoder_type in ["vae", "cvae", "infovae", "dino"]:
        return VAEConfig(
            use_reparameterization=args.use_reparametrization,
            ignore_sigma=args.ignore_sigma,
            latent_scaler=scaler if scaler is not None else ScalerKind.MINMAX,
            latent_scaler_kwargs=scaler_kwargs,
        )

    if autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
        return VQVAEConfig(
            latent_state=args.vq_latent_state,
            latent_scaler=scaler if scaler is not None else (ScalerKind.CONSTANT if args.vq_latent_state=="indices" else ScalerKind.MINMAX),
            latent_scaler_kwargs=scaler_kwargs,
        )

    return AEConfig(
        latent_scaler=scaler if scaler is not None else ScalerKind.MINMAX,
        latent_scaler_kwargs=scaler_kwargs,
    )
    
def manage_seeds(passed_seeds, start_at_run, number_of_runs) -> list[int]:
    """Manages the seeds for an experimental series.
    
       In general: Passed seeds are preffered. If no seeds are passed, internal seeds are preferred. ==> Passed seeds override internal seeds.
       So: If internally internal_seeds == [0, 1, 2, 3] and passed seeds are [10, 11] --> used_seeds == [10, 11, 2, 3]
       If not enough seeds are set an exception is raised.
       So if we wanted 4 seeds in our example a resulting set of seeds could be:
       [10, 11, 2, 3]
       
       Returns:
           list[int]: The seeds that should be used for this experimental series
    """
    seeds = []
    if passed_seeds:
        logging.info(f"Following seeds were passed to the experiment: {passed_seeds}.")
        seeds = passed_seeds
    
    passed_seed_num = len(seeds)
    # start_at_run is usually 0 but if not it is interpreted as start at seed i and do as many runs as wanted from i
    num_seeds_required = start_at_run + number_of_runs
    seeds_missing = max(num_seeds_required - passed_seed_num, 0)
    
    # First fill up the number of needed seeds with internally set seeds
    if seeds_missing > 0 and len(INTERNAL_SEEDS) > passed_seed_num:
        seeds += INTERNAL_SEEDS[passed_seed_num:]
        seeds_missing -= len(INTERNAL_SEEDS)
    
    # If the series still requires more seeds then fill them up with random seeds as long as the option for this is set. Otherwise raise an error
    if seeds_missing > 0: 
            raise Exception(f"Not enough seeds were internally set or passed. The system still requires {seeds_missing} additional seeds. "
                             f"Passed seeds override internal seeds - they are not a prefix to them.")
    return seeds

def set_more_determinism(seed: int) -> None:
    """
    Function for increasing the chance of reproducibility.
    Sets pytorch seed and pytorch to deterministic if desired.
    Args:
        seed (int): the chosen or generated seed
    """
    if MORE_DETERMINISM:
        torch.use_deterministic_algorithms(True)

AUTOENCODER = "cvae" # ("ae, "vae", "cae", "mae", "cvae")
ENV_NAME = None
ENV_CONTINUOUS = False

INTERNAL_SEEDS: list[int] = []
MORE_DETERMINISM = False

# image parameter (changing these also changes the behavior of every image generated)
X_DIMS = 64
Y_DIMS = 64
N_CHANNELS = 1

# Dataset Params
AXN_EPISODES = 2000
MAX_STEPS = 1000 # maximum steps a agent can take in one episode

ATARI_PREPROCESSING = False

TEST_AUTOENCODER = False

state_function = None

METRIC_SAVE_DIR = "./Results/"
MODEL_SAVE_DIR = "./Results/"

#AEXCSF Params
number_threads = 8 #Number of threads. Defaults to 8.
random_seed = None #Random seed. Defaults to None.
population_file = "" #File to save the population. Defaults to "".
pop_init = False #If the population should get initialized. Defaults to False.
pop_size = 5000 #Population size. Defaults to 5000.
subsumption = True #If subsumption should get used. Defaults to True.
theta_sub = 100 #Subsumption threshold. Defaults to 100.
loss_function = "mae" #Loss function. Defaults to "mae" ("mae", "mse", "rmse", "log", "binary_log", "onehot", "huber").
huber_delta = 1 #Huber delta. Defaults to 1.
gamma = 0.93 #Gamma. Defaults to 0.93.
p_explore = "linear" #Exploration function (linear, exponential, sigmoid, constant). Defaults to "linear" ("linear", "exponential", "sigmoid", "constant").
p_max = 1 #Maximum exploration probability. Defaults to 1.
p_min = 0.1 #Minimum exploration probability. Defaults to 0.1.
e0 = 0.001 #E0. Defaults to 0.1.
alpha = 0.1 #Alpha. Defaults to 0.1.
nu = 5 #Nu. Defaults to 5.
beta = 0.15 #Beta. Defaults to 0.15.
delta = 0.15 #Delta. Defaults to 0.15.
theta_del = 20 #Theta del. Defaults to 20.
init_fitness = 0.01 #Initial fitness. Defaults to 0.01.
init_error = 0 #Initial error. Defaults to 0
m_probation = 10000 #M probation. Defaults to 10000.
stateful = True #If stateful. Defaults to True.
compaction = False #If compaction should get used. Defaults to False.
ea_select_size = 0.4 #Size of the selection. Defaults to 0.4.
ea_theta = 50 #Theta. Defaults to 50.
ea_lambda = 2 #Lambda. Defaults to 2.
ea_p_crossover = 0.8 #Crossover probability. Defaults to 0.8.
ea_fitness_reduction = 0.1 #Fitness reduction. Defaults to 0.5.
ea_error_reduction = 1.0 #Error reduction. Defaults to 0.5.
condition = "hyperrectangle_csr" #Condition type ("ternary", "hyperrectangle_csr", "hyperrectangle_ubr", "hyperellipsoid", "tree_gp", "dgp"). Defaults to "hyperrectangle_csr".
c_eta = 0.0 #Eta. Defaults to 0.0.
c_min = 0.0 #Min. Defaults to 0.0.
c_max = 1.0 #Max. Defaults to 1.0.
c_spread_min = 0.1 #Spread min. Defaults to 0.15.
prediction = "rls_linear" #Prediction type (constant, (rls, nlms), linear, quadratic). Defaults to "rls_linear".
ae_learning_rate = 0.001 #Learning rate of the autoencoder (if autoencoder should learn parallel). Defaults to 0.001.
ae_batch_size = 64 #Batch size of the autoencoder (if autoencoder should learn parallel). Defaults to 64.

# Argument parser
parser = argparse.ArgumentParser(description="Generate a dataset from a gym environment.")
parser.add_argument("--env_name", type=str, required=True, help="Name of the gym environment.")
parser.add_argument("--autoencoder_path", type=str, required=True, help="Path to the autoencoder model.")
parser.add_argument("--train_autoencoder", action="store_true", default=False, help="Whether to train the autoencoder while training the axn.")
parser.add_argument("--autoencoder", type=str, default=AUTOENCODER, help="Type of autoencoder to use (ae, vae, cae, mae, cvae)")
parser.add_argument("--test_autoencoder", action="store_true", default=TEST_AUTOENCODER, help="Whether to test the autoencoder with a DQN Agent.")
parser.add_argument("--metric_save_dir", type=str, default=METRIC_SAVE_DIR, help="Path to where the metrics are saved.")
parser.add_argument("--model_save_dir", type=str, default=MODEL_SAVE_DIR, help="Path to the model save directory.")
parser.add_argument("--axn_episodes", type=int, default=AXN_EPISODES, help="Number of episodes the axn agent should train.")
parser.add_argument("--env_continuous", action="store_true", default=ENV_CONTINUOUS, help="Whether the environment is continuous.")
parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Maximum steps an agent can take in one episode.")
parser.add_argument("--x_dims", type=int, default=X_DIMS, help="Width of the generated images.")
parser.add_argument("--y_dims", type=int, default=Y_DIMS, help="Height of the generated images.")
parser.add_argument("--n_channels", type=int, default=N_CHANNELS, help="Number of channels in the generated images.")
parser.add_argument("--atari_preprocessing", action="store_true", default=False, help="Whether to use the Atari preprocessing for the environment.")
parser.add_argument("--use_render_as_state", action="store_true", default=False, help="Whether to use the render as state function.")
parser.add_argument("--number_threads", type=int, default=number_threads, help="Number of threads. Defaults to 8.")
parser.add_argument("--random_seed", type=int, default=random_seed, help="Random seed")
parser.add_argument("--population_file", type=str, default=population_file, help="File to save the population. Defaults to ''.")
parser.add_argument("--pop_init", action="store_true", default=pop_init, help="If the population should get initialized. Defaults to False.")
parser.add_argument("--pop_size", type=int, default=pop_size, help="Population size. Defaults to 5000.")
parser.add_argument("--no-subsumption", action="store_true", default=False, help="If subsumption should get used. Defaults to True.")
parser.add_argument("--theta_sub", type=int, default=theta_sub, help="Subsumption threshold. Defaults to 100.")
parser.add_argument("--loss_function", type=str, default=loss_function, help="Loss function. Defaults to 'mae'.")
parser.add_argument("--huber_delta", type=float, default=huber_delta, help="Huber delta. Defaults to 1.")
parser.add_argument("--gamma", type=float, default=gamma, help="Gamma. Defaults to 0.93.")
parser.add_argument("--p_explore", type=str, default=p_explore, help="Exploration function (linear, exponential, sigmoid, constant). Defaults to 'linear'.")
parser.add_argument("--p_max", type=float, default=p_max, help="Maximum exploration probability. Defaults to 1.")
parser.add_argument("--p_min", type=float, default=p_min, help="Minimum exploration probability. Defaults to 0.1.")
parser.add_argument("--max_exploration_trials", type=int, default=-1, help="Maximum exploration trials. Defaults to max_trials - max_trials // 10.")
parser.add_argument("--e0", type=float, default=e0, help="E0. Defaults to 0.1.")
parser.add_argument("--alpha", type=float, default=alpha, help="Alpha. Defaults to 0.1.")
parser.add_argument("--nu", type=int, default=nu, help="Nu. Defaults to 5.")
parser.add_argument("--beta", type=float, default=beta, help="Beta. Defaults to 0.15.")
parser.add_argument("--delta", type=float, default=delta, help="Delta. Defaults to 0.15.")
parser.add_argument("--theta_del", type=int, default=theta_del, help="Theta del. Defaults to 20.")
parser.add_argument("--init_fitness", type=float, default=init_fitness, help="Initial fitness. Defaults to 0.01.")
parser.add_argument("--init_error", type=float, default=init_error, help="Initial error. Defaults to 0.")
parser.add_argument("--m_probation", type=int, default=m_probation, help="M probation. Defaults to 10000.")
parser.add_argument("--stateful", action="store_true", default=stateful, help="If stateful. Defaults to True.")
parser.add_argument("--compaction", action="store_true", default=compaction, help="If compaction should get used. Defaults to False.")
parser.add_argument("--ea_roulette", action="store_true", default=False)
parser.add_argument("--ea_select_size", type=float, default=ea_select_size, help="Size of the selection. Defaults to 0.4.")
parser.add_argument("--ea_theta", type=int, default=ea_theta, help="Theta. Defaults to 50.")
parser.add_argument("--ea_lambda", type=int, default=ea_lambda, help="Lambda. Defaults to 2.")
parser.add_argument("--ea_p_crossover", type=float, default=ea_p_crossover, help="Crossover probability. Defaults to 0.8.")
parser.add_argument("--ea_fitness_reduction", type=float, default=ea_fitness_reduction, help="Fitness reduction. Defaults to 0.5.")
parser.add_argument("--ea_error_reduction", type=float, default=ea_error_reduction, help="Error reduction. Defaults to 0.5.")
parser.add_argument("--condition", type=str, default=condition, help="Condition type ('ternary', 'hyperrectangle_csr', 'hyperrectangle_ubr', 'hyperellipsoid', 'tree_gp', 'dgp'). Defaults to 'hyperrectangle_csr'.")
parser.add_argument("--c_eta", type=float, default=c_eta, help="Eta. Defaults to 0.0.")
parser.add_argument("--c_min", type=float, default=c_min, help="Min. Defaults to 0.0.")
parser.add_argument("--c_max", type=float, default=c_max, help="Max. Defaults to 1.0.")
parser.add_argument("--c_spread_min", type=float, default=c_spread_min, help="Spread min. Defaults to 0.15.")
parser.add_argument("--prediction", type=str, default=prediction, help="Prediction type (constant, (rls, nlms), linear, quadratic). Defaults to 'rls_linear'.")
parser.add_argument("--ae_learning_rate", type=float, default=ae_learning_rate, help="Learning rate of the autoencoder (if autoencoder should learn parallel). Defaults to 0.001.")
parser.add_argument("--ae_batch_size", type=int, default=ae_batch_size, help="Batch size of the autoencoder (if autoencoder should learn parallel). Defaults to 64.")
parser.add_argument("--env_flags", dest="env_flags", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help="Environment flags to pass to the environment.")
parser.add_argument("--use_reparametrization", action="store_true", default=False, help="Whether to use the reparametrization trick for the VAE as State.")
parser.add_argument("--ignore_sigma", action="store_true", default=False, help="Whether to ignore the sigma when forwarding latent to XCS.")
parser.add_argument("--normalize_latent", action="store_true", default=False, help="Whether to normalize the latent space into a range of 0 and 1.")
parser.add_argument("--legacy_input_scaling", action="store_true", default=False, help="Temporarily re-enable the old input preprocessing scaling behavior.")

parser.add_argument("--save_populations", action="store_true", default=False, help="Save the populations at the end of a run to the harddrive.")
parser.add_argument("--save_plots", action="store_true", default=False, help="Activates generating reward image plots for the single runs")
parser.add_argument("--use_tensorboard", action="store_true", default=False, help="Activates tensorboard and stores the events.")
parser.add_argument("--runs", type=int, required=False, default=1)
parser.add_argument("--start_at_run", type=int, required=False, default=None, help="Set this argument so that the experiment starts not at the first internal or passed seed but at the i-th one.")
parser.add_argument("--seeds", action="extend", type=int, nargs="+", required=False, default=[],
                        help="Allows to pass a fix seed for every run of an experimental series." 
                        "If no seed is set for run i, the experiment tries to take an internal_seed[i]."
                        "If such a seed does not exist it will generate a random one if the option for this is set")
parser.add_argument("--debug_images", action="store_true", default=False, help="Whether to show debug images of the autoencoder.")
parser.add_argument('--name', type=str, default="", help='Adds a custom name to the aexcsf directory')

parser.add_argument(
    "--latent_scaler",
    type=str,
    default=None,
    choices=["minmax", "rolling_minmax", "constant"],
    help="Scaler used for latent normalization. If omitted, the AE-specific default is used, if normalize_latent is set."
)
parser.add_argument("--vq_latent_state", type=str, default="indices", choices=["indices", "quantized"], help="Latent representation for VQ-based autoencoders.")
parser.add_argument(
    "--latent_scaler_kwargs",
    dest="latent_scaler_kwargs",
    action=StoreDictKeyPair,
    metavar="KEY1=VAL1,KEY2=VAL2...",
    default=None,
    help="Keyword args for latent scaler, e.g. window=500,eps=1e-6 or min=0,max=1024."
)


args = parser.parse_args()


# Override variables with parsed arguments
ENV_NAME = args.env_name
ENV_CONTINUOUS = args.env_continuous
X_DIMS = args.x_dims
Y_DIMS = args.y_dims
N_CHANNELS = args.n_channels
AXN_EPISODES = args.axn_episodes
MAX_STEPS = args.max_steps
TEST_AUTOENCODER = args.test_autoencoder
AUTOENCODER = args.autoencoder
NAME = args.name
use_reparametrization = args.use_reparametrization
ignore_sigma = args.ignore_sigma
normalize_latent = args.normalize_latent
ae_config = build_ae_config_from_args(AUTOENCODER, args)


trial_name = "_".join(("AEXCSF", NAME, ENV_NAME, AUTOENCODER, time.strftime("%Y%m%d-%H%M%S"))).replace("/", "_")
metric_save_path = os.path.join(METRIC_SAVE_DIR, trial_name)
os.makedirs(metric_save_path, exist_ok=True)
save_python_env_info(os.path.join(METRIC_SAVE_DIR, trial_name))
setup_logging(os.path.join(METRIC_SAVE_DIR, trial_name))

START_AT_RUN = 0 if args.start_at_run is None else args.start_at_run
passed_seeds: list[int] | None = None if not args.seeds else args.seeds
NUMBERS_OF_RUNS: int = args.runs
SEEDS = manage_seeds(passed_seeds=passed_seeds, start_at_run=START_AT_RUN, number_of_runs=NUMBERS_OF_RUNS)


ENV_ARGS = {"render_mode" : "rgb_array", "max_episode_steps" : MAX_STEPS}

if args.env_flags is not None:
    ENV_ARGS.update(args.env_flags)
    
print(f"Environment Flags: {ENV_ARGS}")

number_threads = args.number_threads
random_seed = args.random_seed
population_file = args.population_file
pop_init = args.pop_init
pop_size = args.pop_size
subsumption = not args.no_subsumption
theta_sub = args.theta_sub
loss_function = args.loss_function
huber_delta = args.huber_delta
gamma = args.gamma
p_explore = args.p_explore
p_max = args.p_max
p_min = args.p_min
max_exploration_trials = args.max_exploration_trials
e0 = args.e0
alpha = args.alpha
nu = args.nu
beta = args.beta
delta = args.delta
theta_del = args.theta_del
init_fitness = args.init_fitness
init_error = args.init_error
m_probation = args.m_probation
stateful = args.stateful
compaction = args.compaction
ea_select_size = args.ea_select_size
ea_theta = args.ea_theta
ea_lambda = args.ea_lambda
ea_p_crossover = args.ea_p_crossover
ea_fitness_reduction = args.ea_fitness_reduction
ea_error_reduction = args.ea_error_reduction
condition = args.condition
c_eta = args.c_eta
c_min = args.c_min
c_max = args.c_max
c_spread_min = args.c_spread_min
prediction = args.prediction
ae_learning_rate = args.ae_learning_rate
ae_batch_size = args.ae_batch_size
use_roulette = args.ea_roulette
debug_images = args.debug_images

use_tensorboard: bool = args.use_tensorboard
save_image_plots: bool = args.save_plots
save_populations: bool = args.save_populations

if args.use_render_as_state:
    def state_function(env, state):
        return env.render()

# Initiliase the RL environment using gymnasium
try:
    env = gymnasium.make(ENV_NAME, **ENV_ARGS)
except:
    try:
        env = gymnasium.make(ENV_NAME, max_episode_steps=MAX_STEPS, render_mode=ENV_ARGS["render_mode"])
    except Exception as e:
        print(f"Environment {ENV_NAME} not found or the Environment arguments {ENV_ARGS} do not fit. Please check the name of the environment.")
        e.print_stacktrace()
        exit(1)
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

# Reset the environment
state = env.reset()

# Load the autoencoder
# torch.serialization.add_safe_globals([ConvVariationalAutoEncoder, dill._dill._load_type, torch.nn.modules.container.Sequential])
autoencoder = torch.load(args.autoencoder_path, weights_only=False, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

# TODO: Remember to distinguish Action set subsumption and ga subsumption
#Experiment logic        
#NOTE Start_at_run is by default 0
for i in range(START_AT_RUN, START_AT_RUN + NUMBERS_OF_RUNS):
    current_run = i
    current_seed = SEEDS[i]
    axn = Axnet(autoencoder=autoencoder, env_name=ENV_NAME, autoencoder_type=AUTOENCODER, x_dim=X_DIMS, y_dim=Y_DIMS, channel=N_CHANNELS, n_actions=int(env.action_space.n), max_trials=AXN_EPISODES, max_steps=MAX_STEPS,
            number_threads=number_threads, random_seed=current_seed, population_file=population_file, pop_init=pop_init, pop_size=pop_size, subsumption=subsumption, theta_sub=theta_sub, loss_function=loss_function,
            ea="tournament" if not use_roulette else "roulette", huber_delta=huber_delta, gamma=gamma, p_explore=p_explore, p_max=p_max, p_min=p_min, e0=e0, alpha=alpha, nu=nu, beta=beta, delta=delta, theta_del=theta_del, init_fitness=init_fitness, init_error=init_error, m_probation=m_probation, 
            stateful=stateful, compaction=compaction, condition=condition, prediction=prediction, ae_learning_rate=ae_learning_rate, ae_batch_size=ae_batch_size, train_autoencoder=args.train_autoencoder, max_exploration_trials=max_exploration_trials,
            ea_select_size=ea_select_size, ea_theta=ea_theta, ea_lambda=ea_lambda, ea_p_crossover=ea_p_crossover, ea_fitness_reduction=ea_fitness_reduction, ea_error_reduction=ea_error_reduction, c_eta=c_eta, c_min=c_min, c_max=c_max, 
            c_spread_min=c_spread_min, normalize_latent=normalize_latent, legacy_input_scaling=args.legacy_input_scaling, debug_images=debug_images, ae_config=ae_config)

    if current_run == START_AT_RUN:
        logging.info("AEXCSF successful initialized with autoencoder: ", axn.get_autoencoder_type(), "\n")
        logging.info(f"Commandline arguments passed to the system: {str(args)}\n")
        logging.info(f"AEConfig is {str(ae_config)}\n")
        logging.info(f"Internally set hyperparameters of the XCSF:\n {str(axn.internal_params())}")
    
        ## test preprocess state function
        state, _ = env.reset()
        if state_function:
            state = state_function(env, state)
        state = axn.preprocess_state(state)

    #NOTE: First reset to set the seed of the enviornment
    env.reset(seed=current_seed)
    
    metrics_file_name = "_".join(("Experimental_Run_AEXCSF_in_", ENV_NAME, "seed", str(current_seed), "run", str(current_run))).replace("/", "_")
    axn.train_env(env=env, state_function=state_function, metrics_save_path=os.path.join("./Results", trial_name), 
                  tensorboard_path=os.path.join("./Results", trial_name, "tensorboard") if use_tensorboard else None, 
                  model_save_path=os.path.join("./Results", trial_name), model_file_name=f"result_population_seed_{current_seed}_run_{current_run}" if save_populations else None,
                  notebook=False, metrics_file_name=metrics_file_name)

    if save_populations:
        print("AEXCSF training finished, population saved in:", os.path.abspath(MODEL_SAVE_DIR + trial_name + ".pth"))

    metric_path = os.path.join(METRIC_SAVE_DIR, trial_name)
    if save_image_plots:
        # read in metrics and plot them
        parser = pd.read_csv(os.path.join(metric_path, metrics_file_name + ".csv"))
        statistic_savepath = os.path.join(os.path.join(metric_path, "stat_images"))
        os.makedirs(statistic_savepath, exist_ok=True)
        #calc the mean reward over 100 episodes
        mean_rewards = parser["rollout/ep_rew"].rolling(window=100).mean()
        plt.plot(mean_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward")
        plt.savefig(os.path.join(statistic_savepath, f"Mean_Reward_seed_{current_seed}_run{current_run}.pdf"))
        plt.clf()
    
        # plot the steps
        mean_steps = parser["rollout/ep_len"].rolling(window=100).mean()
        plt.plot(mean_steps)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Steps")
        plt.savefig(os.path.join(statistic_savepath, f"Mean_Steps_seed_{current_seed}_run{current_run}.pdf"))
        plt.clf()

    evaluation_utils.eval_and_store_experimental_series(metric_path, False, False)
