import argparse
import ast
import logging
import os
import subprocess
import time

import ale_py
import evaluation_utils
import gymnasium
from gymnasium import wrappers
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from vizdoom import gymnasium_wrapper
from env_wrapper.doom import init_custom_vizdoom_env
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import numpy as np
import pandas as pd
from torchvision.transforms import v2
from torch import float32, use_deterministic_algorithms
from tqdm import tqdm
import xcsf

from DatasetMaker.DatasetGenerator import MetricTracker
from DatasetMaker.DatasetParser import MinMaxNormalize
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
    result = subprocess.run(["pip3", "freeze"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return_val = result.stdout.decode("utf-8")
    with open(save_path + "/python_env_info.txt", "w") as fp:
        fp.write(str(return_val))

def setup_logging(save_path):
    logging.basicConfig(filename=save_path + "/experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.INFO)
    
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

def set_more_determinism() -> None:
    """
    Function for increasing the chance of reproducibility.
    Sets pytorch seed and pytorch to deterministic if desired.
    Args:
        seed (int): the chosen or generated seed
    """
    if MORE_DETERMINISM:
        use_deterministic_algorithms(True)


ENV_NAME = None
ENV_CONTINUOUS = False

INTERNAL_SEEDS: list[int] = []
MORE_DETERMINISM = False

# image parameter (changing these also changes the behavior of every image generated)
X_DIM = 64
Y_DIM = 64
N_CHANNELS = 1

# Dataset Params
TRAINING_EPISODES = 2000
MAX_STEPS = 1000 # maximum steps a agent can take in one episode

ATARI_PREPROCESSING = False
state_function = None

METRIC_SAVE_DIR = "./Results/"
MODEL_SAVE_DIR = "./Results/"

#AXN Params
number_threads = 8 #Number of threads. Defaults to 8.
random_seed = None #Random seed. Defaults to None.
population_file = "" #File to save the population. Defaults to "".
pop_init = False #If the population should get initialized. Defaults to False.
pop_size = 5000 #Population size. Defaults to 5000.
subsumption = True #If subsumption should get used. Defaults to True.
theta_sub = 100 #Subsumption threshold. Defaults to 100.
loss_function = "mae" #Loss function. Defaults to "mae" ("mae", "mse", "rmse", "log", "binary_log", "onehot", "huber").
huber_delta = 1 #Huber delta. Defaults to 1.
gamma = 0.95 #Gamma. Defaults to 0.95.
p_explore = "linear" #Exploration function (linear, exponential, sigmoid, constant). Defaults to "linear" ("linear", "exponential", "sigmoid", "constant").
p_max = 1 #Maximum exploration probability. Defaults to 1.
p_min = 0.01 #Minimum exploration probability. Defaults to 0.1.
e0 = 0.001 #E0. Defaults to 0.1.
alpha = 0.1 #Alpha. Defaults to 0.1.
nu = 5 #Nu. Defaults to 5.
beta = 0.1 #Beta. Defaults to 0.1.
delta = 0.1 #Delta. Defaults to 0.1.
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
condition = "hyperrectangle_ubr" #Condition type ("ternary", "hyperrectangle_csr", "hyperrectangle_ubr", "hyperellipsoid", "tree_gp", "dgp"). Defaults to "hyperrectangle_csr".
c_eta = 0.0 #Eta. Defaults to 0.0.
c_min = 0.0 #Min. Defaults to 0.0.
c_max = 1.0 #Max. Defaults to 1.0.
c_spread_min = 0.1 #Spread min. Defaults to 0.15.
prediction = "rls_linear" #Prediction type (constant, (rls, nlms), linear, quadratic). Defaults to "rls_linear".

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, required=True, help="Name of the gym environment.")
parser.add_argument("--metric_save_dir", type=str, default=METRIC_SAVE_DIR, help="Path to where the metrics are saved.")
parser.add_argument("--model_save_dir", type=str, default=MODEL_SAVE_DIR, help="Path to the model save directory.")
parser.add_argument("--axn_episodes", type=int, default=TRAINING_EPISODES, help="Number of episodes the axn agent should train.")
parser.add_argument("--env_continuous", action="store_true", default=ENV_CONTINUOUS, help="Whether the environment is continuous.")
parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Maximum steps an agent can take in one episode.")
parser.add_argument("--x_dims", type=int, default=X_DIM, help="Width of the generated images.")
parser.add_argument("--y_dims", type=int, default=Y_DIM, help="Height of the generated images.")
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
parser.add_argument("--ea_select_size", type=float, default=0.4, help="Size of the selection. Defaults to 0.4.")
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
parser.add_argument("--env_flags", dest="env_flags", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help="Environment flags to pass to the environment.")
parser.add_argument("--no_pixel_rescaling", action="store_true", default=False, help="Deactivate the rescaling of pixel values to [0, 1]")
parser.add_argument("--save_populations", action="store_true", default=False, help="Save the populations at the end of a run to the harddrive.")
parser.add_argument("--save_plots", action="store_true", default=False, help="Activates generating reward image plots for the single runs")
parser.add_argument("--use_tensorboard", action="store_true", default=False, help="Activates tensorboard and stores the events.")
parser.add_argument("--runs", type=int, required=False, default=1)
parser.add_argument("--start_at_run", type=int, required=False, default=None, help="Set this argument so that the experiment starts not at the first internal or passed seed but at the i-th one.")
parser.add_argument("--seeds", action="extend", type=int, nargs="+", required=False, default=[],
                        help="Allows to pass a fix seed for every run of an experimental series." 
                        "If no seed is set for run i, the experiment tries to take an internal_seed[i]."
                        "If such a seed does not exist it will generate a random one if the option for this is set")
parser.add_argument('--name', type=str, default="", help='Adds a custom name to the xcs directory')

args = parser.parse_args()

# Override variables with parsed arguments
ENV_NAME = args.env_name
ENV_CONTINUOUS = args.env_continuous
X_DIM = args.x_dims
Y_DIM = args.y_dims
N_CHANNELS = args.n_channels
TRAINING_EPISODES = args.axn_episodes
MAX_STEPS = args.max_steps
NAME = args.name


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
no_pixel_rescaling = args.no_pixel_rescaling

use_tensorboard: bool = args.use_tensorboard
save_image_plots: bool = args.save_plots
save_populations: bool = args.save_populations


trial_name = "_".join(("XCSF_Training", NAME, ENV_NAME, str(X_DIM), "x", str(Y_DIM), time.strftime("%Y%m%d-%H%M%S")))
trial_name = trial_name.replace("/", "_")
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


def update_p(xcs, p_max, p_min, trial, max_trials, exploration_function = "linear") -> None:
    """Update the exploration probability inside the axn.
    Raises:
        ValueError: Exploration function not implemented.
    """
    if trial > max_trials:
        return
    elif exploration_function == "linear":
        p_explr = max(p_max - p_min, 0) * (1 - trial / max_trials) + p_min
    elif exploration_function == "exponential":
        p_explr = p_max * (p_min / p_max) ** (trial / max_trials)
    elif exploration_function == "sigmoid":
        p_explr = p_max / (1 + np.exp(0.1 * (trial - max_trials / 2)))
    elif exploration_function == "constant":
        p_explr = p_max
    else:
        raise ValueError("Exploration function not implemented.")
    xcs.set_params(p_explore=p_explr)


if args.use_render_as_state:
    def state_function(env, state):
        return env.render()
    
# rescale the image to the desired size
transformer = v2.Compose([
            v2.ToPILImage(),
            v2.Resize((X_DIM, Y_DIM)),
            v2.Grayscale(num_output_channels=1) if N_CHANNELS == 1 else v2.Identity(),
            v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=True)])
        ])

# NOTE: Superfluos - v2.toDtype scales already
# if no_pixel_rescaling:
#     pixel_scaler = lambda state: state
# else:
#     pixel_scaler = MinMaxNormalize(per_channel=False)

# Create the environment
try:
    env = gymnasium.make(ENV_NAME, **ENV_ARGS)
except:
    try:
        env = gymnasium.make(ENV_NAME, max_episode_steps=MAX_STEPS, render_mode=ENV_ARGS["render_mode"])
    except Exception as e:
        print(f"Environment {ENV_NAME} not found or the Environment arguments {ENV_ARGS} do not fit. Please check the name of the environment.")
        e.print_stacktrace()
        exit(1)
if ENV_NAME.startswith("MiniGrid"):
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


for i in range(START_AT_RUN, START_AT_RUN + NUMBERS_OF_RUNS):
    current_run = i
    current_seed = SEEDS[i]

    # Instantiate the XCSF
    xcs = xcsf.XCS(x_dim=X_DIM * Y_DIM * N_CHANNELS, y_dim=1, n_actions=int(env.action_space.n), omp_num_threads=number_threads, random_state=current_seed, population_file=population_file, pop_init=pop_init,
                   max_trials=TRAINING_EPISODES, perf_trials=1000000, pop_size=pop_size,teletransportation=MAX_STEPS, set_subsumption=False, ea={"theta_ea": ea_theta, "select_type": "tournament", "subsumption": subsumption}, theta_sub=theta_sub, loss_func=loss_function, huber_delta=huber_delta, gamma=gamma, p_explore=p_max, 
                   e0=e0, alpha=alpha, nu=nu, beta=beta, delta=delta, theta_del=theta_del, init_fitness=init_fitness, init_error=init_error, m_probation=m_probation, stateful=stateful, compaction=compaction, 
                   condition={"type" : condition, "args": {"eta" : c_eta, "min" : c_min, "max" : c_max, "spread_min" : c_spread_min}}, prediction={"type" : prediction})

    if current_run == START_AT_RUN:
        logging.info(f"Commandline arguments passed to the system: {str(args)}\n")
        logging.info(f"Internally set hyperparameters of the XCSF:\n {str(xcs.internal_params())}")
    
    # Train the XCSF
    metrics_file_name = "_".join(("Experimental_Run_XCSF_in", ENV_NAME, "seed", str(current_seed), "run", str(current_run))).replace("/", "_")
    print(metrics_file_name)
    metricTrack = MetricTracker("XCSF_training_metrics" if metrics_file_name is None else metrics_file_name, tensorboard_log=METRIC_SAVE_DIR + trial_name if use_tensorboard else None)
    
    # Set the seed for the environment
    env.reset(seed=current_seed)

    training_pbar = tqdm(range(TRAINING_EPISODES), desc='Training XCSF...')
    for i in range(TRAINING_EPISODES):
        state, _ = env.reset()
        xcs.init_trial()
        err = 0
        cnt = 0
        overall_reward = 0
        for _ in range(MAX_STEPS):
            if state_function:
                state = state_function(env, state)
            state = np.asarray(transformer(state), dtype=np.float64).copy().flatten()
            xcs.init_step()
            action = xcs.decision(state, explore=True)
            next_state, reward, done, trunc, _ = env.step(action)
            done = done or trunc # axn cannot handle truncation, so we treat it as done, but we still learn from it
            xcs.update(reward, done) # update the current action set and/or previous action set
            overall_reward += reward
            cnt += 1
            # gymnasium environments dont have a max_payoff, so we use the max_payoff of the environment
            try: # try to get the max_payoff from the environment
                max_payoff = env.get_wrapper_attr('max_payoff')() # env specific
            except: # if it fails, we use the max_payoff of the environment
                max_payoff = 10 # env specific
            error = xcs.error(reward, done, max_payoff) # system prediction error, max_payoff is the maximum reward possible
            err += error
            xcs.end_step()
            if done:
                break
            state = next_state
        xcs.end_trial()
        update_p(xcs, p_max, p_min, i, max_exploration_trials, p_explore)
        metric = {"episode" : i, "rollout/ep_len" : cnt, "rollout/ep_rew" : overall_reward,
                  "train/sys_error" : err / cnt, "train/exploration_rate" : xcs.internal_params()["p_explore"], "train/mset" : xcs.mset_size(), "train/pop_size" : xcs.pset_size()}
        metricTrack.add_metrics(metric)
        avg_reward = np.mean((metricTrack.metrics["rollout/ep_rew"])[-100:])
        avg_ep_len = np.mean((metricTrack.metrics["rollout/ep_len"])[-100:])
        mean_metric = {"rollout/mean100_ep_rew" :  avg_reward, "rollout/mean100_ep_len" :  avg_ep_len}
        metricTrack.add_metrics(mean_metric)
        postfix = {"episode" : i, "p_xplr" : xcs.internal_params()["p_explore"], "avg_reward": avg_reward, "ep_reward":overall_reward, "avg_steps": avg_ep_len, "ep_steps" : cnt,"train/pop_size" : xcs.pset_size()}
        training_pbar.set_postfix(postfix)
        training_pbar.update()
    training_pbar.close()
    
    metric_path = os.path.join(METRIC_SAVE_DIR, trial_name)
    complete_save_path = os.path.join(metric_path, metrics_file_name + ".csv")
    
    if save_populations:
        xcs.save(os.path.join(metric_path, f"result_population_seed_{current_seed}_run_{current_run}.xcs"))
    metricTrack.save_csv(metric_path)
    print("XCSF runs finished")

    if save_image_plots:
        # read in metrics and plot them
        parser = pd.read_csv(os.path.join(METRIC_SAVE_DIR, trial_name, metrics_file_name + ".csv"))
        statistic_savepath = os.path.join(METRIC_SAVE_DIR, trial_name, "stat_images")
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