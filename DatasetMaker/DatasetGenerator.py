import time
import random
import os
import queue
import threading

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from torchvision.transforms import v2
import torchvision

import gymnasium

from DatasetMaker.ppo import PPO

from tqdm import tqdm

import pandas as pd

from torch.utils.tensorboard import SummaryWriter


class BackgroundImageWriter:
    """
    Streams image writes on a single background thread.

    The queue is bounded so dataset generation cannot accumulate the full
    dataset in memory if disk writes fall behind.
    """

    def __init__(self, save_fn, max_queue_size: int = 32) -> None:
        self.save_fn = save_fn
        self.items = queue.Queue(maxsize=max_queue_size)
        self.stop_token = object()
        self.error = None
        self.worker = threading.Thread(target=self._run, name="dataset-image-writer", daemon=True)
        self.worker.start()

    def submit(self, state, path: str) -> None:
        frozen_state = self._freeze_state(state)
        while True:
            self._raise_if_failed()
            try:
                self.items.put((frozen_state, path), timeout=0.1)
                break
            except queue.Full:
                continue
        self._raise_if_failed()

    def close(self) -> None:
        while self.worker.is_alive():
            try:
                self.items.put(self.stop_token, timeout=0.1)
                break
            except queue.Full:
                self._raise_if_failed()
                continue
        self.worker.join()
        self._raise_if_failed()

    def _run(self) -> None:
        while True:
            item = self.items.get()
            try:
                if item is self.stop_token:
                    return
                state, path = item
                self.save_fn(state, path)
            except Exception as exc:
                self.error = exc
                return
            finally:
                self.items.task_done()

    def _freeze_state(self, state):
        if torch.is_tensor(state):
            return state.detach().cpu().clone()
        if isinstance(state, np.ndarray):
            return state.copy()
        return state

    def _raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("Background image writer failed") from self.error

class AbstractAgent():
    """
    Abstract class defining the structure and behavior of an RL agent.
    """

    def __init__(self, state_shape, action_count):
        self.state_shape = state_shape
        self.action_count = action_count
    
    def get_action(self, state):
        """
        Select an action based on the current state.
        """
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done):
        """
        Update the parameters based on the observed reward and transition to the next state.
        """
        raise NotImplementedError

    def save_model(self, path, filename):
        """
        Serializes and saves the model as a file
        """
        raise NotImplementedError

    def load_model(self, path, filename):
        """
        Loads the model from a file
        """
        raise NotImplementedError


class RandomAgent(AbstractAgent):
    """a random agent that selects actions randomly.

    Args:
        AbstractAgent (_type_): AbstractAgent class
        state_shape (tuple): shape of the state
        action_count (int): number of actions
        seed (float): seed to use for random number generation
    """
    def __init__(self, state_shape = (224, 244), action_count = 4, seed : float = time.time()) -> None:
        super().__init__(state_shape, action_count)
        random.seed(seed)
        
    def get_action(self, state):
        return random.randint(0, self.action_count - 1)
    
    def learn(self, state, action, reward, next_state, done):
        pass
    
    def save_model(self, path, filename):
        pass
    
    def load_model(self, path, filename):
        pass
    
class PPOAgent(AbstractAgent):
    """
    Class to implement an agent using the PPO algorithm.
    Args:
        AbstractAgent (_type_): AbstractAgent class
        state_shape (tuple): shape of the state
        action_count (int): number of actions
        is_continiuos (bool): whether the action space is continious or discrete
        neural_net (str): neural network to use
        load_model (str): path to the model to load
        random_seed (float): seed to use for random number generation
    """
    def __init__(self, state_shape, action_count : int, is_continiuos : bool, neural_net, load_model : str = "", random_seed : float = time.time()) -> None:
        random_seed = int(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        super().__init__(state_shape, action_count)
        self.is_continiuos = is_continiuos
        if load_model:
            self.model = self.load_model(load_model)
        else:
            self.model = PPO(state_dim=state_shape, action_dim=action_count, is_continious=is_continiuos, neural_net=neural_net)
        
    def get_action(self, state):
        return self.model.select_action(state.unsqueeze(0))
    
    def learn(self, state, action, reward, next_state, done):
        self.model.buffer.rewards.append(reward)
        self.model.buffer.is_terminals.append(done)
        
    def update(self):
        self.model.update()
        if self.is_continiuos:
            self.model.decay_action_std(action_std_decay_rate=0.05, min_action_std=0.1)
        
    def save_model(self, path, filename):
        self.model.save(os.path.join(path, filename))
        
    def load_model(self, path, filename):
        return self.model.load(os.path.join(path, filename))
    

class DatasetGenerator():
    """a class to generate a dataset from an environment using a given method.
    Args:
        env : gymnasium.Env : environment to use
        method : str : method to use for generating the dataset
        image_size : tuple : size of the image
        image_channel : int : number of channels in the image
        action_count : int : number of actions
        is_continous : bool : whether the action space is continious or discrete
        load_model : str : path to the model to load
        net : str : neural network to use
        state_function : function : function to preprocess
        random_seed : float : seed to use for random number generation
    """
    def __init__(self, env : gymnasium.Env, method : str = "random", image_size : tuple[int, int] = (224, 224), image_channel : int = 3, is_continous : bool = False, load_model : str = "", net : str = "cnn", state_function = None, random_seed : float = time.time(), legacy_preprocessing : bool = False) -> None:
        # maybe use weights for runs that went well?
        self.method = method
        self.image_size = image_size
        self.seed = random_seed
        self.env = env
        action_count = env.action_space.n

        if legacy_preprocessing:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(self.image_size),
                v2.Grayscale(num_output_channels=1) if image_channel == 1 else v2.Identity(),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            ])
        else:
            transform_steps = [
                v2.ToImage(),
                v2.Resize(self.image_size),
            ]
            if image_channel == 1:
                transform_steps.append(v2.Grayscale(num_output_channels=1))
            transform_steps.append(v2.ToDtype(torch.float32, scale=True))
            self.transform = v2.Compose(transform_steps)
        
        self.state_function = state_function
        
        if self.method == "random":
            self.agent = RandomAgent(state_shape = (image_channel, *image_size), action_count = action_count)
        elif self.method == "ppo":
            self.agent = PPOAgent(state_shape = (image_channel, *image_size), action_count = action_count, is_continiuos = is_continous, neural_net = net, random_seed = random_seed, load_model = load_model)
        else:
            raise ValueError(f"Method {method} not implemented")
            
    def generate_dataset(self, num_episodes : int, max_steps : int = 400, update_interval : int = None, save_preprocessed : bool = True, save_path : str = "./Dataset", dataset_name="", async_image_writes : bool = True, image_write_queue_size : int = 32):     
        ### set environment default variables
        if update_interval is None:
            update_interval = max_steps - 1 # set the update interval to the maximum number of steps
        if self.method == "ppo":
            self.agent.K_epochs = num_episodes # set the number of epochs for the PPO algorithm
        
        summary_steps = 0
        save_path = os.path.join(save_path, self.method)
        save_path = os.path.join(save_path, self.env.unwrapped.spec.id.replace("/", "_"))
        save_path = os.path.join(save_path, dataset_name + time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(save_path, exist_ok=True)
        metric_tracker = MetricTracker("training_metrics_" + self.method)
        image_writer = BackgroundImageWriter(self.save_image, max_queue_size=image_write_queue_size) if async_image_writes else None
        try:
            for episode in tqdm(range(num_episodes), desc="filling dataset...", bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', colour='red', leave=False):
                state : torch.Tensor = self.env.reset()[0]
                if self.state_function:
                    try:
                        state = self.state_function(self.env, state)
                    except:
                        raise Exception("State function failed")
                if not save_preprocessed:
                    self._store_image(state, os.path.join(save_path, f"{summary_steps}.jpg"), image_writer) # Save the image into the dataset
                state = self.preprocess(state)
                if save_preprocessed:
                    self._store_image(state, os.path.join(save_path, f"{summary_steps}.jpg"), image_writer)
                done = False
                i = 0
                total_reward = 0
                total_loss = 0
                while not done and i < max_steps:
                    action = self.agent.get_action(state)
                    next_state, reward, done, trunc, _ = self.env.step(action)
                    if self.state_function:
                        next_state = self.state_function(self.env, next_state)
                    done = done or trunc
                    if not save_preprocessed:
                        self._store_image(next_state, os.path.join(save_path, f"{summary_steps}.jpg"), image_writer) # Save the image into the dataset
                    next_state = self.preprocess(next_state)
                    if save_preprocessed:
                        self._store_image(next_state, os.path.join(save_path, f"{summary_steps}.jpg"), image_writer)
                    self.agent.learn(state, action, reward, next_state, done)
                    state = next_state
                    i += 1
                    summary_steps += 1
                    total_reward += reward
                    if self.method != "random":
                        total_loss += self.agent.model.loss
                    if summary_steps % update_interval == 0 and self.method != "random":
                        self.agent.update()
                metric_tracker.add_metrics({"episode": episode, "episode_steps": i, "episode_reward": total_reward, "episode_loss": total_loss})  
        finally:
            if image_writer is not None:
                image_writer.close()
        self.agent.save_model(save_path, "model.pt")
        metric_tracker.save_csv(save_path)
        return save_path
        
    def preprocess(self, state):
        return self.transform(state)
        
    def save_image(self, state : Image, path):
        torchvision.utils.save_image(state, path)

    def _store_image(self, state, path, image_writer = None):
        if image_writer is None:
            self.save_image(state, path)
            return
        image_writer.submit(state, path)
        
    def show_image(self, state):
        plt.imshow(state.permute(1, 2, 0))
        plt.show()
        
class MetricTracker():
    def __init__(self, trial_name : str, tensorboard_log = None) -> None:
        self.trial_name = trial_name
        self.metrics = {}
        self.tensorboard_log = tensorboard_log
        if tensorboard_log:
            # create tensorboard log writer
            self.writer = SummaryWriter(os.path.join(tensorboard_log, trial_name))
    
    def add_metric(self, metric_name : str, metric_value : float):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(metric_value)
        if self.tensorboard_log:
            global_step = len(self.metrics[metric_name])
            self.writer.add_scalar(metric_name, metric_value, global_step)
 
    def add_metrics(self, metrics : dict):
        for metric_name, metric_value in metrics.items():
            self.add_metric(metric_name, metric_value)
        
    def save_csv(self, root_dir : str, delete_cache : bool = True):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        pd.DataFrame(self.metrics).to_csv(os.path.join(root_dir, self.trial_name + ".csv"))
        if delete_cache:
            self.clear_cache()

    def clear_cache(self):
        self.metrics = {}
