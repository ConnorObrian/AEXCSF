import xcsf
import numpy as np
import torch
from torchvision.transforms import v2
import os
import warnings
import gymnasium
import time
from dataclasses import dataclass, field
from DatasetMaker.DatasetGenerator import MetricTracker
from DatasetMaker.DatasetParser import ConstantMinMaxScaler, MinMaxNormalize, MinMaxRollingObsScaler, ScalerKind
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from typing import Deque

import matplotlib.pyplot as plt

import dill  # for saving the autoencoder with torch.save

@dataclass
class AEConfig:
    latent_scaler: ScalerKind | str | int = ScalerKind.MINMAX
    latent_scaler_kwargs: dict = field(default_factory=dict)


@dataclass
class VAEConfig(AEConfig):
    use_reparameterization: bool = False
    ignore_sigma: bool = True


@dataclass
class VQVAEConfig(AEConfig):
    latent_state: str = "indices"
    latent_scaler: ScalerKind | str | int | None = None


class Axnet():
    """
    A XCS combined with an Autoencoder for state representation.
    Parameters:
        autoencoder (torch.nn.Module): The autoencoder model.
        autoencoder_type (str): Type of the autoencoder (cae, mae, vae, cvae, ae, transformer).
        x_dim (int): Width of the input image.
        y_dim (int): Height of the input image.
        channel (int): Number of channels in the input image.
        n_actions (int): Number of actions.
        train_autoencoder (bool): Flag to indicate if the autoencoder should be trained.
        number_threads (int): Number of threads to use for XCS.
        random_seed (int): Random seed for reproducibility.
        population_file (str): File to save the population.
        pop_init (bool): Flag to indicate if the population should be initialized.
        max_trials (int): Maximum number of trials.
        perf_trials (int): Performance trials.
        pop_size (int): Population size.
        subsumption (bool): Flag to indicate if subsumption should be used.
        set_subsumption (bool): Flag activating/deactivating action set subsumption (not recommended to use)
        theta_sub (int): Subsumption threshold.
        loss_function (str): Loss function for the XCS (mae, mse, rmse, log, binary_log, onehot, huber).
        huber_delta (float): Delta value for the Huber loss.
        max_steps (int): Maximum number of steps per trial.
        gamma (float): Gamma value for the XCS.
        p_explore (str): Exploration function.
        p_max (float): Maximum exploration probability.
        p_min (float): Minimum exploration probability.
        max_exploration_trials (int): Maximum number of exploration trials.
        e0 (float): Epsilon value.
        alpha (float): Alpha value.
        nu (float): Nu value.
        beta (float): Beta value.
        delta (float): Delta value.
        theta_del (int): Deletion threshold.
        init_fitness (float): Initial fitness value.
        init_error (float): Initial error value.
        m_probation (int): M-probation value.
        stateful (bool): Flag to indicate if the XCS should be stateful.
        compaction (bool): Flag to indicate if compaction should be used.
        ea (str): Evolution algorithm.
        ea_select_size (float): Selection size for the evolution algorithm.
        ea_theta (int): Theta value for the evolution algorithm.
        ea_lambda (int): Lambda value for the evolution algorithm.
        ea_p_crossover (float): Crossover probability for the evolution algorithm.
        ea_fitness_reduction (float): Fitness reduction for the evolution algorithm.
        ea_error_reduction (float): Error reduction for the evolution algorithm.
        ea_pred_reset (bool): Flag to indicate if the prediction should be reset.
        action_type (str): Type of action.
        condition (str): Type of condition (ternary, hyperrectangle_csr, hyperrectangle_ubr, hyperellipsoid, tree_gp, dgp).
        c_eta (float): Eta value for the condition.
        c_min (float): Minimum value for the condition.
        c_max (float): Maximum value for the condition.
        c_spread_min (float): Minimum spread value for the condition.
        prediction (str): Type of prediction (constant, nlms_linear / quadratic, rls_linear / quadratic).
        transform (v2.Compose): Additional transformations for the input state.
        ae_learning_rate (float): Learning rate for the autoencoder.
        ae_batch_size (int): Batch size for the autoencoder.
        use_reparameterization (bool): Flag to indicate if the reparameterization trick should be used as state representation.
    Attributes:
        device (torch.device): The device to run the model on (CPU or GPU).
        train_autoencoder (bool): Flag to indicate if the autoencoder should be trained.
        use_reparameterization (bool): Flag to indicate if the reparameterization trick should be used as state representation.
        pre_transform (v2.Compose): Preprocessing transformations for the input state.
        exploration_function (str): The exploration function type.
        p_min (float): Minimum exploration probability.
        p_max (float): Maximum exploration probability.
        trial (int): Current trial number.
        max_trials (int): Maximum number of trials.
        max_exploration_trials (int): Maximum number of exploration trials.
        max_steps (int): Maximum number of steps per trial.
        train_mode (bool): Mode of the model ("train" or "eval").
        transform (v2.Compose): Additional transformations for the input state.
        autoencoder_type (str): Type of the autoencoder.
        autoencoder (torch.nn.Module): The autoencoder model.
        optimizer (torch.optim.Optimizer): Optimizer for training the autoencoder.
        image_buffer (list): Buffer to store images for training the autoencoder.
        batch_size (int): Batch size for training the autoencoder.
        autoencoder_loss (float): Loss value of the autoencoder.
        xcs (xcsf.XCS): The XCS model.
    Methods:
        set_autoencoder(autoencoder):
            Set the autoencoder.
        get_autoencoder_type() -> type:
            Get the autoencoder type.
        _update_p():
            Update the exploration probability inside the AXN.
        select_action(state: torch.Tensor) -> int:
            Select an action based on the current state.
        init_trial():
            Initialize the trial.
        end_trial():
            End the trial and update exploration probability.
        init_step():
            Initialize the step.
        end_step():
            End the step.
        train():
            Set mode to train.
        eval():
            Set mode to eval.
        get_error(reward: float, done: bool, max_reward: float) -> tuple[float, float]:
            Get the error of AXN.
        learn(reward: float, done: bool):
            Learn one step of an episode.
        preprocess_state(state: np.ndarray) -> np.ndarray:
            Preprocess the state.
        _train_autoencoder():
            Train the autoencoder from the already seen images this episode.
        select_single(state: torch.Tensor) -> int:
            Select an action for single action environments.
        learn_single(state: torch.Tensor, action: int, reward: float):
            Learn for single action environments.
        save(root_dir: str, filename: str):
            Save the model.
        load(root_dir: str, filename: str):
            Load a model.
        internal_params() -> dict:
            Get the internal parameters of the XCS.
        pset_size() -> int:
            Get the size of the PSet.
        mset_size() -> float:
            Get the size of the MSet.
        train_env(env: gymnasium.Env, state_function=None, metrics_save_path: str = None, tensorboard_path: str = None, model_save_path: str = None, notebook: bool = False):
            Train the AXN on a given environment.
    """
    def __init__(self, autoencoder, autoencoder_type : str, x_dim : int, y_dim : int, channel : int, n_actions : int, train_autoencoder : bool = False, number_threads : int = 8, random_seed : int = None, 
                 population_file : str = "", pop_init : bool = False, max_trials : int = 10000, perf_trials : int = 10000000, pop_size : int = 5000, subsumption : bool = True, set_subsumption: bool = False,
                 theta_sub : int = 100, loss_function : str = "mae", huber_delta : float = 1, max_steps : int = 300, gamma : float = 0.93, p_explore : str = "linear", 
                 p_max : float = 1, p_min = 0.05, max_exploration_trials : int = -1, e0 : float = 0.01, alpha : float = 0.1, nu : float = 5, beta : float = 0.15, delta : float = 0.15, theta_del : int = 25, 
                 init_fitness : float = 0.01, init_error : float = 0, m_probation : int = 10000, stateful : bool = True, compaction : bool = False, ea : str = "roulette", ea_select_size : float = 0.4, ea_theta : int = 25,
                 ea_lambda : int = 2, ea_p_crossover : float = 0.8, ea_fitness_reduction : float = 0.1, ea_error_reduction : float = 1.0, ea_pred_reset : bool = False, action_type : str = "integer", 
                 condition : str = "hyperrectangle_ubr", c_eta : float = 0.0, c_min : float = 0.001, c_max : float = 1.0, c_spread_min : float = 0.001, prediction : str = "rls_linear", transform : v2.Compose = None, ae_learning_rate : float = 0.001, ae_batch_size : int = 64,
                 normalize_latent : bool = True, debug_images: bool = False, env_name:str="", legacy_input_scaling: bool = False,
                 ae_config: AEConfig | None = None) -> None:

        action = {
            "type": action_type
        }
        condition = {
            "type": condition,
            "args":{
                "eta": c_eta,
                "min": c_min,
                "max": c_max,
                "spread_min": c_spread_min
            }
        }
        prediction = {
            "type": prediction
        }
        ea = {
            "select_type" : ea,
            "subsumption" : subsumption,
            "select_size" : ea_select_size,
            "theta_ea" : ea_theta,
            "lambda" : ea_lambda,
            "p_crossover" : ea_p_crossover,
            "err_reduc" : ea_error_reduction,
            "fit_reduc" : ea_fitness_reduction,
            "pred_reset" : ea_pred_reset
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_autoencoder = train_autoencoder

        if random_seed is None:
            random_seed = time.time()
        torch.manual_seed(random_seed)
        np.random.seed(int(random_seed))
        
        pre_transform_steps = [v2.Resize((x_dim, y_dim))]
        if channel == 1:
            pre_transform_steps.append(v2.Grayscale(num_output_channels=1))
        if legacy_input_scaling:
            pre_transform_steps.append(MinMaxNormalize(per_channel=False))
            pre_transform_steps.append(v2.ToDtype(torch.float32, scale=True))
        self.pre_transform = v2.Compose(pre_transform_steps).to(self.device)

        # exploration - exploitation
        self.exploration_function = p_explore
        self.p_min = p_min
        self.p_max = p_max
        self.max_exploration_trials = max_exploration_trials if max_exploration_trials > 0 else max_trials - max_trials // 10
        self.trial = 0
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.train_mode = True
        self.env_name = env_name
        
        self.debugging_images = debug_images

        
        self.transform = transform
        
        # autoencoder
        self.autoencoder_type = autoencoder_type
        self.autoencoder = autoencoder.to(self.device)
        if self.train_autoencoder:
            warnings.warn("Training the autoencoder is not recommended for large datasets. Consider using a pre-trained autoencoder.")
            if self.autoencoder_type == "vae" or self.autoencoder_type == "cvae" and not hasattr(self.autoencoder, "loss_function"):
                raise ValueError("Given VAE does not have any loss_function as class function.")
            if self.autoencoder_type == "vae" or self.autoencoder_type == "cvae" and not hasattr(self.autoencoder, "reparameterize"):
                raise ValueError("Given VAE does not have any reparameterize as class function.")
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=ae_learning_rate)
        self.image_buffer = Deque(maxlen=10000) # buffer for the training of the ae
        self.batch_size = ae_batch_size
        self.autoencoder_loss = 0
        self.normalize_latent = normalize_latent
        self.ae_config = self._resolve_ae_config(ae_config)
        self.latent_scaler = self._build_latent_scaler(self.ae_config.latent_scaler, self.ae_config.latent_scaler_kwargs)

        self.use_reparameterization = getattr(self.ae_config, "use_reparameterization", False)
        self.ignore_sigma = getattr(self.ae_config, "ignore_sigma", False)
        if self.ignore_sigma and self.use_reparameterization:
            warnings.warn("Both ignore sigma and reparametrization trick are enabled. Ignoring ignore sigma.")
            self.ignore_sigma = False
        
        
        x = torch.rand(1, channel, x_dim, y_dim).to(self.device) # random input
        try:
            x = autoencoder.encode(x) # test if the encoder works
        except:
            raise ValueError("Given encoder does not have the right input size or is missing the encode function.")
        x = self._extract_xcs_latent(x)
        print("Encoder output size: ", x.shape)
        if len(x.shape) > 1:
            x = torch.flatten(x)
            print("Flattened encoder output size: ", x.shape)
        x_dim = x.shape[0]
        print("Input size: ", x_dim)
        self.xcs = xcsf.XCS(x_dim=x_dim, y_dim=1, n_actions=n_actions, omp_num_threads=number_threads, random_state=random_seed, population_file=population_file, 
                         pop_init=pop_init, max_trials=max_trials, perf_trials=perf_trials, pop_size=pop_size, set_subsumption=set_subsumption, theta_sub=theta_sub, 
                         loss_func=loss_function, huber_delta=huber_delta, teletransportation=max_steps, gamma=gamma, p_explore=p_max, e0=e0, alpha=alpha, nu=nu, 
                         beta=beta, delta=delta, theta_del=theta_del,init_fitness=init_fitness, init_error=init_error, m_probation=m_probation, stateful=stateful, 
                         compaction=compaction, ea=ea, action=action, condition=condition, prediction=prediction)
        


    def _resolve_ae_config(self, ae_config: AEConfig | None) -> AEConfig:
        if ae_config is None:
            if self.autoencoder_type in ["vae", "cvae", "infovae", "dino"]:
                return VAEConfig()
            if self.autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
                return VQVAEConfig()
            return AEConfig()

        if self.autoencoder_type in ["vae", "cvae", "infovae", "dino"]:
            if not isinstance(ae_config, VAEConfig):
                raise TypeError(f"{self.autoencoder_type} requires a VAEConfig.")
            return ae_config

        if self.autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
            if not isinstance(ae_config, VQVAEConfig):
                raise TypeError(f"{self.autoencoder_type} requires a VQVAEConfig.")
            if ae_config.latent_state not in ["indices", "quantized"]:
                raise ValueError(f"Unsupported VQ latent_state '{ae_config.latent_state}'. Valid: ['indices', 'quantized']")
            if ae_config.latent_scaler is None:
                ae_config.latent_scaler = self._default_vq_scaler_kind(ae_config.latent_state)
            return ae_config

        if isinstance(ae_config, (VAEConfig, VQVAEConfig)):
            raise TypeError(f"{type(ae_config).__name__} is not supported for autoencoder_type '{self.autoencoder_type}'.")

        return ae_config

    def _default_vq_scaler_kind(self, latent_state: str) -> ScalerKind:
        if latent_state == "indices":
            return ScalerKind.CONSTANT
        if latent_state == "quantized":
            return ScalerKind.ROLLING_MINMAX
        raise ValueError(f"Unsupported VQ latent_state '{latent_state}'.")

    def _extract_xcs_latent(self, latent):
        if self.autoencoder_type == "vae" or self.autoencoder_type == "cvae":
            if self.use_reparameterization:
                return self.autoencoder.reparameterize(latent[0].detach(), latent[1].detach())
            return torch.cat((latent[0].detach(), latent[1].detach())) if not self.ignore_sigma else latent[0].detach()


        if self.autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
            vq_latent_state = getattr(self.ae_config, "latent_state", "indices")
            if vq_latent_state == "indices":
                return latent[3].detach()
            if vq_latent_state == "quantized":
                return latent[1].detach()
            raise ValueError(f"Unsupported VQ latent_state '{vq_latent_state}'.")

        latent = latent.detach()
        return latent[0]

    def _build_latent_scaler(self, kind: ScalerKind | str | int, kwargs: dict):
        # Optional backward compatibility for int
        if isinstance(kind, int):
            int_map = {0: ScalerKind.MINMAX, 1: ScalerKind.ROLLING_MINMAX, 2: ScalerKind.CONSTANT}
            if kind not in int_map:
                raise ValueError(f"Unknown scaler id {kind}. Valid ids: {list(int_map.keys())}")
            kind = int_map[kind]

        if isinstance(kind, str):
            try:
                kind = ScalerKind(kind.lower())
            except ValueError:
                valid = [k.value for k in ScalerKind]
                raise ValueError(f"Unknown scaler '{kind}'. Valid: {valid}")

        if kind == ScalerKind.MINMAX:
            return MinMaxNormalize(**kwargs)

        if kind == ScalerKind.ROLLING_MINMAX:
            return MinMaxRollingObsScaler(**kwargs)

        if kind == ScalerKind.CONSTANT:
            if self.autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
                return ConstantMinMaxScaler(min=0, max=self.autoencoder.vq.codebook_size)
            else:
                return ConstantMinMaxScaler(**kwargs)
            
        raise ValueError(f"Unhandled scaler kind: {kind}")

    def set_autoencoder(self, autoencoder) -> None:
        """Set the autoencoder.

        Args:
            autoencoder (_type_): Autoencoder.
        """
        self.autoencoder = autoencoder
        
    def get_autoencoder_type(self) -> type:
        """Get the autoencoder type.

        Returns:
            _type_: Autoencoder type.
        """
        return type(self.autoencoder)
        
    def _update_p(self) -> None:
        """Update the exploration probability inside the axn.

        Raises:
            ValueError: Exploration function not implemented.
        """
        if self.trial > self.max_exploration_trials:
            return
        elif self.exploration_function == "linear":
            p_explr = max(self.p_max - self.p_min, 0) * (1 - self.trial / self.max_exploration_trials) + self.p_min
        elif self.exploration_function == "exponential":
            p_explr = self.p_max * (self.p_min / self.p_max) ** (self.trial / self.max_exploration_trials)
        elif self.exploration_function == "sigmoid":
            p_explr = self.p_max / (1 + np.exp(0.1 * (self.trial - self.max_exploration_trials / 2)))
        elif self.exploration_function == "constant":
            p_explr = self.p_max
        else:
            raise ValueError("Exploration function not implemented.")
        self.xcs.set_params(p_explore=p_explr)
        
    def select_action(self, state : torch.Tensor) -> int:
        """Select an action.

        Args:
            state (torch.Tensor): current State.

        Returns:
            _type_: Action.
        """
        x = self.preprocess_state(state)
        action = self.xcs.decision(x, self.train_mode)
        return action
    
    def set_seed(self, seed : int) -> None:
        """Set the seed of the xcsf.

        Args:
            seed (int): Seed.
        """
        self.xcs.set_seed(seed)
    
    def init_trial(self) -> None:
        """Initialize the trial."""
        self.xcs.init_trial()
        
    def end_trial(self) -> None:
        """End the trial."""
        self.xcs.end_trial()
        self.trial += 1
        self._update_p()
        if self.train_autoencoder:
            self._train_autoencoder()
        
    def init_step(self) -> None:
        """Initialize the step."""
        self.xcs.init_step()
    
    def end_step(self) -> None:
        """End the step."""
        self.xcs.end_step()
        
    def train(self) -> None:
        """set mode to train"""
        self.trial = 0
        self.train_mode = True
    
    def eval(self) -> None:
        """set mode to eval"""
        self.mode = "eval"
        
    def get_error(self, reward : float, done : bool, max_reward : float) -> tuple[float, float]:
        """Get the error of AXN.

        Args:
            reward (float): actual reward.
            done (bool): if the episode is done.
            max_reward (float): expected maximum reward.

        Returns:
            _tuple_: error and autoencoder loss.
        """
        return self.xcs.error(reward, done, max_reward), self.autoencoder_loss
        
    
    def learn(self, reward : float, done : bool) -> None:
        """Learn one step of an episode.

        Args:
            reward (float): actual reward.
            done (bool): if the episode is done.
        """
        if self.train_mode:
            self.xcs.update(reward, done)
        else:
            warnings.warn("Trying to learn in eval mode.")
        
    def preprocess_state(self, state : np.ndarray) -> np.ndarray:
        """Preprocess the state.

        Args:
            state (torch.Tensor): tensor of a single image state.

        Returns:
            np.ndarray: Array of the detached state.
        """
        self.autoencoder.eval()
        state = to_tensor(state)
        state = self.pre_transform(state)
        if self.transform:
            state = self.transform(state)
        if self.train_autoencoder:
            self.image_buffer.append(state.copy() if hasattr(state, "copy") else state.clone())
        # add batch dimension if needed
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        # encode the state
        state = state.to(self.device)
        latent = self.autoencoder.encode(state)
        if self.autoencoder_type == "vae" or self.autoencoder_type == "cvae" or self.autoencoder_type == "dino":
            recon = self.autoencoder.decode(self.autoencoder.reparameterize(latent[0], latent[1]))
        elif self.autoencoder_type == "transformer":
            recon = self.autoencoder.decode(latent[0], latent[1])
            recon = self.autoencoder.reconstruct_image(recon)
        elif self.autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
            recon = self.autoencoder.decode(latent[1])
        elif self.autoencoder_type == "infovae":
            recon = self.autoencoder.decode(self.autoencoder.reparameterize(latent[0], latent[1]))
        else:
            recon = self.autoencoder.decode(latent)
        # show state and recon as image
        if self.debugging_images:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(state[0].cpu().detach().numpy().transpose(1,2,0))
            ax[0].set_title("State")
            ax[1].imshow(recon[0].cpu().detach().numpy().transpose(1,2,0))
            ax[1].set_title("Reconstructed")
            plt.show()
        # calc the loss
        loss = self.loss_func(state, recon)
        self.autoencoder_loss = loss.item()
        x = self._extract_xcs_latent(latent)

        # Test the new scaler for improved behaviour!            
        x = self.latent_scaler(x) if self.normalize_latent else x
        x = x.cpu()
        if len(x.shape) > 1:
            x = torch.flatten(x)
        return x.numpy()
    
    def _train_autoencoder(self) -> None:
        """Train the autoencoder from the already seen images this episode."""
        self.autoencoder.train()
        if len(self.image_buffer) == 0:
            return
        # create batches of size batch_size
        for i in range(0, len(self.image_buffer), self.batch_size):
            self.optimizer.zero_grad()
            batch = self.image_buffer[i:i+self.batch_size]
            batch = torch.stack(batch)
            if self.autoencoder_type == "cvae":
                x_hat, mu, logvar = self.autoencoder(batch.to(self.device))
                loss = self.autoencoder.loss_function(batch, x_hat, mu, logvar)
            elif self.autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
                x_hat, vq_loss = self.autoencoder(batch.to(self.device))
                loss = self.autoencoder.loss_function(batch, x_hat, vq_loss)
            else:
                x_hat = self.autoencoder(batch.to(self.device))
                loss = self.loss_func(x_hat, batch.to(self.device))
            loss.backward()
            self.optimizer.step()
        self.autoencoder.eval()
        
    def select_single(self, state : torch.Tensor) -> int:
        """Select an action for single action environments.

        Args:
            state (torch.Tensor): current State.

        Returns:
            _type_: action/prediction.
        """
        # add batch dimension if needed
        return self.xcs.predict(self.preprocess_state(state).reshape(1,-1))[0]
    
    def learn_single(self, state : torch.Tensor, action : int, reward : float) -> None:
        """Learn for single action environments

        Args:
            state (torch.Tensor): current State.
            action (int): taken action.
            reward (float): actual reward.
        """
        if self.train_mode:
            self.xcs.fit(self.preprocess_state(state), action, reward)
        else:
            warnings.warn("Trying to learn in eval mode.")
        
    def save(self, root_dir : str, filename : str) -> None:
        """Save the model.

        Args:
            root_dir (str): The path where the model should get saved in.
            filename (str): The trial/filename of the model.
        """
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        dir = os.path.join(root_dir, filename)
        self.xcs.save(dir + "_xcsf.pth")
        torch.save(self.autoencoder, dir + "_autoencoder.pth", pickle_module=dill)
        
    def load(self, root_dir : str, filename : str) -> None:
        """Load a model.

        Args:
            root_dir (str): The path where the model is saved.
            filename (str): The trial/filename of the model.
        """
        dir = os.path.join(root_dir, filename)
        self.xcs.load(dir + "_xcsf.pth")
        self.autoencoder = (torch.load(dir + "_autoencoder.pth", weights_only=False))
        
    def internal_params(self) -> dict:
        """Get the internal parameters of the XCS.

        Returns:
            _dict_: Dictionary of the internal parameters.
        """
        return self.xcs.internal_params()
    
    def pset_size(self) -> int:
        """Get the size of the PSet.

        Returns:
            _int_: Size of the PSet.
        """
        return self.xcs.pset_size()
    
    def mset_size(self) -> float:
        """Get the size of the MSet.

        Returns:
            _float_: Size of the MSet.
        """
        return self.xcs.mset_size()
    
    def train_env(self, env : gymnasium.Env, state_function = None, metrics_save_path : str = None, metrics_file_name: str=None,
                  model_file_name: str=None, tensorboard_path : str = None, model_save_path : str = None, notebook : bool = False) -> None:
        """Train the AEXCSF on a given environment.

        Args:
            env (gymnasium.Env): The environment to train on.
            state_function (_type_, optional): A optional function to retrieve the current state of the env. Defaults to None.
            metrics_save_path (str, optional): The path in which the metrics are saved after training. Defaults to None.
            tensorboard_path (str, optional): If you want to use tensorboard define a path. Defaults to None.
            model_save_path (str, optional): Path to the path in which the trained model is saved. Defaults to None.
            notebook (bool, optional): If this function is used inside a Jupyter Notebook. Defaults to False.
        """
        ENV_NAME = env.unwrapped.spec.id.replace("/", "_")
        AUTOENCODER = self.get_autoencoder_type().__name__
        trial_name = "AEXCSFTraining_" + ENV_NAME + "_" + AUTOENCODER + "_" + time.strftime("%Y%m%d-%H%M%S")
        metricTrack = MetricTracker("AEXCSF_training_metrics" if metrics_file_name is None else metrics_file_name, tensorboard_log=tensorboard_path)
        training_pbar = tqdm_notebook(range(self.max_trials), desc='Training AEXCSF...') if notebook else tqdm(range(self.max_trials), desc='Training AEXCSF...')
        self.train()
        for i in range(self.max_trials):
            state, _ = env.reset()
            self.init_trial()
            err = 0
            cnt = 0
            overall_reward = 0
            autoencoder_error = 0
            for _ in range(self.max_steps):
                if state_function:
                    state = state_function(env, state)
                self.init_step()
                action = self.select_action(state)
                next_state, reward, done, trunc, _ = env.step(action)
                done = done or trunc # axn cannot handle truncation, so we treat it as done, but we still learn from it
                self.learn(reward, done) # update the current action set and/or previous action set
                overall_reward += reward
                cnt += 1
                # gymnasium environments dont have a max_payoff, so we use the max_payoff of the environment
                try: # try to get the max_payoff from the environment
                    max_payoff = env.get_wrapper_attr('max_payoff')() # env specific
                except: # if it fails, we use the max_payoff of the environment
                    max_payoff = 1 # env specific
                error = self.get_error(reward, done, max_payoff) # system prediction error, max_payoff is the maximum reward possible
                err += error[0]
                autoencoder_error += error[1]
                self.end_step()
                if done:
                    break
                state = next_state
            self.end_trial()
            metric = {"rollout/ep_len" : cnt, "rollout/ep_rew" : overall_reward,
                      "train/sys_error" : err / cnt, "train/exploration_rate" : self.internal_params()["p_explore"], "train/pop_size" : self.pset_size()}
            metricTrack.add_metrics(metric)
            avg_reward = np.mean((metricTrack.metrics["rollout/ep_rew"])[-100:])
            avg_ep_len = np.mean((metricTrack.metrics["rollout/ep_len"])[-100:])
            mean_metric = { "episode" : i, "rollout/mean100_ep_rew" :  avg_reward, 
                           "rollout/mean100_ep_len" :  avg_ep_len, "train/autoencoder_error" : autoencoder_error / cnt, "train/mset" : self.mset_size()}
            metricTrack.add_metrics(mean_metric)
            metric["rollout/mean100_ep_rew"] = mean_metric["rollout/mean100_ep_rew"]

            postfix = {"episode" : i, "p_xplr" : self.internal_params()["p_explore"], "avg_reward": avg_reward, "ep_reward":overall_reward, "avg_steps": avg_ep_len, "ep_steps" : cnt,"train/pop_size" : self.pset_size()}
            training_pbar.set_postfix(postfix)
            training_pbar.update()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        training_pbar.close()
        self.save(model_save_path, "res" if model_file_name is None else model_file_name) if model_save_path else self.save("./Results_", trial_name)
        metricTrack.save_csv(metrics_save_path, trial_name) if metrics_save_path else metricTrack.save_csv("./Results_", trial_name)
