from gymnasium import Env, spaces
import numpy as np
import torch
from torchvision.transforms import v2
from DatasetMaker.DatasetParser import MinMaxNormalize, ConstantMinMaxScaler

from torchvision.transforms.functional import to_tensor

class AxnEnv(Env):
    """Environment wrapper that uses latents of autoencoder as state representation.

    Args:
        env (Env): The environment to wrap.
        autoencoder (torch.nn.Module): The autoencoder to use for encoding the state.
        is_vae (bool): Whether the autoencoder is a VAE or not.
        x_dim (int, optional): Width of the input image. Defaults to 224.
        y_dim (int, optional): Height of the input image. Defaults to 224.
        channel (int, optional): Number of channels in the input image. Defaults to 3.
        get_state (callable, optional): Function to get the state from the environment. Defaults to None.
        transform (callable, optional): Transform to apply to the state. Defaults to None.
        use_reparameterization (bool, optional): Whether to use reparameterization trick. Defaults to True.
        ignore_sigma (bool, optional): Whether to ignore sigma in VAE. Defaults to False.
        normalize_latent (bool, optional): Whether to normalize the latent space. Defaults to False.
    """
    def __init__(self, env : Env, autoencoder : torch.nn.Module, is_vae : bool, x_dim : int = 224, y_dim : int = 224, channel : int = 3, get_state = None
                 , transform = None, use_reparameterization = True, ignore_sigma = False, normalize_latent = False):
        super(AxnEnv, self).__init__()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vae = is_vae
        self.autoencoder = autoencoder.to(self.device)
        
        self.pre_transform = v2.Compose([
            v2.Resize((x_dim, y_dim)),
            v2.Grayscale(num_output_channels=1) if channel == 1 else v2.Identity(),
            MinMaxNormalize(per_channel=False if channel == 1 else True),
            v2.ToDtype(torch.float32, scale=True),
        ]).to(self.device)
        
        x = torch.rand(1, channel, x_dim, y_dim).to(self.device) # some AE does not accept single batches (Resnet VAE)
        try:
            x = autoencoder.encode(x)
        except:
            raise ValueError("Given encoder does not have the right input size or is missing the encode function.")
        if self.vae:
            # concatenate the mean and variance
            x = autoencoder.reparameterize(x[0].detach(), x[1].detach())
        else:
            x = x.detach()
            x = x[0] # get only the first batch
        print("Encoder output size: ", x.shape)
        if len(x.shape) > 1:
            x = torch.flatten(x)
            print("Flattened encoder output size: ", x.shape)
        x_dim = x.shape[0]
        self.transform = transform
        self.use_reparameterization = use_reparameterization
        self.ignore_sigma = ignore_sigma
        self.normalize_latent = normalize_latent
        self.get_state = get_state
        self.loss_func = torch.nn.MSELoss() if not self.vae else autoencoder.loss_function
        
        # Assuming the autoencoder has an attribute `latent_dim` for the dimension of the latent space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(x_dim,), dtype=np.float32)
        self.action_space = self.env.action_space
        
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
        # add batch dimension if needed
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        # encode the state
        state = state.to(self.device)
        latent = self.autoencoder.encode(state)
        recon = self.autoencoder.decode(self.autoencoder.reparameterize(latent[0], latent[1])) if self.vae else self.autoencoder.decode(latent)
        # calc the loss
        loss = self.loss_func(recon, state) if not self.vae else self.autoencoder.loss_function(state, recon, latent[0], latent[1])
        self.autoencoder_loss = loss.item()
        if self.vae:
            if self.use_reparameterization:
                x = self.autoencoder.reparameterize(latent[0].detach(), latent[1].detach())
            else:
                x = torch.cat((latent[0].detach(), latent[1].detach())) if not self.ignore_sigma else latent[0].detach()
        else:
            x = latent.detach()
            x = x[0] # get only the first batch
        
        # TODO: Try new rolling scaler
        x = MinMaxNormalize(per_channel=False)(x) if self.normalize_latent else x
        x = x.cpu()
        if len(x.shape) > 1:
            x = torch.flatten(x)
        return x.numpy()

    def step(self, action):
        tup = self.env.step(action) 
        trunc = False
        if not isinstance(tup, tuple):
            state = tup
        elif len(tup) == 5:
            state, reward, done, trunc, info = tup
        elif len(tup) == 4:
            state, reward, done, info = tup
        else:
            raise ValueError("The environment should return a tuple of length 3 or 4.")
        state = self.get_state(self.env, state) if self.get_state else state
        latent_state = self.preprocess_state(state)
        return latent_state, reward, done, trunc, info

    def reset(self, seed: int | None = None, options = None) -> tuple:
        state, _ = self.env.reset(seed = seed, options = options) 
        state = state if not self.get_state else self.get_state(self.env)
        latent_state = self.preprocess_state(state)
        return latent_state, {}

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()