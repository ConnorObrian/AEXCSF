
# cite:
# @misc{pytorch_minimal_ppo,
#     author = {Barhate, Nikhil},
#     title = {Minimal PyTorch Implementation of Proximal Policy Optimization},
#     year = {2021},
#     publisher = {GitHub},
#     journal = {GitHub repository},
#     howpublished = {\url{https://github.com/nikhilbarhate99/PPO-PyTorch}},
# }

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from torchshape import tensorshape

################################## set device ##################################

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0')



################################## PPO Policy ##################################


class RolloutBuffer:
    """A class to store the rollout buffer information."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """
    A class to implement the Actor-Critic network.
    Args:
        state_dim (tuple): shape of the state
        action_dim (int): number of actions
        has_continuous_action_space (bool): whether the action space is continuous or discrete
        action_std_init (float): initial standard deviation of the action
        neural_net (str): type of neural network to use (mlp, cnn)
        """
    def __init__(self, state_dim : tuple[int, int, int], action_dim : int, has_continuous_action_space : bool = False, action_std_init : float = 0.6, neural_net : str = "cnn"):
        super(ActorCritic, self).__init__()

        flatten_shape = tensorshape(nn.Flatten(), tensorshape(nn.Conv2d(16, 32, 4, stride=2), tensorshape(nn.Conv2d(3, 16, 8, stride=4), (1, 3, *state_dim[1:]))))
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        if neural_net != "cnn" and neural_net != "mlp":
            raise ValueError("Invalid network type. Choose between 'cnn' and 'mlp")

        # actor
        if has_continuous_action_space:
            if neural_net == "mlp":
                self.actor = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(state_dim[1] * state_dim[2] * state_dim[0], 64),
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                                nn.Linear(64, action_dim),
                                nn.Tanh()
                            )
            elif neural_net == "cnn":
                self.actor = nn.Sequential(
                                nn.Conv2d(state_dim[0], 16, 8, stride=4),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, 4, stride=2),
                                nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(flatten_shape[1], 256),
                                nn.ReLU(),
                                nn.Linear(256, action_dim),
                                nn.Tanh()
                            )
        else:
            if neural_net == "mlp":
                self.actor = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(state_dim[1] * state_dim[2] * state_dim[0], 64),
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                                nn.Linear(64, action_dim),
                                nn.Softmax(dim=-1)
                            )
            elif neural_net == "cnn":
                self.actor = nn.Sequential(
                                nn.Conv2d(state_dim[0], 16, 8, stride=4),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, 4, stride=2),
                                nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(flatten_shape[1], 256),
                                nn.ReLU(),
                                nn.Linear(256, action_dim),
                                nn.Softmax(dim=-1)
                )

        
        # critic
        if neural_net == "mlp":
            self.critic = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(state_dim[1] * state_dim[2] * state_dim[0], 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        elif neural_net == "cnn":
            self.critic = nn.Sequential(
                        nn.Conv2d(state_dim[0], 16, 8, stride=4),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, 4, stride=2),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(flatten_shape[1], 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                    )
        
    def set_action_std(self, new_action_std):
        """
        A function to set the standard deviation of the action."""
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        """
        A function to get the action, log probability of the action, and the state value."""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):
        """
        A function to evaluate the state and action."""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    """A class to implement the Proximal Policy Optimization algorithm.
    Args:
        state_dim (tuple): shape of the state
        action_dim (int): number of actions
        lr_actor (float): learning rate for the actor network
        lr_critic (float): learning rate for the critic network
        gamma (float): discount factor
        K_epochs (int): update policy for K epochs
        eps_clip (float): clip parameter for PPO
        has_continuous_action_space (bool): whether the action space is continuous or discrete
        action_std_init (float): initial standard deviation of the action
        neural_net (str): type of neural network to use (mlp, cnn)
    """
    def __init__(self, state_dim : tuple[int, int, int], action_dim : int, lr_actor : float = 0.0003, lr_critic : float = 0.001, gamma : float = 0.99, K_epochs : int = 40, eps_clip : float = 0.2, is_continious : bool = False, action_std_init : float = 0.6, neural_net : str = "cnn"):

        self.has_continuous_action_space = is_continious

        if is_continious:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.state_dim = state_dim
        
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, is_continious, action_std_init, neural_net).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, is_continious, action_std_init, neural_net).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.loss = 0


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device) if not self.state_dim[0] == 1 else torch.squeeze(torch.stack(self.buffer.states, dim=0), 1).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        
        losses = []
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            losses.append(loss.mean().item())
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        self.loss = sum(losses) / len(losses)
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        