#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import argparse
import numpy as np
# np.bool8 = np.bool  # Deprecated, update to bool
import pandas as pd
import torch
import time
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from torch.autograd import grad as torch_grad

import h5py
import os
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard Writer
writer = SummaryWriter()


# In[2]:


class LearningRate:
    """
    Attributes:
        lr (float)
        decay_factor (float)
        training_step (int)
    """
    __instance = None

    def __init__(self):
        if LearningRate.__instance is not None:
            raise Exception("Singleton instantiation called twice")
        else:
            LearningRate.__instance = self
            self.lr = None
            self.decay_factor = None
            self.training_step = 0

    @staticmethod
    def get_instance():
        """Get the singleton instance.

        Returns:
            (LearningRate)
        """
        if LearningRate.__instance is None:
            LearningRate()
        return LearningRate.__instance

    def set_learning_rate(self, lr):
        self.lr = lr

    def get_learning_rate(self):
        return self.lr

    def increment_step(self):
        self.training_step += 1

    def get_step(self):
        return self.training_step

    def set_decay(self, d):
        self.decay_factor = d

    def decay(self):
        if self.lr is None:
            raise ValueError("Learning rate has not been set.")
        self.lr = self.lr * self.decay_factor


# In[3]:
class SerializedBuffer:

    def __init__(self, path=None, device=None):
        if path is not None:
            tmp = torch.load(path)
            self.buffer_size = self._n = tmp['state'].size(0)
            self.device = device

            self.states = tmp['state'].clone().to(self.device)
            self.actions = tmp['action'].clone().to(self.device)
            self.rewards = tmp['reward'].clone().to(self.device)
            self.dones = tmp['done'].clone().to(self.device)
            self.next_states = tmp['next_state'].clone().to(self.device)
        else:
            pass  # For Buffer subclass which initializes empty buffers

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

        # For absorbing states
        self.absorbing_state = np.zeros(state_shape, dtype=np.float32)
        self.zero_action = np.zeros(action_shape, dtype=np.float32)

    def __len__(self):
        return self._n

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def add(self, data, reward, done):
        state, action, next_state = data
        self.append(state, action, reward, done, next_state)

    def addAbsorbing(self):
        self.append(self.absorbing_state, self.zero_action, 0, False, self.absorbing_state)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


# In[4]:


# entropy_weight = 0.001 from openAI/imitation
class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=100, lamb=10, entropy_weight=0.001):
        """

        Args:
            num_inputs:
            hidden_size:
            lamb:
            entropy_weight:
        """
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)
        self.criterion = nn.BCEWithLogitsLoss()
        self.entropy_weight = entropy_weight
        self.optimizer = torch.optim.Adam(self.parameters())
        self.LAMBDA = lamb  # used in gradient penalty
        self.use_cuda = torch.cuda.is_available()

        self.loss = self.ce_loss

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        out = self.linear3(x)
        return out

    def reward(self, x):
        out = self(x)
        probs = torch.sigmoid(out)
        return torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)

    def adjust_adversary_learning_rate(self, lr):
        print("Setting adversary learning rate to: {}".format(lr))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def logit_bernoulli_entropy(self, logits):
        ent = (1. - torch.sigmoid(logits)) * logits - self.logsigmoid(logits)
        return ent

    def logsigmoid(self, a):
        return torch.log(torch.sigmoid(a))

    def logsigmoidminus(self, a):
        return torch.log(1 - torch.sigmoid(a))

    def ce_loss(self, pred_on_learner, pred_on_expert, expert_weights):
        """Binary cross entropy loss.
        We believe this is the loss function the authors to communicate.

        Args:
            pred_on_learner (torch.Tensor): The discriminator's prediction on the learner.
            pred_on_expert (torch.Tensor): The discriminator's prediction on the expert.
            expert_weights (torch.Tensor): The weighting to apply to the expert loss

        Returns:
            (torch.Tensor)
        """
        learner_loss = torch.log(1 - torch.sigmoid(pred_on_learner))
        expert_loss = torch.log(torch.sigmoid(pred_on_expert)) * expert_weights
        return -torch.sum(learner_loss + expert_loss)

    def learn(self, replay_buf, expert_buf, iterations, batch_size=100):
        self.adjust_adversary_learning_rate(LearningRate.get_instance().lr)

        total_losses = []

        for it in range(iterations):
            # Sample replay buffer
            x, y, r, d, u = replay_buf.sample(batch_size)
            state = x
            action = y
            next_state = u

            # Sample expert buffer
            expert_obs, expert_act, expert_weights = expert_buf.get_next_batch(batch_size)
            expert_obs = torch.tensor(expert_obs, dtype=torch.float32, device=device)
            expert_act = torch.tensor(expert_act, dtype=torch.float32, device=device)
            expert_weights = torch.tensor(expert_weights, dtype=torch.float32, device=device).view(-1, 1)

            # Predict
            state_action = torch.cat([state, action], 1).to(device)
            expert_state_action = torch.cat([expert_obs, expert_act], 1).to(device)

            # Align batch sizes
            min_batch_size = min(state_action.size(0), expert_state_action.size(0))
            state_action = state_action[:min_batch_size]
            expert_state_action = expert_state_action[:min_batch_size]
            expert_weights = expert_weights[:min_batch_size]

            fake = self(state_action)
            real = self(expert_state_action)

            # Gradient penalty for regularization.
            gradient_penalty = self._gradient_penalty(expert_state_action, state_action)

            # The main discriminator loss
            main_loss = self.loss(fake, real, expert_weights)

            self.optimizer.zero_grad()

            total_loss = main_loss + gradient_penalty
            total_losses.append(total_loss.item())

            if it == 0 or it == iterations - 1:
                print("Discr Iteration:  {:03} ---- Loss: {:.5f} | Learner Prob: {:.5f} | Expert Prob: {:.5f}".format(
                    it, total_loss.item(), torch.sigmoid(fake[0]).item(), torch.sigmoid(real[0]).item()
                ))
            total_loss.backward()
            self.optimizer.step()

        return total_losses

    def _gradient_penalty(self, real_data, generated_data):
        """
        Compute the gradient penalty for the current update.
        """
        batch_size = min(real_data.size(0), generated_data.size(0))
        device = real_data.device  # Get the device (CPU or GPU)

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, device=device)  # Move alpha to the same device as real_data
        interpolated = alpha * real_data[:batch_size] + (1 - alpha) * generated_data[:batch_size]

        # Ensure gradients are calculated for the interpolated data
        interpolated.requires_grad_(True)

        # Calculate probability of interpolated examples
        prob_interpolated = self(interpolated)  # Forward pass through the discriminator

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                        grad_outputs=torch.ones_like(prob_interpolated),
                        create_graph=True, retain_graph=True)[0]

        # Flatten gradients to compute the L2 norm per example in the batch
        gradients = gradients.view(batch_size, -1)

        # Compute gradient penalty
        gradients_norm = gradients.norm(2, dim=1)  # L2 norm of the gradients
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()  # (||grad||_2 - 1)^2

        return self.LAMBDA * gradient_penalty


# In[5]:


class Actor(nn.Module):
    '''
    \pi(a|s): Given a sequence of states, return a sequence of actions
    '''
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        x = x.to(device).float()
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

    def act(self, x: Tensor) -> Tensor:
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self(x)


# In[6]:


class Critic(nn.Module):
    '''
    Given a sequence of (state,action) pairs, return (Q1,Q2)
    '''
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # Q1 network architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 network architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Concatenate the state (x) and action (y)
        xy = torch.cat([x, y], dim=1)

        # Q1 computation
        x1 = torch.relu(self.l1(xy))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)

        # Q2 computation
        x2 = torch.relu(self.l4(xy))
        x2 = torch.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, x: Tensor, y: Tensor) -> Tensor:
        # Compute Q1 value alone (without Q2)
        xy = torch.cat([x, y], dim=1)
        x1 = torch.relu(self.l1(xy))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


# In[7]:


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, actor_clipping, decay_steps):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.decay_steps = decay_steps
        self.actor_grad_clipping = actor_clipping
        self.max_action = max_action
        self.actor_steps = 0
        self.critic_steps = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def adjust_critic_learning_rate(self, lr):
        print("Setting critic learning rate to: {}".format(lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def adjust_actor_learning_rate(self, lr):
        print("Setting actor learning rate to: {}".format(lr))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def reward(self, discriminator, states, actions):
        states_actions = torch.cat([states, actions], 1).to(device)
        return discriminator.reward(states_actions)

    def train(self, discriminator, replay_buf, iterations, batch_size=100, discount=0.8, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        lr_tracker = LearningRate.get_instance()
        lr = lr_tracker.lr

        self.adjust_actor_learning_rate(lr)
        self.adjust_critic_learning_rate(lr)

        actor_losses = []
        critic_losses = []

        for iteration in range(iterations):
            # Sample replay buffer
            x, y, r, d, u = replay_buf.sample(batch_size)
            state = x
            action = y
            next_state = u
            reward = self.reward(discriminator, state, action)

            # Align batch sizes
            min_batch_size = min(state.size(0), action.size(0), next_state.size(0))
            state = state[:min_batch_size]
            action = action[:min_batch_size]
            next_state = next_state[:min_batch_size]
            reward = reward[:min_batch_size]

            # Select action according to policy and add clipped noise
            # Generate clipped noise
            noise = torch.randn_like(action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)

            # Add noise to the action selected by the target actor network
            next_action = self.actor_target(next_state) + noise
            # Clamp the action to the valid action space defined by the max action
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # detach() is used to prevent backpropagation to the target network
            target_Q = reward + (discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            if iteration == 0 or iteration == iterations - 1:
                print("Critic Iteration: {:3} ---- Loss: {:.5f}".format(iteration, critic_loss.item()))
            critic_losses.append(critic_loss.item())
            writer.add_scalar('Loss/Critic', critic_loss.item(), lr_tracker.training_step)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if iteration % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                if iteration == 0 or iteration == iterations - 1 or iteration == iterations - 2:
                    print("Actor Iteration:  {:3} ---- Loss: {:.5f}".format(iteration, actor_loss.item()))
                actor_losses.append(actor_loss.item())
                writer.add_scalar('Loss/Actor', actor_loss.item(), lr_tracker.training_step)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                # Clip, like the paper
                clip_grad_value_(self.actor.parameters(), self.actor_grad_clipping)

                self.actor_optimizer.step()
                lr_tracker.training_step += 1
                step = lr_tracker.training_step

                if step != 0 and step % self.decay_steps == 0:
                    print("Decaying learning rate at step: {}".format(step))
                    lr_tracker.decay()

                    self.adjust_actor_learning_rate(lr_tracker.lr)
                    self.adjust_critic_learning_rate(lr_tracker.lr)

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return actor_losses, critic_losses

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


# In[8]:


def store_results(evaluations, number_of_timesteps, actor_losses, critic_losses):
    """Store the results of a run.

    Args:
        evaluations:
        number_of_timesteps (int):
        actor_losses (list):
        critic_losses (list):

    Returns:
        None
    """

    df = pd.DataFrame.from_records(evaluations)
    number_of_trajectories = len(evaluations[0]) - 1
    columns = ["reward_trajectory_{}".format(i + 1) for i in range(number_of_trajectories)]
    columns.append("timestep")
    df.columns = columns

    # Add actor and critic losses to dataframe
    df['actor_loss'] = pd.Series(actor_losses)
    df['critic_loss'] = pd.Series(critic_losses)

    timestamp = time.time()
    results_fname = 'DAC_{}_tsteps_{}_results.csv'.format(number_of_timesteps, timestamp)
    df.to_csv(str(results_fname), index=False)


# In[9]:


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, time_step, evaluation_trajectories=6):
    """

    Args:
        env: The environment being trained on.
        policy:    The policy being evaluated
        time_step (int): The number of time steps the policy has been trained for.
        evaluation_trajectories (int): The number of trajectories on which to evaluate.

    Returns:
        (list)    - The time_step, followed by all the rewards.
    """
    rewards = []
    for _ in range(evaluation_trajectories):
        r = 0
        obs = env.reset()  # Adjusted for gym version
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, info = env.step(action)  # Adjusted for gym version
            r += reward
        rewards.append(r)
    print("Average reward at timestep {}: {}".format(time_step, np.mean(rewards)))

    rewards.append(time_step)
    return rewards


# In[10]:


env = gym.make('InvertedPendulum-v2')


# In[11]:


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


# In[12]:


trajectory_length = 200
batch_size = 500
num_steps = 20000


# In[13]:


lr = LearningRate.get_instance()
lr.lr = 10 ** (-3)
lr.decay_factor = 0.5


# In[14]:


def load_dataset(path, limit_trajs=None, data_subsamp_freq=1):
    tmp = torch.load(path)
    full_dset_size = tmp['state'].size(0)

    # Dynamically get state and action dimensions
    state_dim = tmp['state'].size(1)
    action_dim = tmp['action'].size(1)

    # Determine steps per trajectory
    steps_per_traj = 200  # Adjust this based on your data
    num_trajs = full_dset_size // steps_per_traj

    # Adjust dataset size to be divisible by steps_per_traj
    dset_size = num_trajs * steps_per_traj

    # Reshape data
    states = tmp['state'][:dset_size].reshape(num_trajs, steps_per_traj, state_dim).clone()
    actions = tmp['action'][:dset_size].reshape(num_trajs, steps_per_traj, action_dim).clone()
    rewards = tmp['reward'][:dset_size].reshape(num_trajs, steps_per_traj, 1).clone()
    dones = tmp['done'][:dset_size].reshape(num_trajs, steps_per_traj, 1).clone()
    next_states = tmp['next_state'][:dset_size].reshape(num_trajs, steps_per_traj, state_dim).clone()
    return states, actions, rewards

class Dset(object):
    def __init__(self, obs, acs, num_traj, absorbing_state, absorbing_action):
        self.obs = obs
        self.acs = acs
        self.num_traj = num_traj
        assert len(self.obs) == len(self.acs)
        assert self.num_traj > 0
        self.steps_per_traj = int(len(self.obs) / num_traj)

        self.absorbing_state = absorbing_state
        self.absorbing_action = absorbing_action

    def get_next_batch(self, batch_size):
        assert batch_size <= len(self.obs)
        num_samples_per_traj = max(1, batch_size // self.num_traj)

        if num_samples_per_traj * self.num_traj != batch_size:
            batch_size = num_samples_per_traj * self.num_traj  # Adjust to the closest valid batch size

        N = self.steps_per_traj / num_samples_per_traj  # This is the importance weight for
        j = num_samples_per_traj
        num_samples_per_traj = num_samples_per_traj - 1  # make room for absorbing

        obs = None
        acs = None
        weights = [1 for i in range(batch_size)]
        while j <= batch_size:
            weights[j - 1] = 1.0 / N
            j = j + num_samples_per_traj + 1

        for i in range(self.num_traj):
            indicies = np.sort(
                np.random.choice(range(self.steps_per_traj * i, self.steps_per_traj * (i + 1)), num_samples_per_traj,
                                 replace=False))
            if obs is None:
                obs = np.concatenate((self.obs[indicies, :], self.absorbing_state), axis=0)

            else:
                obs = np.concatenate((obs, self.obs[indicies, :], self.absorbing_state), axis=0)

            if acs is None:
                acs = np.concatenate((self.acs[indicies, :], self.absorbing_action), axis=0)
            else:
                acs = np.concatenate((acs, self.acs[indicies, :], self.absorbing_action), axis=0)

        return obs, acs, weights

class Mujoco_Dset(object):
    def __init__(self, env, expert_path, traj_limitation=-1):
        obs, acs, rets = load_dataset(expert_path, traj_limitation)
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        self.rets = rets.sum(axis=1)
        self.avg_ret = sum(self.rets) / len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        assert len(self.obs) == len(self.acs)
        self.num_traj = len(rets)
        self.num_transition = len(self.obs)

        absorbing_state = np.zeros((1,env.observation_space.shape[0]), dtype=np.float32)  # Adjusted shape
        zero_action = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)  # Ensuring correct shape
        self.dset = Dset(self.obs, self.acs, self.num_traj, absorbing_state, zero_action)
        self.log_info()

    def log_info(self):
        print("Total trajs: %d" % self.num_traj)
        print("Total transitions: %d" % self.num_transition)
        print("Average returns: %f" % self.avg_ret)
        print("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size):
        return self.dset.get_next_batch(batch_size)

# In[17]:

expert_buffer = Mujoco_Dset(env, 'size1000000_std0.01_prand0.0.pth', 20000)
state_shape = env.observation_space.shape
action_shape = env.action_space.shape

actor_replay_buffer = Buffer(buffer_size=num_steps, state_shape=state_shape, action_shape=action_shape, device=device)

# In[ ]:

# TD3(state_dim, action_dim, max_action, actor_clipping, decay_steps)
td3_policy = TD3(state_dim, action_dim, max_action, 40, 10 ** 5)

# Input dim = state_dim + action_dim
discriminator = Discriminator(state_dim + action_dim).to(device)

# For storing temporary evaluations
evaluations = [evaluate_policy(env, td3_policy, 0)]
evaluate_every = 1000
steps_since_eval = 0

# In[ ]:

env.reset()

# In[ ]:

actor_losses = []
critic_losses = []

while len(actor_replay_buffer) < num_steps:
    print("\nCurrent step: {}".format(len(actor_replay_buffer)))
    current_state = env.reset()  # Adjusted for gym version
    # Sample from policy; maybe we don't reset the environment -> since this may bias the policy toward initial observations
    for j in range(trajectory_length):
        action = td3_policy.select_action(np.array(current_state))
        obs, reward, done, info = env.step(action)  # Adjusted for gym version

        if done:
            actor_replay_buffer.addAbsorbing()
            current_state = env.reset()
        else:
            actor_replay_buffer.add((current_state, action, obs), reward, done)
            current_state = obs

    # TODO: LOSS FUNCTION
    discriminator_losses = discriminator.learn(actor_replay_buffer, expert_buffer, trajectory_length, batch_size)
    td3_actor_losses, td3_critic_losses = td3_policy.train(discriminator, actor_replay_buffer, trajectory_length, batch_size)

    actor_losses.extend(td3_actor_losses)
    critic_losses.extend(td3_critic_losses)

    if steps_since_eval >= evaluate_every:
        steps_since_eval = 0

        evaluation = evaluate_policy(env, td3_policy, len(actor_replay_buffer))
        evaluations.append(evaluation)

    steps_since_eval += trajectory_length

last_evaluation = evaluate_policy(env, td3_policy, len(actor_replay_buffer))
evaluations.append(last_evaluation)

store_results(evaluations, len(actor_replay_buffer), actor_losses, critic_losses)

# Close the tensorboard writer
writer.close()


# Record a 100 steps video of the trained agent
frames = []

# Create a new environment with render_mode set to 'rgb_array'
video_env = gym.make('InvertedPendulum-v2')

state = video_env.reset()
done = False
steps = 0

while not done and steps < 100:
    action = td3_policy.select_action(np.array(state))
    state, reward, done, info = video_env.step(action)
    # Render the frame and append to frames list
    frame = video_env.render(mode='rgb_array')
    frames.append(frame)
    steps += 1

video_env.close()

# Save frames as a video
video_filename = 'trained_agent_video.mp4'
imageio.mimsave(video_filename, frames, fps=30)