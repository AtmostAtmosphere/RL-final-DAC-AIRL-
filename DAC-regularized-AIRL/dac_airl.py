import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.network import AIRLDiscrim
from .DAC_test import Discriminator, Actor, Critic, Mujoco_Dset

device = torch.device("cpu")

class AIRL_DAC(PPO):
    # -------------------------
    # change env, path, alpha
    # -------------------------
    def __init__(self, env, path, alpha, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )
        

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc_airl = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.disc_dac = Discriminator(state_shape[0] + action_shape[0]).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc_airl.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.alpha = alpha

        self.buffer_exp_dac = Mujoco_Dset(
            env = env, 
            expert_path = path, 
            traj_limitation = 20000
        )

    def update(self, writer):
        self.learning_steps += 1
        loss_airl = 0
        for i in range(self.epoch_disc-1):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, action, _, dones, log_pis, next_states = \
                self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.sample(self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(
                    states_exp, actions_exp)
                

            loss_dac = self.disc_dac.learn(
                states, action, 
                expert_buf = self.buffer_exp_dac,
                batch_size = self.batch_size
            )

            # Update discriminator.
            loss_airl = self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, writer
            )
            
            loss = self.alpha * loss_dac + (1 - self.alpha) * loss_airl

            self.optim_disc.zero_grad()
            loss.backward()
            self.optim_disc.step()

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc_airl.calculate_reward(
            states, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc_airl(states, dones, log_pis, next_states)
        logits_exp = self.disc_airl(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_airl = loss_pi + loss_exp

        '''loss_dac = self.disc_dac.learn(
            replay_buf=self.buffer,
            expert_buf=self.buffer_exp_dac,
            iterations = 10,
            batch_size = self.batch_size
        )'''

        return loss_airl
    

        '''if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)'''
        
