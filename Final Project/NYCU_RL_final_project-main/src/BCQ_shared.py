import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor_and_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor_and_Critic, self).__init__()
		self.shared_l1 = nn.Linear(state_dim + action_dim, 400)
		self.actor_l1 = nn.Linear(400, 300)
		self.actor_l2 = nn.Linear(300, action_dim)
		
		self.critic_l1 = nn.Linear(400, 300)
		self.critic_l2 = nn.Linear(300, 1)

		self.critic_l3 = nn.Linear(400, 300)
		self.critic_l4 = nn.Linear(300, 1)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.shared_l1(torch.cat([state, action], 1)))
		a = F.relu(self.actor_l1(a))
		a = self.phi * self.max_action * torch.tanh(self.actor_l2(a))
        
		q1 = F.relu(self.shared_l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.critic_l1(q1))
		q1 = self.critic_l2(q1)

		q2 = F.relu(self.shared_l1(torch.cat([state, action], 1)))
		q2 = F.relu(self.critic_l3(q2))
		q2 = self.critic_l4(q2)

		return (a + action).clamp(-self.max_action, self.max_action), q1, q2
	
	def perturb_action(self, state, action):
		a = F.relu(self.shared_l1(torch.cat([state, action], 1)))
		a = F.relu(self.actor_l1(a))
		a = self.phi * self.max_action * torch.tanh(self.actor_l2(a))
		
		return (a + action).clamp(-self.max_action, self.max_action)
	
	def get_value(self, state, action):
		q1 = F.relu(self.shared_l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.critic_l1(q1))
		q1 = self.critic_l2(q1)

		q2 = F.relu(self.shared_l1(torch.cat([state, action], 1)))
		q2 = F.relu(self.critic_l3(q2))
		q2 = self.critic_l4(q2)
		
		return q1, q2
	
	def q1(self, state, action):
		q1 = F.relu(self.shared_l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.critic_l1(q1))
		q1 = self.critic_l2(q1)
		return q1

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.model = Actor_and_Critic(state_dim, action_dim, max_action, phi).to(device)
		self.model_target = copy.deepcopy(self.model)
		self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.model.perturb_action(state, self.vae.decode(state))
			q1 = self.model.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()


			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.model_target.get_value(next_state, self.model_target.perturb_action(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			current_Q1, current_Q2 = self.model.get_value(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.model.perturb_action(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.model.q1(state, perturbed_actions).mean()
		 	 
			loss = critic_loss + actor_loss
			
			self.model_optimizer.zero_grad()
			loss.backward()
			self.model_optimizer.step()

			# Update Target Networks 
			for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
