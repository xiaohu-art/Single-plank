import torch
import torch.nn as nn
import numpy as np

from legged_gym.reference.dataset import RefDataset, collate_fn

class CosineScheduler:
    def __init__(self, start_iter, end_iter, start_val=0, peak_val=0.5):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_val = start_val
        self.peak_val = peak_val

    def __call__(self, iter):
        if iter < self.start_iter:
            return self.start_val
        elif iter >= self.end_iter:
            return 0
        else:
            rp = (iter - self.start_iter) / (self.end_iter - self.start_iter)
            value = 0.5 * (1 - np.cos(np.pi * rp)) * (self.peak_val - self.start_val) + self.start_val
            return value

class Discriminator(nn.Module):
    def __init__(self, num_envs, num_env_steps, num_mini_batch, device,
                        lr=1e-4, data_file='./legged_gym/reference/state_action.npz'
                ):
        super().__init__()

        self.device = device

        self._init_data(data_file)

        self.batch_size = num_envs * num_env_steps
        self.num_mini_batches = num_mini_batch
        self.mini_batch_size = self.batch_size // num_mini_batch

        self.dataset = RefDataset(data_file)
        self.data_len = len(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        self.state_dim, self.action_dim = self.dataset[0]['state'].shape[0], self.dataset[0]['action'].shape[0]
        input_dim = self.state_dim + self.action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ).to(self.device)

        self.activation = nn.Tanh()

        self.eta = torch.tensor(0.3, requires_grad=False, device=self.device)
        self.lambda_gp = torch.tensor(10.0, requires_grad=False, device=self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()

    def genarator(self):
        for _ in range(self.num_mini_batches):
            sample_idx = np.random.randint(0, self.data_len, self.mini_batch_size)
            yield self.state[sample_idx].to(self.device), self.action[sample_idx].to(self.device)
            
    def forward(self, x):
        x = self.net(x)
        x = self.eta * x
        x = self.activation(x)
        return x
    
    def init_coeffi_scheduler(self, start_iter, end_iter):
        self.coeffi_scheduler = CosineScheduler(start_iter, end_iter)

    def coeffi(self, it):
        return self.coeffi_scheduler(it)
        
    def imitation_reward(self, state, action):
        with torch.no_grad():
            state = state.to(self.device)
            action = action.to(self.device)
            x = torch.cat([state, action], dim=1)
            reward = torch.clamp_max(self.net(x), 0.)
            reward = torch.exp(reward)
        return reward.view(-1)
    
    def expert_loss(self, expert_transitions):
        return -torch.mean(self.forward(expert_transitions))
    
    def agent_loss(self, agent_transitions):
        return torch.mean(self.forward(agent_transitions))
    
    def gradient_penalty(self, expert_transitions, agent_transitions):
        alpha = torch.rand(expert_transitions.shape[0], 1, device=self.device)
        interpolates = alpha * expert_transitions + (1 - alpha) * agent_transitions
        interpolates.requires_grad = True
        disc_interpolates = self.forward(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(self.batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def _init_data(self, data_file):
        self.data = np.load(data_file)
        self.data = dict(self.data)

        idx_reodrder = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8] 
        self.base_lin_vel = torch.tensor(self.data['base_lin_vel'], dtype=torch.float32)
        self.base_ang_vel = torch.tensor(self.data['base_ang_vel'], dtype=torch.float32)
        self.projected_gravity = torch.tensor(self.data['projected_gravity'], dtype=torch.float32)
        self.command = torch.tensor(self.data['command'], dtype=torch.float32)
        self.dof_pos = torch.tensor(self.data['dof_pos'][:, idx_reodrder], dtype=torch.float32)
        self.dof_vel = torch.tensor(self.data['dof_vel'][:, idx_reodrder], dtype=torch.float32)
        self.last_action = torch.tensor(self.data['last_action'][:, idx_reodrder], dtype=torch.float32)

        self.state = torch.cat([
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.command,
            self.dof_pos,
            self.dof_vel,
            self.last_action
        ], dim=1)
        self.action = torch.tensor(self.data['action'], dtype=torch.float32)