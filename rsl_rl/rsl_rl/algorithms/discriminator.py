import torch
import torch.nn as nn

from legged_gym.reference.dataset import RefDataset, collate_fn

class Discriminator(nn.Module):
    def __init__(self, num_envs, num_env_steps, num_mini_batch, device,
                        lr=1e-3, data_file='./legged_gym/reference/state_action.npz'
                ):
        super().__init__()

        self.device = device

        self.batch_size = num_envs * num_env_steps // num_mini_batch

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
        for data in self.dataloader:
            yield data

    def forward(self, x):
        x = self.net(x)
        x = self.eta * x
        x = self.activation(x)
        return x
    
    def imitation_reward(self, state, action):
        with torch.no_grad():
            state = state.to(self.device)
            action = action.to(self.device)
            x = torch.cat([state, action], dim=1)
            reward = self.net(x)
            reward = torch.exp(reward)
        return reward.view(-1)
    
    def expert_loss(self, expert_transitions):
        return -torch.mean(self.forward(expert_transitions))
    
    def agent_loss(self, agent_transitions):
        return torch.mean(self.forward(agent_transitions))
    
    def gradient_penalty(self, expert_transitions, agent_transitions):
        alpha = torch.rand(self.batch_size, 1, device=self.device)
        interpolates = alpha * expert_transitions + (1 - alpha) * agent_transitions
        interpolates.requires_grad = True
        disc_interpolates = self.forward(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(self.batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty