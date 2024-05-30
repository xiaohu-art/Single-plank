import torch
import torch.nn as nn

from legged_gym.reference.dataset import RefDataset, collate_fn

class Discriminator(nn.Module):
    def __init__(self, num_envs, num_mini_batch, num_env_steps, device,
                        state_dim, action_dim, 
                        lr=1e-3, data_file='./legged_gym/reference/state_action.npz'
                ):
        super().__init__()

        self.device = device

        self.batch_size = num_envs * num_env_steps // num_mini_batch

        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        self.dataset = RefDataset(data_file)
        self.data_len = len(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()

    def genarator(self):
        for data in self.dataloader:
            yield data

    def forward(self, x):
        return self.net(x)