import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
the joint order in IsaacGym is:
[
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
]

but the joint order in PyBullet is:
[
    "FR_hip_joint", "FR_upper_joint", "FR_lower_joint",
    "FL_hip_joint", "FL_upper_joint", "FL_lower_joint",
    "RR_hip_joint", "RR_upper_joint", "RR_lower_joint",
    "RL_hip_joint", "RL_upper_joint", "RL_lower_joint",
]

so we need to reorder the joint angles from Pybullet to IsaacGym
'''

# state_action = np.load('state_action.npz')
# state_action = dict(state_action)

# for key in state_action.keys():
#     print(key, state_action[key].shape)

class RefDataset(Dataset):
    def __init__(self, npz_file):
        super().__init__()

        self.data = np.load(npz_file)
        self.data = dict(self.data)

        # reorder the joint angles from Pybullet to IsaacGym
        idx_reodrder = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8] 
        self.base_lin_vel = torch.tensor(self.data['base_lin_vel'], dtype=torch.float32)
        self.base_ang_vel = torch.tensor(self.data['base_ang_vel'], dtype=torch.float32)
        self.projected_gravity = torch.tensor(self.data['projected_gravity'], dtype=torch.float32)
        self.command = torch.tensor(self.data['command'], dtype=torch.float32)
        self.dof_pos = torch.tensor(self.data['dof_pos'][:, idx_reodrder], dtype=torch.float32)
        self.dof_vel = torch.tensor(self.data['dof_vel'][:, idx_reodrder], dtype=torch.float32)
        self.last_action = torch.tensor(self.data['last_action'][:, idx_reodrder], dtype=torch.float32)

        self.action = torch.tensor(self.data['action'], dtype=torch.float32)

    def __len__(self):
        return self.base_lin_vel.shape[0]
    
    def __getitem__(self, idx):
        state = torch.cat([
            self.base_lin_vel[idx],
            self.base_ang_vel[idx],
            self.projected_gravity[idx],
            self.command[idx],
            self.dof_pos[idx],
            self.dof_vel[idx],
            self.last_action[idx]
        ])

        action = self.action[idx]

        return {
            'state': state,
            'action': action
        }
    
def collate_fn(batch):
    state = torch.stack([sample['state'] for sample in batch])
    action = torch.stack([sample['action'] for sample in batch])

    return {
        'state': state,
        'action': action
    }
    
if __name__ == '__main__':
    dataset = RefDataset('state_action.npz')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch['state'].shape, batch['action'].shape)
        break