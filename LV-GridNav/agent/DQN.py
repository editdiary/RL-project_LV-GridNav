import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, action_size)

        # Dropout 추가
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.dropout(x)
        x = torch.relu(self.l2(x))
        x = self.dropout(x)
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data], dtype=np.float32)
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data], dtype=np.float32)
        return state, action, reward, next_state, done