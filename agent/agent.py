from collections import defaultdict
from environment.gridworld import Action
from typing import Tuple
import numpy as np
from utils.utils import greedy_probs

import torch
import torch.nn as nn
import torch.optim as optim
from agent.DQN import QNet, ReplayBuffer


class QLearningAgent:
    """환경 내에서 움직이는 Agent 클래스"""
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        self.random_actions = {
            Action.UP: 0.25,
            Action.RIGHT: 0.25,
            Action.DOWN: 0.25,
            Action.LEFT: 0.25,
        }
        self.Q = defaultdict(lambda: 0)
        self.pi = defaultdict(lambda: self.random_actions.copy())
        self.b = defaultdict(lambda: self.random_actions.copy())    # 행동 정책

    def get_action(self, state: Tuple[int, int]) -> Action:
        action_probs = self.b[state]
        actions = list(Action)
        probs = [action_probs[action] for action in actions]
        return np.random.choice(actions, p=probs)
    
    def update_policy(self, state: Tuple[int, int], action: Action, next_state: Tuple[int, int], reward: float, done: bool):
        if done:    # 목표 도달
            next_q_max = 0
        else:       # 그 외에는 다음 상태에서 Q 함수의 최댓값 계산
            next_qs = [self.Q[(next_state, a)] for a in Action]
            next_q_max = max(next_qs)
        
        # Q 함수 갱신
        target = reward + self.gamma * next_q_max
        self.Q[(state, action)] += (target - self.Q[(state, action)]) * self.alpha

        # 행동 정책과 대상 정책 갱신
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, epsilon=self.epsilon)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 1.0          # 초기 epsilon 값
        self.epsilon_min = 0.01     # 최소 epsilon 값
        self.epsilon_decay = 0.992  # epsilon 감소 속도
        self.buffer_size = 20000  # 경험 재생 버퍼 크기
        self.batch_size = 64      # 미니배치 크기
        self.action_size = action_size
        self.state_size = state_size
        
        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Q-Network
        self.qnet = QNet(self.state_size, self.action_size)
        self.qnet_target = QNet(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                qs = self.qnet(state_tensor)
                return qs.argmax().item()
        
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        action = torch.from_numpy(action).long()
        reward = torch.from_numpy(reward).float()
        done = torch.from_numpy(done).float()
        
        # 현재 Q 값 계산
        qs = self.qnet(state)
        q = qs.gather(1, action.unsqueeze(1)).squeeze(1)

        # 타겟 Q 값 계산
        with torch.no_grad():
            next_qs = self.qnet_target(next_state)
            next_q = next_qs.max(1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        # 손실 계산 및 역전파
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
