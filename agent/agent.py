from collections import defaultdict
from environment.gridworld import Action
from typing import Tuple
import numpy as np
from utils.utils import greedy_probs

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