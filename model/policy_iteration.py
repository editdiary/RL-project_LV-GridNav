import numpy as np
from typing import Tuple, List, Dict
from environment import GridMap
from environment.config import RewardConfig
from agent import Action

class PolicyIteration:
    """정책 반복 알고리즘을 구현한 클래스"""
    def __init__(self, grid_map: GridMap, goals: List[Tuple[int, int]], 
                 gamma: float = 0.9, theta: float = 0.0001):
        """
        Args:
            grid_map (GridMap): 그리드 맵
            goals (List[Tuple[int, int]]): 목표 지점들의 좌표 리스트
            gamma (float): 할인율 (기본값: 0.9)
            theta (float): 수렴 판단 기준값 (기본값: 0.0001)
        """
        self.grid_map = grid_map
        self.goals = goals
        self.gamma = gamma
        self.theta = theta
        
        # 상태-가치 함수 초기화
        self.V = np.zeros((grid_map.padded_size, grid_map.padded_size))
        # 정책 초기화 (모든 상태에서 모든 행동을 동일한 확률로 선택)
        self.policy = np.ones((grid_map.padded_size, grid_map.padded_size, len(Action))) / len(Action)
        
        # 목표 지점의 가치를 최대로 설정
        for goal in goals:
            self.V[goal] = RewardConfig.GOAL_REWARD
    
    def get_next_state(self, state: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """주어진 상태와 행동에 따른 다음 상태를 반환합니다."""
        x, y = state
        
        if action == Action.UP:
            y -= 1
        elif action == Action.RIGHT:
            x += 1
        elif action == Action.DOWN:
            y += 1
        elif action == Action.LEFT:
            x -= 1
            
        # 맵 범위를 벗어나면 원래 상태 반환
        if not (0 <= x < self.grid_map.padded_size and 0 <= y < self.grid_map.padded_size):
            return state
            
        # 벽이나 공사중인 경로면 원래 상태 반환
        if self.grid_map.is_wall(x, y) or self.grid_map.is_construction(x, y):
            return state
            
        return (x, y)
    
    def get_reward(self, state: Tuple[int, int]) -> float:
        """주어진 상태에 대한 보상을 반환합니다."""
        x, y = state
        reward = RewardConfig.STEP_PENALTY
        
        # 목표 도달
        if state in self.goals:
            reward += RewardConfig.GOAL_REWARD
        # 벽이나 공사중인 경로
        elif self.grid_map.is_wall(x, y):
            reward += RewardConfig.WALL_PENALTY
        elif self.grid_map.is_construction(x, y):
            reward += RewardConfig.CONSTRUCTION_PENALTY
            
        return reward
    
    def policy_evaluation(self):
        """정책 평가 단계"""
        while True:
            delta = 0
            # 모든 상태에 대해 반복
            for x in range(self.grid_map.padded_size):
                for y in range(self.grid_map.padded_size):
                    state = (x, y)
                    v = self.V[x, y]
                    
                    # 현재 정책에 따른 행동들의 가치 합산
                    value = 0
                    for action in Action:
                        next_state = self.get_next_state(state, action)
                        next_x, next_y = next_state
                        value += self.policy[x, y, action.value] * (
                            self.get_reward(next_state) + 
                            self.gamma * self.V[next_x, next_y]
                        )
                    
                    self.V[x, y] = value
                    delta = max(delta, abs(v - value))
            
            if delta < self.theta:
                break
    
    def policy_improvement(self) -> bool:
        """정책 개선 단계"""
        policy_stable = True
        
        for x in range(self.grid_map.padded_size):
            for y in range(self.grid_map.padded_size):
                state = (x, y)
                old_action = np.argmax(self.policy[x, y])
                
                # 각 행동의 가치 계산
                action_values = np.zeros(len(Action))
                for action in Action:
                    next_state = self.get_next_state(state, action)
                    next_x, next_y = next_state
                    action_values[action.value] = (
                        self.get_reward(next_state) + 
                        self.gamma * self.V[next_x, next_y]
                    )
                
                # 최적 행동 선택
                best_action = np.argmax(action_values)
                self.policy[x, y] = np.zeros(len(Action))
                self.policy[x, y, best_action] = 1
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def train(self, max_iterations: int = 1000):
        """정책 반복 알고리즘 실행"""
        for i in range(max_iterations):
            self.policy_evaluation()
            if self.policy_improvement():
                print(f"정책이 {i+1}번째 반복에서 수렴했습니다.")
                break
    
    def get_action(self, state: Tuple[int, int]) -> Action:
        """주어진 상태에서 최적 행동을 반환합니다."""
        x, y = state
        return Action(np.argmax(self.policy[x, y])) 