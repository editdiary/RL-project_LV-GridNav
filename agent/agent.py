from enum import Enum
from typing import Tuple, List, Dict
import numpy as np
from environment import GridMap, TileType
from environment.config import RewardConfig

class Action(Enum):
    """Agent가 취할 수 있는 행동을 정의하는 열거형"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Agent:
    """환경 내에서 움직이는 Agent 클래스"""
    def __init__(self, agent_pos: Tuple[int, int]):
        """
        Args:
            x (int): Agent의 초기 x 좌표
            y (int): Agent의 초기 y 좌표
        """
        self.pos = agent_pos
        self.actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        self.visited_positions = set([agent_pos])  # 방문한 위치 기록
        self.last_distance_to_goal = float('inf')  # 목표까지의 이전 거리
    
    def move(self, action: Action) -> Tuple[int, int]:
        """
        Agent를 주어진 행동에 따라 이동시킵니다.
        
        Args:
            action (Action): 수행할 행동
            
        Returns:
            Tuple[int, int]: 이동 후의 새로운 위치
        """
        x, y = self.pos
        
        if action == Action.UP:
            y -= 1
        elif action == Action.RIGHT:
            x += 1
        elif action == Action.DOWN:
            y += 1
        elif action == Action.LEFT:
            x -= 1
            
        self.pos = (x, y)
        return self.pos
    
    @property
    def possible_actions(self) -> list:
        """현재 상태에서 가능한 모든 행동을 반환합니다."""
        return self.actions
    
    def get_state(self, grid_map: GridMap) -> np.ndarray:
        """
        전체 맵의 상태를 반환합니다.
        
        Args:
            grid_map (GridMap): 현재 그리드 맵
            
        Returns:
            np.ndarray: 전체 맵의 상태 배열
        """
        state = np.zeros((grid_map.padded_size, grid_map.padded_size))
        
        for x in range(grid_map.padded_size):
            for y in range(grid_map.padded_size):
                tile_type = grid_map.get_tile_type(x, y)
                if tile_type == TileType.WALL:
                    state[x, y] = 1
                elif tile_type == TileType.CONSTRUCTION:
                    state[x, y] = 2
        
        return state
    
    def calculate_reward(self, grid_map: GridMap, goals: List[Tuple[int, int]], 
                        scenario_type: str) -> float:
        """
        현재 상태에 대한 보상을 계산합니다.
        
        Args:
            grid_map (GridMap): 현재 그리드 맵
            goals (List[Tuple[int, int]]): 목표 지점들의 좌표 리스트
            scenario_type (str): 현재 시나리오 타입
            
        Returns:
            float: 계산된 보상
        """
        reward = 0.0
        x, y = self.pos
        
        # 1. 기본 보상 계산
        # 스텝 패널티
        reward += RewardConfig.STEP_PENALTY
        
        # 벽이나 공사중인 경로에 있는지 확인
        if grid_map.is_wall(x, y):
            reward += RewardConfig.WALL_PENALTY
        elif grid_map.is_construction(x, y):
            reward += RewardConfig.CONSTRUCTION_PENALTY
        
        # 2. 목표 관련 보상
        if scenario_type == "시나리오1":
            # 단일 목표
            goal = goals[0]
            if self.pos == goal:
                reward += RewardConfig.GOAL_REWARD
            else:
                # 목표와의 거리에 따른 보상
                current_distance = abs(x - goal[0]) + abs(y - goal[1])
                if current_distance < self.last_distance_to_goal:
                    reward += RewardConfig.GOAL_PROXIMITY_REWARD
                self.last_distance_to_goal = current_distance
        
        return reward