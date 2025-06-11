from enum import Enum
from environment.grid_map import GridMap
from scenarios.scenarios import ScenarioFactory
from typing import Tuple
import numpy as np
from environment.grid_map import TileType
from environment.config import RewardConfig
from collections import defaultdict

class Action(Enum):
    """Agent가 취할 수 있는 행동을 정의하는 열거형"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class GridWorld:
    def __init__(
        self,
        size: int = 7,
        padding: int = 2,
        seed: int = None,
        num_walls: int = 5,
        num_construction: int = 5,
        scenario_type: str = "시나리오1",
        agent_pos: Tuple[int, int] = None,
    ):
        # 1. 환경 초기화
        self.grid_map = GridMap(size, padding, seed)
    
        # 2. 벽과 공사중인 경로 추가
        self.grid_map.add_random_walls(num_walls=num_walls)
        self.grid_map.add_random_construction(num_construction=num_construction)

        # 3. 시나리오 생성
        self.scenario = ScenarioFactory.create_scenario(
            scenario_type=scenario_type,
            grid_map=self.grid_map,
            agent_pos=self.grid_map.agent_pos,
        )
        self.scenario.generate_goals()
        # 생성된 목표 지점의 TileType을 GOAL로 변경
        for goal in self.scenario._goals:
            self.grid_map.tiles[goal[0]][goal[1]].type = TileType.GOAL

        # 4. reward map 생성
        self.reward_map = np.zeros((self.grid_map.padded_size, self.grid_map.padded_size))
        for x in range(self.grid_map.padded_size):
            for y in range(self.grid_map.padded_size):
                self.reward_map[x, y] = self._get_reward(x, y)

        # 5. 기타 속성 초기화
        self.goals = self.scenario._goals
        self.agent_pos = self.grid_map.agent_pos
        self.scenario_type = scenario_type

        # 재방문 영역 초기화
        self.visited_states = defaultdict(int)
        self.visited_states[self.grid_map.agent_pos] = 1    # 초기 상태 방문 횟수 1로 설정

        # 행동 가능 영역
        self.action_space = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

    def _get_reward(self, x: int, y: int) -> float:
        """주어진 위치의 보상을 반환합니다."""
        # 기본 보상 초기화
        reward = 0
        
        # 목표 도달
        if self.grid_map.get_tile_type(x, y) == TileType.GOAL:
            reward = RewardConfig.GOAL_REWARD
        # 벽이나 공사중인 경로
        elif self.grid_map.is_wall(x, y):
            reward = RewardConfig.WALL_PENALTY
        elif self.grid_map.is_construction(x, y):
            reward = RewardConfig.CONSTRUCTION_PENALTY
            
        return reward

    @property
    def possible_actions(self) -> list:
        """현재 상태에서 가능한 모든 행동을 반환합니다."""
        return self.action_space
        
    def next_state(self, state: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """
        Agent를 주어진 행동에 따라 이동시킵니다.
        
        Args:
            state (Tuple[int, int]): 현재 상태 (x, y)
            action (Action): 수행할 행동
            
        Returns:
            Tuple[int, int]: 이동 후의 새로운 위치
        """
        action_move_map = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        move = action_move_map[action.value]
        next_state = (state[0] + move[0], state[1] + move[1])
        nx, ny = next_state

        if nx < 0 or nx >= self.grid_map.padded_size or ny < 0 or ny >= self.grid_map.padded_size:
            next_state = state
        elif self.grid_map.is_wall(nx, ny):
            next_state = state

        return next_state
        
    def reward(self, state: Tuple[int, int], action: Action, next_state: Tuple[int, int]) -> float:
        """
        주어진 상태와 행동에 따른 보상을 반환합니다.
        
        Args:
            state (Tuple[int, int]): 현재 상태 (x, y)
            action (Action): 수행할 행동
            next_state (Tuple[int, int]): 이동 후의 상태 (x, y)
        """
        # 1. 기본 보상
        reward = self.reward_map[next_state]

        # 2. step penalty
        reward += RewardConfig.STEP_PENALTY

        # 3. 중복 방문 패널티 (방문 횟수에 따라 증가)
        visit_count = self.visited_states[next_state]
        # 이미 방문한 경우 방문 횟수의 제곱만큼 패널티
        if visit_count > 1:
            duplicate_penalty = RewardConfig.VISITED_PENALTY * (2 ** (visit_count - 2))
            reward += duplicate_penalty

        # 4. 목표 도달 보상
        if next_state in self.goals:
            reward += RewardConfig.GOAL_REWARD

        return reward
        
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool]:
        """
        Agent를 주어진 행동에 따라 이동시키고 보상을 반환합니다.
        
        Args:
            action (Action): 수행할 행동
        """
        state = self.agent_pos
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state in self.goals)

        self.agent_pos = next_state
        self.visited_states[next_state] += 1    # 방문 횟수 증가

        return next_state, reward, done
        
    def reset(self) -> Tuple[int, int]:
        """
        Agent를 초기 위치로 이동시킵니다.
        """
        self.agent_pos = self.grid_map.agent_pos
        self.visited_states.clear()
        self.visited_states[self.agent_pos] = 1

        return self.agent_pos