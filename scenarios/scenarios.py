from enum import Enum
from typing import List, Tuple, Optional
import numpy as np
from environment import GridMap

class ScenarioType(Enum):
    """시나리오 타입을 정의하는 열거형"""
    SCENARIO_1 = "시나리오1"    # 단일 목표
    SCENARIO_2 = "시나리오2"    # 가장 가까운 목표
    SCENARIO_3 = "시나리오3"    # 모든 목표 방문
    
class Scenario:
    """시나리오 기본 클래스"""
    def __init__(self, grid_map: GridMap, num_goals: int = 1, agent_pos: Tuple[int, int] = None):
        self.grid_map = grid_map
        self.num_goals = num_goals
        self.goals: List[Tuple[int, int]] = []
        self.agent_pos = agent_pos  # Agent의 초기 위치

    def generate_goals(self) -> None:
        """목표 지점을 생성하는 메서드"""
        raise NotImplementedError("Subclasses must implement generate_goals()")
    
    def get_goals(self) -> List[Tuple[int, int]]:
        """생성된 목표 지점들을 반환"""
        return self.goals
    
    def _is_far_enough_from_other_goals(self, x: int, y: int, min_distance: int) -> bool:
        """새로운 목표 지점이 기존 목표 지점들과 충분히 떨어져 있는지 확인합니다."""
        for goal_x, goal_y in self.goals:
            distance = abs(x - goal_x) + abs(y - goal_y)  # 맨해튼 거리
            if distance < min_distance:
                return False
        return True

    def _is_far_enough_from_agent(self, x: int, y: int, min_distance: int) -> bool:
        """새로운 목표 지점이 Agent와 충분히 떨어져 있는지 확인합니다."""
        if self.agent_pos is None:
            return True
        
        agent_x, agent_y = self.agent_pos
        distance = abs(x - agent_x) + abs(y - agent_y)  # 맨해튼 거리
        return distance >= min_distance

class SingleGoalScenario(Scenario):
    """시나리오1: 단일 목표 시나리오"""
    def __init__(self, grid_map: GridMap, seed: int = None, agent_pos: Tuple[int, int] = None):
        super().__init__(grid_map=grid_map, num_goals=1, agent_pos=agent_pos)
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.rng = np.random.RandomState(self.seed)

    @property
    def scenario_type(self) -> str:
        return "시나리오1"

    def generate_goals(self) -> None:
        attempts = 0
        max_attempts = 100  # 무한 루프 방지
        min_distance = 5  # Agent와의 최소 거리
        
        while attempts < max_attempts:
            x = np.random.randint(self.grid_map.padding, 
                                self.grid_map.padded_size - self.grid_map.padding)
            y = np.random.randint(self.grid_map.padding, 
                                self.grid_map.padded_size - self.grid_map.padding)
            
            # 해당 위치가 비어있는지 확인
            if self.grid_map._is_area_empty(x, y, 1, 1):
                if self._is_far_enough_from_agent(x, y, min_distance):
                    self.goals = [(x, y)]
                    return
            
            attempts += 1
        
        raise RuntimeError("비어있는 목표 지점을 찾을 수 없습니다.")
    
class NearestGoalScenario(Scenario):
    """시나리오2: 가장 가까운 목표 시나리오"""
    def __init__(self, grid_map: GridMap, seed: int = None, agent_pos: Tuple[int, int] = None):
        super().__init__(grid_map=grid_map, num_goals=3, agent_pos=agent_pos)
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.rng = np.random.RandomState(self.seed)

    @property
    def scenario_type(self) -> str:
        return "시나리오2"

    def generate_goals(self) -> None:
        """가장 가까운 목표를 찾기 위한 목표 지점들을 생성"""
        self.goals = []
        attempts = 0
        max_attempts = self.num_goals * 100  # 무한 루프 방지
        min_distance = 3  # 목표 지점들 사이의 최소 거리
        
        while len(self.goals) < self.num_goals and attempts < max_attempts:
            x = self.rng.randint(self.grid_map.padding, 
                               self.grid_map.padded_size - self.grid_map.padding)
            y = self.rng.randint(self.grid_map.padding, 
                               self.grid_map.padded_size - self.grid_map.padding)
            
            # 해당 위치가 비어있는지 확인
            if self.grid_map._is_area_empty(x, y, 1, 1):
                if (self._is_far_enough_from_other_goals(x, y, min_distance) and 
                    self._is_far_enough_from_agent(x, y, min_distance)):
                    self.goals.append((x, y))
            
            attempts += 1
        
        if len(self.goals) < self.num_goals:
            raise RuntimeError(f"요청한 {self.num_goals}개의 목표 지점 중 {len(self.goals)}개만 생성할 수 있었습니다.")

class MultipleGoalsScenario(Scenario):
    """시나리오3: 모든 목표 방문 시나리오"""
    def __init__(self, grid_map: GridMap, seed: int = None, agent_pos: Tuple[int, int] = None):
        super().__init__(grid_map=grid_map, num_goals=3, agent_pos=agent_pos)
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.rng = np.random.RandomState(self.seed)

    @property
    def scenario_type(self) -> str:
        return "시나리오3"

    def generate_goals(self) -> None:
        """모든 목표를 방문해야 하는 목표 지점들을 생성합니다."""
        self.goals = []
        attempts = 0
        max_attempts = self.num_goals * 100
        min_distance = 3  # 목표 지점들 사이의 최소 거리
        
        while len(self.goals) < self.num_goals and attempts < max_attempts:
            x = self.rng.randint(self.grid_map.padding, 
                               self.grid_map.padded_size - self.grid_map.padding)
            y = self.rng.randint(self.grid_map.padding, 
                               self.grid_map.padded_size - self.grid_map.padding)
            
            if self.grid_map._is_area_empty(x, y, 1, 1):
                if (self._is_far_enough_from_other_goals(x, y, min_distance) and 
                    self._is_far_enough_from_agent(x, y, min_distance)):
                    self.goals.append((x, y))
            
            attempts += 1
        
        if len(self.goals) < self.num_goals:
            raise RuntimeError(f"요청한 {self.num_goals}개의 목표 지점 중 {len(self.goals)}개만 생성할 수 있었습니다.")


class ScenarioFactory:
    """시나리오 생성 팩토리 클래스"""
    @staticmethod
    def create_scenario(scenario_type: str,
                        grid_map: GridMap,
                        num_goals: int = None,
                        agent_pos: Tuple[int, int] = None) -> Optional[Scenario]:
        """
        시나리오 타입에 따라 적절한 시나리오 객체를 생성합니다.
        
        Args:
            scenario_type (str): 생성할 시나리오 타입 ("시나리오1", "시나리오2", "시나리오3")
            grid_map (GridMap): 시나리오가 적용될 그리드 맵
            num_goals (int): 생성할 목표 지점의 개수 (None인 경우 시나리오별 기본값 사용)
            agent_pos (Tuple[int, int]): Agent의 초기 위치

        Returns:
            Optional[Scenario]: 생성된 시나리오 객체
            
        Raises:
            ValueError: 잘못된 시나리오 타입이 입력된 경우
        """
        try:
            scenario_type = ScenarioType(scenario_type)
        except ValueError:
            raise ValueError(f"잘못된 시나리오 타입입니다. '시나리오1', '시나리오2', '시나리오3' 중 하나를 선택해주세요.")
            
        # 시나리오 타입에 따라 다른 시드값 생성
        base_seed = np.random.randint(0, 10000)  # 기본 시드값

        if scenario_type == ScenarioType.SCENARIO_1:
            return SingleGoalScenario(grid_map, base_seed, agent_pos)   # 시나리오1은 항상 목표 지점 1개
        elif scenario_type == ScenarioType.SCENARIO_2:
            return NearestGoalScenario(grid_map, base_seed + 1, agent_pos)  # 시나리오2는 목표 지점 기본값 5개
        elif scenario_type == ScenarioType.SCENARIO_3:
            return MultipleGoalsScenario(grid_map, base_seed + 2, agent_pos)  # 시나리오3는 목표 지점 기본값 3개