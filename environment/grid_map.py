import numpy as np
from enum import Enum
from typing import Dict, Tuple, List
from .config import MapConfig, ElementConfig
from agent import Agent


class Tile:
    """그리드 맵의 각 타일을 표현하는 클래스"""
    def __init__(self, x: int, y: int, type: str):
        self.x = x
        self.y = y
        self.type = type

class TileType(Enum):
    EMPTY = "empty"
    WALL = "wall"
    CONSTRUCTION = "construction"


class GridMap:
    def __init__(self, size: int = MapConfig.DEFAULT_SIZE,
                 padding: int = MapConfig.DEFAULT_PADDING,
                 seed: int = MapConfig.DEFAULT_SEED):
        """
        2D 그리드 환경을 표현하는 클래스
        
        이 클래스는 강화학습을 위한 2D 그리드 환경을 구현합니다.
        패딩이 포함된 맵을 생성하고, 벽, 지름길, 공사중인 경로 등을 관리합니다.
        
        Args:
            size (int): 실제 맵의 크기 (기본값: 11)
            padding (int): 경계에 추가할 padding의 크기 (기본값: 2)
            seed (int): 랜덤 시드 (기본값: 42)

        Attributes:
            size (int): 실제 맵의 크기
            padding (int): 경계에 추가할 padding의 크기
            padded_size (int): 패딩을 포함한 전체 맵의 크기
            tiles (List[List[Tile]]): 타일 객체들의 2차원 리스트
            agent (Agent): 환경 내 에이전트
            shortcuts (Dict[Tuple[int, int], Tuple[int, int]]): 지름길 정보
        """
        self.size = size
        self.padding = padding
        self.padded_size = size + 2 * padding
        
        # 기본 타일 생성 (패딩 포함)
        self.tiles: List[List[Tile]] = [[Tile(x, y, TileType.EMPTY) for y in range(self.padded_size)] 
                                       for x in range(self.padded_size)]
        
        # 에이전트 초기화 (중앙 위치)
        self.agent = Agent(self.padded_size // 2, self.padded_size // 2)
        
        # 지름길 저장
        self.shortcuts: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        np.random.seed(seed)
        self._add_padding()     # 패딩 영역을 벽으로 설정

    def _add_padding(self):
        """
        맵의 경계에 padding을 추가합니다.
        padding 영역은 모두 벽(1)으로 설정됩니다.
        """
        for x in range(self.padded_size):
            for y in range(self.padded_size):
                if (x < self.padding or x >= self.padded_size - self.padding or
                    y < self.padding or y >= self.padded_size - self.padding):
                    self.tiles[x][y].type = TileType.WALL

    def _add_wall(self, start_x: int, start_y: int, width: int, height: int) -> None:
        """
        특정 영역에 벽을 추가합니다.
        
        Args:
            start_x (int): 시작 x 좌표
            start_y (int): 시작 y 좌표
            width (int): 벽의 너비
            height (int): 벽의 높이
        """
        for x in range(start_x, start_x + width):
            for y in range(start_y, start_y + height):
                self.tiles[x][y].type = TileType.WALL

    def add_random_walls(self, num_walls: int) -> None:
        """
        맵에 랜덤한 크기의 벽을 추가합니다.
        
        Args:
            num_walls (int): 추가할 벽의 개수
            
        Note:
            - 패딩 영역에는 벽이 추가되지 않습니다.
            - 에이전트가 있는 위치에는 벽이 추가되지 않습니다.
            - 이미 벽이나 공사중인 경로가 있는 위치에는 벽이 추가되지 않습니다.
            - 벽은 WallSize.SIZES에 정의된 크기 중 하나로 랜덤하게 생성됩니다.
        """
        attempts = 0
        max_attempts = num_walls * MapConfig.MAX_ATTEMPTS_MULTIPLIER      # 무한루프 방지
        walls_added = 0

        while walls_added < num_walls and attempts < max_attempts:
            # 랜덤한 벽 크기 선택
            width, height = ElementConfig.WALL_SIZES[np.random.randint(len(ElementConfig.WALL_SIZES))]
            
            # 패딩을 제외한 영역에서 랜덤한 시작 위치 선택
            # 벽의 크기를 고려하여 시작 위치 범위 조정
            start_x = np.random.randint(self.padding, self.padded_size - self.padding - width + 1)
            start_y = np.random.randint(self.padding, self.padded_size - self.padding - height + 1)
            
            # 에이전트 위치와 겹치지 않는지 확인
            agent_x, agent_y = self.agent.pos
            if (start_x <= agent_x < start_x + width and 
                start_y <= agent_y < start_y + height):
                attempts += 1
                continue
            
            # 해당 영역이 모두 비어있는지 확인
            if self._is_area_empty(start_x, start_y, width, height):
                self._add_wall(start_x, start_y, width, height)
                walls_added += 1
            
            attempts += 1

        if walls_added < num_walls:
            print(f"Warning: {num_walls - walls_added}개의 벽을 추가하지 못했습니다.")


    def _add_construction(self, start_x: int, start_y: int, width: int, height: int) -> None:
        """
        특정 영역에 공사중인 경로를 추가합니다.
        
        Args:
            start_x (int): 시작 x 좌표
            start_y (int): 시작 y 좌표
            width (int): 공사중인 경로의 너비
            height (int): 공사중인 경로의 높이
        """
        for x in range(start_x, start_x + width):
            for y in range(start_y, start_y + height):
                self.tiles[x][y].type = TileType.CONSTRUCTION

    def add_random_construction(self, num_construction: int) -> None:
        """
        맵에 랜덤한 크기의 공사중인 경로를 추가합니다.
        
        Args:
            num_construction (int): 추가할 공사중인 경로의 개수
            
        Note:
            - 패딩 영역에는 공사중인 경로가 추가되지 않습니다.
            - 에이전트가 있는 위치에는 공사중인 경로가 추가되지 않습니다.
            - 이미 벽이나 공사중인 경로가 있는 위치에는 추가되지 않습니다.
            - 공사중인 경로는 ConstructionSize.SIZES에 정의된 크기 중 하나로 랜덤하게 생성됩니다.
        """
        attempts = 0
        max_attempts = num_construction * MapConfig.MAX_ATTEMPTS_MULTIPLIER
        constructions_added = 0

        while constructions_added < num_construction and attempts < max_attempts:
            # 랜덤한 공사중인 경로 크기 선택
            width, height = ElementConfig.CONSTRUCTION_SIZES[np.random.randint(len(ElementConfig.CONSTRUCTION_SIZES))]
            
            # 패딩을 제외한 영역에서 랜덤한 시작 위치 선택
            # 공사중인 경로의 크기를 고려하여 시작 위치 범위 조정
            start_x = np.random.randint(self.padding, self.padded_size - self.padding - width + 1)
            start_y = np.random.randint(self.padding, self.padded_size - self.padding - height + 1)
            
            # 에이전트 위치와 겹치지 않는지 확인
            agent_x, agent_y = self.agent.pos
            if (start_x <= agent_x < start_x + width and 
                start_y <= agent_y < start_y + height):
                attempts += 1
                continue
            
            # 해당 영역이 모두 비어있는지 확인
            if self._is_area_empty(start_x, start_y, width, height):
                self._add_construction(start_x, start_y, width, height)
                constructions_added += 1
            
            attempts += 1

        if constructions_added < num_construction:
            print(f"Warning: {num_construction - constructions_added}개의 공사중인 경로를 추가하지 못했습니다.")

    def get_tile_type(self, x: int, y: int) -> str:
        """특정 위치의 타일 타입을 반환합니다."""
        if not (0 <= x < self.padded_size and 0 <= y < self.padded_size):
            raise IndexError(f"좌표 ({x}, {y})가 맵 범위를 벗어났습니다.")
        return self.tiles[x][y].type

    def is_wall(self, x: int, y: int) -> bool:
        """특정 위치가 벽인지 확인합니다."""
        return self.get_tile_type(x, y) == TileType.WALL
    
    def is_construction(self, x: int, y: int) -> bool:
        """특정 위치가 공사중인 경로인지 확인합니다."""
        return self.get_tile_type(x, y) == TileType.CONSTRUCTION

    def _is_area_empty(self, start_x: int, start_y: int, width: int, height: int) -> bool:
        """
        특정 영역이 모두 비어있는지 확인합니다.
        
        Args:
            start_x (int): 시작 x 좌표
            start_y (int): 시작 y 좌표
            width (int): 영역의 너비
            height (int): 영역의 높이
            
        Returns:
            bool: 영역이 모두 비어있으면 True, 아니면 False
        """
        # 맵 범위를 벗어나는지 확인
        if (start_x < self.padding or 
            start_y < self.padding or 
            start_x + width > self.padded_size - self.padding or 
            start_y + height > self.padded_size - self.padding):
            return False
            
        # 영역 내 모든 타일이 비어있는지 확인
        for x in range(start_x, start_x + width):
            for y in range(start_y, start_y + height):
                if self.tiles[x][y].type != TileType.EMPTY:
                    return False
        return True