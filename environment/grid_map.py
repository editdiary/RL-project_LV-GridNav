import numpy as np
from enum import Enum
from typing import Dict, Tuple, List
from .config import MapConfig, ElementConfig


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
        
        Args:
            size (int): 실제 맵의 크기 (기본값: 11)
            padding (int): 경계에 추가할 padding의 크기 (기본값: 2)
            seed (int): 랜덤 시드 (기본값: 42)
        """
        self.size = size
        self.padding = padding
        self.padded_size = size + 2 * padding
        
        # 기본 타일 생성 (패딩 포함)
        self.tiles: List[List[Tile]] = [[Tile(x, y, TileType.EMPTY) for y in range(self.padded_size)] 
                                       for x in range(self.padded_size)]
        
        # 에이전트 초기 위치 (중앙)
        self.agent_pos = (self.padded_size // 2, self.padded_size // 2)
        
        # 지름길 저장
        self.shortcuts: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        np.random.seed(seed)
        self._add_padding()     # 패딩 영역을 벽으로 설정

    def _add_padding(self):
        """맵의 경계에 padding을 추가합니다."""
        for x in range(self.padded_size):
            for y in range(self.padded_size):
                if (x < self.padding or x >= self.padded_size - self.padding or
                    y < self.padding or y >= self.padded_size - self.padding):
                    self.tiles[x][y].type = TileType.WALL

    def _add_wall(self, start_x: int, start_y: int, width: int, height: int) -> None:
        """특정 영역에 벽을 추가합니다."""
        for x in range(start_x, start_x + width):
            for y in range(start_y, start_y + height):
                self.tiles[x][y].type = TileType.WALL

    def add_random_walls(self, num_walls: int) -> None:
        """맵에 랜덤한 크기의 벽을 추가합니다."""
        attempts = 0
        max_attempts = num_walls * MapConfig.MAX_ATTEMPTS_MULTIPLIER
        walls_added = 0

        while walls_added < num_walls and attempts < max_attempts:
            width, height = ElementConfig.WALL_SIZES[np.random.randint(len(ElementConfig.WALL_SIZES))]
            start_x = np.random.randint(self.padding, self.padded_size - self.padding - width + 1)
            start_y = np.random.randint(self.padding, self.padded_size - self.padding - height + 1)
            
            agent_x, agent_y = self.agent_pos
            if (start_x <= agent_x < start_x + width and 
                start_y <= agent_y < start_y + height):
                attempts += 1
                continue
            
            if self._is_area_empty(start_x, start_y, width, height):
                self._add_wall(start_x, start_y, width, height)
                walls_added += 1
            
            attempts += 1

    def _add_construction(self, start_x: int, start_y: int, width: int, height: int) -> None:
        """특정 영역에 공사중인 경로를 추가합니다."""
        for x in range(start_x, start_x + width):
            for y in range(start_y, start_y + height):
                self.tiles[x][y].type = TileType.CONSTRUCTION

    def add_random_construction(self, num_construction: int) -> None:
        """맵에 랜덤한 크기의 공사중인 경로를 추가합니다."""
        attempts = 0
        max_attempts = num_construction * MapConfig.MAX_ATTEMPTS_MULTIPLIER
        constructions_added = 0

        while constructions_added < num_construction and attempts < max_attempts:
            width, height = ElementConfig.CONSTRUCTION_SIZES[np.random.randint(len(ElementConfig.CONSTRUCTION_SIZES))]
            start_x = np.random.randint(self.padding, self.padded_size - self.padding - width + 1)
            start_y = np.random.randint(self.padding, self.padded_size - self.padding - height + 1)
            
            agent_x, agent_y = self.agent_pos
            if (start_x <= agent_x < start_x + width and 
                start_y <= agent_y < start_y + height):
                attempts += 1
                continue
            
            if self._is_area_empty(start_x, start_y, width, height):
                self._add_construction(start_x, start_y, width, height)
                constructions_added += 1
            
            attempts += 1

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
        """특정 영역이 비어있는지 확인합니다."""
        for x in range(start_x, start_x + width):
            for y in range(start_y, start_y + height):
                if not (0 <= x < self.padded_size and 0 <= y < self.padded_size):
                    return False
                if self.tiles[x][y].type != TileType.EMPTY:
                    return False
        return True