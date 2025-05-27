from typing import Dict, List, Tuple

class MapConfig:
    """맵 관련 기본 설정"""
    DEFAULT_SIZE: int = 11
    DEFAULT_PADDING: int = 2
    DEFAULT_SEED: int = 42
    MAX_ATTEMPTS_MULTIPLIER: int = 100

class ElementConfig:
    """맵 요소(벽, 공사중인 경로 등) 관련 설정"""
    
    # 벽의 크기 정의
    WALL_SIZES: List[Tuple[int, int]] = [
        (1, 1),  # 1x1
        (2, 1),  # 2x1
        (1, 2),  # 1x2
        (2, 2),  # 2x2
        (1, 3),  # 1x3
        (3, 1),  # 3x1
        (2, 3),  # 2x3
        (3, 2),  # 3x2
    ]
    
    # 공사중인 경로의 크기 정의
    CONSTRUCTION_SIZES: List[Tuple[int, int]] = [
        (1, 1),  # 1x1
        (1, 2),  # 1x2
        (2, 1),  # 2x1
        (1, 3),  # 1x3
        (3, 1),  # 3x1
    ]