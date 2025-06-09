from typing import Dict, List, Tuple

class MapConfig:
    """맵 관련 기본 설정"""
    DEFAULT_SIZE: int = 7
    DEFAULT_PADDING: int = 2
    DEFAULT_SEED: int = 40
    MAX_ATTEMPTS_MULTIPLIER: int = 100

class ElementConfig:
    """맵 요소(벽, 공사중인 경로 등) 관련 설정"""
    
    # 벽의 크기 정의
    WALL_SIZES: List[Tuple[int, int]] = [
        (1, 1),  # 1x1
        (2, 1),  # 2x1
        (1, 2),  # 1x2
        (2, 2),  # 2x2
    ]
    
    # 공사중인 경로의 크기 정의
    CONSTRUCTION_SIZES: List[Tuple[int, int]] = [
        (1, 1),  # 1x1
        (1, 2),  # 1x2
        (2, 1),  # 2x1
    ]

class RewardConfig:
    """보상 관련 설정"""
    # 기본 보상
    GOAL_REWARD = 10.0          # 목표 도달
    WALL_PENALTY = -5.0         # 벽 충돌
    CONSTRUCTION_PENALTY = -2.0  # 공사중인 경로 통과
    STEP_PENALTY = -1.0          # 한 스텝당 패널티
    
    # 추가 보상
    GOAL_PROXIMITY_REWARD = 0.5  # 목표에 가까워질 때마다
    VISITED_PENALTY = -0.2       # 이미 방문한 타일 재방문시