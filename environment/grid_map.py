import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class GridMap:
    def __init__(self, size: int = 11, padding: int = 2, seed: int = 42):
        """
        2D 그리드 환경을 표현하는 클래스
        
        CNN 모델의 3x3 커널을 고려하여 경계에 2칸의 padding을 추가합니다.
        
        Attributes:
            size (int): 실제 맵의 크기 (기본값: 11)
            padding (int): 경계에 추가할 padding의 크기 (기본값: 2)
            grid (numpy.ndarray): 맵의 상태 (0: 일반 경로, 1: 벽, 2: 공사중인 경로)
        """
        self.size = size
        self.padding = padding
        self.padded_size = size + 2 * padding
        
        self.agent_pos = (self.padded_size // 2, self.padded_size // 2)     # default 에이전트 위치 (패딩 포함 좌표로 중앙 설정)
        self.grid = np.zeros((self.padded_size, self.padded_size))      # 0: 일반 경로, 1: 벽(이동 불가), 2: 공사중인 경로
        self.shortcuts: Dict[Tuple[int, int], Tuple[int, int]] = {}     # 지름길 정보를 저장할 딕셔너리 {(start_row, start_col): (end_row, end_col)} 형태로 저장
    
        np.random.seed(seed)    # 랜덤 시드 설정
        self._add_padding()     # padding 영역을 벽으로 설정

    def _add_padding(self):
        """
        CNN 모델이 맵의 경계에서도 주변을 인식할 수 있도록 맵의 경계에 padding을 추가합니다.
        padding 영역은 모두 벽(1)으로 설정됩니다.
        """
        self.grid[0:self.padding, :] = 1     # 상단 padding
        self.grid[-self.padding:, :] = 1     # 하단 padding
        self.grid[:, 0:self.padding] = 1     # 좌측 padding
        self.grid[:, -self.padding:] = 1     # 우측 padding


    def add_random_shortcuts(self, num_shortcuts: int):
        """
        랜덤한 시작점과 끝점을 연결하는 지름길들을 생성
        끝점은 시작점으로부터 대각선 방향 오프셋 [-3, -2, 2, 3] 범위 내에서 선택됩니다.
        즉, 직선 이동(상하좌우)이나 1칸 대각선 이동(오프셋 ±1, ±1), 제자리 이동(0,0)은 제외됩니다.
        (패딩 포함 좌표 사용)
        """
        successful_shortcuts = 0
        max_attempts = num_shortcuts * 500  # 무한 루프 방지

        # 패딩을 제외한 유효 영역의 좌표 범위 (패딩 포함 좌표계 기준)
        min_coord = self.padding
        max_coord = self.padded_size - self.padding - 1

        attempts = 0
        while successful_shortcuts < num_shortcuts and attempts < max_attempts:
            # 유효 영역 내에서 랜덤한 시작점 생성 (패딩 포함 좌표)
            start_row = np.random.randint(min_coord, max_coord + 1)
            start_col = np.random.randint(min_coord, max_coord + 1)

            # 시작점으로부터의 랜덤 오프셋 (dr, dc) 생성
            # dr, dc는 {-3, -2, 2, 3} 범위 내에서 선택 => 대각선 오프셋만 생성
            possible_offsets = [-3, -2, 2, 3]
            dr = np.random.choice(possible_offsets)
            dc = np.random.choice(possible_offsets)

            # 끝점 좌표 계산 (패딩 포함 좌표)
            end_row = start_row + dr
            end_col = start_col + dc

            # 계산된 끝점이 유효 영역 내에 있는지 확인 (패딩 포함 좌표 기준)
            if end_row < min_coord or end_row > max_coord or \
               end_col < min_coord or end_col > max_coord:
                attempts += 1
                continue

            # 에이전트 위치와 시작점 또는 끝점이 겹치는지 확인 (self.agent_pos는 이미 패딩 포함 좌표)
            agent_row, agent_col = self.agent_pos[0], self.agent_pos[1]
            if (start_row == agent_row and start_col == agent_col) or \
               (end_row == agent_row and end_col == agent_col):
               attempts += 1
               continue

            # 시작점 또는 끝점 칸이 벽인지 확인 (self.grid 접근은 패딩 포함 좌표 사용)
            if self.grid[start_row, start_col] == 1 or self.grid[end_row, end_col] == 1:
                attempts += 1
                continue

            # 이미 존재하는 지름길의 시작점/끝점과 겹치는지 확인 (점 겹침 로직)
            # 기존 shortcuts 딕셔너리의 키(시작점 좌표)와 값(끝점 좌표)들을 모두 확인
            existing_points = set(self.shortcuts.keys()) | set(self.shortcuts.values())
            if (start_row, start_col) in existing_points or \
               (end_row, end_col) in existing_points:
               attempts += 1
               continue

            # 모든 조건을 만족하면 지름길 추가 (양방향, 패딩 포함 좌표 사용)
            self.shortcuts[(start_row, start_col)] = (end_row, end_col)
            self.shortcuts[(end_row, end_col)] = (start_row, start_col) # 양방향
            successful_shortcuts += 1
            attempts += 1

        if successful_shortcuts < num_shortcuts:
            print(f"Warning: Only {successful_shortcuts} shortcuts successfully added out of {attempts} attempts.")


    def _add_wall(self, row: int, col: int, height: int = 1, width: int = 1):
        """
        그리드 맵에 벽을 추가합니다.

        Args:
            row (int): 벽의 시작 row 좌표 (패딩 **포함** 좌표계 기준)
            col (int): 벽의 시작 col 좌표 (패딩 **포함** 좌표계 기준)
            height (int): 벽의 세로 크기
            width (int): 벽의 가로 크기

        Note:
            - 입력 좌표 (row, col)는 패딩이 **포함된** 내부 좌표계 기준입니다.
            - height와 width는 1~3 사이로 제한됩니다.
            - 맵 범위를 벗어나는 경우 벽이 추가되지 않습니다.
            - 벽이 추가된 영역은 1로 표시됩니다.
        """
        # 맵 범위 확인 (벽 영역의 끝 좌표가 padded_size를 넘지 않아야 함)
        if row < 0 or row + height > self.padded_size or \
           col < 0 or col + width > self.padded_size:
            return

        # 장애물 영역 설정 (패딩 포함 좌표 사용)
        self.grid[row : row + height, col : col + width] = 1

    def add_random_wall(self, num_walls: int):
        """
        랜덤한 크기와 위치의 벽들을 생성합니다. 에이전트 초기 위치와 겹치지 않도록 합니다.
        (패딩 포함 좌표 사용)

        Args:
            num_walls (int): 생성할 벽의 개수
        """
        attempts = 0
        max_attempts = num_walls * 500      # 무한 루프 방지를 위한 최대 시도 횟수

        walls_added = 0     # 실제로 추가된 벽의 개수를 세는 카운터 사용

        # 벽 생성을 시도할 유효 영역 (패딩 포함 좌표 기준)
        min_coord = self.padding
        max_start_row = self.padded_size - self.padding
        max_start_col = self.padded_size - self.padding

        while walls_added < num_walls and attempts < max_attempts:
            # 랜덤한 크기 생성 (1~3)
            height = np.random.randint(1, 4) # 세로 크기
            width = np.random.randint(1, 4)  # 가로 크기

            # 랜덤한 시작 위치 생성 (패딩 포함 좌표 기준)
            # 시작 위치 + 크기가 맵의 패딩 제외 영역 끝을 넘지 않도록 조정
            start_row = np.random.randint(min_coord, max_start_row - height + 1)
            start_col = np.random.randint(min_coord, max_start_col - width + 1)

            # 에이전트 위치와 겹치는지 확인 (self.agent_pos는 이미 패딩 포함 좌표)
            agent_row, agent_col = self.agent_pos[0], self.agent_pos[1]

            # 생성하려는 벽 영역와 에이전트 위치 겹치는지 확인
            if (start_row <= agent_row < start_row + height) and \
               (start_col <= agent_col < start_col + width):
                attempts += 1
                continue

            # 이미 벽(1)이나 공사중인 경로(2)가 있는 영역인지 확인
            if self._is_area_occupied(start_row, start_col, height, width):
                 attempts += 1
                 continue

            # 벽 추가
            self._add_wall(start_row, start_col, height, width)
            walls_added += 1
            attempts += 1

        if walls_added < num_walls:
             print(f"Warning: Only {walls_added} walls successfully added out of {attempts} attempts.")


    def add_random_construction(self, num_construction: int):
        """
        랜덤한 크기와 위치의 공사중인 경로들을 생성. 에이전트 초기 위치와 겹치지 않도록 합니다.
        (패딩 포함 좌표 사용)

        Args:
            num_construction (int): 생성할 공사중인 경로의 개수
        """
        attempts = 0
        max_attempts = num_construction * 500   # 무한 루프 방지

        constructions_added = 0     # 실제로 추가된 공사중 경로 개수

        # 공사중 경로 생성을 시도할 유효 영역 (패딩 포함 좌표 기준)
        min_coord = self.padding
        max_start_row = self.padded_size - self.padding
        max_start_col = self.padded_size - self.padding

        while constructions_added < num_construction and attempts < max_attempts:
            # 가로 또는 세로 방향 랜덤 선택
            is_horizontal = np.random.choice([True, False])

            # 크기 설정 (하나의 축은 1, 다른 축은 1~3)
            if is_horizontal:
                width, height = np.random.randint(1, 3), 1  # 가로로 긴 형태 (1x1 또는 2x1)
            else:
                width, height = 1, np.random.randint(1, 3)  # 세로로 긴 형태 (1x1 또는 1x2)

            # 랜덤한 시작 위치 생성 (패딩 포함 좌표 기준)
            # 시작 위치 + 크기가 맵의 패딩 제외 영역 끝을 넘지 않도록 조정
            start_row = np.random.randint(min_coord, max_start_row - height + 1)
            start_col = np.random.randint(min_coord, max_start_col - width + 1)

            # 에이전트 위치와 겹치는지 확인 (self.agent_pos는 이미 패딩 포함 좌표)
            agent_row, agent_col = self.agent_pos[0], self.agent_pos[1]

            # 생성하려는 영역 [start_row:start_row+height, start_col:start_col+width] 와 에이전트 위치 겹치는지 확인
            if (start_row <= agent_row < start_row + height) and \
               (start_col <= agent_col < start_col + width):
                attempts += 1
                continue

            # 이미 벽(1)이나 공사중인 경로(2)가 있는 영역인지 확인
            # _is_area_occupied는 이제 패딩 포함 좌표를 기대하므로 start_row, start_col를 그대로 전달
            if self._is_area_occupied(start_row, start_col, height, width): # height, width 순서 맞춤
                 attempts += 1
                 continue

            # 공사중인 경로 추가 (패딩 포함 좌표 사용)
            self.grid[start_row : start_row + height, start_col : start_col + width] = 2

            constructions_added += 1 # 성공적으로 추가되면 카운트 증가
            attempts += 1

        if constructions_added < num_construction:
             print(f"Warning: Only {constructions_added} constructions successfully added out of {attempts} attempts.")


    def _is_area_occupied(self, row: int, col: int, height: int, width: int) -> bool:
        """
        특정 영역에 장애물(1)이나 공사중인 경로(2)가 있는지 확인

        Args:
            row (int): 시작 row 좌표 (패딩 **포함** 좌표계 기준)
            col (int): 시작 col 좌표 (패딩 **포함** 좌표계 기준)
            height (int): 영역의 세로 크기
            width (int): 영역의 가로 크기

        Returns:
            bool: 영역이 점유되어 있으면 True, 아니면 False
        """
        # 맵 범위를 벗어나는지 확인
        if row < 0 or row + height > self.padded_size or \
           col < 0 or col + width > self.padded_size:
             return True

        # 해당 영역에 1(벽) 또는 2(공사중) 값이 있는지 확인 (패딩 포함 좌표 사용)
        return np.any(self.grid[row : row + height, col : col + width] > 0)


    def visualize(self):
        """맵 시각화"""
        plt.figure(figsize=(10, 10))

        # 커스텀 컬러맵 생성
        colors = ['white', 'black', 'yellow']   # 0: 일반 경로, 1: 장애물, 2: 공사중인 경로
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N) # 값 범위에 맞게 정규화

        # 그리드 시각화
        plt.imshow(self.grid, cmap=cmap, norm=norm, extent=[0, self.padded_size, self.padded_size, 0], origin='upper', alpha=0.8)

        # 그리드 선 추가
        plt.grid(True, which='both', color='gray', linewidth=0.5)
        # 축 눈금 설정 (각 칸의 경계에 오도록: 0, 1, 2, ...)
        plt.xticks(np.arange(0, self.padded_size + 1, 1))
        plt.yticks(np.arange(0, self.padded_size + 1, 1))
        # 축 라벨 제거
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        # 축 범위 설정 (패딩 포함)
        plt.xlim(0, self.padded_size)
        plt.ylim(self.padded_size, 0)

        # 지름길 표시 (shortcuts는 패딩 포함 좌표 사용, (row, col) 형태)
        for start, end in self.shortcuts.items():
            start_col, start_row = start[1], start[0]   # (col, row)
            end_col, end_row = end[1], end[0]           # (col, row)

            # 선 그리기 (시작점 중앙에서 끝점 중앙까지)
            plt.plot([start_col + 0.5, end_col + 0.5], [start_row + 0.5, end_row + 0.5], 'g-', alpha=1)
            # 시작점과 끝점에 점 찍기 (셀 중앙)
            plt.plot(start_col + 0.5, start_row + 0.5, 'go', markersize=5)
            plt.plot(end_col + 0.5, end_row + 0.5, 'go', markersize=5)

        # 에이전트 위치 표시 (self.agent_pos는 패딩 포함 좌표, (row, col) 형태)
        agent_row, agent_col = self.agent_pos
        plt.plot(agent_col + 0.5, agent_row + 0.5, 'bo', markersize=10) # 파란색 점

        plt.title('Grid Map Visualization')
        plt.show()
