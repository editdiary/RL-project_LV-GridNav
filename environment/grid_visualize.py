import matplotlib.pyplot as plt
import numpy as np

def visualize_map(grid_map):
    """
    GridMap 객체의 현재 상태를 시각화합니다.
    Args:
        grid_map (GridMap): 시각화할 GridMap 인스턴스
    """
    plt.figure(figsize=(10, 10))

    # 커스텀 컬러맵 생성
    colors = ['white', 'black', 'yellow']   # 0: 일반 경로, 1: 장애물, 2: 공사중인 경로
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N) # 값 범위에 맞게 정규화

    # 그리드 시각화
    plt.imshow(grid_map.grid, cmap=cmap, norm=norm, extent=[0, grid_map.padded_size, grid_map.padded_size, 0], origin='upper', alpha=0.8)

    # 그리드 선 추가
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    # 축 눈금 설정 (각 칸의 경계에 오도록: 0, 1, 2, ...)
    plt.xticks(np.arange(0, grid_map.padded_size + 1, 1))
    plt.yticks(np.arange(0, grid_map.padded_size + 1, 1))
    # 축 라벨 제거
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    # 축 범위 설정 (패딩 포함)
    plt.xlim(0, grid_map.padded_size)
    plt.ylim(grid_map.padded_size, 0)

    # 지름길 표시 (shortcuts는 패딩 포함 좌표 사용, (row, col) 형태)
    for start, end in grid_map.shortcuts.items():
        start_col, start_row = start[1], start[0]   # (col, row)
        end_col, end_row = end[1], end[0]           # (col, row)

        # 선 그리기 (시작점 중앙에서 끝점 중앙까지)
        plt.plot([start_col + 0.5, end_col + 0.5], [start_row + 0.5, end_row + 0.5], 'g-', alpha=1)
        # 시작점과 끝점에 점 찍기 (셀 중앙)
        plt.plot(start_col + 0.5, start_row + 0.5, 'go', markersize=5)
        plt.plot(end_col + 0.5, end_row + 0.5, 'go', markersize=5)

    # 에이전트 위치 표시 (self.agent_pos는 패딩 포함 좌표, (row, col) 형태)
    agent_row, agent_col = grid_map.agent_pos
    plt.plot(agent_col + 0.5, agent_row + 0.5, 'bo', markersize=10) # 파란색 점

    plt.title('Grid Map Visualization')
    plt.show()
