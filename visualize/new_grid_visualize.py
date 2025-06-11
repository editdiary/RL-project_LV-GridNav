import matplotlib.pyplot as plt
import numpy as np
from environment import GridWorld, Action
from typing import List, Tuple

def visualize_gridworld(grid_world: GridWorld, agent_pos: Tuple[int, int] = None):
    plt.figure(figsize=(10, 10))
    
    # 커스텀 컬러맵 생성 (초록색 제거)
    colors = ['white', 'black', 'yellow']  # 0: 일반 경로, 1: 벽, 2: 공사중인 경로
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    
    # 타일 타입을 숫자로 변환
    grid = np.zeros((grid_world.grid_map.padded_size, grid_world.grid_map.padded_size))
    for x in range(grid_world.grid_map.padded_size):
        for y in range(grid_world.grid_map.padded_size):
            if grid_world.grid_map.is_wall(x, y):
                grid[x, y] = 1
            elif grid_world.grid_map.is_construction(x, y):
                grid[x, y] = 2
    
    # 그리드 시각화 (좌표계 변환)
    plt.imshow(grid.T, cmap=cmap, norm=norm,
               extent=[0, grid_world.grid_map.padded_size, grid_world.grid_map.padded_size, 0],
               origin='upper', alpha=0.8)
    
    # 그리드 선 추가
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.xticks(np.arange(0, grid_world.grid_map.padded_size + 1, 1))
    plt.yticks(np.arange(0, grid_world.grid_map.padded_size + 1, 1))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    # 에이전트 위치 표시 (좌표계 변환)
    agent_x, agent_y = agent_pos or grid_world.agent_pos
    plt.plot(agent_x + 0.5, agent_y + 0.5, 'bo', markersize=10)

    # 목표 지점들 표시 (좌표계 변환)
    if grid_world.scenario_type == "시나리오3":
        # 시나리오3의 경우 순서대로 다른 모양으로 표시
        markers = ['*', 's', '^', 'D', 'o', 'v', '<', '>', 'p', 'h']
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (x, y) in enumerate(grid_world.goals):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.plot(x + 0.5, y + 0.5, marker=marker, color=color, markersize=15)
    else:
        # 다른 시나리오의 경우 모두 동일한 모양으로 표시
        for x, y in grid_world.goals:
            plt.plot(x + 0.5, y + 0.5, 'r*', markersize=15)
    
    plt.show()