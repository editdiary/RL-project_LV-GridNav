import matplotlib.pyplot as plt
import numpy as np
from .grid_map import GridMap
from .tile_type import TileType

def visualize_map(grid_map: GridMap):
    """
    GridMap 객체를 시각화합니다.
    
    Args:
        grid_map (GridMap): 시각화할 GridMap 인스턴스
    """
    plt.figure(figsize=(10, 10))
    
    # 커스텀 컬러맵 생성
    colors = ['white', 'black', 'yellow']  # 0: 일반 경로, 1: 벽, 2: 공사중인 경로
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    
    # 타일 타입을 숫자로 변환
    grid = np.zeros((grid_map.padded_size, grid_map.padded_size))
    for x in range(grid_map.padded_size):
        for y in range(grid_map.padded_size):
            if grid_map.tiles[x][y].type == TileType.WALL:
                grid[x, y] = 1
            elif grid_map.tiles[x][y].type == TileType.CONSTRUCTION:
                grid[x, y] = 2
    
    # 그리드 시각화
    plt.imshow(grid, cmap=cmap, norm=norm,
               extent=[0, grid_map.padded_size, grid_map.padded_size, 0],
               origin='upper', alpha=0.8)
    
    # 그리드 선 추가
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.xticks(np.arange(0, grid_map.padded_size + 1, 1))
    plt.yticks(np.arange(0, grid_map.padded_size + 1, 1))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    # 에이전트 위치 표시
    agent_x, agent_y = grid_map.agent.pos
    plt.plot(agent_y + 0.5, agent_x + 0.5, 'bo', markersize=10)
    
    plt.title('Grid Map Visualization')
    plt.show()