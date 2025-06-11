import matplotlib.pyplot as plt
import numpy as np
from environment.gridworld import Action

def visualize_policy(grid_world, agent):
    """
    학습된 정책을 시각화합니다.
    
    Args:
        grid_world: GridWorld 환경
        agent: 학습된 QLearningAgent
    """
    plt.figure(figsize=(10, 10))
    
    # 커스텀 컬러맵 생성
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
    
    # 화살표 방향 매핑
    arrow_map = {
        Action.UP: '↑',
        Action.RIGHT: '→',
        Action.DOWN: '↓',
        Action.LEFT: '←'
    }
    
    # 색상 매핑
    color_map = {
        Action.UP: 'red',
        Action.RIGHT: 'blue',
        Action.DOWN: 'green',
        Action.LEFT: 'purple'
    }
    
    # 정책 시각화
    for x in range(grid_world.grid_map.padded_size):
        for y in range(grid_world.grid_map.padded_size):
            state = (x, y)
            if not grid_world.grid_map.is_wall(x, y) and state in agent.pi:
                action_probs = agent.pi[state]
                best_action = max(action_probs.items(), key=lambda x: x[1])[0]
                
                # 화살표 그리기 (좌표계 변환)
                plt.text(x + 0.5, y + 0.5, arrow_map[best_action], 
                        ha='center', va='center', fontsize=15,
                        color=color_map[best_action])
    
    # 목표 지점들 표시 (좌표계 변환)
    if grid_world.scenario_type == "시나리오3":
        markers = ['*', 's', '^', 'D', 'o', 'v', '<', '>', 'p', 'h']
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (x, y) in enumerate(grid_world.goals):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.plot(x + 0.5, y + 0.5, marker=marker, color=color, markersize=15)
    else:
        for x, y in grid_world.goals:
            plt.plot(x + 0.5, y + 0.5, 'r*', markersize=15)
    
    # 그리드 선 추가
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.xticks(np.arange(0, grid_world.grid_map.padded_size + 1, 1))
    plt.yticks(np.arange(0, grid_world.grid_map.padded_size + 1, 1))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    plt.show()