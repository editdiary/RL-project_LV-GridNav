import matplotlib.pyplot as plt
import numpy as np
from environment.gridworld import Action
import time

def visualize_agent_movement(grid_world, agent, max_steps=100):
    """
    에이전트가 최적 정책에 따라 움직이는 것을 시각화합니다.
    
    Args:
        grid_world: GridWorld 환경
        agent: 학습된 QLearningAgent
        max_steps: 최대 스텝 수
    """
    state = grid_world.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        # 현재 상태 시각화
        plt.figure(figsize=(10, 10))
        
        # 커스텀 컬러맵 생성
        colors = ['white', 'black', 'yellow']
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
        
        # 그리드 시각화
        plt.imshow(grid.T, cmap=cmap, norm=norm,
                   extent=[0, grid_world.grid_map.padded_size, grid_world.grid_map.padded_size, 0],
                   origin='upper', alpha=0.8)
        
        # 목표 지점들 표시
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
        
        # 에이전트 위치 표시
        agent_x, agent_y = state
        plt.plot(agent_x + 0.5, agent_y + 0.5, 'bo', markersize=10)
        
        # 그리드 선 추가
        plt.grid(True, which='both', color='gray', linewidth=0.5)
        plt.xticks(np.arange(0, grid_world.grid_map.padded_size + 1, 1))
        plt.yticks(np.arange(0, grid_world.grid_map.padded_size + 1, 1))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        
        # 현재 상태의 최적 행동 표시
        if state in agent.pi:
            action_probs = agent.pi[state]
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            arrow_map = {
                Action.UP: '↑',
                Action.RIGHT: '→',
                Action.DOWN: '↓',
                Action.LEFT: '←'
            }
            plt.text(agent_x + 0.5, agent_y + 0.5, arrow_map[best_action], 
                    ha='center', va='center', fontsize=15, color='red')
        
        plt.title(f'Step {step + 1}')
        plt.show()
        
        # 다음 상태로 이동
        action = agent.get_action(state)
        next_state, reward, done = grid_world.step(action)
        state = next_state
        step += 1
        
        # 잠시 대기 (시각화를 천천히 보기 위해)
        time.sleep(0.5)
    
    print(f"에이전트가 {'목표에 도달했습니다.' if done else '최대 스텝 수에 도달했습니다.'}")