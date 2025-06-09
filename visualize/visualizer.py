import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from environment.grid_map import TileType

class Visualizer:
    def __init__(self, grid_map):
        """그리드 월드 시각화를 위한 초기화
        
        Args:
            grid_map: 시각화할 그리드 맵 객체
        """
        self.grid_map = grid_map
        self.ys = grid_map.padded_size
        self.xs = grid_map.padded_size
        
        self.ax = None
        self.fig = None

    def set_figure(self, figsize=(10, 10)):
        """그래프 설정"""
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)
        ax = self.ax
        ax.clear()
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.set_xticks(range(self.xs))
        ax.set_yticks(range(self.ys))
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)
        ax.grid(True)

    def get_reward(self, x: int, y: int) -> float:
        """타일 타입에 따른 보상을 반환합니다."""
        tile_type = self.grid_map.get_tile_type(x, y)
        return self.grid_map.get_reward(x, y, tile_type == TileType.GOAL)

    def render(self, V=None, policy=None, print_value=True, goals=None):
        """그리드 월드 시각화
        
        Args:
            V: 상태 가치 함수 (numpy array)
            policy: 정책 (numpy array)
            print_value: 가치 함수 값을 표시할지 여부
        """
        self.set_figure()
        ax = self.ax

        # 가치 함수 시각화
        if V is not None:
            color_list = ['red', 'white', 'green']
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'colormap_name', color_list)

            vmax, vmin = V.max(), V.min()
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            vmax = 1 if vmax < 1 else vmax
            vmin = -1 if vmin > -1 else vmin

            ax.pcolormesh(np.flipud(V), cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)

        # 그리드 타일과 보상 표시
        for y in range(self.ys):
            for x in range(self.xs):
                # 타일 타입에 따른 색상 설정
                tile_type = self.grid_map.get_tile_type(x, y)
                if goals and (x, y) in goals:
                    tile_type = TileType.GOAL
                
                if tile_type == TileType.WALL:
                    ax.add_patch(plt.Rectangle((x, self.ys-y-1), 1, 1, fc=(0.0, 0.0, 0.0, 0.9)))  # 검정색
                elif tile_type == TileType.CONSTRUCTION:
                    ax.add_patch(plt.Rectangle((x, self.ys-y-1), 1, 1, fc=(1.0, 1.0, 0.0, 1.0)))  # 노란색
                elif tile_type == TileType.GOAL:
                    ax.add_patch(plt.Rectangle((x, self.ys-y-1), 1, 1, fc=(0.0, 0.0, 0.9, 0.5)))  # 파란색 (목표 지점)
                
                # # 보상 표시
                # reward = self.get_reward(x, y)
                # if tile_type not in [TileType.WALL, TileType.CONSTRUCTION, TileType.GOAL]:  # 벽, 공사길, 목표지점이 아닌 경우에만 보상 표시
                #     txt = f'R {reward:.1f}'
                #     ax.text(x+.1, self.ys-y-0.9, txt)

                # 가치 함수 값 표시
                if V is not None and tile_type not in [TileType.WALL]:
                    if print_value:
                        if tile_type == TileType.GOAL:
                            ax.text(x+0.52, self.ys-y-0.48, "(GOAL)", fontsize=10, ha='center', va='center')
                        else:
                            ax.text(x+0.5, self.ys-y-0.2, f"{V[y, x]:.2f}", fontsize=8)

                # 정책에 따른 화살표 표시
                if policy is not None and tile_type not in [TileType.WALL, TileType.GOAL]:
                    best_action = np.argmax(policy[x, y])  # x, y 순서로!
                    arrows = ["↑", "→", "↓", "←"]  # Action Enum 순서와 일치!
                    offsets = [(0, 0.1), (0.1, 0), (0, -0.1), (-0.1, 0)]
                    arrow = arrows[best_action]
                    offset = offsets[best_action]
                    ax.text(x+0.5+offset[0], self.ys-y-0.5+offset[1], arrow, fontsize=12, ha='center', va='center')

        plt.show() 