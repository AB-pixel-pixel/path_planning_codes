import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import heapq
from typing import List, Tuple, Set, Dict, Optional

class GridMap:
    """栅格地图类"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        
    def add_obstacle(self, x: int, y: int):
        """添加障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
            
    def add_obstacles_rect(self, x: int, y: int, w: int, h: int):
        """添加矩形障碍物区域"""
        for i in range(y, min(y + h, self.height)):
            for j in range(x, min(x + w, self.width)):
                self.grid[i, j] = 1
    
    def set_start(self, x: int, y: int):
        """设置起点"""
        self.start = (x, y)
        
    def set_goal(self, x: int, y: int):
        """设置终点"""
        self.goal = (x, y)
        
    def is_valid(self, x: int, y: int) -> bool:
        """检查坐标是否有效且无障碍"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.grid[y, x] == 0)
    
    def get_neighbors(self, x: int, y: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        """获取邻居节点"""
        neighbors = []
        # 4-连通
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 8-连通
        if diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors


class PathPlanner:
    """路径规划器"""
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        self.visited = set()
        self.path = []
        
    def bfs(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List, Set]:
        """广度优先搜索"""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == goal:
                return path, visited
            
            for nx, ny in self.grid_map.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return [], visited
    
    def dfs(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List, Set]:
        """深度优先搜索"""
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            (x, y), path = stack.pop()
            
            if (x, y) == goal:
                return path, visited
            
            for nx, ny in self.grid_map.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append(((nx, ny), path + [(nx, ny)]))
        
        return [], visited
    
    def dijkstra(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List, Set, Dict]:
        """Dijkstra算法"""
        pq = [(0, start, [start])]
        visited = set()
        distances = {start: 0}
        
        while pq:
            dist, (x, y), path = heapq.heappop(pq)
            
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            if (x, y) == goal:
                return path, visited, distances
            
            for nx, ny in self.grid_map.get_neighbors(x, y, diagonal=True):
                if (nx, ny) not in visited:
                    # 对角线移动成本为sqrt(2)，否则为1
                    cost = 1.414 if abs(nx - x) + abs(ny - y) == 2 else 1.0
                    new_dist = dist + cost
                    
                    if (nx, ny) not in distances or new_dist < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_dist
                        heapq.heappush(pq, (new_dist, (nx, ny), path + [(nx, ny)]))
        
        return [], visited, distances
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发式函数 (欧几里得距离)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List, Set, Dict]:
        """A*算法"""
        pq = [(0, start, [start], 0)]  # (f, node, path, g)
        visited = set()
        g_scores = {start: 0}
        
        while pq:
            f, (x, y), path, g = heapq.heappop(pq)
            
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            if (x, y) == goal:
                return path, visited, g_scores
            
            for nx, ny in self.grid_map.get_neighbors(x, y, diagonal=True):
                if (nx, ny) not in visited:
                    cost = 1.414 if abs(nx - x) + abs(ny - y) == 2 else 1.0
                    new_g = g + cost
                    
                    if (nx, ny) not in g_scores or new_g < g_scores[(nx, ny)]:
                        g_scores[(nx, ny)] = new_g
                        h = self.heuristic((nx, ny), goal)
                        f = new_g + h
                        heapq.heappush(pq, (f, (nx, ny), path + [(nx, ny)], new_g))
        
        return [], visited, g_scores


class Visualizer:
    """可视化工具"""
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        
    def visualize_map_and_path(self, path: List, visited: Set, title: str = "Path Planning"):
        """可视化地图和路径"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图：栅格地图视图
        self._draw_grid_map(ax1, path, visited)
        ax1.set_title(f'{title} - Grid Map View', fontsize=14, fontweight='bold')
        
        # 右图：图结构视图
        self._draw_graph_view(ax2, path, visited)
        ax2.set_title(f'{title} - Graph View', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _draw_grid_map(self, ax, path: List, visited: Set):
        """绘制栅格地图"""
        # 绘制网格
        for i in range(self.grid_map.height):
            for j in range(self.grid_map.width):
                if self.grid_map.grid[i, j] == 1:
                    # 障碍物
                    rect = patches.Rectangle((j, i), 1, 1, 
                                            linewidth=1, edgecolor='gray',
                                            facecolor='black')
                    ax.add_patch(rect)
                else:
                    # 空白格子
                    rect = patches.Rectangle((j, i), 1, 1, 
                                            linewidth=0.5, edgecolor='lightgray',
                                            facecolor='white')
                    ax.add_patch(rect)
        
        # 绘制访问过的节点
        for (x, y) in visited:
            if (x, y) not in path:
                rect = patches.Rectangle((x, y), 1, 1, 
                                        linewidth=0.5, edgecolor='lightgray',
                                        facecolor='lightblue', alpha=0.5)
                ax.add_patch(rect)
        
        # 绘制路径
        if path:
            for i, (x, y) in enumerate(path):
                if i == 0:  # 起点
                    circle = patches.Circle((x + 0.5, y + 0.5), 0.3, 
                                          color='green', zorder=10)
                    ax.add_patch(circle)
                    ax.text(x + 0.5, y + 0.5, 'S', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='white')
                elif i == len(path) - 1:  # 终点
                    circle = patches.Circle((x + 0.5, y + 0.5), 0.3, 
                                          color='red', zorder=10)
                    ax.add_patch(circle)
                    ax.text(x + 0.5, y + 0.5, 'G', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='white')
                else:
                    rect = patches.Rectangle((x, y), 1, 1, 
                                            linewidth=0.5, edgecolor='lightgray',
                                            facecolor='yellow', alpha=0.7)
                    ax.add_patch(rect)
            
            # 绘制路径线
            path_x = [x + 0.5 for x, y in path]
            path_y = [y + 0.5 for x, y in path]
            ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.6, zorder=5)
        
        ax.set_xlim(0, self.grid_map.width)
        ax.set_ylim(0, self.grid_map.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 添加图例
        legend_elements = [
            patches.Patch(facecolor='black', edgecolor='gray', label='Obstacle'),
            patches.Patch(facecolor='lightblue', alpha=0.5, label='Visited'),
            patches.Patch(facecolor='yellow', alpha=0.7, label='Path'),
            patches.Patch(facecolor='green', label='Start'),
            patches.Patch(facecolor='red', label='Goal')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _draw_graph_view(self, ax, path: List, visited: Set):
        """绘制图结构视图"""
        # 收集所有可达节点（非障碍物）
        nodes = []
        for i in range(self.grid_map.height):
            for j in range(self.grid_map.width):
                if self.grid_map.grid[i, j] == 0:
                    nodes.append((j, i))
        
        # 绘制边（连接关系）
        for (x, y) in nodes:
            neighbors = self.grid_map.get_neighbors(x, y, diagonal=True)
            for nx, ny in neighbors:
                ax.plot([x + 0.5, nx + 0.5], [y + 0.5, ny + 0.5], 
                       'lightgray', linewidth=0.5, alpha=0.3, zorder=1)
        
        # 绘制节点
        for (x, y) in nodes:
            if (x, y) in visited:
                if (x, y) in path:
                    ax.plot(x + 0.5, y + 0.5, 'o', color='yellow', 
                           markersize=8, zorder=5)
                else:
                    ax.plot(x + 0.5, y + 0.5, 'o', color='lightblue', 
                           markersize=6, alpha=0.7, zorder=3)
            else:
                ax.plot(x + 0.5, y + 0.5, 'o', color='lightgray', 
                       markersize=4, alpha=0.5, zorder=2)
        
        # 绘制路径上的边
        if path:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                ax.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], 
                       'r-', linewidth=3, alpha=0.8, zorder=6)
            
            # 绘制起点和终点
            start_x, start_y = path[0]
            goal_x, goal_y = path[-1]
            ax.plot(start_x + 0.5, start_y + 0.5, 'o', color='green', 
                   markersize=15, zorder=10, label='Start')
            ax.plot(goal_x + 0.5, goal_y + 0.5, 's', color='red', 
                   markersize=15, zorder=10, label='Goal')
        
        ax.set_xlim(0, self.grid_map.width)
        ax.set_ylim(0, self.grid_map.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def compare_algorithms(self, algorithms_results: Dict):
        """比较多个算法的结果"""
        n_algorithms = len(algorithms_results)
        fig, axes = plt.subplots(2, n_algorithms, figsize=(6*n_algorithms, 12))
        
        if n_algorithms == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, (path, visited)) in enumerate(algorithms_results.items()):
            # 上排：栅格地图视图
            self._draw_grid_map(axes[0, idx], path, visited)
            axes[0, idx].set_title(f'{name}\nGrid Map', fontsize=12, fontweight='bold')
            
            # 下排：图视图
            self._draw_graph_view(axes[1, idx], path, visited)
            axes[1, idx].set_title(f'{name}\nGraph View', fontsize=12, fontweight='bold')
            
            # 添加统计信息
            path_length = len(path) - 1 if path else 0
            nodes_explored = len(visited)
            info_text = f'Path Length: {path_length}\nNodes Explored: {nodes_explored}'
            axes[0, idx].text(0.02, 0.98, info_text, transform=axes[0, idx].transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()


# ==================== 使用示例 ====================

def main():
    # 创建栅格地图
    grid_map = GridMap(width=20, height=15)
    
    # 添加障碍物
    grid_map.add_obstacles_rect(5, 3, 2, 8)
    grid_map.add_obstacles_rect(10, 1, 3, 6)
    grid_map.add_obstacles_rect(15, 8, 2, 5)
    grid_map.add_obstacles_rect(8, 10, 5, 2)
    
    # 设置起点和终点
    start = (2, 2)
    goal = (17, 12)
    grid_map.set_start(*start)
    grid_map.set_goal(*goal)
    
    # 创建路径规划器
    planner = PathPlanner(grid_map)
    
    # 创建可视化器
    visualizer = Visualizer(grid_map)
    
    # 运行不同的算法
    print("运行 BFS...")
    path_bfs, visited_bfs = planner.bfs(start, goal)
    
    print("运行 DFS...")
    path_dfs, visited_dfs = planner.dfs(start, goal)
    
    print("运行 Dijkstra...")
    path_dijkstra, visited_dijkstra, _ = planner.dijkstra(start, goal)
    
    print("运行 A*...")
    path_astar, visited_astar, _ = planner.astar(start, goal)
    
    # 比较所有算法
    algorithms_results = {
        'BFS': (path_bfs, visited_bfs),
        'DFS': (path_dfs, visited_dfs),
        'Dijkstra': (path_dijkstra, visited_dijkstra),
        'A*': (path_astar, visited_astar)
    }
    
    visualizer.compare_algorithms(algorithms_results)
    
    # 打印统计信息
    print("\n算法性能比较:")
    print("-" * 60)
    for name, (path, visited) in algorithms_results.items():
        path_length = len(path) - 1 if path else 0
        print(f"{name:12s} | 路径长度: {path_length:3d} | 探索节点: {len(visited):4d}")


if __name__ == "__main__":
    main()
