import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import heapq
import time
import random
from matplotlib.gridspec import GridSpec

# ==================== A* 算法实现 ====================
class AStarPlanner:
    def __init__(self, grid_map, start, goal):
        self.grid_map = grid_map
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid_map.shape
        self.explored_nodes = []
        
    def heuristic(self, pos1, pos2):
        """欧几里得距离启发函数"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, pos):
        """获取8邻域的邻居节点"""
        neighbors = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            if (0 <= new_x < self.rows and 0 <= new_y < self.cols and 
                self.grid_map[new_x, new_y] == 0):
                if dx != 0 and dy != 0:
                    if (self.grid_map[pos[0] + dx, pos[1]] == 0 and 
                        self.grid_map[pos[0], pos[1] + dy] == 0):
                        neighbors.append((new_x, new_y))
                else:
                    neighbors.append((new_x, new_y))
                    
        return neighbors
    
    def plan(self):
        """A*路径规划"""
        start_time = time.time()
        
        counter = 0
        open_set = [(0, counter, self.start)]
        counter += 1
        
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            self.explored_nodes.append(current)
            closed_set.add(current)
            
            if current == self.goal:
                path = self.reconstruct_path(came_from, current)
                end_time = time.time()
                return {
                    'path': path,
                    'explored_nodes': self.explored_nodes,
                    'time': end_time - start_time,
                    'path_length': self.calculate_path_length(path),
                    'nodes_explored': len(self.explored_nodes)
                }
            
            for neighbor in self.get_neighbors(current):
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = np.sqrt(2) if (dx + dy == 2) else 1.0
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self.heuristic(neighbor, self.goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
        
        end_time = time.time()
        return {
            'path': None,
            'explored_nodes': self.explored_nodes,
            'time': end_time - start_time,
            'path_length': float('inf'),
            'nodes_explored': len(self.explored_nodes)
        }
    
    def reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def calculate_path_length(self, path):
        """计算路径长度"""
        if not path:
            return 0
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
        return length


# ==================== RRT 算法实现 ====================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRTPlanner:
    def __init__(self, grid_map, start, goal, step_size=2.0, max_iter=5000, goal_sample_rate=0.15):
        self.grid_map = grid_map
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.rows, self.cols = grid_map.shape
        self.node_list = [self.start]
        self.explored_nodes = []
        
    def plan(self):
        """RRT路径规划"""
        start_time = time.time()
        
        for i in range(self.max_iter):
            if random.random() < self.goal_sample_rate:
                rnd_node = Node(self.goal.x, self.goal.y)
            else:
                rnd_node = self.get_random_node()
            
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)
            
            if self.check_collision(nearest_node, new_node):
                self.node_list.append(new_node)
                self.explored_nodes.append((int(new_node.x), int(new_node.y)))
                
                if self.calc_distance(new_node, self.goal) <= self.step_size:
                    final_node = self.steer(new_node, self.goal)
                    if self.check_collision(new_node, final_node):
                        end_time = time.time()
                        path = self.generate_final_path(final_node)
                        return {
                            'path': path,
                            'explored_nodes': self.explored_nodes,
                            'tree': self.node_list,
                            'time': end_time - start_time,
                            'path_length': self.calculate_path_length(path),
                            'nodes_explored': len(self.explored_nodes)
                        }
        
        end_time = time.time()
        return {
            'path': None,
            'explored_nodes': self.explored_nodes,
            'tree': self.node_list,
            'time': end_time - start_time,
            'path_length': float('inf'),
            'nodes_explored': len(self.explored_nodes)
        }
    
    def get_random_node(self):
        x = random.uniform(0, self.rows - 1)
        y = random.uniform(0, self.cols - 1)
        return Node(x, y)
    
    def get_nearest_node(self, rnd_node):
        distances = [self.calc_distance(node, rnd_node) for node in self.node_list]
        min_index = distances.index(min(distances))
        return self.node_list[min_index]
    
    def steer(self, from_node, to_node):
        new_node = Node(from_node.x, from_node.y)
        distance = self.calc_distance(from_node, to_node)
        
        if distance <= self.step_size:
            new_node.x = to_node.x
            new_node.y = to_node.y
        else:
            theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node.x = from_node.x + self.step_size * np.cos(theta)
            new_node.y = from_node.y + self.step_size * np.sin(theta)
        
        new_node.parent = from_node
        return new_node
    
    def calc_distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def check_collision(self, from_node, to_node):
        steps = int(self.calc_distance(from_node, to_node) / 0.5)
        if steps == 0:
            steps = 1
            
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
                return False
            
            if self.grid_map[int(x), int(y)] == 1:
                return False
        
        return True
    
    def generate_final_path(self, goal_node):
        path = [(int(goal_node.x), int(goal_node.y))]
        node = goal_node
        
        while node.parent is not None:
            node = node.parent
            path.append((int(node.x), int(node.y)))
        
        path.reverse()
        return path
    
    def calculate_path_length(self, path):
        if not path:
            return 0
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
        return length


# ==================== 创建RRT优势场景的地图 ====================
def create_rrt_favorable_map(size=(80, 80)):
    """
    创建一个对RRT更有利的复杂迷宫环境
    特点：
    1. 大量稀疏分布的障碍物
    2. 多条可选路径但相互干扰
    3. 狭窄通道
    4. A*需要探索大量节点，而RRT可以快速采样通过
    """
    grid_map = np.zeros(size)
    
    # 添加边界
    grid_map[0, :] = 1
    grid_map[-1, :] = 1
    grid_map[:, 0] = 1
    grid_map[:, -1] = 1
    
    # 创建复杂的"之"字形障碍物墙，但留有多个小缺口
    # 这种结构会让A*探索大量死路，但RRT可能更快找到缺口
    
    # 第一层横墙 (上部)
    grid_map[15, 10:70] = 1
    grid_map[15, 30:35] = 0  # 小缺口1
    grid_map[15, 55:60] = 0  # 小缺口2
    
    # 第二层竖墙 (右侧)
    grid_map[15:45, 65] = 1
    grid_map[30:35, 65] = 0  # 小缺口
    
    # 第三层横墙 (中部)
    grid_map[40, 15:70] = 1
    grid_map[40, 25:30] = 0  # 小缺口1
    grid_map[40, 48:53] = 0  # 小缺口2
    
    # 第四层竖墙 (左侧)
    grid_map[40:65, 20] = 1
    grid_map[52:57, 20] = 0  # 小缺口
    
    # 第五层横墙 (下部)
    grid_map[60, 20:75] = 1
    grid_map[60, 40:45] = 0  # 小缺口1
    grid_map[60, 62:67] = 0  # 小缺口2
    
    # 添加一些随机散布的小障碍物增加复杂度
    random.seed(42)  # 固定随机种子保证可重现
    for _ in range(30):
        obs_size = random.randint(2, 4)
        obs_x = random.randint(5, size[0] - obs_size - 5)
        obs_y = random.randint(5, size[1] - obs_size - 5)
        grid_map[obs_x:obs_x+obs_size, obs_y:obs_y+obs_size] = 1
    
    return grid_map


def visualize_comparison(grid_map, start, goal, astar_result, rrt_result):
    """可视化对比两种算法"""
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })
    
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. A*算法可视化
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(grid_map, cmap='Greys', origin='lower')
    ax1.set_title(f'A* Algorithm\nTime: {astar_result["time"]:.4f}s', 
                  fontsize=22, fontweight='bold', pad=15)
    
    if astar_result['explored_nodes']:
        explored = np.array(astar_result['explored_nodes'])
        ax1.scatter(explored[:, 1], explored[:, 0], c='lightblue', s=8, alpha=0.4, label='Explored')
    
    if astar_result['path']:
        path = np.array(astar_result['path'])
        ax1.plot(path[:, 1], path[:, 0], 'r-', linewidth=3.5, label='Path')
    
    ax1.scatter(start[1], start[0], c='green', s=300, marker='o', 
                edgecolors='black', linewidth=2.5, label='Start', zorder=5)
    ax1.scatter(goal[1], goal[0], c='red', s=300, marker='*', 
                edgecolors='black', linewidth=2.5, label='Goal', zorder=5)
    ax1.legend(loc='upper right', fontsize=16, framealpha=0.9)
    ax1.set_xlabel('Y', fontsize=18, fontweight='bold')
    ax1.set_ylabel('X', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    
    # 2. RRT算法可视化
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(grid_map, cmap='Greys', origin='lower')
    ax2.set_title(f'RRT Algorithm\nTime: {rrt_result["time"]:.4f}s', 
                  fontsize=22, fontweight='bold', pad=15)
    
    if 'tree' in rrt_result:
        for node in rrt_result['tree'][::2]:  # 每隔一个绘制，避免太密集
            if node.parent:
                ax2.plot([node.y, node.parent.y], [node.x, node.parent.x], 
                        'c-', linewidth=0.5, alpha=0.3)
    
    if rrt_result['path']:
        path = np.array(rrt_result['path'])
        ax2.plot(path[:, 1], path[:, 0], 'r-', linewidth=3.5, label='Path')
    
    ax2.scatter(start[1], start[0], c='green', s=300, marker='o', 
                edgecolors='black', linewidth=2.5, label='Start', zorder=5)
    ax2.scatter(goal[1], goal[0], c='red', s=300, marker='*', 
                edgecolors='black', linewidth=2.5, label='Goal', zorder=5)
    ax2.legend(loc='upper right', fontsize=16, framealpha=0.9)
    ax2.set_xlabel('Y', fontsize=18, fontweight='bold')
    ax2.set_ylabel('X', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    
    # 3. 路径对比
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(grid_map, cmap='Greys', origin='lower')
    ax3.set_title('Path Comparison', fontsize=22, fontweight='bold', pad=15)
    
    if astar_result['path']:
        path_astar = np.array(astar_result['path'])
        ax3.plot(path_astar[:, 1], path_astar[:, 0], 'b-', linewidth=3, 
                label='A* Path', alpha=0.8)
    
    if rrt_result['path']:
        path_rrt = np.array(rrt_result['path'])
        ax3.plot(path_rrt[:, 1], path_rrt[:, 0], 'orange', linewidth=3, 
                label='RRT Path', alpha=0.8)
    
    ax3.scatter(start[1], start[0], c='green', s=300, marker='o', 
                edgecolors='black', linewidth=2.5, label='Start', zorder=5)
    ax3.scatter(goal[1], goal[0], c='red', s=300, marker='*', 
                edgecolors='black', linewidth=2.5, label='Goal', zorder=5)
    ax3.legend(loc='upper right', fontsize=16, framealpha=0.9)
    ax3.set_xlabel('Y', fontsize=18, fontweight='bold')
    ax3.set_ylabel('X', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3, linewidth=1.5)
    
    # 4-6. 性能指标对比
    metrics = ['Time (s)', 'Path Length', 'Nodes Explored']
    astar_metrics = [astar_result['time'], astar_result['path_length'], astar_result['nodes_explored']]
    rrt_metrics = [rrt_result['time'], rrt_result['path_length'], rrt_result['nodes_explored']]
    
    colors = ['#3498db', '#e74c3c']
    
    for idx, (metric_name, astar_val, rrt_val) in enumerate(zip(metrics, astar_metrics, rrt_metrics)):
        ax = fig.add_subplot(gs[1, idx])
        
        bars = ax.bar(['A*', 'RRT'], [astar_val, rrt_val], color=colors, 
                     edgecolor='black', linewidth=2, width=0.6)
        ax.set_title(metric_name, fontsize=22, fontweight='bold', pad=15)
        ax.set_ylabel('Value', fontsize=18, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=20)
        
        # 标注获胜者
        if idx == 0:  # Time
            winner_idx = 0 if astar_val < rrt_val else 1
        elif idx == 1:  # Path length
            winner_idx = 0 if astar_val < rrt_val else 1
        else:  # Nodes explored
            winner_idx = 0 if astar_val < rrt_val else 1
        
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(4)
    
    plt.suptitle('RRT Favorable Scenario: Complex Maze Environment', 
                fontsize=28, fontweight='bold', y=0.98)
    
    return fig


def main():
    print("=" * 70)
    print("RRT Favorable Scenario - Complex Maze Comparison")
    print("=" * 70)
    
    print("\n[1/4] Creating complex maze environment (RRT favorable)...")
    grid_map = create_rrt_favorable_map(size=(80, 80))
    
    # 设置起点和终点 - 穿越复杂迷宫
    start = (5, 5)
    goal = (75, 75)
    
    grid_map[start] = 0
    grid_map[goal] = 0
    
    print(f"      Start: {start}, Goal: {goal}")
    print(f"      Map size: {grid_map.shape}")
    print(f"      Scenario: Complex maze with narrow passages")
    
    print("\n[2/4] Running A* algorithm...")
    astar = AStarPlanner(grid_map, start, goal)
    astar_result = astar.plan()
    
    if astar_result['path']:
        print(f"      ✓ Path found!")
        print(f"      - Time: {astar_result['time']:.4f} seconds")
        print(f"      - Path length: {astar_result['path_length']:.2f}")
        print(f"      - Nodes explored: {astar_result['nodes_explored']}")
    else:
        print(f"      ✗ No path found")
    
    print("\n[3/4] Running RRT algorithm...")
    rrt = RRTPlanner(grid_map, start, goal, step_size=3.0, max_iter=8000, goal_sample_rate=0.15)
    rrt_result = rrt.plan()
    
    if rrt_result['path']:
        print(f"      ✓ Path found!")
        print(f"      - Time: {rrt_result['time']:.4f} seconds")
        print(f"      - Path length: {rrt_result['path_length']:.2f}")
        print(f"      - Nodes explored: {rrt_result['nodes_explored']}")
    else:
        print(f"      ✗ No path found")
    
    print("\n[4/4] Generating comparison visualization...")
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY - RRT FAVORABLE SCENARIO")
    print("=" * 70)
    
    if astar_result['path'] and rrt_result['path']:
        print(f"\n{'Metric':<25} {'A*':<18} {'RRT':<18} {'Advantage':<15}")
        print("-" * 70)
        
        time_diff = ((astar_result['time'] - rrt_result['time']) / rrt_result['time'] * 100)
        time_winner = "RRT" if rrt_result['time'] < astar_result['time'] else "A*"
        print(f"{'Time (s)':<25} {astar_result['time']:<18.4f} {rrt_result['time']:<18.4f} {time_winner} ({abs(time_diff):.1f}% faster)")
        
        length_diff = ((rrt_result['path_length'] - astar_result['path_length']) / astar_result['path_length'] * 100)
        length_winner = "A*" if astar_result['path_length'] < rrt_result['path_length'] else "RRT"
        print(f"{'Path Length':<25} {astar_result['path_length']:<18.2f} {rrt_result['path_length']:<18.2f} {length_winner}")
        
        nodes_diff = ((astar_result['nodes_explored'] - rrt_result['nodes_explored']) / astar_result['nodes_explored'] * 100)
        nodes_winner = "RRT" if rrt_result['nodes_explored'] < astar_result['nodes_explored'] else "A*"
        print(f"{'Nodes Explored':<25} {astar_result['nodes_explored']:<18} {rrt_result['nodes_explored']:<18} {nodes_winner} ({abs(nodes_diff):.1f}% fewer)")
        
        print("\n" + "=" * 70)
        print("🎯 Key Insights:")
        print(f"   - RRT's random sampling helps it find narrow passages faster")
        print(f"   - A* explores systematically but may check many dead ends")
        print(f"   - In sparse, complex environments, RRT can be more efficient")
        print("=" * 70)
    
    fig = visualize_comparison(grid_map, start, goal, astar_result, rrt_result)
    plt.show()
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()
