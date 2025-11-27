import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import heapq
from typing import List, Tuple, Set, Dict, Optional
import matplotlib.lines as mlines

class DijkstraVisualizer:
    """Dijkstra算法可视化器(支持8邻域)"""
    
    # 颜色方案
    COLORS = {
        'obstacle': '#2C3E50',
        'start': '#2ECC71',
        'goal': '#E74C3C',
        'current': '#FF6B6B',
        'visited': '#AED6F1',
        'in_queue': '#F9E79F',
        'exploring': '#82E0AA',
        'path': '#FF0000',
        'edge': 'lightgray',
    }
    
    def __init__(self, width: int, height: int, use_8_neighbors: bool = True):
        """
        初始化可视化器
        
        Args:
            width: 网格宽度
            height: 网格高度
            use_8_neighbors: 是否使用8邻域(True)或4邻域(False)
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        self.use_8_neighbors = use_8_neighbors
        
        # 定义邻域方向和代价
        if use_8_neighbors:
            # 8邻域:4个正交方向 + 4个对角线方向
            self.directions = [
                (0, -1, 1.0),    # 上
                (1, 0, 1.0),     # 右
                (0, 1, 1.0),     # 下
                (-1, 0, 1.0),    # 左
                (1, -1, 1.414),  # 右上
                (1, 1, 1.414),   # 右下
                (-1, 1, 1.414),  # 左下
                (-1, -1, 1.414), # 左上
            ]
        else:
            # 4邻域
            self.directions = [
                (0, -1, 1.0),    # 上
                (1, 0, 1.0),     # 右
                (0, 1, 1.0),     # 下
                (-1, 0, 1.0),    # 左
            ]
    
    def add_obstacle(self, x: int, y: int) -> None:
        """添加障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
    
    def add_obstacles_rect(self, x: int, y: int, w: int, h: int) -> None:
        """添加矩形障碍物"""
        for i in range(y, min(y + h, self.height)):
            for j in range(x, min(x + w, self.width)):
                self.grid[i, j] = 1
    
    def set_start(self, x: int, y: int) -> None:
        """设置起点"""
        self.start = (x, y)
    
    def set_goal(self, x: int, y: int) -> None:
        """设置终点"""
        self.goal = (x, y)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int, float]]:
        """
        获取邻居节点及其代价
        
        Returns:
            List of (nx, ny, cost) tuples
        """
        neighbors = []
        for dx, dy, cost in self.directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.grid[ny, nx] == 0):
                neighbors.append((nx, ny, cost))
        return neighbors
    
    def dijkstra_step_by_step(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Dict]:
        """
        Dijkstra逐步搜索,返回每步的状态
        
        Returns:
            每步状态信息的列表
        """
        pq = [(0, start)]
        distances = {start: 0}
        visited = set()
        parent = {start: None}
        steps = []
        
        # 初始状态
        steps.append(self._create_step_info(
            'init', start, pq, distances, visited, parent, None, False, []
        ))
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # 记录当前探索节点
            steps.append(self._create_step_info(
                'dequeue', current, pq, distances, visited, parent, 
                None, False, [], current_dist
            ))
            
            # 找到目标
            if current == goal:
                path = self._reconstruct_path(parent, goal)
                steps.append(self._create_step_info(
                    'found', current, pq, distances, visited, parent,
                    path, True, [], current_dist
                ))
                return steps
            
            # 探索邻居
            new_neighbors = []
            for nx, ny, cost in self.get_neighbors(*current):
                neighbor = (nx, ny)
                new_distance = current_dist + cost
                
                if neighbor not in visited:
                    if neighbor not in distances or new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parent[neighbor] = current
                        heapq.heappush(pq, (new_distance, neighbor))
                        new_neighbors.append(neighbor)
            
            # 记录探索邻居后的状态
            if new_neighbors:
                steps.append(self._create_step_info(
                    'explore', current, pq, distances, visited, parent,
                    None, False, new_neighbors, current_dist
                ))
        
        # 未找到路径
        steps.append(self._create_step_info(
            'no_path', None, [], distances, visited, parent, None, False, []
        ))
        
        return steps
    
    def _create_step_info(self, step_type: str, current: Optional[Tuple[int, int]], 
                         pq: List, distances: Dict, visited: Set, parent: Dict,
                         path: Optional[List], found: bool, exploring: List,
                         current_dist: float = 0) -> Dict:
        """创建步骤信息字典"""
        info = {
            'step_type': step_type,
            'current': current,
            'pq': list(pq),
            'distances': distances.copy(),
            'visited': visited.copy(),
            'parent': parent.copy(),
            'path': path,
            'found': found,
            'exploring': exploring,
        }
        if current_dist > 0 or step_type in ['dequeue', 'explore', 'found']:
            info['current_dist'] = current_dist
        return info
    
    def _reconstruct_path(self, parent: Dict, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重构路径"""
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path
    
    def _draw_grid_and_obstacles(self, ax) -> None:
        """绘制网格和障碍物"""
        # 绘制图结构的边
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    for nx, ny, cost in self.get_neighbors(j, i):
                        ax.plot([j + 0.5, nx + 0.5], [i + 0.5, ny + 0.5],
                               color=self.COLORS['edge'], linewidth=1, 
                               alpha=0.3, zorder=1)
        
        # 绘制障碍物
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle(
                        (j, i), 1, 1, linewidth=1, 
                        edgecolor='black', facecolor=self.COLORS['obstacle']
                    )
                    ax.add_patch(rect)
    
    def _draw_node_with_distance(self, ax, x: int, y: int, distance: float,
                                 color: str, marker: str, size: int, 
                                 zorder: int, label_bg: str = 'white') -> None:
        """绘制带距离标签的节点"""
        ax.plot(x + 0.5, y + 0.5, marker, color=color, 
               markersize=size, zorder=zorder, alpha=0.8)
        
        # 显示距离
        ax.text(x + 0.5, y + 0.85, f'{distance:.1f}',
               ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor=label_bg, alpha=0.9))
    
    def _draw_dijkstra_state(self, ax, step_info: Dict, title: str) -> None:
        """绘制Dijkstra某一步的状态"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # 提取状态信息
        current = step_info['current']
        visited = step_info['visited']
        exploring = step_info['exploring']
        path = step_info['path']
        distances = step_info['distances']
        pq_nodes = [node for dist, node in step_info['pq']]
        
        # 绘制网格和障碍物
        self._draw_grid_and_obstacles(ax)
        
        # 绘制已访问节点
        for (x, y) in visited:
            if (x, y) not in [self.start, self.goal]:
                if (x, y) in distances:
                    self._draw_node_with_distance(
                        ax, x, y, distances[(x, y)], 
                        self.COLORS['visited'], 'o', 15, 5
                    )
        
        # 绘制优先队列中的节点
        for (x, y) in pq_nodes:
            if (x, y) not in [self.start, self.goal]:
                if (x, y) in distances:
                    self._draw_node_with_distance(
                        ax, x, y, distances[(x, y)],
                        self.COLORS['in_queue'], 's', 14, 6, 'yellow'
                    )
        
        # 绘制正在探索的邻居
        for (x, y) in exploring:
            ax.plot(x + 0.5, y + 0.5, 'D', color=self.COLORS['exploring'],
                   markersize=12, zorder=7, alpha=0.9)
        
        # 绘制当前节点
        if current:
            cx, cy = current
            ax.plot(cx + 0.5, cy + 0.5, 'o', color=self.COLORS['current'],
                   markersize=20, zorder=8,
                   markeredgecolor='darkred', markeredgewidth=2)
            if 'current_dist' in step_info:
                ax.text(cx + 0.5, cy + 0.85, f'{step_info["current_dist"]:.1f}',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='red', alpha=0.7, edgecolor='darkred'))
        
        # 绘制起点
        self._draw_special_point(ax, *self.start, 'S', self.COLORS['start'], 'darkgreen')
        
        # 绘制终点
        self._draw_special_point(ax, *self.goal, 'G', self.COLORS['goal'], 'darkred', 's')
        
        # 绘制路径
        if path:
            path_x = [x + 0.5 for x, y in path]
            path_y = [y + 0.5 for x, y in path]
            ax.plot(path_x, path_y, color=self.COLORS['path'], 
                   linewidth=4, alpha=0.7, zorder=9)
        
        # 设置坐标轴
        self._setup_axes(ax)
        
        # 添加图例
        self._add_legend(ax, path, distances)
        
        # 添加统计信息
        self._add_statistics(ax, visited, pq_nodes, exploring, distances, path)
    
    def _draw_special_point(self, ax, x: int, y: int, label: str, 
                           color: str, edge_color: str, marker: str = 'o') -> None:
        """绘制特殊点(起点/终点)"""
        ax.plot(x + 0.5, y + 0.5, marker, color=color,
               markersize=22, zorder=10,
               markeredgecolor=edge_color, markeredgewidth=3)
        ax.text(x + 0.5, y + 0.5, label, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
    
    def _setup_axes(self, ax) -> None:
        """设置坐标轴"""
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
    
    def _add_legend(self, ax, path: Optional[List], distances: Dict) -> None:
        """添加图例"""
        neighbor_type = "8-Connected" if self.use_8_neighbors else "4-Connected"
        
        legend_elements = [
            mlines.Line2D([], [], color=self.COLORS['start'], marker='o', 
                         linestyle='None', markersize=12, label='Start',
                         markeredgecolor='darkgreen', markeredgewidth=2),
            mlines.Line2D([], [], color=self.COLORS['goal'], marker='s',
                         linestyle='None', markersize=12, label='Goal',
                         markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color=self.COLORS['current'], marker='o',
                         linestyle='None', markersize=12, label='Current',
                         markeredgecolor='darkred', markeredgewidth=2),
            mlines.Line2D([], [], color=self.COLORS['in_queue'], marker='s',
                         linestyle='None', markersize=10, label='In PQ'),
            mlines.Line2D([], [], color=self.COLORS['exploring'], marker='D',
                         linestyle='None', markersize=8, label='Exploring'),
            mlines.Line2D([], [], color=self.COLORS['visited'], marker='o',
                         linestyle='None', markersize=10, label='Visited'),
            mlines.Line2D([], [], color='gray', linestyle='-',
                         linewidth=1, label=neighbor_type),
        ]
        
        if path and self.goal in distances:
            legend_elements.append(
                mlines.Line2D([], [], color=self.COLORS['path'], linewidth=3,
                             label=f'Path (Cost: {distances[self.goal]:.1f})')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def _add_statistics(self, ax, visited: Set, pq_nodes: List, 
                       exploring: List, distances: Dict, path: Optional[List]) -> None:
        """添加统计信息"""
        info_text = f"Visited: {len(visited)} nodes\n"
        info_text += f"PQ Size: {len(pq_nodes)}\n"
        if exploring:
            info_text += f"Exploring: {len(exploring)} nodes"
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def visualize_static_explanation(self, filename: str = 'dijkstra_steps.png') -> None:
        """创建静态解释图"""
        steps = self.dijkstra_step_by_step(self.start, self.goal)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        axes = axes.flatten()
        
        # 选择关键帧
        key_frames = [
            (0, "Step 1: Initialization"),
            (len(steps) // 3, f"Step {len(steps)//3}: Expanding"),
            (2 * len(steps) // 3, f"Step {2*len(steps)//3}: Approaching Goal"),
            (len(steps) - 1, "Final: Path Found" if steps[-1]['found'] else "Final: No Path"),
        ]
        
        for idx, (frame_idx, title) in enumerate(key_frames):
            self._draw_dijkstra_state(axes[idx], steps[frame_idx], title)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Static diagram saved: {filename}")
    
    def create_dijkstra_animation(self, filename: str = 'dijkstra.gif', fps: int = 2) -> None:
        """创建Dijkstra搜索过程动画"""
        steps = self.dijkstra_step_by_step(self.start, self.goal)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        step_titles = {
            'init': 'Initialize: Start from Begin',
            'dequeue': 'Dequeue: Extract Min Distance Node',
            'explore': 'Explore: Update Neighbor Distances',
            'found': '✓ Path Found!',
            'no_path': '✗ No Path Exists'
        }
        
        def animate(frame_num):
            ax.clear()
            step_info = steps[frame_num]
            title = f"Step {frame_num + 1}/{len(steps)}: {step_titles.get(step_info['step_type'], '')}"
            self._draw_dijkstra_state(ax, step_info, title)
        
        anim = FuncAnimation(fig, animate, frames=len(steps),
                           interval=1000/fps, repeat=True)
        
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)
        plt.close()
        
        print(f"✓ Animation saved: {filename}")
        print(f"  Total steps: {len(steps)}")
        print(f"  Neighbors: {'8-connected' if self.use_8_neighbors else '4-connected'}")


# ==================== 使用示例 ====================

def demo_8_neighbor_simple():
    """8邻域简单场景"""
    print("=" * 60)
    print("Example: 8-Neighbor Dijkstra Visualization")
    print("=" * 60)
    
    viz = DijkstraVisualizer(width=12, height=10, use_8_neighbors=True)
    
    # 添加障碍物
    viz.add_obstacles_rect(3, 2, 2, 4)
    viz.add_obstacles_rect(7, 1, 1, 3)
    viz.add_obstacles_rect(7, 6, 2, 3)
    
    viz.set_start(1, 3)
    viz.set_goal(10, 7)
    
    # 生成可视化
    viz.visualize_static_explanation('dijkstra_8neighbor_steps.png')
    viz.create_dijkstra_animation('dijkstra_8neighbor.gif', fps=2)


def demo_comparison():
    """4邻域vs 8邻域对比"""
    print("\n" + "=" * 60)
    print("Comparison: 4-Neighbor vs 8-Neighbor")
    print("=" * 60)
    
    # 相同的地图配置
    def setup_map(viz):
        viz.add_obstacles_rect(3, 2, 2, 4)
        viz.add_obstacles_rect(7, 3, 1, 4)
        viz.set_start(1, 3)
        viz.set_goal(10, 5)
    
    # 4邻域
    viz4 = DijkstraVisualizer(width=12, height=8, use_8_neighbors=False)
    setup_map(viz4)
    viz4.create_dijkstra_animation('dijkstra_4neighbor.gif', fps=2)
    
    # 8邻域
    viz8 = DijkstraVisualizer(width=12, height=8, use_8_neighbors=True)
    setup_map(viz8)
    viz8.create_dijkstra_animation('dijkstra_8neighbor_compare.gif', fps=2)
    
    print("✓ Comparison animations created!")


if __name__ == "__main__":
    print("🎓 Dijkstra Algorithm Visualization (8-Neighbor Support)")
    print("=" * 60)
    
    # 8邻域示例
    demo_8_neighbor_simple()
    
    # 对比示例
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("✓ All visualizations completed!")
    print("=" * 60)
