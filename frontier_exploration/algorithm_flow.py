import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from collections import deque
import heapq

class TeachingFrontierExplorer:
    """教学版边界探索算法 - 强调分步骤可视化"""
    
    def __init__(self, map_size=30, sensor_range=4):
        self.map_size = map_size
        self.sensor_range = sensor_range
        
        # 地图状态: 0=未知, 1=自由空间, 2=障碍物
        self.map = np.zeros((map_size, map_size))
        
        # 真实环境
        self.ground_truth = self._generate_simple_environment()
        
        # 机器人位置
        self.robot_pos = np.array([map_size//2, map_size//2])
        self.path_history = [self.robot_pos.copy()]
        
        # 算法状态
        self.current_path = []
        self.frontiers = []
        self.frontier_clusters = []  # 用于可视化聚类
        self.target_frontier = None
        self.candidate_frontiers = []  # 候选边界点
        
        # 步骤控制
        self.step_index = 0
        self.steps = [
            "Initialization",
            "Step 1: Sensing (LiDAR Scanning)",
            "Step 2: Frontier Detection", 
            "Step 3: Frontier Clustering",
            "Step 4: Target Selection & Scoring",
            "Step 5: Path Planning (A*)",
            "Step 6: Robot Movement",
            "Exploration Complete"
        ]
        self.pause_for_teaching = True
        
    def _generate_simple_environment(self):
        """生成简单教学环境"""
        env = np.ones((self.map_size, self.map_size))
        
        # 边界墙
        env[0, :] = 2
        env[-1, :] = 2
        env[:, 0] = 2
        env[:, -1] = 2
        
        # 添加几个简单的障碍物便于教学
        # L型障碍
        env[8:15, 10:12] = 2
        env[13:15, 10:18] = 2
        
        # 矩形房间
        env[20:25, 15:20] = 2
        env[22:23, 17:18] = 1  # 门
        
        # 小障碍
        env[10:12, 20:22] = 2
        
        return env
    
    def sense_environment(self):
        """步骤1: 环境感知"""
        self.step_index = 1
        x, y = self.robot_pos
        
        # 圆形扫描区域
        for i in range(max(0, x-self.sensor_range), 
                      min(self.map_size, x+self.sensor_range+1)):
            for j in range(max(0, y-self.sensor_range), 
                          min(self.map_size, y+self.sensor_range+1)):
                dist = np.sqrt((i-x)**2 + (j-y)**2)
                if dist <= self.sensor_range:
                    self.map[i, j] = self.ground_truth[i, j]
    
    def detect_frontiers(self):
        """步骤2: 边界检测"""
        self.step_index = 2
        self.candidate_frontiers = []
        
        for i in range(1, self.map_size-1):
            for j in range(1, self.map_size-1):
                if self.map[i, j] == 1:  # 已知自由空间
                    # 检查8邻域
                    neighbors = [
                        self.map[i-1:i+2, j-1:j+2].flatten()
                    ]
                    if 0 in neighbors[0]:  # 邻接未知区域
                        self.candidate_frontiers.append((i, j))
    
    def cluster_frontiers(self):
        """步骤3: 边界聚类"""
        self.step_index = 3
        
        if not self.candidate_frontiers:
            self.frontiers = []
            self.frontier_clusters = []
            return
        
        visited = set()
        self.frontier_clusters = []
        
        for point in self.candidate_frontiers:
            if point in visited:
                continue
            
            cluster = []
            queue = deque([point])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                
                # 查找相邻边界点
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (current[0]+dx, current[1]+dy)
                        if neighbor in self.candidate_frontiers and neighbor not in visited:
                            queue.append(neighbor)
            
            if len(cluster) >= 3:  # 最小聚类大小
                self.frontier_clusters.append(cluster)
        
        # 计算每个聚类的中心点
        self.frontiers = [self._cluster_center(c) for c in self.frontier_clusters]
    
    def _cluster_center(self, cluster):
        """计算聚类中心 - 改进版：选择最靠近未知区域的边界点"""
        if not cluster:
            return None
        
        # 方法1：选择聚类中最靠近未知区域的点
        best_point = None
        max_unknown_neighbors = 0
        
        for point in cluster:
            # 计算该点周围的未知区域数量
            x, y = point
            unknown_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.map_size and 
                        0 <= ny < self.map_size and 
                        self.map[nx, ny] == 0):  # 未知区域
                        unknown_count += 1
            
            if unknown_count > max_unknown_neighbors:
                max_unknown_neighbors = unknown_count
                best_point = point
        
        return best_point if best_point else cluster[0]

    def select_target(self):
        """步骤4: 目标选择与评分"""
        self.step_index = 4
        
        if not self.frontiers:
            self.target_frontier = None
            return False
        
        # 计算每个边界的得分
        scores = []
        for frontier in self.frontiers:
            # 距离成本
            dist = np.linalg.norm(np.array(frontier) - self.robot_pos)
            
            # 信息增益：计算该frontier周围的未知区域数量
            info_gain = self._compute_info_gain(frontier)
            
            # 聚类大小作为额外增益
            cluster_size = len([c for c in self.frontier_clusters 
                            if self._cluster_center(c) == frontier])
            
            # 得分公式: 信息增益 + 聚类大小奖励 - λ*距离
            score = info_gain + 0.5 * cluster_size - 0.3 * dist
            scores.append(score)
        
        # 选择得分最高的
        best_idx = np.argmax(scores)
        self.target_frontier = self.frontiers[best_idx]
        
        # 额外检查：确保目标不是当前位置
        if tuple(self.robot_pos) == self.target_frontier:
            print("⚠️ Warning: Target is current position, selecting alternative...")
            scores[best_idx] = -float('inf')  # 排除当前最佳
            if max(scores) > -float('inf'):
                best_idx = np.argmax(scores)
                self.target_frontier = self.frontiers[best_idx]
            else:
                return False
        
        return True

    def _compute_info_gain(self, point):
        """计算信息增益：统计该点周围可探索的未知区域"""
        x, y = point
        unknown_count = 0
        search_range = 3  # 搜索半径
        
        for i in range(max(0, x - search_range), 
                    min(self.map_size, x + search_range + 1)):
            for j in range(max(0, y - search_range), 
                        min(self.map_size, y + search_range + 1)):
                if self.map[i, j] == 0:  # 未知区域
                    dist = np.sqrt((i - x)**2 + (j - y)**2)
                    if dist <= search_range:
                        unknown_count += 1
        
        return unknown_count
    
    def plan_path(self):
        """步骤5: 路径规划 (A*)"""
        self.step_index = 5
        
        if self.target_frontier is None:
            self.current_path = []
            return False
        
        self.current_path = self._astar_planning(
            tuple(self.robot_pos), 
            self.target_frontier
        )
        
        return len(self.current_path) > 0
    
    def _astar_planning(self, start, goal):
        """A*算法"""
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])  # 曼哈顿距离
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (current[0]+dx, current[1]+dy)
                
                if (0 <= neighbor[0] < self.map_size and 
                    0 <= neighbor[1] < self.map_size and
                    self.map[neighbor] != 2):
                    
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
        
        return []
    
    def move_robot(self):
        """步骤6: 机器人移动"""
        self.step_index = 6
        
        if self.current_path:
            next_pos = self.current_path.pop(0)
            self.robot_pos = np.array(next_pos)
            self.path_history.append(self.robot_pos.copy())
            
            # 移动时持续扫描
            self.sense_environment()
            return True
        return False


class TeachingVisualizer:
    """教学可视化工具"""
    
    def __init__(self, explorer):
        self.explorer = explorer
        self.fig = plt.figure(figsize=(20, 11))  # 稍微增大画布
        
        # 创建网格布局
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        self.ax_truth = self.fig.add_subplot(gs[0, 0])     # 真实环境
        self.ax_robot = self.fig.add_subplot(gs[0, 1])     # 机器人视角
        # self.ax_detail = self.fig.add_subplot(gs[0, 2])    # 细节放大
        self.ax_flowchart = self.fig.add_subplot(gs[1, :]) # 流程图
        
        self.step_count = 0
        self.auto_play = False
        
    def init_plot(self):
        """初始化绘图"""
        self.fig.suptitle('Frontier Exploration Algorithm - Step-by-Step Teaching Demo', 
                         fontsize=22, fontweight='bold', color='darkblue')  # 16→22
        
        # 添加控制说明
        self.fig.text(0.5, 0.02, 
                     'Press SPACE to advance | Press A for auto-play | Press R to restart',
                     ha='center', fontsize=14, style='italic',  # 10→14
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def on_key_press(self, event):
        """键盘控制"""
        if event.key == ' ':  # 空格键前进
            self.advance_step()
        elif event.key == 'a':  # A键自动播放
            self.auto_play = not self.auto_play
        elif event.key == 'r':  # R键重启
            self.explorer.__init__(self.explorer.map_size, self.explorer.sensor_range)
            self.step_count = 0
    
    def advance_step(self):
        """前进一步"""
        step = self.explorer.step_index
        
        if step == 0:
            self.explorer.sense_environment()
        elif step == 1:
            self.explorer.detect_frontiers()
        elif step == 2:
            self.explorer.cluster_frontiers()
        elif step == 3:
            self.explorer.select_target()
        elif step == 4:
            self.explorer.plan_path()
        elif step == 5:
            # 执行移动
            if not self.explorer.move_robot():
                # 路径完成,重新开始循环
                self.explorer.step_index = 0
        
        self.step_count += 1
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        # 清空所有子图
        for ax in [self.ax_truth, self.ax_robot,  self.ax_flowchart]: # self.ax_detail,
            ax.clear()
        
        # 1. 真实环境
        self._draw_ground_truth()
        
        # 2. 机器人视角
        self._draw_robot_view()
        
        # 3. 细节视图
        # self._draw_detail_view()
        
        # 4. 算法流程图
        self._draw_flowchart()
        
        plt.draw()
    
    def _draw_ground_truth(self):
        """绘制真实环境"""
        ax = self.ax_truth
        ax.imshow(self.explorer.ground_truth, cmap='gray_r', origin='lower', vmin=0, vmax=2)
        ax.set_title('Ground Truth Environment', fontsize=16, fontweight='bold')  # 12→16
        
        # 显示机器人
        ax.plot(self.explorer.robot_pos[1], self.explorer.robot_pos[0], 
               'go', markersize=15, markeredgecolor='darkgreen', markeredgewidth=2.5)  # 12→15
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', fontsize=14)  # 添加字体大小
        ax.set_ylabel('Y', fontsize=14)  # 添加字体大小
        ax.tick_params(labelsize=12)  # 刻度标签大小
    
    def _draw_robot_view(self):
        """绘制机器人视角"""
        ax = self.ax_robot
        
        # 创建显示地图
        display_map = np.ones((self.explorer.map_size, self.explorer.map_size, 3)) * 0.7
        display_map[self.explorer.map == 1] = [1, 1, 1]  # 自由空间-白色
        display_map[self.explorer.map == 2] = [0.2, 0.2, 0.2]  # 障碍物-深灰
        
        ax.imshow(display_map, origin='lower')
        
        step = self.explorer.step_index
        
        # 根据当前步骤显示不同内容
        if step >= 1:  # 传感器扫描
            sensor_circle = Circle((self.explorer.robot_pos[1], self.explorer.robot_pos[0]),
                                  self.explorer.sensor_range, fill=False, 
                                  edgecolor='cyan', linewidth=4, linestyle='--', alpha=0.8)  # 3→4
            ax.add_patch(sensor_circle)
        
        if step == 2 and self.explorer.candidate_frontiers:  # 边界检测
            frontiers = np.array(self.explorer.candidate_frontiers)
            ax.scatter(frontiers[:, 1], frontiers[:, 0], 
                      c='yellow', s=50, marker='s', alpha=0.6, label='Frontier Cells')  # 30→50
        
        if step == 3 and self.explorer.frontier_clusters:  # 聚类
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.explorer.frontier_clusters)))
            for cluster, color in zip(self.explorer.frontier_clusters, colors):
                cluster_array = np.array(cluster)
                ax.scatter(cluster_array[:, 1], cluster_array[:, 0], 
                          c=[color], s=80, marker='o', alpha=0.7, label = "cluster")  # 50→80
            
            # 显示中心点
            if self.explorer.frontiers:
                centers = np.array(self.explorer.frontiers)
                ax.scatter(centers[:, 1], centers[:, 0], 
                          c='orange', s=300, marker='*',  # 200→300
                          edgecolors='red', linewidths=3, label='representitive point', zorder=5)  # 2→3
        
        if step == 4 and self.explorer.target_frontier:  # 目标选择
            ax.scatter(self.explorer.target_frontier[1], self.explorer.target_frontier[0],
                      c='red', s=450, marker='*',  # 300→450
                      edgecolors='darkred', linewidths=4,  # 3→4
                      label='Selected Target', zorder=6)
        
        if step >= 5 and self.explorer.current_path:  # 路径规划
            path = np.array(self.explorer.current_path)
            ax.plot(path[:, 1], path[:, 0], 'b-', linewidth=4,  # 3→4
                   label='Planned Path', alpha=0.7)
            # 标注路径点
            for i, p in enumerate(path[::3]):  # 每3个点标注一次
                ax.text(p[1], p[0], str(i), fontsize=11,  # 8→11
                       bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))
        
        # 历史轨迹
        if len(self.explorer.path_history) > 1:
            history = np.array(self.explorer.path_history)
            ax.plot(history[:, 1], history[:, 0], 'g-', 
                   linewidth=3, label='Trajectory', alpha=0.5)  # 2→3
        
        # 机器人
        robot_circle = Circle((self.explorer.robot_pos[1], self.explorer.robot_pos[0]),
                             0.6, color='green', zorder=10)
        ax.add_patch(robot_circle)
        
        current_step_name = self.explorer.steps[step]
        ax.set_title(f'Robot View: {current_step_name}', 
                    fontsize=16, fontweight='bold', color='red')  # 12→16
        ax.legend(loc='upper right', fontsize=12)  # 8→12
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        # 统计信息
        explored = np.sum(self.explorer.map > 0) / (self.explorer.map_size ** 2) * 100
        info = f'Steps: {self.step_count}\nExplored: {explored:.1f}%\nFrontiers: {len(self.explorer.frontiers)}'
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=14,  # 10→14
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    def _draw_detail_view(self):
        """细节放大视图"""
        ax = self.ax_detail
        
        # 聚焦机器人周围区域
        x, y = self.explorer.robot_pos
        margin = 8
        x_min, x_max = max(0, x-margin), min(self.explorer.map_size, x+margin)
        y_min, y_max = max(0, y-margin), min(self.explorer.map_size, y+margin)
        
        # 创建放大地图
        zoom_map = np.ones((x_max-x_min, y_max-y_min, 3)) * 0.7
        zoom_region = self.explorer.map[x_min:x_max, y_min:y_max]
        zoom_map[zoom_region == 1] = [1, 1, 1]
        zoom_map[zoom_region == 2] = [0.2, 0.2, 0.2]
        
        ax.imshow(zoom_map, origin='lower', extent=[y_min, y_max, x_min, x_max])
        
        # 显示局部边界点
        if self.explorer.step_index >= 2:
            for fx, fy in self.explorer.candidate_frontiers:
                if x_min <= fx < x_max and y_min <= fy < y_max:
                    ax.plot(fy, fx, 'y^', markersize=12)  # 8→12
                    # 标注邻域
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = fx+dx, fy+dy
                        if (x_min <= nx < x_max and y_min <= ny < y_max and 
                            self.explorer.map[nx, ny] == 0):
                            ax.plot(ny, nx, 'r.', markersize=6, alpha=0.5)  # 4→6
        
        # 机器人
        ax.plot(y, x, 'go', markersize=18, markeredgecolor='darkgreen', markeredgewidth=3)  # 15→18, 2→3
        
        ax.set_title('Detail View (Zoom In)', fontsize=15, fontweight='bold')  # 11→15
        ax.grid(True, alpha=0.5, linestyle=':')
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(x_min, x_max)
        ax.tick_params(labelsize=12)
    
    def _draw_flowchart(self):
        """绘制算法流程图"""
        ax = self.ax_flowchart
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 3)
        ax.axis('off')
        
        # 定义步骤
        steps = [
            "1 Sensing",
            "2 Detect\nFrontiers", 
            "3 Cluster",
            "4 Select\nTarget",
            "5 Plan\nPath",
            "6 Move"
        ]
        
        current = self.explorer.step_index
        
        # 绘制流程框
        for i, step_text in enumerate(steps):
            x = i * 1.1 + 0.5
            y = 1.5
            
            # 高亮当前步骤
            if i == current:
                box_color = 'lightcoral'
                edge_color = 'red'
                linewidth = 4  # 3→4
            elif i < current:
                box_color = 'lightgreen'
                edge_color = 'green'
                linewidth = 3  # 2→3
            else:
                box_color = 'lightgray'
                edge_color = 'gray'
                linewidth = 2  # 1→2
            
            bbox = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                 boxstyle="round,pad=0.05", 
                                 facecolor=box_color, edgecolor=edge_color,
                                 linewidth=linewidth)
            ax.add_patch(bbox)
            
            ax.text(x, y, step_text, ha='center', va='center', 
                   fontsize=13, fontweight='bold')  # 9→13
            
            # 箭头
            if i < len(steps) - 1:
                ax.arrow(x+0.4, y, 0.25, 0, head_width=0.1, 
                        head_length=0.08, fc='black', ec='black', linewidth=2)
        
        # 循环箭头
        # ax.annotate('', xy=(0.5, 1.8), xytext=(6.5, 1.8),
        #            arrowprops=dict(arrowstyle='->', lw=3, color='blue',  # 2→3
        #                          connectionstyle="arc3,rad=.5"))
        # ax.text(3.5, 2.5, 'Repeat Until Complete', 
        #        ha='center', fontsize=14, color='blue', fontweight='bold')  # 10→14
        
        # 当前步骤说明
        step_descriptions = [
            "Robot scans environment with LiDAR sensor",
            "Find cells between known and unknown areas",
            "Group adjacent frontier cells into regions",
            "Score frontiers by distance and information gain",
            "Use A* to find safe path to target",
            "Follow path and update map continuously"
        ]
        
        if current < len(step_descriptions):
            ax.text(3.5, 0.5, f"{step_descriptions[current]}", 
                   ha='center', fontsize=14, style='italic',  # 10→14
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


def run_teaching_demo():
    """运行教学演示"""
    print("=" * 70)
    print("FRONTIER EXPLORATION - Interactive Teaching Demo")
    print("=" * 70)
    print("\n Algorithm Steps:")
    print("  1. SENSING: Robot scans environment with sensors")
    print("  2. DETECT: Find frontier cells (known ↔ unknown boundary)")
    print("  3. CLUSTER: Group adjacent frontiers into regions")
    print("  4. SELECT: Choose best frontier (score = gain - λ×distance)")
    print("  5. PLAN: Use A* to compute path to target")
    print("  6. MOVE: Follow path and repeat")
    print("\nControls:")
    print("  SPACE: Advance one step")
    print("  A: Toggle auto-play")
    print("  R: Restart")
    print("=" * 70)
    print("\nPress SPACE to begin...\n")
    
    # 创建演示
    explorer = TeachingFrontierExplorer(map_size=30, sensor_range=4)
    visualizer = TeachingVisualizer(explorer)
    
    # 初始化
    visualizer.init_plot()
    visualizer.update_display()
    
    # 自动播放循环
    def auto_update(frame):
        if visualizer.auto_play:
            visualizer.advance_step()
        return []
    
    anim = FuncAnimation(visualizer.fig, auto_update, 
                        frames=1000, interval=800, repeat=True)
    
    plt.show()


if __name__ == "__main__":
    run_teaching_demo()
