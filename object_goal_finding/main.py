import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import label
from collections import deque
import heapq
import imageio
import copy
from tqdm import tqdm

PERCEPTION_LENGTH = 20

class ObjectGoalNavigationDemo:
    def __init__(self, width=100, height=100, fov=90):
        """
        初始化导航演示
        width, height: 地图大小
        fov: 视场角（度）
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.fov_rad = np.radians(fov)
        
        # 机器人状态
        self.robot_x = width // 4
        self.robot_y = height // 4
        self.robot_angle = 0  # 弧度
        
        # 目标物体位置
        self.target_x = int(width * 0.75)
        self.target_y = int(height * 0.75)
        
        # 机器人已探索的地图（语义地图）
        self.semantic_map = np.zeros((height, width), dtype=int)  # 0=未知, 1=空闲, 2=障碍, 3=找到目标
        self.semantic_map[:, :] = 0  # 初始全部未知
        
        # 机器人走过的路径
        self.robot_path = [(self.robot_x, self.robot_y)]
        
        # 路径规划
        self.planned_path = []
        self.current_goal = None
        self.steps_to_goal = 0
        
        # 历史记录
        self.history = []
        self.target_found = False
        self.step_count = 0

        # Ground truth 地图
        self.ground_truth_map = self._generate_ground_truth_map()
        
    def _generate_ground_truth_map(self):
        """生成随机的ground truth地图（包含障碍和空闲空间）"""
        ground_truth = np.ones((self.height, self.width), dtype=int)  # 1=空闲
        
        # 添加一些随机障碍
        for _ in range(5):
            x = np.random.randint(self.width // 3, int(self.width * 0.8))
            y = np.random.randint(self.height // 3, int(self.height * 0.8))
            size = np.random.randint(5, 15)
            ground_truth[max(0, y-size):min(self.height, y+size),
                        max(0, x-size):min(self.width, x+size)] = 2  # 2=障碍
        
        # 添加目标物体周围的空闲区域
        ground_truth[max(0, self.target_y-10):min(self.height, self.target_y+10),
                    max(0, self.target_x-10):min(self.width, self.target_x+10)] = 1
        
        return ground_truth
    

    def _find_frontiers(self):
        """
        查找frontier点（已探索空闲区域与未探索区域的边界）
        返回: frontier点列表 [(x, y), ...]
        """
        frontiers = []
        
        # 遍历地图寻找frontier
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                # frontier定义: 该点是已知空闲区域(1)
                if self.semantic_map[y, x] == 1:
                    # 检查8邻域是否有未探索区域(0)
                    has_unknown_neighbor = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.height and 0 <= nx < self.width:
                                if self.semantic_map[ny, nx] == 0:
                                    has_unknown_neighbor = True
                                    break
                        if has_unknown_neighbor:
                            break
                    
                    if has_unknown_neighbor:
                        frontiers.append((x, y))
        
        return frontiers

    def _cluster_frontiers(self, frontiers, min_cluster_size=3):
        """
        将frontier点聚类，返回聚类中心点
        frontiers: frontier点列表
        min_cluster_size: 最小聚类大小
        返回: 聚类中心点列表 [(x, y), ...]
        """
        if not frontiers:
            return []
        
        # 使用连通域标记进行聚类
        frontier_map = np.zeros((self.height, self.width), dtype=bool)
        for x, y in frontiers:
            frontier_map[y, x] = True
        
        # 标记连通域
        labeled_map, num_clusters = label(frontier_map)
        
        # 计算每个聚类的中心点
        cluster_centers = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_points = np.argwhere(labeled_map == cluster_id)
            
            # 过滤太小的聚类
            if len(cluster_points) < min_cluster_size:
                continue
            
            # 计算聚类中心
            center_y = int(np.mean(cluster_points[:, 0]))
            center_x = int(np.mean(cluster_points[:, 1]))
            
            cluster_centers.append((center_x, center_y))
        
        return cluster_centers

    def _select_frontier_goal(self):
        """
        基于Frontier的目标点选择
        优先选择距离适中的frontier聚类中心
        """
        # 1. 查找所有frontier点
        frontiers = self._find_frontiers()
        
        if not frontiers:
            # 没有frontier，说明可能探索完成或被困住
            return self._select_fallback_goal()
        
        # 2. 对frontier进行聚类
        cluster_centers = self._cluster_frontiers(frontiers, min_cluster_size=5)
        
        if not cluster_centers:
            # 聚类失败，直接从frontier中随机选择
            return frontiers[np.random.randint(len(frontiers))]
        
        # 3. 评估每个聚类中心的得分
        best_goal = None
        best_score = -np.inf
        
        robot_pos = np.array([self.robot_x, self.robot_y])
        
        for goal_x, goal_y in cluster_centers:
            # 检查该点是否可达（不在障碍物中）
            if self.semantic_map[goal_y, goal_x] != 1:
                continue
            
            goal_pos = np.array([goal_x, goal_y])
            distance = np.linalg.norm(goal_pos - robot_pos)
            
            # 计算得分（考虑距离和探索价值）
            # 距离适中的点得分更高（20-60单位）
            if distance < 10:
                distance_score = distance / 10  # 太近得分低
            elif distance > 60:
                distance_score = 60 / distance  # 太远得分低
            else:
                distance_score = 1.0  # 中等距离得分高
            
            # 计算周围未探索区域的数量（探索价值）
            unknown_count = 0
            search_radius = 5
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = goal_y + dy, goal_x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if self.semantic_map[ny, nx] == 0:
                            unknown_count += 1
            
            exploration_score = unknown_count / (search_radius * 2 + 1) ** 2
            
            # 综合得分
            score = distance_score * 0.6 + exploration_score * 0.4
            
            if score > best_score:
                best_score = score
                best_goal = (goal_x, goal_y)
        
        # 4. 返回最佳目标点
        if best_goal is not None:
            return best_goal
        
        # 5. 如果没有找到合适的目标，使用fallback策略
        return self._select_fallback_goal()

    def _select_fallback_goal(self):
        """
        Fallback目标选择策略
        当没有frontier时使用
        """
        # 策略1: 寻找未探索区域
        unknown_cells = []
        for y in range(5, self.height - 5, 3):
            for x in range(5, self.width - 5, 3):
                if self.semantic_map[y, x] == 0:
                    # 检查ground truth是否可行
                    if self.ground_truth_map[y, x] == 1:
                        unknown_cells.append((x, y))
        
        if unknown_cells:
            # 选择距离适中的未探索点
            robot_pos = np.array([self.robot_x, self.robot_y])
            valid_goals = []
            
            for x, y in unknown_cells:
                distance = np.linalg.norm(np.array([x, y]) - robot_pos)
                if 5 < distance < 50:
                    valid_goals.append((x, y))
            
            if valid_goals:
                return unknown_cells[np.random.randint(len(valid_goals))]
            
            # 如果没有距离适中的，随机选择一个
            return unknown_cells[np.random.randint(len(unknown_cells))]
        exit()
        # # 策略2: 在已知空闲区域随机游走
        # free_cells = []
        # for y in range(5, self.height - 5):
        #     for x in range(5, self.width - 5):
        #         if self.semantic_map[y, x] == 1:
        #             free_cells.append((x, y))
        
        # if free_cells:
        #     return free_cells[np.random.randint(len(free_cells))]
        
        # # 策略3: 最终fallback - 在机器人周围寻找
        # for radius in range(5, 30, 5):
        #     for angle in np.linspace(0, 2 * np.pi, 8):
        #         test_x = int(self.robot_x + radius * np.cos(angle))
        #         test_y = int(self.robot_y + radius * np.sin(angle))
                
        #         if (0 <= test_x < self.width and 0 <= test_y < self.height and
        #             self.ground_truth_map[test_y, test_x] == 1):
        #             return (test_x, test_y)
        
        # # 绝对最后的fallback
        # return (int(self.robot_x), int(self.robot_y))

    def _select_random_goal(self):
        """
        改进的目标点选择方法
        优先使用frontier-based探索，fallback到随机策略
        """
        # 首先尝试frontier-based探索
        goal = self._select_frontier_goal()
        
        # 验证目标点的有效性
        if goal:
            goal_x, goal_y = goal
            
            # 确保在地图范围内
            if not (0 <= goal_x < self.width and 0 <= goal_y < self.height):
                print("check map")
                return self._select_fallback_goal()
            
            # 检查是否是障碍物
            if self.semantic_map[goal_y, goal_x] == 2:
                print("check obstracle")
                return self._select_fallback_goal()
            
            return goal
        
        # 如果frontier方法失败，使用fallback
        return self._select_fallback_goal()

    
    def _plan_path_astar(self, start, goal):
        """A*路径规划"""
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)}
        
        closed_set = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # 重构路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            closed_set.add(current)
            
            # 8个方向
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # 边界检查
                    if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                        continue
                    
                    # 障碍检查（只考虑已知的障碍）
                    if self.semantic_map[neighbor[1], neighbor[0]] == 2:
                        continue
                    
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g = g_score[current] + (np.sqrt(2) if dx != 0 and dy != 0 else 1)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = np.sqrt((goal[0] - neighbor[0])**2 + (goal[1] - neighbor[1])**2)
                        f_score[neighbor] = g_score[neighbor] + h
                        
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # 无法到达
    
    def _get_sensor_observation(self):
        """模拟半圆形FOV传感器观察"""
        observation = {}
        
        # 半圆形FOV，90度视场角
        angle_range = self.fov_rad / 2  # 45度
        observed_points = []
        for angle in np.linspace(self.robot_angle - angle_range, 
                                 self.robot_angle + angle_range, 45):
            for dist in np.linspace(1, PERCEPTION_LENGTH, int(PERCEPTION_LENGTH*0.8)):
                # 计算射线端点
                x = int(self.robot_x + dist * np.cos(angle))
                y = int(self.robot_y + dist * np.sin(angle))
                
                # 边界检查
                if 0 <= x < self.width and 0 <= y < self.height:
                    cell_type = self.ground_truth_map[y, x]
                    
                    # 更新语义地图
                    if self.semantic_map[y, x] == 0:  # 未知
                        self.semantic_map[y, x] = cell_type
                        
                        # # 检查是否发现目标
                        if abs(x - self.target_x) < 2  and abs(y - self.target_y) < 2:
                            self.semantic_map[y, x] = 3  # 标记为目标
                        #     self.target_found = True
                    
                    # 遇到障碍停止这条射线
                    if cell_type == 2:
                        break
                observed_points.append((x, y))
                centroid_x = np.mean([p[0] for p in observed_points])
                centroid_y = np.mean([p[1] for p in observed_points])

                centroid_dist = np.sqrt((centroid_x - self.target_x)**2 +
                                        (centroid_y - self.target_y)**2)

                if centroid_dist < 1:
                    self.target_found = True
    def step(self):
        """执行一步探索"""
        self.step_count += 1
        
        # 1. 感知环境
        self._get_sensor_observation()
        
        if self.target_found:
            self.history.append({
                'step': self.step_count,
                'current_goal': copy.deepcopy(self.current_goal),
                'path': self.planned_path.copy() if self.planned_path else [],
                'robot_pos': (self.robot_x, self.robot_y),
                'robot_angle': copy.deepcopy(self.robot_angle),
                'target_found': self.target_found,
                'semantic_map': copy.deepcopy(self.semantic_map),
                'robot_path' : copy.deepcopy(self.robot_path)
            })
            return True  # 探索完成
        
        # 2. 检查是否到达当前目标
        reached_goal = False
        if self.current_goal is not None:
            # 计算与目标的距离
            distance_to_goal = np.sqrt(
                (self.robot_x - self.current_goal[0])**2 + 
                (self.robot_y - self.current_goal[1])**2
            )
            
            # 如果距离小于0.5，认为已到达
            if distance_to_goal < 0.5:
                reached_goal = True
        
        # 如果没有当前目标、已经到达目标、或步数用尽，选择新的目标
        if self.current_goal is None or reached_goal or self.steps_to_goal <= 0:
            if reached_goal:
                print(f"Step {self.step_count}: Reached goal at ({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f})")
            
            self.current_goal = self._select_random_goal()
            self.steps_to_goal = 20  # 向该目标移动最多20步
            
            # 规划路径到新目标
            self.planned_path = self._plan_path_astar(
                (int(self.robot_x), int(self.robot_y)), 
                self.current_goal
            )
            
            # 如果无法规划路径，立即选择新目标
            if not self.planned_path:
                self.current_goal = None
                self.steps_to_goal = 0
                return False
        
        # 3. 沿着路径移动
        if self.planned_path and len(self.planned_path) > 1:
            # 移动到路径的下一个点
            step_size = 2
            next_pos = self.planned_path[min(1, len(self.planned_path) - 1)]
            
            # 计算新位置和方向
            dx = next_pos[0] - self.robot_x
            dy = next_pos[1] - self.robot_y
            
            if dx != 0 or dy != 0:
                self.robot_angle = np.arctan2(dy, dx)
                
                # 移动
                distance = np.sqrt(dx**2 + dy**2)
                move_distance = min(step_size, distance)
                self.robot_x += move_distance * np.cos(self.robot_angle)
                self.robot_y += move_distance * np.sin(self.robot_angle)
                
                # 更新路径（移除已经走过的点）
                if distance < step_size and len(self.planned_path) > 1:
                    self.planned_path.pop(0)
            
            self.robot_path.append((self.robot_x, self.robot_y))
            self.steps_to_goal -= 1
        
        # 记录历史
        self.history.append({
            'step': self.step_count,
            'current_goal': copy.deepcopy(self.current_goal),
            'path': self.planned_path.copy() if self.planned_path else [],
            'robot_pos': (self.robot_x, self.robot_y),
            'robot_angle': copy.deepcopy(self.robot_angle),
            'target_found': self.target_found,
            'semantic_map': copy.deepcopy(self.semantic_map),
            'robot_path' : copy.deepcopy(self.robot_path)
        })
        
        return False

    def render_frame(self, step_idx=None):
        """渲染一帧，返回RGB数组"""
        if step_idx is None:
            step_idx = len(self.history) - 1
        
        if step_idx < 0 or step_idx >= len(self.history):
            return None
        
        history_item = self.history[step_idx]
        robot_x, robot_y = history_item['robot_pos']

        self.semantic_map = history_item['semantic_map']
        planned_path = history_item['path']
        self.robot_path = history_item['robot_path']
        self.robot_angle = history_item['robot_angle']

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
        fig.patch.set_facecolor('white')
        
        # ===== 左图：Ground Truth 地图 =====
        ax_left = axes[0]

        ground_truth_display = np.zeros((self.height, self.width, 3))

        ground_truth_display[self.ground_truth_map == 1] = [1, 1, 1]  # 白色=空闲
        ground_truth_display[self.ground_truth_map == 2] = [0, 0, 0]  # 黑色=障碍
        # ground_truth_display = np.zeros((self.height, self.width, 3))        
        # for y in range(self.height):
        #     for x in range(self.width):
        #         if self.ground_truth_map[y, x] == 1:
        #             ground_truth_display[y, x] = [1, 1, 1]  # 白色=空闲
        #         elif self.ground_truth_map[y, x] == 2:
        #             ground_truth_display[y, x] = [0, 0, 0]  # 黑色=障碍
        
        # 标记目标
        y_target, x_target = int(self.target_y), int(self.target_x)
        ground_truth_display[max(0, y_target-2):min(self.height, y_target+3),
                            max(0, x_target-2):min(self.width, x_target+3)] = [1, 0, 0]  # 红色=目标
        
        ax_left.imshow(ground_truth_display, origin='lower')
        
        # 绘制机器人历史路径（GT图上）
        if len(self.robot_path) > 1:
            path_array = np.array(self.robot_path)
            ax_left.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=1.5, alpha=0.6, label='Robot Path')
        
        # 绘制当前机器人位置
        ax_left.plot(robot_x, robot_y, 'bo', markersize=10, label='Robot', zorder=5)
        
        # 绘制机器人FOV
        angle_range = self.fov_rad / 2
        angles = np.linspace(self.robot_angle - angle_range, self.robot_angle + angle_range, 20)
        sensor_range = 10
        
        for angle in angles[::2]:
            end_x = robot_x + sensor_range * np.cos(angle)
            end_y = robot_y + sensor_range * np.sin(angle)
            ax_left.plot([robot_x, end_x], [robot_y, end_y], 'g-', alpha=0.2, linewidth=0.8)
        
        ax_left.set_xlim(0, self.width)
        ax_left.set_ylim(0, self.height)
        ax_left.set_title('Ground Truth Map', fontsize=14, fontweight='bold')
        ax_left.set_xlabel('X', fontsize=12)
        ax_left.set_ylabel('Y', fontsize=12)
        ax_left.legend(loc='upper right', fontsize=10)
        ax_left.grid(True, alpha=0.3)
        
        # ===== 右图：语义地图 =====
        ax_right = axes[1]

        frontiers = self._find_frontiers()
        if frontiers:
            frontier_array = np.array(frontiers)
            ax_right.scatter(frontier_array[:, 0], frontier_array[:, 1], 
                            c='yellow', s=10, marker='.', alpha=0.5, 
                            label='Frontiers', zorder=3)

        # 方法1: 使用 np.where 或索引(推荐)
        semantic_display = np.zeros((self.height, self.width, 3))

        # 定义颜色映射
        semantic_display[self.semantic_map == 0] = [0.5, 0.5, 0.5]  # 灰色=未知
        semantic_display[self.semantic_map == 1] = [1, 1, 1]        # 白色=空闲
        semantic_display[self.semantic_map == 2] = [0, 0, 0]        # 黑色=障碍
        semantic_display[self.semantic_map == 3] = [1, 0, 0]        # 红色=目标

        
        ax_right.imshow(semantic_display, origin='lower')
        
        # 绘制当前随机目标点
        if history_item['current_goal']:
            ax_right.scatter([history_item['current_goal'][0]], 
                        [history_item['current_goal'][1]], 
                        c='orange', s=200, marker='*', 
                        edgecolors='darkorange', linewidth=2, label='Random Goal', zorder=4)
        
        # 绘制规划路径
        if history_item['path']:
            path_array = np.array(history_item['path'])
            ax_right.plot(path_array[:, 0], path_array[:, 1], 'c--', linewidth=2, alpha=0.8, label='Planned Path')
        
        # 绘制机器人历史路径
        if len(self.robot_path) > 1:
            path_array = np.array(self.robot_path)
            ax_right.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=1.5, alpha=0.6, label='Executed Path')
        
        # 绘制当前机器人位置
        ax_right.plot(robot_x, robot_y, 'bo', markersize=10, label='Robot', zorder=5)
        
        # 绘制机器人FOV
        for angle in angles[::2]:
            end_x = robot_x + sensor_range * np.cos(angle)
            end_y = robot_y + sensor_range * np.sin(angle)
            ax_right.plot([robot_x, end_x], [robot_y, end_y], 'g-', alpha=0.2, linewidth=0.8)
        
        ax_right.set_xlim(0, self.width)
        ax_right.set_ylim(0, self.height)
        ax_right.set_title(f'Semantic Map - Random Exploration (Step: {history_item["step"]})', 
                        fontsize=14, fontweight='bold')
        ax_right.set_xlabel('X', fontsize=12)
        ax_right.set_ylabel('Y', fontsize=12)
        ax_right.legend(loc='upper right', fontsize=10)
        ax_right.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 转换为RGB数组 - 修复的部分
        fig.canvas.draw()
        # 使用 buffer_rgba() 替代 tostring_rgb()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)
        # 转换RGBA为RGB
        image = image[:, :, :3]
        
        plt.close(fig)
        return image



def run_demo_with_gif_export():
    """运行演示并导出GIF - 延长结尾版本"""
    print("=" * 60)
    print("Object Goal Navigation - Frontier Exploration Demo")
    print("=" * 60)
    
    demo = ObjectGoalNavigationDemo(width=100, height=100, fov=90)
    
    # 执行探索
    max_steps = 300
    step = 0
    
    print("\n正在执行探索...")
    while step < max_steps:
        done = demo.step()
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}")
        
        if done:
            print(f"\n✓ 探索完成 (步数: {step})")
            break
    
    # 生成GIF
    print("\n正在生成GIF...")
    total_frames = len(demo.history)
    
    # 每N帧取1帧
    frame_skip = max(1, total_frames // 200)
    
    frames = []
    for idx in tqdm(range(0, total_frames, frame_skip), desc="渲染帧"):
        frame = demo.render_frame(idx)
        if frame is not None:
            frames.append(frame)
    
    # ===== 新增：延长最后一帧 =====
    if frames:
        last_frame = frames[-1]
        # 重复最后一帧30次（约6秒，假设每帧200ms）
        repeat_count = 30
        print(f"\n添加 {repeat_count} 帧结尾停留...")
        for _ in range(repeat_count):
            frames.append(last_frame.copy())
    # ================================
    
    # 保存GIF
    gif_filename = 'robot_exploration4.gif'
    print(f"\n正在保存GIF...")
    
    imageio.mimsave(
        gif_filename, 
        frames, 
        duration=0.2,  # 每帧200ms
        loop=0
    )
    
    print(f"✓ GIF已保存: {gif_filename}")
    print(f"  总帧数: {len(frames)} (原始: {total_frames}, 结尾延长: {repeat_count}帧)")
    print(f"  文件大小: {os.path.getsize(gif_filename) / (1024*1024):.2f} MB")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Exploration Statistics:")
    print("=" * 60)
    print(f"Total steps: {step}")
    print(f"Target found: {'✓ YES' if demo.target_found else '✗ NO'}")
    print(f"Explored cells: {np.sum(demo.semantic_map > 0)}")
    print(f"Exploration rate: {np.sum(demo.semantic_map > 0) / (demo.width * demo.height) * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    import os
    run_demo_with_gif_export()
