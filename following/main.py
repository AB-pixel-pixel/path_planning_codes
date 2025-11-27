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
FOLLOW_DISTANCE = 15  # 理想跟随距离

class DynamicFollowingDemo:
    def __init__(self, width=100, height=100, fov=90):
        """
        初始化动态跟随演示
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
        
        # 目标人物初始位置（会移动）
        self.target_x = int(width * 0.75)
        self.target_y = int(height * 0.75)
        self.target_angle = np.pi  # 人物移动方向
        self.target_speed = 1.5  # 人物移动速度
        
        # 人物移动历史（用于预测）
        self.target_history = deque(maxlen=10)
        self.target_history.append((self.target_x, self.target_y))
        
        # 机器人已探索的地图（语义地图）
        self.semantic_map = np.zeros((height, width), dtype=int)
        
        # 机器人走过的路径
        self.robot_path = [(self.robot_x, self.robot_y)]
        
        # 路径规划
        self.planned_path = []
        self.current_goal = None
        self.steps_to_goal = 0
        
        # 跟随状态
        self.target_visible = False  # 目标是否在视野内
        self.target_last_seen_pos = None  # 目标最后可见位置
        self.steps_since_seen = 0  # 未见到目标的步数
        
        # 历史记录
        self.history = []
        self.step_count = 0

        # Ground truth 地图
        self.ground_truth_map = self._generate_ground_truth_map()
        
    def _generate_ground_truth_map(self):
        """生成随机的ground truth地图"""
        ground_truth = np.ones((self.height, self.width), dtype=int)
        
        # 添加障碍物
        for _ in range(5):
            x = np.random.randint(self.width // 3, int(self.width * 0.8))
            y = np.random.randint(self.height // 3, int(self.height * 0.8))
            size = np.random.randint(5, 15)
            ground_truth[max(0, y-size):min(self.height, y+size),
                        max(0, x-size):min(self.width, x+size)] = 2
        
        # 确保起点和终点周围空闲
        ground_truth[max(0, self.robot_y-10):min(self.height, self.robot_y+10),
                    max(0, self.robot_x-10):min(self.width, self.robot_x+10)] = 1
        ground_truth[max(0, self.target_y-10):min(self.height, self.target_y+10),
                    max(0, self.target_x-10):min(self.width, self.target_x+10)] = 1
        
        return ground_truth
    
    def _move_target_person(self):
        """移动目标人物（随机游走+避障）"""
        # 随机改变方向（小概率）
        if np.random.rand() < 0.05:
            self.target_angle += np.random.uniform(-np.pi/4, np.pi/4)
        
        # 尝试移动
        new_x = self.target_x + self.target_speed * np.cos(self.target_angle)
        new_y = self.target_y + self.target_speed * np.sin(self.target_angle)
        
        # 边界检查
        if new_x < 5 or new_x >= self.width - 5:
            self.target_angle = np.pi - self.target_angle
            new_x = self.target_x + self.target_speed * np.cos(self.target_angle)
        
        if new_y < 5 or new_y >= self.height - 5:
            self.target_angle = -self.target_angle
            new_y = self.target_y + self.target_speed * np.sin(self.target_angle)
        
        # 障碍检查
        new_x_int = int(new_x)
        new_y_int = int(new_y)
        
        if (0 <= new_x_int < self.width and 0 <= new_y_int < self.height and
            self.ground_truth_map[new_y_int, new_x_int] == 1):
            # 无障碍，更新位置
            self.target_x = new_x
            self.target_y = new_y
            self.target_history.append((self.target_x, self.target_y))
        else:
            # 遇到障碍，改变方向
            self.target_angle += np.random.uniform(np.pi/2, np.pi)
    
    def _predict_target_position(self):
        """基于历史预测目标未来位置"""
        if len(self.target_history) < 3:
            return (self.target_x, self.target_y)
        
        # 计算平均速度向量
        recent_positions = list(self.target_history)[-5:]
        velocities = []
        
        for i in range(1, len(recent_positions)):
            vx = recent_positions[i][0] - recent_positions[i-1][0]
            vy = recent_positions[i][1] - recent_positions[i-1][1]
            velocities.append((vx, vy))
        
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        # 预测未来5步的位置
        prediction_steps = 5
        pred_x = self.target_x + avg_vx * prediction_steps
        pred_y = self.target_y + avg_vy * prediction_steps
        
        # 边界限制
        pred_x = np.clip(pred_x, 5, self.width - 5)
        pred_y = np.clip(pred_y, 5, self.height - 5)
        
        return (pred_x, pred_y)
    
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
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            closed_set.add(current)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                        continue
                    
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
        
        return []
    
    def _get_sensor_observation(self):
        """模拟半圆形FOV传感器观察"""
        self.target_visible = False
        
        angle_range = self.fov_rad / 2
        observed_points = []
        
        for angle in np.linspace(self.robot_angle - angle_range, 
                                 self.robot_angle + angle_range, 45):
            for dist in np.linspace(1, PERCEPTION_LENGTH, int(PERCEPTION_LENGTH*0.8)):
                x = int(self.robot_x + dist * np.cos(angle))
                y = int(self.robot_y + dist * np.sin(angle))
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    cell_type = self.ground_truth_map[y, x]
                    
                    if self.semantic_map[y, x] == 0:
                        self.semantic_map[y, x] = cell_type
                    
                    # 检测目标人物
                    if abs(x - self.target_x) < 3 and abs(y - self.target_y) < 3:
                        self.target_visible = True
                        self.target_last_seen_pos = (self.target_x, self.target_y)
                        self.steps_since_seen = 0
                        self.semantic_map[y, x] = 3  # 标记目标
                    
                    if cell_type == 2:
                        break
                
                observed_points.append((x, y))
        
        if not self.target_visible:
            self.steps_since_seen += 1
    
    def _select_following_goal(self):
        """选择跟随目标点"""
        # 情况1：目标可见 - 保持理想距离
        if self.target_visible:
            # 计算当前距离
            current_dist = np.sqrt(
                (self.robot_x - self.target_x)**2 + 
                (self.robot_y - self.target_y)**2
            )
            
            # 如果距离合适，预测未来位置
            if abs(current_dist - FOLLOW_DISTANCE) < 3:
                pred_pos = self._predict_target_position()
                
                # 计算跟随点（保持在目标后方FOLLOW_DISTANCE距离）
                target_angle = np.arctan2(
                    pred_pos[1] - self.robot_y,
                    pred_pos[0] - self.robot_x
                )
                
                goal_x = pred_pos[0] - FOLLOW_DISTANCE * np.cos(target_angle)
                goal_y = pred_pos[1] - FOLLOW_DISTANCE * np.sin(target_angle)
            else:
                # 距离不合适，直接靠近
                direction = np.arctan2(
                    self.target_y - self.robot_y,
                    self.target_x - self.robot_x
                )
                
                goal_x = self.target_x - FOLLOW_DISTANCE * np.cos(direction)
                goal_y = self.target_y - FOLLOW_DISTANCE * np.sin(direction)
            
            goal_x = int(np.clip(goal_x, 0, self.width - 1))
            goal_y = int(np.clip(goal_y, 0, self.height - 1))
            
            return (goal_x, goal_y)
        
        # 情况2：目标不可见但最近见过 - 前往最后位置
        elif self.target_last_seen_pos and self.steps_since_seen < 20:
            return self.target_last_seen_pos
        
        # 情况3：长时间未见 - 搜索策略
        else:
            # 在最后可见位置周围搜索
            if self.target_last_seen_pos:
                search_radius = min(30, 10 + self.steps_since_seen)
                angle = np.random.uniform(0, 2*np.pi)
                
                goal_x = int(self.target_last_seen_pos[0] + search_radius * np.cos(angle))
                goal_y = int(self.target_last_seen_pos[1] + search_radius * np.sin(angle))
                
                goal_x = np.clip(goal_x, 5, self.width - 5)
                goal_y = np.clip(goal_y, 5, self.height - 5)
                
                return (goal_x, goal_y)
            else:
                # 完全迷失 - 随机搜索
                return (np.random.randint(10, self.width-10),
                       np.random.randint(10, self.height-10))
    
    def step(self):
        """执行一步跟随"""
        self.step_count += 1
        
        # 1. 移动目标人物
        self._move_target_person()
        
        # 2. 感知环境
        self._get_sensor_observation()
        
        # 3. 检查是否到达当前目标
        reached_goal = False
        if self.current_goal is not None:
            distance_to_goal = np.sqrt(
                (self.robot_x - self.current_goal[0])**2 + 
                (self.robot_y - self.current_goal[1])**2
            )
            
            if distance_to_goal < 2.0:
                reached_goal = True
        
        # 4. 重新规划（每步都重新规划以应对动态目标）
        if self.current_goal is None or reached_goal or self.step_count % 3 == 0:
            self.current_goal = self._select_following_goal()
            
            self.planned_path = self._plan_path_astar(
                (int(self.robot_x), int(self.robot_y)), 
                self.current_goal
            )
        
        # 5. 沿着路径移动
        if self.planned_path and len(self.planned_path) > 1:
            step_size = 2.0
            next_pos = self.planned_path[min(1, len(self.planned_path) - 1)]
            
            dx = next_pos[0] - self.robot_x
            dy = next_pos[1] - self.robot_y
            
            if dx != 0 or dy != 0:
                self.robot_angle = np.arctan2(dy, dx)
                
                distance = np.sqrt(dx**2 + dy**2)
                move_distance = min(step_size, distance)
                self.robot_x += move_distance * np.cos(self.robot_angle)
                self.robot_y += move_distance * np.sin(self.robot_angle)
                
                if distance < step_size and len(self.planned_path) > 1:
                    self.planned_path.pop(0)
            
            self.robot_path.append((self.robot_x, self.robot_y))
        
        # 6. 记录历史
        self.history.append({
            'step': self.step_count,
            'current_goal': copy.deepcopy(self.current_goal),
            'path': self.planned_path.copy() if self.planned_path else [],
            'robot_pos': (self.robot_x, self.robot_y),
            'target_pos': (self.target_x, self.target_y),
            'robot_angle': copy.deepcopy(self.robot_angle),
            'target_visible': self.target_visible,
            'semantic_map': copy.deepcopy(self.semantic_map),
            'robot_path': copy.deepcopy(self.robot_path),
            'target_history': list(copy.deepcopy(self.target_history))
        })
        
        return False

    def render_frame(self, step_idx=None):
        """渲染一帧"""
        if step_idx is None:
            step_idx = len(self.history) - 1
        
        if step_idx < 0 or step_idx >= len(self.history):
            return None
        
        history_item = self.history[step_idx]
        robot_x, robot_y = history_item['robot_pos']
        target_x, target_y = history_item['target_pos']

        self.semantic_map = history_item['semantic_map']
        planned_path = history_item['path']
        self.robot_path = history_item['robot_path']
        self.robot_angle = history_item['robot_angle']
        target_history = history_item['target_history']

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
        fig.patch.set_facecolor('white')
        
        # ===== 左图：Ground Truth 地图 =====
        ax_left = axes[0]

        ground_truth_display = np.zeros((self.height, self.width, 3))
        ground_truth_display[self.ground_truth_map == 1] = [1, 1, 1]
        ground_truth_display[self.ground_truth_map == 2] = [0, 0, 0]
        
        ax_left.imshow(ground_truth_display, origin='lower')
        
        # 绘制目标人物轨迹
        if len(target_history) > 1:
            path_array = np.array(target_history)
            ax_left.plot(path_array[:, 0], path_array[:, 1], 
                        'r--', linewidth=1.5, alpha=0.6, label='Target Path')
        
        # 绘制目标人物（人形标记）
        ax_left.plot(target_x, target_y, 'r^', markersize=15, 
                    label='Target Person', zorder=5)
        
        # 绘制机器人历史路径
        if len(self.robot_path) > 1:
            path_array = np.array(self.robot_path)
            ax_left.plot(path_array[:, 0], path_array[:, 1], 
                        'b-', linewidth=1.5, alpha=0.6, label='Robot Path')
        
        # 绘制机器人
        ax_left.plot(robot_x, robot_y, 'bo', markersize=10, label='Robot', zorder=5)
        
        # 绘制FOV
        angle_range = self.fov_rad / 2
        angles = np.linspace(self.robot_angle - angle_range, 
                            self.robot_angle + angle_range, 20)
        sensor_range = PERCEPTION_LENGTH
        
        for angle in angles[::2]:
            end_x = robot_x + sensor_range * np.cos(angle)
            end_y = robot_y + sensor_range * np.sin(angle)
            ax_left.plot([robot_x, end_x], [robot_y, end_y], 
                        'g-', alpha=0.2, linewidth=0.8)
        
        # 绘制跟随距离圆
        circle = plt.Circle((target_x, target_y), FOLLOW_DISTANCE, 
                           color='orange', fill=False, linestyle='--', 
                           linewidth=2, alpha=0.5, label='Follow Distance')
        ax_left.add_patch(circle)
        
        ax_left.set_xlim(0, self.width)
        ax_left.set_ylim(0, self.height)
        ax_left.set_title('Ground Truth Map', fontsize=14, fontweight='bold')
        ax_left.set_xlabel('X', fontsize=12)
        ax_left.set_ylabel('Y', fontsize=12)
        ax_left.legend(loc='upper right', fontsize=9)
        ax_left.grid(True, alpha=0.3)
        
        # ===== 右图：语义地图 =====
        ax_right = axes[1]

        semantic_display = np.zeros((self.height, self.width, 3))
        semantic_display[self.semantic_map == 0] = [0.5, 0.5, 0.5]
        semantic_display[self.semantic_map == 1] = [1, 1, 1]
        semantic_display[self.semantic_map == 2] = [0, 0, 0]
        semantic_display[self.semantic_map == 3] = [1, 0, 0]
        
        ax_right.imshow(semantic_display, origin='lower')
        
        # 绘制跟随目标点
        if history_item['current_goal']:
            ax_right.scatter([history_item['current_goal'][0]], 
                        [history_item['current_goal'][1]], 
                        c='orange', s=200, marker='*', 
                        edgecolors='darkorange', linewidth=2, 
                        label='Following Goal', zorder=4)
        
        # 绘制规划路径
        if history_item['path']:
            path_array = np.array(history_item['path'])
            ax_right.plot(path_array[:, 0], path_array[:, 1], 
                         'c--', linewidth=2, alpha=0.8, label='Planned Path')
        
        # 绘制机器人路径
        if len(self.robot_path) > 1:
            path_array = np.array(self.robot_path)
            ax_right.plot(path_array[:, 0], path_array[:, 1], 
                         'b-', linewidth=1.5, alpha=0.6, label='Executed Path')
        
        # 绘制机器人
        ax_right.plot(robot_x, robot_y, 'bo', markersize=10, label='Robot', zorder=5)
        
        # 绘制FOV
        for angle in angles[::2]:
            end_x = robot_x + sensor_range * np.cos(angle)
            end_y = robot_y + sensor_range * np.sin(angle)
            ax_right.plot([robot_x, end_x], [robot_y, end_y], 
                         'g-', alpha=0.2, linewidth=0.8)
        
        # 状态文本
        status_text = f"Step: {history_item['step']}\n"
        status_text += f"Target {'VISIBLE' if history_item['target_visible'] else 'LOST'}"
        ax_right.text(0.02, 0.98, status_text, transform=ax_right.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax_right.set_xlim(0, self.width)
        ax_right.set_ylim(0, self.height)
        ax_right.set_title(f'Semantic Map - Dynamic Following', 
                        fontsize=14, fontweight='bold')
        ax_right.set_xlabel('X', fontsize=12)
        ax_right.set_ylabel('Y', fontsize=12)
        ax_right.legend(loc='upper right', fontsize=9)
        ax_right.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        
        plt.close(fig)
        return image


def run_demo_with_gif_export():
    """运行演示并导出GIF"""
    print("=" * 60)
    print("Dynamic Person Following Demo")
    print("=" * 60)
    
    demo = DynamicFollowingDemo(width=100, height=100, fov=90)
    
    max_steps = 300
    step = 0
    
    print("\n正在执行跟随任务...")
    while step < max_steps:
        demo.step()
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step} - Target {'VISIBLE' if demo.target_visible else 'LOST'}")
    
    # 生成GIF
    print("\n正在生成GIF...")
    total_frames = len(demo.history)
    frame_skip = max(1, total_frames // 200)
    
    frames = []
    for idx in tqdm(range(0, total_frames, frame_skip), desc="渲染帧"):
        frame = demo.render_frame(idx)
        if frame is not None:
            frames.append(frame)
    
    # 延长最后一帧
    if frames:
        last_frame = frames[-1]
        repeat_count = 30
        for _ in range(repeat_count):
            frames.append(last_frame.copy())
    
    # 保存GIF
    gif_filename = 'robot_following.gif'
    print(f"\n正在保存GIF...")
    
    imageio.mimsave(
        gif_filename, 
        frames, 
        duration=0.2,
        loop=0
    )
    
    print(f"✓ GIF已保存: {gif_filename}")
    print(f"  总帧数: {len(frames)}")
    
    # 统计信息
    visible_count = sum(1 for h in demo.history if h['target_visible'])
    print("\n" + "=" * 60)
    print("Following Statistics:")
    print("=" * 60)
    print(f"Total steps: {step}")
    print(f"Target visible: {visible_count}/{step} ({visible_count/step*100:.1f}%)")
    print(f"Explored cells: {np.sum(demo.semantic_map > 0)}")
    print("=" * 60)

if __name__ == "__main__":
    import os
    run_demo_with_gif_export()
