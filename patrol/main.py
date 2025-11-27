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

# 全局参数
PERCEPTION_RANGE = 40  # 感知范围
ROBOT_RADIUS = 1.5  # 机器人半径
PATROL_SPEED = 2.0  # 巡检速度

class PatrolInspectionDemo:
    def __init__(self, width=100, height=100, fov=120):
        """
        初始化巡检演示
        width, height: 地图大小
        fov: 视场角（度）
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.fov_rad = np.radians(fov)
        
        # 机器人状态
        self.robot_x = 20
        self.robot_y = 20
        self.robot_angle = 0
        
        # 巡检路径（矩形巡检路线）
        self.patrol_waypoints = self._generate_patrol_path()
        self.current_waypoint_idx = 0
        
        # 地图和障碍物
        self.ground_truth_map = self._generate_facility_map()
        self.semantic_map = np.zeros((height, width), dtype=int)
        
        # 物品（红色=可疑，绿色=正常）
        self.objects = self._generate_random_objects()
        
        # 检测记录
        self.detected_objects = []  # 存储检测到的物体
        self.alarm_active = False  # 当前是否报警
        self.alarm_position = None  # 报警位置
        self.alarm_cooldown = 0  # 报警冷却
        
        # 历史记录
        self.robot_path = [(self.robot_x, self.robot_y)]
        self.history = []
        self.step_count = 0
        
    def _generate_patrol_path(self):
        """生成巡检路径（矩形）"""
        margin = 15
        waypoints = [
            (margin, margin),
            (self.width - margin, margin),
            (self.width - margin, self.height - margin),
            (margin, self.height - margin),
            (margin, margin)  # 回到起点
        ]
        return waypoints
    
    def _generate_facility_map(self):
        """生成设施地图（包含一些障碍物）"""
        facility_map = np.ones((self.height, self.width), dtype=int)
        
        # 添加一些设施/障碍物
        obstacles = [
            (40, 30, 8, 8),   # (x, y, width, height)
            (60, 60, 10, 6),
            (25, 70, 6, 8),
            (75, 25, 5, 12)
        ]
        
        for x, y, w, h in obstacles:
            x1, x2 = max(0, x), min(self.width, x + w)
            y1, y2 = max(0, y), min(self.height, y + h)
            facility_map[y1:y2, x1:x2] = 2
        
        return facility_map
    
    def _generate_random_objects(self):
        """生成随机物品（红色=可疑，绿色=正常）"""
        objects = []
        
        # 生成5-8个绿色物品（正常）
        for _ in range(np.random.randint(5, 9)):
            x = np.random.randint(20, self.width - 20)
            y = np.random.randint(20, self.height - 20)
            
            # 确保不在障碍物上
            if self.ground_truth_map[y, x] == 1:
                objects.append({
                    'x': x,
                    'y': y,
                    'type': 'normal',  # 绿色
                    'color': 'green',
                    'detected': False,
                    'id': len(objects)
                })
        
        # 生成2-4个红色物品（可疑）
        for _ in range(np.random.randint(2, 5)):
            x = np.random.randint(20, self.width - 20)
            y = np.random.randint(20, self.height - 20)
            
            if self.ground_truth_map[y, x] == 1:
                objects.append({
                    'x': x,
                    'y': y,
                    'type': 'suspicious',  # 红色
                    'color': 'red',
                    'detected': False,
                    'id': len(objects)
                })
        
        return objects
    
    def _get_current_target_waypoint(self):
        """获取当前目标路径点"""
        return self.patrol_waypoints[self.current_waypoint_idx]
    
    def _move_towards_waypoint(self):
        """向当前路径点移动"""
        target_x, target_y = self._get_current_target_waypoint()
        
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 2.0:
            # 到达路径点，切换到下一个
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.patrol_waypoints)
            return
        
        # 更新朝向
        self.robot_angle = np.arctan2(dy, dx)
        
        # 移动
        move_distance = min(PATROL_SPEED, distance)
        self.robot_x += move_distance * np.cos(self.robot_angle)
        self.robot_y += move_distance * np.sin(self.robot_angle)
        
        self.robot_path.append((self.robot_x, self.robot_y))
    
    def _detect_objects(self):
        """检测视野内的物品"""
        self.alarm_active = False
        
        # 更新报警冷却
        if self.alarm_cooldown > 0:
            self.alarm_cooldown -= 1
        
        angle_range = self.fov_rad / 2
        
        for obj in self.objects:
            if obj['detected']:
                continue
            
            # 计算物体相对机器人的位置
            dx = obj['x'] - self.robot_x
            dy = obj['y'] - self.robot_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # 检查是否在感知范围内
            if distance > PERCEPTION_RANGE:
                continue
            
            # 检查是否在视野角度内
            obj_angle = np.arctan2(dy, dx)
            angle_diff = np.abs(np.arctan2(np.sin(obj_angle - self.robot_angle),
                                           np.cos(obj_angle - self.robot_angle)))
            
            if angle_diff > angle_range:
                continue
            
            # 检查是否有障碍物遮挡
            if self._is_path_clear(self.robot_x, self.robot_y, obj['x'], obj['y']):
                # 检测到物品！
                obj['detected'] = True
                obj['detected_step'] = self.step_count
                obj['detected_position'] = (self.robot_x, self.robot_y)
                
                self.detected_objects.append(obj)
                
                # 如果是可疑物品，触发报警
                if obj['type'] == 'suspicious' and self.alarm_cooldown == 0:
                    self.alarm_active = True
                    self.alarm_position = (obj['x'], obj['y'])
                    self.alarm_cooldown = 20  # 报警持续时间
    
    def _is_path_clear(self, x1, y1, x2, y2, step_size=1.0):
        """检查两点之间路径是否畅通"""
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if distance < 0.1:
            return True
        
        steps = int(distance / step_size) + 1
        
        for i in range(steps + 1):
            t = i / steps
            check_x = int(x1 + t * (x2 - x1))
            check_y = int(y1 + t * (y2 - y1))
            
            if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                return False
            
            if self.ground_truth_map[check_y, check_x] == 2:
                return False
        
        return True
    
    def _update_semantic_map(self):
        """更新语义地图"""
        angle_range = self.fov_rad / 2
        
        for angle in np.linspace(self.robot_angle - angle_range, 
                                 self.robot_angle + angle_range, 60):
            for dist in np.linspace(1, PERCEPTION_RANGE, 40):
                x = int(self.robot_x + dist * np.cos(angle))
                y = int(self.robot_y + dist * np.sin(angle))
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.semantic_map[y, x] == 0:
                        self.semantic_map[y, x] = self.ground_truth_map[y, x]
                    
                    # 如果遇到障碍物，停止该方向的扫描
                    if self.ground_truth_map[y, x] == 2:
                        break
    
    def step(self):
        """执行一步巡检"""
        self.step_count += 1
        
        # 1. 移动到下一个路径点
        self._move_towards_waypoint()
        
        # 2. 更新语义地图
        self._update_semantic_map()
        
        # 3. 检测物品
        self._detect_objects()
        
        # 4. 记录历史
        self.history.append({
            'step': self.step_count,
            'robot_pos': (self.robot_x, self.robot_y),
            'robot_angle': self.robot_angle,
            'semantic_map': copy.deepcopy(self.semantic_map),
            'robot_path': copy.deepcopy(self.robot_path),
            'objects': copy.deepcopy(self.objects),
            'detected_objects': copy.deepcopy(self.detected_objects),
            'alarm_active': self.alarm_active,
            'alarm_position': self.alarm_position,
            'current_waypoint': self._get_current_target_waypoint()
        })
        
        return self.step_count < 500  # 最多500步
    
    def render_frame(self, step_idx=None):
        """渲染一帧"""
        if step_idx is None:
            step_idx = len(self.history) - 1
        
        if step_idx < 0 or step_idx >= len(self.history):
            return None
        
        history_item = self.history[step_idx]
        robot_x, robot_y = history_item['robot_pos']
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
        fig.patch.set_facecolor('white')
        
        # ===== 左图：Ground Truth =====
        ax_left = axes[0]
        
        # 绘制地图
        ground_truth_display = np.zeros((self.height, self.width, 3))
        ground_truth_display[self.ground_truth_map == 1] = [0.95, 0.95, 0.95]
        ground_truth_display[self.ground_truth_map == 2] = [0.3, 0.3, 0.3]
        
        ax_left.imshow(ground_truth_display, origin='lower')
        
        # 绘制巡检路径
        waypoints = np.array(self.patrol_waypoints)
        ax_left.plot(waypoints[:, 0], waypoints[:, 1], 
                    'b--', linewidth=2, alpha=0.5, label='Patrol Route')
        ax_left.scatter(waypoints[:, 0], waypoints[:, 1], 
                       c='blue', s=100, marker='s', alpha=0.5)
        
        # 绘制所有物品
        for obj in history_item['objects']:
            marker_size = 200 if obj['detected'] else 150
            alpha = 0.8 if obj['detected'] else 1.0
            
            ax_left.scatter(obj['x'], obj['y'], 
                          c=obj['color'], s=marker_size, 
                          marker='o', alpha=alpha,
                          edgecolors='black', linewidth=2)
            
            if obj['detected']:
                # 标记已检测
                circle = plt.Circle((obj['x'], obj['y']), 3, 
                                  color='yellow', fill=False, 
                                  linewidth=2, linestyle='--')
                ax_left.add_patch(circle)
        
        # 绘制机器人路径
        if len(history_item['robot_path']) > 1:
            path_array = np.array(history_item['robot_path'])
            ax_left.plot(path_array[:, 0], path_array[:, 1], 
                        'cyan', linewidth=2, alpha=0.6, label='Robot Trail')
        
        # 绘制机器人
        robot_circle = plt.Circle((robot_x, robot_y), ROBOT_RADIUS, 
                                 color='blue', fill=True, alpha=0.5)
        ax_left.add_patch(robot_circle)
        ax_left.plot(robot_x, robot_y, 'bo', markersize=12, label='Robot')
        
        # 绘制视野
        angle_range = self.fov_rad / 2
        robot_angle = history_item['robot_angle']
        angles = np.linspace(robot_angle - angle_range, 
                            robot_angle + angle_range, 25)
        
        for angle in angles[::3]:
            end_x = robot_x + PERCEPTION_RANGE * np.cos(angle)
            end_y = robot_y + PERCEPTION_RANGE * np.sin(angle)
            ax_left.plot([robot_x, end_x], [robot_y, end_y], 
                        'lime', alpha=0.15, linewidth=1)
        
        # 报警效果
        if history_item['alarm_active'] and history_item['alarm_position']:
            alarm_x, alarm_y = history_item['alarm_position']
            for radius in [5, 8, 11]:
                alarm_circle = plt.Circle((alarm_x, alarm_y), radius, 
                                        color='red', fill=False, 
                                        linewidth=3, alpha=0.7)
                ax_left.add_patch(alarm_circle)
            
            ax_left.text(alarm_x, alarm_y + 15, '⚠️ ALARM!', 
                        fontsize=16, fontweight='bold', color='red',
                        ha='center', bbox=dict(boxstyle='round', 
                        facecolor='yellow', alpha=0.8))
        
        ax_left.set_xlim(0, self.width)
        ax_left.set_ylim(0, self.height)
        ax_left.set_title('Ground Truth - Facility Patrol Inspection', 
                         fontsize=14, fontweight='bold')
        ax_left.legend(loc='upper right', fontsize=10)
        ax_left.grid(True, alpha=0.3)
        
        # ===== 右图：语义地图 =====
        ax_right = axes[1]
        
        semantic_display = np.zeros((self.height, self.width, 3))
        semantic_display[history_item['semantic_map'] == 0] = [0.2, 0.2, 0.2]
        semantic_display[history_item['semantic_map'] == 1] = [0.95, 0.95, 0.95]
        semantic_display[history_item['semantic_map'] == 2] = [0.3, 0.3, 0.3]
        
        ax_right.imshow(semantic_display, origin='lower')
        
        # 绘制已检测物品
        for obj in history_item['detected_objects']:
            ax_right.scatter(obj['x'], obj['y'], 
                           c=obj['color'], s=200, 
                           marker='o', alpha=0.9,
                           edgecolors='white', linewidth=2)
        
        # 绘制机器人
        robot_circle = plt.Circle((robot_x, robot_y), ROBOT_RADIUS, 
                                 color='blue', fill=True, alpha=0.5)
        ax_right.add_patch(robot_circle)
        ax_right.plot(robot_x, robot_y, 'bo', markersize=12)
        
        # 绘制视野
        for angle in angles[::3]:
            end_x = robot_x + PERCEPTION_RANGE * np.cos(angle)
            end_y = robot_y + PERCEPTION_RANGE * np.sin(angle)
            ax_right.plot([robot_x, end_x], [robot_y, end_y], 
                         'lime', alpha=0.15, linewidth=1)
        
        # 状态信息
        total_objects = len(self.objects)
        detected_count = len(history_item['detected_objects'])
        suspicious_detected = sum(1 for obj in history_item['detected_objects'] 
                                 if obj['type'] == 'suspicious')
        normal_detected = detected_count - suspicious_detected
        
        status_text = f"Step: {history_item['step']}\n"
        status_text += f"Detected: {detected_count}/{total_objects}\n"
        status_text += f"Normal: {normal_detected}\n"
        status_text += f"Suspicious: {suspicious_detected}\n"
        
        if history_item['alarm_active']:
            status_text += "\nALARM ACTIVE!"
        
        bbox_color = 'red' if history_item['alarm_active'] else 'wheat'
        ax_right.text(0.02, 0.98, status_text, transform=ax_right.transAxes,
                     fontsize=12, verticalalignment='top', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.9))
        
        ax_right.set_xlim(0, self.width)
        ax_right.set_ylim(0, self.height)
        ax_right.set_title('Semantic Map - Detection Results', 
                          fontsize=14, fontweight='bold')
        ax_right.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        
        plt.close(fig)
        return image


def run_patrol_demo():
    """运行巡检演示并导出GIF"""
    print("=" * 60)
    print("机器人设施巡检演示")
    print("=" * 60)
    print(f"感知范围: {PERCEPTION_RANGE}m")
    print(f"巡检速度: {PATROL_SPEED}m/s")
    print("🟢 绿色物体 = 正常物品（不报警）")
    print("🔴 红色物体 = 可疑物品（报警）")
    print("=" * 60)
    
    demo = PatrolInspectionDemo(width=100, height=100, fov=120)
    
    print(f"\n生成了 {len(demo.objects)} 个物品:")
    suspicious_count = sum(1 for obj in demo.objects if obj['type'] == 'suspicious')
    normal_count = len(demo.objects) - suspicious_count
    print(f"  🟢 正常物品: {normal_count}")
    print(f"  🔴 可疑物品: {suspicious_count}")
    
    print("\n开始巡检...")
    
    max_steps = 400
    step = 0
    
    while step < max_steps:
        demo.step()
        step += 1
        
        if step % 50 == 0:
            detected = len(demo.detected_objects)
            total = len(demo.objects)
            print(f"Step {step} - 已检测: {detected}/{total}")
    
    print("\n生成GIF...")
    
    total_frames = len(demo.history)
    frame_skip = max(1, total_frames // 200)
    
    frames = []
    for idx in tqdm(range(0, total_frames, frame_skip), desc="渲染帧"):
        frame = demo.render_frame(idx)
        if frame is not None:
            frames.append(frame)
    
    # 延长最后一帧
    if frames:
        for _ in range(30):
            frames.append(frames[-1].copy())
    
    gif_filename = 'robot_patrol_inspection.gif'
    imageio.mimsave(gif_filename, frames, duration=0.15, loop=0)
    
    print(f"\n✓ GIF已保存: {gif_filename}")
    print(f"  总帧数: {len(frames)}")
    
    # 统计
    print("\n" + "=" * 60)
    print("巡检统计:")
    print("=" * 60)
    print(f"总步数: {step}")
    print(f"检测到的物品: {len(demo.detected_objects)}/{len(demo.objects)}")
    
    suspicious_detected = [obj for obj in demo.detected_objects if obj['type'] == 'suspicious']
    normal_detected = [obj for obj in demo.detected_objects if obj['type'] == 'normal']
    
    print(f"正常物品: {len(normal_detected)}")
    print(f"可疑物品: {len(suspicious_detected)}")
    
    alarm_count = sum(1 for h in demo.history if h['alarm_active'])
    print(f"\n报警次数: {alarm_count}")
    print(f"探索区域: {np.sum(demo.semantic_map > 0)} 格")
    print("=" * 60)


if __name__ == "__main__":
    run_patrol_demo()
