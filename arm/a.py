import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import random
import heapq
from matplotlib.gridspec import GridSpec

# ==================== 3D机器人臂运动学 ====================
class RobotArm3DOF:
    """3自由度机器人臂"""
    def __init__(self, link_lengths=[2.0, 1.5, 1.0]):
        self.link_lengths = link_lengths
        self.n_joints = len(link_lengths)
    
    def forward_kinematics(self, joint_angles):
        """正向运动学：从关节角度计算末端执行器位置"""
        positions = [np.array([0, 0, 0])]
        
        x1 = self.link_lengths[0] * np.cos(joint_angles[0])
        y1 = self.link_lengths[0] * np.sin(joint_angles[0])
        z1 = 0
        positions.append(np.array([x1, y1, z1]))
        
        angle_sum_1 = joint_angles[0]
        x2 = x1 + self.link_lengths[1] * np.cos(angle_sum_1 + joint_angles[1]) * np.cos(joint_angles[2])
        y2 = y1 + self.link_lengths[1] * np.sin(angle_sum_1 + joint_angles[1]) * np.cos(joint_angles[2])
        z2 = z1 + self.link_lengths[1] * np.sin(joint_angles[2])
        positions.append(np.array([x2, y2, z2]))
        
        angle_sum_2 = angle_sum_1 + joint_angles[1]
        x3 = x2 + self.link_lengths[2] * np.cos(angle_sum_2) * np.cos(joint_angles[2])
        y3 = y2 + self.link_lengths[2] * np.sin(angle_sum_2) * np.cos(joint_angles[2])
        z3 = z2 + self.link_lengths[2] * np.sin(joint_angles[2])
        positions.append(np.array([x3, y3, z3]))
        
        return positions
    
    def check_collision_with_obstacles(self, joint_angles, obstacles):
        """检查与障碍物的碰撞"""
        positions = self.forward_kinematics(joint_angles)
        
        for i in range(len(positions) - 1):
            for t in np.linspace(0, 1, 10):
                point = positions[i] + t * (positions[i+1] - positions[i])
                for obs in obstacles:
                    if self._point_in_box(point, obs):
                        return True
        return False
    
    def _point_in_box(self, point, box):
        """检查点是否在长方体内"""
        center, size = box
        return (abs(point[0] - center[0]) < size[0]/2 and
                abs(point[1] - center[1]) < size[1]/2 and
                abs(point[2] - center[2]) < size[2]/2)


# ==================== A* 算法（配置空间版本）====================
class AStar3D:
    """3D配置空间的A*算法"""
    def __init__(self, robot, obstacles, start_config, goal_config, 
                 resolution=0.2, max_iter=10000):
        self.robot = robot
        self.obstacles = obstacles
        self.start = tuple(np.round(start_config / resolution).astype(int))
        self.goal = tuple(np.round(goal_config / resolution).astype(int))
        self.resolution = resolution
        self.max_iter = max_iter
        self.explored_nodes = []
        
        # 配置空间边界
        self.config_bounds = [
            (-np.pi, np.pi),
            (-np.pi/2, np.pi/2),
            (-np.pi/2, np.pi/2)
        ]
    
    def plan(self):
        """A*路径规划"""
        start_time = time.time()
        
        counter = 0
        open_set = [(0, counter, self.start)]
        counter += 1
        
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self._heuristic(self.start, self.goal)}
        closed_set = set()
        
        iterations = 0
        while open_set and iterations < self.max_iter:
            iterations += 1
            current_f, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            self.explored_nodes.append(current)
            closed_set.add(current)
            
            # 检查是否到达目标
            if self._is_goal(current):
                path = self._reconstruct_path(came_from, current)
                end_time = time.time()
                return {
                    'success': True,
                    'path': path,
                    'explored_nodes': self.explored_nodes,
                    'time': end_time - start_time,
                    'nodes_explored': len(self.explored_nodes),
                    'path_cost': g_score[current]
                }
            
            # 探索邻居
            for neighbor in self._get_neighbors(current):
                neighbor_config = self._discrete_to_continuous(neighbor)
                
                # 检查边界
                if not self._is_valid_config(neighbor_config):
                    continue
                
                # 检查碰撞
                if self.robot.check_collision_with_obstacles(neighbor_config, self.obstacles):
                    continue
                
                # 计算代价
                move_cost = self._config_distance(current, neighbor)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self._heuristic(neighbor, self.goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
        
        end_time = time.time()
        return {
            'success': False,
            'path': None,
            'explored_nodes': self.explored_nodes,
            'time': end_time - start_time,
            'nodes_explored': len(self.explored_nodes),
            'path_cost': float('inf')
        }
    
    def _get_neighbors(self, config):
        """获取邻居配置（26-连通）"""
        neighbors = []
        for d1 in [-1, 0, 1]:
            for d2 in [-1, 0, 1]:
                for d3 in [-1, 0, 1]:
                    if d1 == 0 and d2 == 0 and d3 == 0:
                        continue
                    neighbor = (config[0] + d1, config[1] + d2, config[2] + d3)
                    neighbors.append(neighbor)
        return neighbors
    
    def _heuristic(self, config1, config2):
        """启发函数：配置空间欧几里得距离"""
        return np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(config1, config2)))
    
    def _config_distance(self, config1, config2):
        """配置空间距离"""
        return np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(config1, config2)))
    
    def _is_goal(self, config):
        """检查是否到达目标"""
        return self._config_distance(config, self.goal) < 2.0  # 容差
    
    def _is_valid_config(self, config):
        """检查配置是否在边界内"""
        for i, (lower, upper) in enumerate(self.config_bounds):
            if config[i] < lower or config[i] > upper:
                return False
        return True
    
    def _discrete_to_continuous(self, discrete_config):
        """离散配置转连续配置"""
        return np.array([c * self.resolution for c in discrete_config])
    
    def _reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [self._discrete_to_continuous(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self._discrete_to_continuous(current))
        path.reverse()
        return path


# ==================== RRT 算法 ====================
class Node3D:
    def __init__(self, config):
        self.config = np.array(config)
        self.parent = None
        self.cost = 0

class RRT3D:
    """3D配置空间的RRT算法"""
    def __init__(self, robot, obstacles, start_config, goal_config, 
                 max_iter=3000, step_size=0.3, goal_sample_rate=0.1):
        self.robot = robot
        self.obstacles = obstacles
        self.start = Node3D(start_config)
        self.goal = Node3D(goal_config)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.node_list = [self.start]
        
        self.config_bounds = [
            (-np.pi, np.pi),
            (-np.pi/2, np.pi/2),
            (-np.pi/2, np.pi/2)
        ]
    
    def plan(self):
        """RRT路径规划"""
        start_time = time.time()
        
        for i in range(self.max_iter):
            if random.random() < self.goal_sample_rate:
                rnd_config = self.goal.config
            else:
                rnd_config = self._sample_random_config()
            
            nearest_node = self._get_nearest_node(rnd_config)
            new_config = self._steer(nearest_node.config, rnd_config)
            
            if not self._is_collision(new_config):
                new_node = Node3D(new_config)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + np.linalg.norm(new_config - nearest_node.config)
                self.node_list.append(new_node)
                
                if np.linalg.norm(new_config - self.goal.config) < self.step_size:
                    if not self._is_collision(self.goal.config):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + np.linalg.norm(self.goal.config - new_config)
                        end_time = time.time()
                        
                        path = self._extract_path()
                        return {
                            'success': True,
                            'path': path,
                            'tree': self.node_list,
                            'time': end_time - start_time,
                            'nodes_explored': len(self.node_list),
                            'path_cost': self.goal.cost
                        }
        
        end_time = time.time()
        return {
            'success': False,
            'path': None,
            'tree': self.node_list,
            'time': end_time - start_time,
            'nodes_explored': len(self.node_list),
            'path_cost': float('inf')
        }
    
    def _sample_random_config(self):
        config = []
        for lower, upper in self.config_bounds:
            config.append(random.uniform(lower, upper))
        return np.array(config)
    
    def _get_nearest_node(self, config):
        distances = [np.linalg.norm(node.config - config) for node in self.node_list]
        return self.node_list[np.argmin(distances)]
    
    def _steer(self, from_config, to_config):
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        if distance < self.step_size:
            return to_config
        else:
            return from_config + (direction / distance) * self.step_size
    
    def _is_collision(self, config):
        for i, (lower, upper) in enumerate(self.config_bounds):
            if config[i] < lower or config[i] > upper:
                return True
        return self.robot.check_collision_with_obstacles(config, self.obstacles)
    
    def _extract_path(self):
        path = [self.goal.config]
        node = self.goal
        while node.parent is not None:
            node = node.parent
            path.append(node.config)
        path.reverse()
        return path


# ==================== 可视化函数 ====================
def create_obstacles():
    """创建3D障碍物"""
    obstacles = [
        (np.array([2.0, 1.5, 0.5]), np.array([0.8, 0.8, 1.0])),
        (np.array([0.5, 2.5, 1.0]), np.array([1.0, 0.6, 0.8])),
        (np.array([-1.5, 1.0, 0.8]), np.array([0.6, 1.0, 1.2])),
    ]
    return obstacles


def draw_box(ax, center, size, color='gray', alpha=0.3):
    """绘制3D长方体"""
    x, y, z = center
    dx, dy, dz = size / 2
    
    vertices = [
        [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz],
        [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
        [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz],
        [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]
    ]
    
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]
    
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidths=1)
    ax.add_collection3d(poly3d)


def draw_robot_arm(ax, robot, config, color='blue', linewidth=3, alpha=1.0):
    """绘制机器人臂"""
    positions = robot.forward_kinematics(config)
    
    for i in range(len(positions) - 1):
        ax.plot3D([positions[i][0], positions[i+1][0]],
                  [positions[i][1], positions[i+1][1]],
                  [positions[i][2], positions[i+1][2]],
                  color=color, linewidth=linewidth, alpha=alpha)
    
    for pos in positions:
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, alpha=alpha, 
                  edgecolors='black', linewidths=1.5)
    
    end_pos = positions[-1]
    ax.scatter(end_pos[0], end_pos[1], end_pos[2], c='red', s=200, marker='*', 
              edgecolors='black', linewidths=2, alpha=alpha, zorder=10)


def visualize_comparison_3d(robot, obstacles, astar_result, rrt_result, 
                           start_config, goal_config):
    """可视化A*和RRT的对比"""
    
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 14
    })
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. A* 可视化
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_title(f'A* Algorithm\nTime: {astar_result["time"]:.3f}s', 
                  fontsize=20, fontweight='bold', pad=15)
    
    for obs in obstacles:
        draw_box(ax1, obs[0], obs[1], color='red', alpha=0.2)
    
    draw_robot_arm(ax1, robot, start_config, color='green', linewidth=2, alpha=0.4)
    draw_robot_arm(ax1, robot, goal_config, color='orange', linewidth=2, alpha=0.4)
    
    if astar_result['success'] and astar_result['path']:
        path_positions = [robot.forward_kinematics(config)[-1] for config in astar_result['path']]
        path_positions = np.array(path_positions)
        ax1.plot3D(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
                  'b-', linewidth=4, label='A* Path', zorder=5)
    
    ax1.set_xlabel('X', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax1.set_zlabel('Z', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([0, 4])
    
    # 2. RRT 可视化
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_title(f'RRT Algorithm\nTime: {rrt_result["time"]:.3f}s', 
                  fontsize=20, fontweight='bold', pad=15)
    
    for obs in obstacles:
        draw_box(ax2, obs[0], obs[1], color='red', alpha=0.2)
    
    draw_robot_arm(ax2, robot, start_config, color='green', linewidth=2, alpha=0.4)
    draw_robot_arm(ax2, robot, goal_config, color='orange', linewidth=2, alpha=0.4)
    
    if rrt_result['success']:
        for node in rrt_result['tree'][::3]:
            if node.parent:
                parent_pos = robot.forward_kinematics(node.parent.config)[-1]
                node_pos = robot.forward_kinematics(node.config)[-1]
                ax2.plot3D([parent_pos[0], node_pos[0]],
                          [parent_pos[1], node_pos[1]],
                          [parent_pos[2], node_pos[2]],
                          'cyan', linewidth=0.5, alpha=0.3)
    
    if rrt_result['success'] and rrt_result['path']:
        path_positions = [robot.forward_kinematics(config)[-1] for config in rrt_result['path']]
        path_positions = np.array(path_positions)
        ax2.plot3D(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
                  'orange', linewidth=4, label='RRT Path', zorder=5)
    
    ax2.set_xlabel('X', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax2.set_zlabel('Z', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    ax2.set_zlim([0, 4])
    
    # 3. 路径对比
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.set_title('Path Comparison', fontsize=20, fontweight='bold', pad=15)
    
    for obs in obstacles:
        draw_box(ax3, obs[0], obs[1], color='red', alpha=0.2)
    
    if astar_result['success'] and astar_result['path']:
        path_positions = [robot.forward_kinematics(config)[-1] for config in astar_result['path']]
        path_positions = np.array(path_positions)
        ax3.plot3D(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
                  'b-', linewidth=3, label='A* Path', alpha=0.8)
    
    if rrt_result['success'] and rrt_result['path']:
        path_positions = [robot.forward_kinematics(config)[-1] for config in rrt_result['path']]
        path_positions = np.array(path_positions)
        ax3.plot3D(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
                  'orange', linewidth=3, label='RRT Path', alpha=0.8)
    
    ax3.set_xlabel('X', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax3.set_zlabel('Z', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.set_xlim([-5, 5])
    ax3.set_ylim([-5, 5])
    ax3.set_zlim([0, 4])
    
    # 4-6. 性能对比柱状图
    metrics = ['Time (s)', 'Path Cost', 'Nodes Explored']
    astar_metrics = [
        astar_result['time'], 
        astar_result['path_cost'] if astar_result['success'] else 0,
        astar_result['nodes_explored']
    ]
    rrt_metrics = [
        rrt_result['time'],
        rrt_result['path_cost'] if rrt_result['success'] else 0,
        rrt_result['nodes_explored']
    ]
    
    colors = ['#3498db', '#e74c3c']
    
    for idx, (metric_name, astar_val, rrt_val) in enumerate(zip(metrics, astar_metrics, rrt_metrics)):
        ax = fig.add_subplot(gs[1, idx])
        
        bars = ax.bar(['A*', 'RRT'], [astar_val, rrt_val], color=colors, 
                     edgecolor='black', linewidth=2, width=0.6)
        ax.set_title(metric_name, fontsize=20, fontweight='bold', pad=15)
        ax.set_ylabel('Value', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}' if idx < 2 else f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=18)
        
        # 标注获胜者
        winner_idx = 0 if astar_val < rrt_val else 1
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(4)
    
    # 7-9. 路径执行可视化
    for idx, (result, title, color) in enumerate([
        (astar_result, 'A* Path Execution', 'blue'),
        (rrt_result, 'RRT Path Execution', 'orange')
    ]):
        ax = fig.add_subplot(gs[2, idx], projection='3d')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        
        for obs in obstacles:
            draw_box(ax, obs[0], obs[1], color='red', alpha=0.3)
        
        if result['success'] and result['path']:
            n_poses = min(6, len(result['path']))
            indices = np.linspace(0, len(result['path'])-1, n_poses, dtype=int)
            
            for pose_idx, i in enumerate(indices):
                alpha_val = 0.3 + 0.7 * (pose_idx / (n_poses - 1))
                draw_robot_arm(ax, robot, result['path'][i], 
                             color=color, linewidth=2.5, alpha=alpha_val)
            
            path_positions = [robot.forward_kinematics(config)[-1] for config in result['path']]
            path_positions = np.array(path_positions)
            ax.plot3D(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
                     'purple', linewidth=3, linestyle='--', alpha=0.8)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 4])
    
    plt.suptitle('A* vs RRT: 3D Robot Arm Motion Planning Comparison', 
                fontsize=26, fontweight='bold', y=0.98)
    
    return fig


# ==================== 主程序 ====================
def main():
    print("=" * 80)
    print("A* vs RRT: 3D Robot Arm Motion Planning Comparison")
    print("=" * 80)
    
    print("\n[1/5] Creating 3-DOF robot arm...")
    robot = RobotArm3DOF(link_lengths=[2.0, 1.5, 1.0])
    
    print("\n[2/5] Creating 3D obstacles...")
    obstacles = create_obstacles()
    
    start_config = np.array([0.0, 0.0, 0.0])
    goal_config = np.array([np.pi/2, np.pi/4, np.pi/6])
    
    print(f"\n      Start: {np.rad2deg(start_config)} degrees")
    print(f"      Goal: {np.rad2deg(goal_config)} degrees")
    
    print("\n[3/5] Running A* algorithm...")
    astar = AStar3D(robot, obstacles, start_config, goal_config, 
                    resolution=0.2, max_iter=10000)
    astar_result = astar.plan()
    
    if astar_result['success']:
        print(f"      ✓ Path found!")
        print(f"      - Time: {astar_result['time']:.3f}s")
        print(f"      - Nodes explored: {astar_result['nodes_explored']}")
        print(f"      - Path cost: {astar_result['path_cost']:.2f}")
    else:
        print(f"      ✗ No path found")
    
    print("\n[4/5] Running RRT algorithm...")
    rrt = RRT3D(robot, obstacles, start_config, goal_config, 
                max_iter=3000, step_size=0.3, goal_sample_rate=0.15)
    rrt_result = rrt.plan()
    
    if rrt_result['success']:
        print(f"      ✓ Path found!")
        print(f"      - Time: {rrt_result['time']:.3f}s")
        print(f"      - Nodes explored: {rrt_result['nodes_explored']}")
        print(f"      - Path cost: {rrt_result['path_cost']:.2f}")
    else:
        print(f"      ✗ No path found")
    
    print("\n[5/5] Generating comparison visualization...")
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    if astar_result['success'] and rrt_result['success']:
        print(f"\n{'Metric':<25} {'A*':<20} {'RRT':<20} {'Winner':<15}")
        print("-" * 80)
        
        time_winner = "A*" if astar_result['time'] < rrt_result['time'] else "RRT"
        time_diff = abs(astar_result['time'] - rrt_result['time']) / min(astar_result['time'], rrt_result['time']) * 100
        print(f"{'Time (s)':<25} {astar_result['time']:<20.3f} {rrt_result['time']:<20.3f} {time_winner} ({time_diff:.1f}% faster)")
        
        cost_winner = "A*" if astar_result['path_cost'] < rrt_result['path_cost'] else "RRT"
        print(f"{'Path Cost':<25} {astar_result['path_cost']:<20.2f} {rrt_result['path_cost']:<20.2f} {cost_winner}")
        
        nodes_winner = "A*" if astar_result['nodes_explored'] < rrt_result['nodes_explored'] else "RRT"
        nodes_diff = abs(astar_result['nodes_explored'] - rrt_result['nodes_explored']) / max(astar_result['nodes_explored'], rrt_result['nodes_explored']) * 100
        print(f"{'Nodes Explored':<25} {astar_result['nodes_explored']:<20} {rrt_result['nodes_explored']:<20} {nodes_winner} ({nodes_diff:.1f}% fewer)")
        
        print("\n" + "=" * 80)
        print("🎯 Key Insights:")
        print("\n" + "=" * 80)
        print("🎯 Key Insights:")
        print("=" * 80)
        print("• A* guarantees optimal paths but explores more nodes systematically")
        print("• RRT is probabilistically complete and often faster in high dimensions")
        print("• A* requires discretization which increases with dimensionality")
        print("• RRT's random sampling is more efficient in continuous spaces")
        print("• Trade-off: A* (optimal but slow) vs RRT (fast but suboptimal)")
        print("=" * 80)
    
    fig = visualize_comparison_3d(robot, obstacles, astar_result, rrt_result, 
                                  start_config, goal_config)
    plt.show()
    
    print("\n✅ 3D Motion Planning comparison complete!")


if __name__ == "__main__":
    main()
