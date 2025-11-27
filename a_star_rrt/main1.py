import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import random
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

# ==================== 3D机器人臂运动学 ====================
class RobotArm3DOF:
    """3自由度机器人臂"""
    def __init__(self, link_lengths=[2.0, 1.5, 1.0]):
        self.link_lengths = link_lengths
        self.n_joints = len(link_lengths)
    
    def forward_kinematics(self, joint_angles):
        """
        正向运动学：从关节角度计算末端执行器位置
        joint_angles: [theta1, theta2, theta3] 单位：弧度
        返回: 各个关节的3D位置
        """
        positions = [np.array([0, 0, 0])]  # 基座位置
        
        # 第一个关节 (绕Z轴旋转)
        x1 = self.link_lengths[0] * np.cos(joint_angles[0])
        y1 = self.link_lengths[0] * np.sin(joint_angles[0])
        z1 = 0
        positions.append(np.array([x1, y1, z1]))
        
        # 第二个关节 (在XY平面上继续延伸，但考虑Z轴抬升)
        angle_sum_1 = joint_angles[0]
        x2 = x1 + self.link_lengths[1] * np.cos(angle_sum_1 + joint_angles[1]) * np.cos(joint_angles[2])
        y2 = y1 + self.link_lengths[1] * np.sin(angle_sum_1 + joint_angles[1]) * np.cos(joint_angles[2])
        z2 = z1 + self.link_lengths[1] * np.sin(joint_angles[2])
        positions.append(np.array([x2, y2, z2]))
        
        # 第三个关节 (末端执行器)
        angle_sum_2 = angle_sum_1 + joint_angles[1]
        x3 = x2 + self.link_lengths[2] * np.cos(angle_sum_2) * np.cos(joint_angles[2])
        y3 = y2 + self.link_lengths[2] * np.sin(angle_sum_2) * np.cos(joint_angles[2])
        z3 = z2 + self.link_lengths[2] * np.sin(joint_angles[2])
        positions.append(np.array([x3, y3, z3]))
        
        return positions
    
    def check_self_collision(self, joint_angles):
        """检查自碰撞"""
        # 简化：这里不做严格的自碰撞检测
        return False
    
    def check_collision_with_obstacles(self, joint_angles, obstacles):
        """检查与障碍物的碰撞"""
        positions = self.forward_kinematics(joint_angles)
        
        # 检查每个连杆与障碍物的碰撞
        for i in range(len(positions) - 1):
            # 检查连杆的多个点
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


# ==================== 3D RRT算法 ====================
class Node3D:
    def __init__(self, config):
        self.config = np.array(config)  # [theta1, theta2, theta3]
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
        
        # 配置空间边界 (关节角度限制)
        self.config_bounds = [
            (-np.pi, np.pi),      # theta1: -180° to 180°
            (-np.pi/2, np.pi/2),  # theta2: -90° to 90°
            (-np.pi/2, np.pi/2)   # theta3: -90° to 90°
        ]
    
    def plan(self):
        """RRT路径规划"""
        start_time = time.time()
        
        for i in range(self.max_iter):
            # 采样
            if random.random() < self.goal_sample_rate:
                rnd_config = self.goal.config
            else:
                rnd_config = self._sample_random_config()
            
            # 找到最近节点
            nearest_node = self._get_nearest_node(rnd_config)
            
            # 扩展
            new_config = self._steer(nearest_node.config, rnd_config)
            
            # 碰撞检测
            if not self._is_collision(new_config):
                new_node = Node3D(new_config)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + np.linalg.norm(new_config - nearest_node.config)
                self.node_list.append(new_node)
                
                # 检查是否到达目标
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
        """在配置空间中随机采样"""
        config = []
        for lower, upper in self.config_bounds:
            config.append(random.uniform(lower, upper))
        return np.array(config)
    
    def _get_nearest_node(self, config):
        """找到树中最近的节点"""
        distances = [np.linalg.norm(node.config - config) for node in self.node_list]
        return self.node_list[np.argmin(distances)]
    
    def _steer(self, from_config, to_config):
        """从from_config向to_config方向扩展step_size"""
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        if distance < self.step_size:
            return to_config
        else:
            return from_config + (direction / distance) * self.step_size
    
    def _is_collision(self, config):
        """检查配置是否碰撞"""
        # 检查关节限制
        for i, (lower, upper) in enumerate(self.config_bounds):
            if config[i] < lower or config[i] > upper:
                return True
        
        # 检查与障碍物碰撞
        return self.robot.check_collision_with_obstacles(config, self.obstacles)
    
    def _extract_path(self):
        """提取路径"""
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
        # (center, size)
        (np.array([2.0, 1.5, 0.5]), np.array([0.8, 0.8, 1.0])),  # 障碍物1
        (np.array([0.5, 2.5, 1.0]), np.array([1.0, 0.6, 0.8])),  # 障碍物2
        (np.array([-1.5, 1.0, 0.8]), np.array([0.6, 1.0, 1.2])), # 障碍物3
    ]
    return obstacles


def draw_box(ax, center, size, color='gray', alpha=0.3):
    """绘制3D长方体"""
    # 计算8个顶点
    x, y, z = center
    dx, dy, dz = size / 2
    
    vertices = [
        [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz],
        [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
        [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz],
        [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]
    ]
    
    # 定义6个面
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]
    
    # 绘制面
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidths=1)
    ax.add_collection3d(poly3d)


def draw_robot_arm(ax, robot, config, color='blue', linewidth=3, alpha=1.0):
    """绘制机器人臂"""
    positions = robot.forward_kinematics(config)
    
    # 绘制连杆
    for i in range(len(positions) - 1):
        ax.plot3D([positions[i][0], positions[i+1][0]],
                  [positions[i][1], positions[i+1][1]],
                  [positions[i][2], positions[i+1][2]],
                  color=color, linewidth=linewidth, alpha=alpha)
    
    # 绘制关节
    for pos in positions:
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, alpha=alpha, edgecolors='black', linewidths=1.5)
    
    # 绘制末端执行器
    end_pos = positions[-1]
    ax.scatter(end_pos[0], end_pos[1], end_pos[2], c='red', s=200, marker='*', 
              edgecolors='black', linewidths=2, alpha=alpha, zorder=10)


def visualize_3d_rrt(robot, obstacles, rrt_result, start_config, goal_config):
    """可视化3D RRT运动规划"""
    
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 14
    })
    
    fig = plt.figure(figsize=(20, 10))
    
    # 左图：RRT树的探索过程
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('RRT Tree Exploration in 3D Configuration Space', 
                  fontsize=20, fontweight='bold', pad=20)
    
    # 绘制障碍物
    for obs in obstacles:
        draw_box(ax1, obs[0], obs[1], color='red', alpha=0.2)
    
    # 绘制起始和目标配置的机器人
    draw_robot_arm(ax1, robot, start_config, color='green', linewidth=2, alpha=0.4)
    draw_robot_arm(ax1, robot, goal_config, color='orange', linewidth=2, alpha=0.4)
    
    # 绘制RRT树
    if rrt_result['success']:
        for node in rrt_result['tree'][::3]:  # 每隔3个绘制以避免太密
            if node.parent:
                # 绘制树的边
                parent_pos = robot.forward_kinematics(node.parent.config)[-1]
                node_pos = robot.forward_kinematics(node.config)[-1]
                ax1.plot3D([parent_pos[0], node_pos[0]],
                          [parent_pos[1], node_pos[1]],
                          [parent_pos[2], node_pos[2]],
                          'cyan', linewidth=0.5, alpha=0.3)
    
    # 绘制找到的路径
    if rrt_result['success'] and rrt_result['path']:
        path_positions = [robot.forward_kinematics(config)[-1] for config in rrt_result['path']]
        path_positions = np.array(path_positions)
        ax1.plot3D(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
                  'b-', linewidth=4, label='RRT Path', zorder=5)
    
    ax1.set_xlabel('X', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=16, fontweight='bold')
    ax1.set_zlabel('Z', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=14)
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([0, 4])
    
    # 右图：路径执行可视化
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Path Execution with Collision Avoidance', 
                  fontsize=20, fontweight='bold', pad=20)
    
    # 绘制障碍物
    for obs in obstacles:
        draw_box(ax2, obs[0], obs[1], color='red', alpha=0.3)
    
    # 绘制路径上的多个姿态
    if rrt_result['success'] and rrt_result['path']:
        n_poses = min(8, len(rrt_result['path']))
        indices = np.linspace(0, len(rrt_result['path'])-1, n_poses, dtype=int)
        
        for idx, i in enumerate(indices):
            alpha_val = 0.3 + 0.7 * (idx / (n_poses - 1))
            if idx == 0:
                color = 'green'
            elif idx == n_poses - 1:
                color = 'red'
            else:
                color = 'blue'
            
            draw_robot_arm(ax2, robot, rrt_result['path'][i], 
                         color=color, linewidth=2.5, alpha=alpha_val)
        
        # 绘制末端执行器轨迹
        path_positions = [robot.forward_kinematics(config)[-1] for config in rrt_result['path']]
        path_positions = np.array(path_positions)
        ax2.plot3D(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2],
                  'purple', linewidth=3, linestyle='--', label='End Effector Trajectory', alpha=0.8)
    
    ax2.set_xlabel('X', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=16, fontweight='bold')
    ax2.set_zlabel('Z', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=14)
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    ax2.set_zlim([0, 4])
    
    # 添加统计信息
    info_text = f"""
    Planning Results:
    • Status: {'Success ✓' if rrt_result['success'] else 'Failed ✗'}
    • Time: {rrt_result['time']:.3f}s
    • Nodes Explored: {rrt_result['nodes_explored']}
    • Path Cost: {rrt_result['path_cost']:.2f}
    • Path Length: {len(rrt_result['path']) if rrt_result['path'] else 0} waypoints
    """
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=14, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('3D Robot Arm Motion Planning with RRT', 
                fontsize=24, fontweight='bold', y=0.98)
    
    return fig


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("3D Robot Arm Motion Planning with RRT")
    print("=" * 70)
    
    # 创建机器人
    print("\n[1/4] Creating 3-DOF robot arm...")
    robot = RobotArm3DOF(link_lengths=[2.0, 1.5, 1.0])
    print(f"      Link lengths: {robot.link_lengths}")
    
    # 创建障碍物
    print("\n[2/4] Creating 3D obstacles...")
    obstacles = create_obstacles()
    print(f"      Number of obstacles: {len(obstacles)}")
    
    # 设置起始和目标配置 (关节角度)
    start_config = np.array([0.0, 0.0, 0.0])  # 初始姿态
    goal_config = np.array([np.pi/2, np.pi/4, np.pi/6])  # 目标姿态
    
    print(f"\n      Start configuration: {np.rad2deg(start_config)} degrees")
    print(f"      Goal configuration: {np.rad2deg(goal_config)} degrees")
    
    # 计算末端位置
    start_end_pos = robot.forward_kinematics(start_config)[-1]
    goal_end_pos = robot.forward_kinematics(goal_config)[-1]
    print(f"      Start end-effector position: ({start_end_pos[0]:.2f}, {start_end_pos[1]:.2f}, {start_end_pos[2]:.2f})")
    print(f"      Goal end-effector position: ({goal_end_pos[0]:.2f}, {goal_end_pos[1]:.2f}, {goal_end_pos[2]:.2f})")
    
    # 运行RRT
    print("\n[3/4] Running RRT in 3D configuration space...")
    rrt = RRT3D(robot, obstacles, start_config, goal_config, 
                max_iter=3000, step_size=0.3, goal_sample_rate=0.15)
    rrt_result = rrt.plan()
    
    if rrt_result['success']:
        print(f"      ✓ Path found!")
        print(f"      - Planning time: {rrt_result['time']:.3f} seconds")
        print(f"      - Nodes explored: {rrt_result['nodes_explored']}")
        print(f"      - Path cost: {rrt_result['path_cost']:.2f}")
        print(f"      - Path waypoints: {len(rrt_result['path'])}")
    else:
        print(f"      ✗ No path found within {rrt.max_iter} iterations")
    
    # 可视化
    print("\n[4/4] Generating 3D visualization...")
    fig = visualize_3d_rrt(robot, obstacles, rrt_result, start_config, goal_config)
    
    print("\n" + "=" * 70)
    print("🎯 Key Insights about 3D Motion Planning:")
    print("=" * 70)
    print("• Configuration space: 3D (3 joint angles)")
    print("• Workspace: 3D physical space")
    print("• RRT explores joint space, avoids obstacles in Cartesian space")
    print("• Each node represents a full robot configuration")
    print("• Path is collision-free in high-dimensional space")
    print("=" * 70)
    
    plt.show()
    
    print("\n✅ 3D Motion Planning visualization complete!")


if __name__ == "__main__":
    main()
