import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Tuple, Optional
import matplotlib.lines as mlines

class RRTNode:
    """RRT树节点"""
    def __init__(self, x: float, y: float, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []
        
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other) -> float:
        """计算到另一个节点的欧氏距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class RRTVisualizer:
    """RRT算法可视化器（采样型算法演示）"""
    
    def __init__(self, width: int, height: int, step_size: float = 0.5):
        self.width = width
        self.height = height
        self.step_size = step_size  # RRT扩展步长
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        self.goal_radius = 0.5  # 目标区域半径
        
    def add_obstacle(self, x: int, y: int):
        """添加障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
    
    def add_obstacles_rect(self, x: int, y: int, w: int, h: int):
        """添加矩形障碍物"""
        for i in range(y, min(y + h, self.height)):
            for j in range(x, min(x + w, self.width)):
                self.grid[i, j] = 1
    
    def add_obstacles_circle(self, cx: float, cy: float, radius: float):
        """添加圆形障碍物"""
        for i in range(self.height):
            for j in range(self.width):
                if np.sqrt((j - cx)**2 + (i - cy)**2) <= radius:
                    self.grid[i, j] = 1
    
    def set_start(self, x: float, y: float):
        """设置起点"""
        self.start = (x, y)
    
    def set_goal(self, x: float, y: float):
        """设置终点"""
        self.goal = (x, y)
    
    def is_collision_free(self, x: float, y: float) -> bool:
        """检查点是否无碰撞"""
        # 边界检查
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # 检查整数坐标及其周围
        grid_x, grid_y = int(x), int(y)
        
        # 检查附近的格子（更精确的碰撞检测）
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x = grid_x + dx
                check_y = grid_y + dy
                if (0 <= check_x < self.width and 
                    0 <= check_y < self.height and 
                    self.grid[check_y, check_x] == 1):
                    # 检查距离
                    if np.sqrt((x - check_x)**2 + (y - check_y)**2) < 0.7:
                        return False
        
        return True
    
    def is_path_collision_free(self, node1: RRTNode, node2: RRTNode, 
                               num_checks: int = 20) -> bool:
        """检查两点之间的路径是否无碰撞"""
        for i in range(num_checks + 1):
            t = i / num_checks
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            if not self.is_collision_free(x, y):
                return False
        return True
    
    def sample_random_point(self) -> Tuple[float, float]:
        """随机采样一个点（关键：体现采样型算法特征）"""
        # 90%概率完全随机采样，10%概率朝目标偏置
        if np.random.random() < 0.9:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
        else:
            # 目标偏置采样
            x = self.goal[0] + np.random.normal(0, 1)
            y = self.goal[1] + np.random.normal(0, 1)
            x = np.clip(x, 0, self.width)
            y = np.clip(y, 0, self.height)
        
        return (x, y)
    
    def find_nearest_node(self, tree: List[RRTNode], 
                         point: Tuple[float, float]) -> RRTNode:
        """在树中找到距离采样点最近的节点"""
        target_node = RRTNode(point[0], point[1])
        nearest = min(tree, key=lambda node: node.distance_to(target_node))
        return nearest
    
    def steer(self, from_node: RRTNode, to_point: Tuple[float, float]) -> RRTNode:
        """从from_node向to_point方向扩展固定步长"""
        direction_x = to_point[0] - from_node.x
        direction_y = to_point[1] - from_node.y
        distance = np.sqrt(direction_x**2 + direction_y**2)
        
        if distance <= self.step_size:
            # 如果距离小于步长，直接到达目标点
            return RRTNode(to_point[0], to_point[1], from_node)
        else:
            # 否则沿方向扩展步长
            ratio = self.step_size / distance
            new_x = from_node.x + ratio * direction_x
            new_y = from_node.y + ratio * direction_y
            return RRTNode(new_x, new_y, from_node)
    
    def is_goal_reached(self, node: RRTNode) -> bool:
        """检查是否到达目标"""
        distance = np.sqrt((node.x - self.goal[0])**2 + 
                          (node.y - self.goal[1])**2)
        return distance <= self.goal_radius
    
    def extract_path(self, goal_node: RRTNode) -> List[Tuple[float, float]]:
        """从目标节点回溯提取路径"""
        path = []
        current = goal_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]  # 反转路径
    
    def rrt_step_by_step(self, max_iterations: int = 500):
        """
        RRT逐步搜索，返回每一步的状态
        
        Returns:
            List of step_info dictionaries
        """
        # 初始化树
        root = RRTNode(self.start[0], self.start[1])
        tree = [root]
        steps = []
        
        # 记录初始状态
        steps.append({
            'tree': [node for node in tree],
            'sampled_point': None,
            'nearest_node': None,
            'new_node': None,
            'path': None,
            'found': False,
            'step_type': 'init',
            'iteration': 0
        })
        
        for iteration in range(max_iterations):
            # 1. 随机采样
            sampled_point = self.sample_random_point()
            
            steps.append({
                'tree': [node for node in tree],
                'sampled_point': sampled_point,
                'nearest_node': None,
                'new_node': None,
                'path': None,
                'found': False,
                'step_type': 'sample',
                'iteration': iteration + 1
            })
            
            # 2. 找最近节点
            nearest_node = self.find_nearest_node(tree, sampled_point)
            
            steps.append({
                'tree': [node for node in tree],
                'sampled_point': sampled_point,
                'nearest_node': nearest_node,
                'new_node': None,
                'path': None,
                'found': False,
                'step_type': 'nearest',
                'iteration': iteration + 1
            })
            
            # 3. 扩展
            new_node = self.steer(nearest_node, sampled_point)
            
            # 4. 碰撞检测
            if (self.is_collision_free(new_node.x, new_node.y) and 
                self.is_path_collision_free(nearest_node, new_node)):
                
                # 添加到树中
                nearest_node.children.append(new_node)
                tree.append(new_node)
                
                steps.append({
                    'tree': [node for node in tree],
                    'sampled_point': sampled_point,
                    'nearest_node': nearest_node,
                    'new_node': new_node,
                    'path': None,
                    'found': False,
                    'step_type': 'extend',
                    'iteration': iteration + 1
                })
                
                # 5. 检查是否到达目标
                if self.is_goal_reached(new_node):
                    path = self.extract_path(new_node)
                    steps.append({
                        'tree': [node for node in tree],
                        'sampled_point': None,
                        'nearest_node': None,
                        'new_node': new_node,
                        'path': path,
                        'found': True,
                        'step_type': 'found',
                        'iteration': iteration + 1
                    })
                    return steps
            else:
                # 碰撞，记录失败的尝试
                steps.append({
                    'tree': [node for node in tree],
                    'sampled_point': sampled_point,
                    'nearest_node': nearest_node,
                    'new_node': new_node,
                    'path': None,
                    'found': False,
                    'step_type': 'collision',
                    'iteration': iteration + 1
                })
        
        # 达到最大迭代次数
        steps.append({
            'tree': [node for node in tree],
            'sampled_point': None,
            'nearest_node': None,
            'new_node': None,
            'path': None,
            'found': False,
            'step_type': 'max_iter',
            'iteration': max_iterations
        })
        
        return steps
    
    def visualize_static_explanation(self):
        """创建静态RRT算法说明图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # 运行RRT获取所有步骤
        steps = self.rrt_step_by_step(max_iterations=500)
        
        # 找关键帧
        init_step = steps[0]
        
        # 找第一次成功扩展
        extend_step = None
        for step in steps:
            if step['step_type'] == 'extend':
                extend_step = step
                break
        
        # 找中间某次扩展
        extend_steps = [s for s in steps if s['step_type'] == 'extend']
        mid_extend = extend_steps[len(extend_steps)//2] if extend_steps else extend_step
        
        # 最终结果
        final_step = steps[-1]
        
        # 绘制四个关键阶段
        self._draw_rrt_state(axes[0, 0], init_step, 
                            "Step 1: Initialization\n(Start from root node)")
        self._draw_rrt_state(axes[0, 1], extend_step if extend_step else init_step, 
                            "Step 2: First Extension\n(Sample → Find Nearest → Extend)")
        self._draw_rrt_state(axes[1, 0], mid_extend if mid_extend else extend_step, 
                            f"Step {len(extend_steps)//2}: Tree Growing\n(Exploring configuration space)")
        self._draw_rrt_state(axes[1, 1], final_step, 
                            "Final: Path Found!\n(Extract solution from tree)")
        
        plt.tight_layout()
        return fig
    
    def _draw_rrt_state(self, ax, step_info, title):
        """绘制RRT在某一步的状态"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        tree = step_info['tree']
        sampled_point = step_info['sampled_point']
        nearest_node = step_info['nearest_node']
        new_node = step_info['new_node']
        path = step_info['path']
        step_type = step_info['step_type']
        
        # 绘制障碍物
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1,
                                            linewidth=1, edgecolor='black',
                                            facecolor='#2C3E50')
                    ax.add_patch(rect)
        
        # 绘制RRT树的所有边（关键：展示树的增长）
        for node in tree:
            if node.parent is not None:
                ax.plot([node.parent.x, node.x], 
                       [node.parent.y, node.y],
                       'b-', linewidth=1.5, alpha=0.4, zorder=3)
        
        # 绘制RRT树的所有节点
        for node in tree:
            ax.plot(node.x, node.y, 'o', color='#3498DB', 
                   markersize=4, zorder=4, alpha=0.6)
        
        # 绘制采样点
        if sampled_point and step_type in ['sample', 'nearest', 'collision']:
            ax.plot(sampled_point[0], sampled_point[1], '*', 
                   color='#F39C12', markersize=20, zorder=7,
                   markeredgecolor='darkorange', markeredgewidth=2,
                   label='Random Sample')
        
        # 绘制最近节点
        if nearest_node and step_type in ['nearest', 'extend', 'collision']:
            ax.plot(nearest_node.x, nearest_node.y, 'o', 
                   color='#9B59B6', markersize=14, zorder=8,
                   markeredgecolor='purple', markeredgewidth=2,
                   label='Nearest Node')
            
            # 绘制连接线
            if sampled_point:
                ax.plot([nearest_node.x, sampled_point[0]], 
                       [nearest_node.y, sampled_point[1]],
                       'g--', linewidth=2, alpha=0.5, zorder=6)
        
        # 绘制新节点
        if new_node and step_type == 'extend':
            ax.plot(new_node.x, new_node.y, 'o', 
                   color='#2ECC71', markersize=12, zorder=9,
                   markeredgecolor='darkgreen', markeredgewidth=2,
                   label='New Node')
            
            # 高亮新添加的边
            if nearest_node:
                ax.plot([nearest_node.x, new_node.x], 
                       [nearest_node.y, new_node.y],
                       'g-', linewidth=3, alpha=0.8, zorder=8)
        
        # 绘制碰撞的尝试
        if new_node and step_type == 'collision':
            ax.plot(new_node.x, new_node.y, 'x', 
                   color='#E74C3C', markersize=12, zorder=9,
                   markeredgewidth=3, label='Collision')
        
        # 绘制起点
        sx, sy = self.start
        ax.plot(sx, sy, 'o', color='#2ECC71', 
               markersize=22, zorder=10,
               markeredgecolor='darkgreen', markeredgewidth=3)
        ax.text(sx, sy, 'S', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
        
        # 绘制目标区域
        gx, gy = self.goal
        goal_circle = plt.Circle((gx, gy), self.goal_radius, 
                                color='#E74C3C', alpha=0.3, zorder=2)
        ax.add_patch(goal_circle)
        ax.plot(gx, gy, 's', color='#E74C3C', 
               markersize=22, zorder=10,
               markeredgecolor='darkred', markeredgewidth=3)
        ax.text(gx, gy, 'G', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
        
        # 如果找到路径，绘制路径
        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            ax.plot(path_x, path_y, 'r-', linewidth=4, alpha=0.8, zorder=11,
                   label=f'Final Path')
            
            # 绘制路径上的节点
            for x, y in path:
                ax.plot(x, y, 'o', color='#FF6B6B',
                       markersize=8, zorder=12, alpha=0.9)
        
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=9)
        
        # 添加统计信息
        info_text = f"Tree Nodes: {len(tree)}\n"
        info_text += f"Iteration: {step_info['iteration']}\n"
        
        if step_type == 'sample':
            info_text += "Action: Random Sampling"
        elif step_type == 'nearest':
            info_text += "Action: Finding Nearest"
        elif step_type == 'extend':
            info_text += "Action: Extending Tree"
        elif step_type == 'collision':
            info_text += "Action: Collision Detected"
        elif step_type == 'found':
            info_text += f"✓ Goal Reached!\nPath Length: {len(path)-1}"
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def create_rrt_animation(self, filename='rrt_animation.gif', 
                            fps=3, max_iterations=500):
        """创建RRT搜索过程的动画"""
        steps = self.rrt_step_by_step(max_iterations=max_iterations)
        
        # 选择关键帧（不是每一步都显示，太慢了）
        key_frames = []
        key_frames.append(steps[0])  # 初始
        
        for i, step in enumerate(steps[1:], 1):
            # 显示所有采样、最近节点查找、扩展和碰撞
            if step['step_type'] in ['sample', 'extend', 'found', 'max_iter']:
                key_frames.append(step)
            # 每5次迭代显示一次其他类型
            elif i % 5 == 0:
                key_frames.append(step)
        
        # 确保最后一帧在内
        if steps[-1] not in key_frames:
            key_frames.append(steps[-1])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame_num):
            ax.clear()
            
            step_info = key_frames[frame_num]
            
            # 设置标题
            step_type = step_info['step_type']
            iteration = step_info['iteration']
            
            title_dict = {
                'init': f'Iteration 0: Initialize RRT\n(Sample-based algorithm starts from root)',
                'sample': f'Iteration {iteration}: Random Sampling\n(Key difference: Random exploration of space)',
                'nearest': f'Iteration {iteration}: Find Nearest Node\n(Search tree for closest node)',
                'extend': f'Iteration {iteration}: Extend Tree\n(Add new branch to tree)',
                'collision': f'Iteration {iteration}: Collision Detected\n(Discard this sample, try again)',
                'found': f'Iteration {iteration}: ✓ Goal Reached!\n(Extract path from tree)',
                'max_iter': f'Iteration {iteration}: Maximum Iterations\n(Search terminated)'
            }
            
            ax.set_title(title_dict.get(step_type, f'Iteration {iteration}'),
                        fontsize=16, fontweight='bold', pad=20)
            
            tree = step_info['tree']
            sampled_point = step_info['sampled_point']
            nearest_node = step_info['nearest_node']
            new_node = step_info['new_node']
            path = step_info['path']
            
            # 绘制障碍物
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == 1:
                        rect = patches.Rectangle((j, i), 1, 1,
                                                linewidth=1.5, edgecolor='black',
                                                facecolor='#2C3E50')
                        ax.add_patch(rect)
            
            # 绘制RRT树的所有边
            for node in tree:
                if node.parent is not None:
                    ax.plot([node.parent.x, node.x], 
                           [node.parent.y, node.y],
                           'b-', linewidth=1.5, alpha=0.5, zorder=3)
            
            # 绘制RRT树的所有节点
            for node in tree:
                ax.plot(node.x, node.y, 'o', color='#3498DB', 
                       markersize=5, zorder=4, alpha=0.7,
                       markeredgecolor='#2874A6', markeredgewidth=0.5)
            
            # 绘制采样点（带动画效果）
            if sampled_point and step_type in ['sample', 'nearest', 'collision']:
                # 外圈脉冲效果
                ax.plot(sampled_point[0], sampled_point[1], 'o', 
                       color='#F39C12', markersize=25, zorder=6, alpha=0.3)
                ax.plot(sampled_point[0], sampled_point[1], '*', 
                       color='#F39C12', markersize=20, zorder=7,
                       markeredgecolor='darkorange', markeredgewidth=2)
                
                # 添加文字标注
                ax.text(sampled_point[0], sampled_point[1] - 0.5, 
                       'Random\nSample', ha='center', va='top',
                       fontsize=9, color='darkorange', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', 
                                alpha=0.7, edgecolor='orange'))
            
            # 绘制最近节点
            if nearest_node and step_type in ['nearest', 'extend', 'collision']:
                ax.plot(nearest_node.x, nearest_node.y, 'o', 
                       color='#9B59B6', markersize=16, zorder=8,
                       markeredgecolor='purple', markeredgewidth=2)
                
                # 绘制连接线（从最近节点到采样点）
                if sampled_point:
                    ax.plot([nearest_node.x, sampled_point[0]], 
                           [nearest_node.y, sampled_point[1]],
                           'g--', linewidth=2.5, alpha=0.6, zorder=6)
                    
                    # 添加箭头
                    ax.annotate('', xy=(sampled_point[0], sampled_point[1]),
                               xytext=(nearest_node.x, nearest_node.y),
                               arrowprops=dict(arrowstyle='->', color='green',
                                             lw=2, alpha=0.6))
            
            # 绘制新节点
            if new_node and step_type == 'extend':
                ax.plot(new_node.x, new_node.y, 'o', 
                       color='#2ECC71', markersize=14, zorder=9,
                       markeredgecolor='darkgreen', markeredgewidth=2)
                
                # 高亮新添加的边
                if nearest_node:
                    ax.plot([nearest_node.x, new_node.x], 
                           [nearest_node.y, new_node.y],
                           'g-', linewidth=4, alpha=0.9, zorder=8)
                
                # 添加文字标注
                ax.text(new_node.x, new_node.y + 0.5, 
                       'New!', ha='center', va='bottom',
                       fontsize=10, color='darkgreen', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', 
                                alpha=0.8))
            
            # 绘制碰撞
            if new_node and step_type == 'collision':
                ax.plot(new_node.x, new_node.y, 'x', 
                       color='#E74C3C', markersize=16, zorder=9,
                       markeredgewidth=4)
                
                # 绘制尝试的路径
                if nearest_node:
                    ax.plot([nearest_node.x, new_node.x], 
                           [nearest_node.y, new_node.y],
                           'r--', linewidth=2, alpha=0.5, zorder=7)
                
                ax.text(new_node.x, new_node.y + 0.5, 
                       'Collision!', ha='center', va='bottom',
                       fontsize=10, color='darkred', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', 
                                alpha=0.8))
            
            # 绘制起点
            sx, sy = self.start
            ax.plot(sx, sy, 'o', color='#2ECC71', 
                   markersize=24, zorder=10,
                   markeredgecolor='darkgreen', markeredgewidth=3)
            ax.text(sx, sy, 'S', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white', zorder=11)
            
            # 绘制目标区域
            gx, gy = self.goal
            goal_circle = plt.Circle((gx, gy), self.goal_radius, 
                                    color='#E74C3C', alpha=0.2, zorder=2)
            ax.add_patch(goal_circle)
            
            # 添加脉冲效果
            goal_circle2 = plt.Circle((gx, gy), self.goal_radius * 1.5, 
                                     color='#E74C3C', alpha=0.1, zorder=1)
            ax.add_patch(goal_circle2)
            
            ax.plot(gx, gy, 's', color='#E74C3C', 
                   markersize=24, zorder=10,
                   markeredgecolor='darkred', markeredgewidth=3)
            ax.text(gx, gy, 'G', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white', zorder=11)
            
            # 如果找到路径，绘制路径
            if path:
                path_x = [x for x, y in path]
                path_y = [y for x, y in path]
                ax.plot(path_x, path_y, 'r-', linewidth=5, alpha=0.8, zorder=11)
                
                # 绘制路径上的节点
                for x, y in path:
                    ax.plot(x, y, 'o', color='#FF6B6B',
                           markersize=9, zorder=12, alpha=0.9,
                           markeredgecolor='darkred', markeredgewidth=1.5)
            
            ax.set_xlim(-0.5, self.width + 0.5)
            ax.set_ylim(-0.5, self.height + 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            
            # 添加详细的RRT信息
            info_text = f"🌳 RRT Algorithm Status:\n"
            info_text += f"{'─'*30}\n"
            info_text += f"Tree Size: {len(tree)} nodes\n"
            info_text += f"Iteration: {iteration}\n"
            info_text += f"{'─'*30}\n"
            
            if step_type == 'sample':
                info_text += f"📍 Sampling random point\n"
                info_text += f"   in configuration space"
            elif step_type == 'nearest':
                info_text += f"🔍 Finding nearest node\n"
                info_text += f"   in existing tree"
            elif step_type == 'extend':
                info_text += f"✅ Successfully extended!\n"
                info_text += f"   New branch added"
            elif step_type == 'collision':
                info_text += f"❌ Collision detected\n"
                info_text += f"   Sample rejected"
            elif step_type == 'found':
                info_text += f"{'─'*30}\n"
                info_text += f"🎯 Goal Reached!\n"
                info_text += f"Path length: {len(path)-1:.2f}\n"
                info_text += f"Success!"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            alpha=0.95, edgecolor='orange', linewidth=2))
            
            # 添加算法特征说明
            feature_text = "🔑 RRT Key Features:\n"
            feature_text += "• Random sampling\n"
            feature_text += "• Tree-based growth\n"
            feature_text += "• Probabilistic complete\n"
            feature_text += "• Non-optimal path"
            
            ax.text(0.98, 0.98, feature_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', 
                            alpha=0.9, edgecolor='blue', linewidth=2))
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=len(key_frames),
                           interval=1000/fps, repeat=True)
        
        # 保存为GIF
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)
        plt.close()
        
        print(f"✓ Animation saved as: {filename}")
        print(f"  Total iterations: {steps[-1]['iteration']}")
        print(f"  Key frames: {len(key_frames)}")
        print(f"  Frame rate: {fps} FPS")
        
        return anim


# ==================== 对比展示函数 ====================

def create_comparison_figure(rrt_viz, bfs_viz=None):
    """创建RRT vs 搜索算法的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # 左侧：RRT（采样型）
    ax_rrt = axes[0]
    ax_rrt.set_title('RRT Algorithm\n(Sample-Based / Probabilistic)', 
                     fontsize=16, fontweight='bold', pad=20)
    
    steps_rrt = rrt_viz.rrt_step_by_step(max_iterations=300)
    final_rrt = steps_rrt[-1]
    
    # 绘制RRT结果
    tree = final_rrt['tree']
    path = final_rrt['path']
    
    # 障碍物
    for i in range(rrt_viz.height):
        for j in range(rrt_viz.width):
            if rrt_viz.grid[i, j] == 1:
                rect = patches.Rectangle((j, i), 1, 1,
                                        linewidth=1, edgecolor='black',
                                        facecolor='#2C3E50')
                ax_rrt.add_patch(rect)
    
    # RRT树
    for node in tree:
        if node.parent is not None:
            ax_rrt.plot([node.parent.x, node.x], 
                       [node.parent.y, node.y],
                       'b-', linewidth=1, alpha=0.4, zorder=3)
    
    for node in tree:
        ax_rrt.plot(node.x, node.y, 'o', color='#3498DB', 
                   markersize=4, zorder=4, alpha=0.6)
    
    # 路径
    if path:
        path_x = [x for x, y in path]
        path_y = [y for x, y in path]
        ax_rrt.plot(path_x, path_y, 'r-', linewidth=4, alpha=0.8, zorder=11)
    
    # 起点和终点
    sx, sy = rrt_viz.start
    ax_rrt.plot(sx, sy, 'o', color='#2ECC71', markersize=20, zorder=10)
    ax_rrt.text(sx, sy, 'S', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
    
    gx, gy = rrt_viz.goal
    ax_rrt.plot(gx, gy, 's', color='#E74C3C', markersize=20, zorder=10)
    ax_rrt.text(gx, gy, 'G', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
    
    ax_rrt.set_xlim(-0.5, rrt_viz.width + 0.5)
    ax_rrt.set_ylim(-0.5, rrt_viz.height + 0.5)
    ax_rrt.set_aspect('equal')
    ax_rrt.invert_yaxis()
    ax_rrt.grid(True, alpha=0.2)
    
    # 添加特征说明
    rrt_features = "✨ RRT Characteristics:\n"
    rrt_features += "━━━━━━━━━━━━━━━━━━━━\n"
    rrt_features += "✓ Random exploration\n"
    rrt_features += "✓ Tree structure\n"
    rrt_features += "✓ Fast in high dimensions\n"
    rrt_features += "✓ Probabilistic complete\n"
    rrt_features += "✓ Continuous space\n"
    rrt_features += "━━━━━━━━━━━━━━━━━━━━\n"
    rrt_features += f"✗ Non-optimal path\n"
    rrt_features += f"✗ Path may be jagged\n"
    rrt_features += f"━━━━━━━━━━━━━━━━━━━━\n"
    rrt_features += f"Tree nodes: {len(tree)}\n"
    if path:
        rrt_features += f"Path length: {len(path)-1}"
    
    ax_rrt.text(0.02, 0.98, rrt_features, transform=ax_rrt.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', 
                        alpha=0.9, edgecolor='blue', linewidth=2))
    
    # 右侧：概念性对比图
    ax_compare = axes[1]
    ax_compare.set_title('Search-Based Algorithm (e.g., BFS)\n(Graph-Based / Deterministic)', 
                         fontsize=16, fontweight='bold', pad=20)
    
    # 创建一个简化的网格来展示搜索算法的特点
    for i in range(rrt_viz.height):
        for j in range(rrt_viz.width):
            if rrt_viz.grid[i, j] == 1:
                rect = patches.Rectangle((j, i), 1, 1,
                                        linewidth=1, edgecolor='black',
                                        facecolor='#2C3E50')
                ax_compare.add_patch(rect)
            else:
                # 绘制网格结构
                rect = patches.Rectangle((j, i), 1, 1,
                                        linewidth=0.5, edgecolor='gray',
                                        facecolor='none', alpha=0.3)
                ax_compare.add_patch(rect)
    
    # 绘制网格连接（展示图结构）
    for i in range(rrt_viz.height):
        for j in range(rrt_viz.width):
            if rrt_viz.grid[i, j] == 0:
                # 向右连接
                if j + 1 < rrt_viz.width and rrt_viz.grid[i, j + 1] == 0:
                    ax_compare.plot([j + 0.5, j + 1.5], [i + 0.5, i + 0.5],
                                   'lightgray', linewidth=1, alpha=0.5)
                # 向下连接
                if i + 1 < rrt_viz.height and rrt_viz.grid[i + 1, j] == 0:
                    ax_compare.plot([j + 0.5, j + 0.5], [i + 0.5, i + 1.5],
                                   'lightgray', linewidth=1, alpha=0.5)
    
    # 绘制所有网格点
    for i in range(rrt_viz.height):
        for j in range(rrt_viz.width):
            if rrt_viz.grid[i, j] == 0:
                ax_compare.plot(j + 0.5, i + 0.5, 'o', color='#AED6F1',
                               markersize=6, alpha=0.7, zorder=5)
    
    # 起点和终点
    sx, sy = rrt_viz.start
    ax_compare.plot(sx + 0.5, sy + 0.5, 'o', color='#2ECC71', 
                    markersize=20, zorder=10)
    ax_compare.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
    
    gx, gy = rrt_viz.goal
    ax_compare.plot(gx + 0.5, gy + 0.5, 's', color='#E74C3C', 
                    markersize=20, zorder=10)
    ax_compare.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
    
    ax_compare.set_xlim(-0.5, rrt_viz.width + 0.5)
    ax_compare.set_ylim(-0.5, rrt_viz.height + 0.5)
    ax_compare.set_aspect('equal')
    ax_compare.invert_yaxis()
    ax_compare.grid(True, alpha=0.2)
    
    # 添加特征说明
    search_features = "✨ Search-Based Characteristics:\n"
    search_features += "━━━━━━━━━━━━━━━━━━━━\n"
    search_features += "✓ Systematic exploration\n"
    search_features += "✓ Graph/Grid structure\n"
    search_features += "✓ Guaranteed optimal (BFS)\n"
    search_features += "✓ Deterministic\n"
    search_features += "✓ Discrete space\n"
    search_features += "━━━━━━━━━━━━━━━━━━━━\n"
    search_features += f"✗ Slow in high dimensions\n"
    search_features += f"✗ Memory intensive\n"
    search_features += "━━━━━━━━━━━━━━━━━━━━\n"
    search_features += f"Structure: Predefined graph\n"
    search_features += f"Exploration: Layer by layer"
    
    ax_compare.text(0.02, 0.98, search_features, transform=ax_compare.transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', 
                            alpha=0.9, edgecolor='green', linewidth=2))
    
    plt.tight_layout()
    return fig


# ==================== 使用示例 ====================

def demo_simple_rrt():
    """简单场景RRT演示"""
    print("=" * 60)
    print("Example 1: Simple RRT Visualization")
    print("=" * 60)
    
    viz = RRTVisualizer(width=10, height=8, step_size=0.6)
    
    # 添加障碍物
    viz.add_obstacles_rect(3, 2, 2, 4)
    viz.add_obstacles_rect(6, 1, 1, 3)
    viz.add_obstacle(7, 5)
    viz.add_obstacle(8, 5)
    
    # 设置起点和终点
    viz.set_start(1.0, 3.0)
    viz.set_goal(8.5, 3.5)
    
    # 创建静态说明图
    print("Generating static explanation diagram...")
    fig = viz.visualize_static_explanation()
    plt.savefig('rrt_steps_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建动画
    print("\nGenerating RRT animation...")
    viz.create_rrt_animation('rrt_simple.gif', fps=3, max_iterations=300)
    print("✓ Done!")


def demo_maze_rrt():
    """迷宫场景RRT演示"""
    print("\n" + "=" * 60)
    print("Example 2: Maze RRT Visualization")
    print("=" * 60)
    
    viz = RRTVisualizer(width=12, height=10, step_size=0.5)
    
    # 创建迷宫障碍物
    viz.add_obstacles_rect(2, 1, 1, 6)
    viz.add_obstacles_rect(4, 3, 1, 6)
    viz.add_obstacles_rect(6, 1, 1, 5)
    viz.add_obstacles_rect(8, 4, 1, 5)
    viz.add_obstacles_rect(10, 2, 1, 4)
    
    # 设置起点和终点
    viz.set_start(0.5, 0.5)
    viz.set_goal(11.0, 9.0)
    
    # 创建动画
    print("Generating maze RRT animation...")
    viz.create_rrt_animation('rrt_maze.gif', fps=4, max_iterations=500)
    print("✓ Done!")


def demo_complex_rrt():
    """复杂场景RRT演示"""
    print("\n" + "=" * 60)
    print("Example 3: Complex RRT Visualization")
    print("=" * 60)
    
    viz = RRTVisualizer(width=15, height=12, step_size=0.6)
    
    # 创建复杂障碍物
    viz.add_obstacles_rect(3, 2, 3, 2)
    viz.add_obstacles_rect(3, 6, 3, 2)
    viz.add_obstacles_rect(8, 3, 2, 5)
    viz.add_obstacles_rect(11, 1, 2, 4)
    viz.add_obstacles_rect(11, 7, 2, 4)
    
    # 添加圆形障碍物（更真实）
    viz.add_obstacles_circle(6.5, 4.5, 1.2)
    
    # 设置起点和终点
    viz.set_start(1.0, 1.0)
    viz.set_goal(13.5, 10.5)
    
    # 创建静态说明图
    print("Generating complex scenario static diagram...")
    fig = viz.visualize_static_explanation()
    plt.savefig('rrt_complex_steps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建动画
    print("\nGenerating complex RRT animation...")
    viz.create_rrt_animation('rrt_complex.gif', fps=4, max_iterations=500)
    print("✓ Done!")


def demo_rrt_vs_search():
    """RRT vs 搜索算法对比演示"""
    print("\n" + "=" * 60)
    print("Example 4: RRT vs Search-Based Comparison")
    print("=" * 60)
    
    viz = RRTVisualizer(width=12, height=10, step_size=0.5)
    
    # 添加障碍物
    viz.add_obstacles_rect(3, 2, 2, 4)
    viz.add_obstacles_rect(7, 3, 2, 5)
    
    viz.set_start(1.0, 2.0)
    viz.set_goal(10.0, 7.0)
    
    # 创建对比图
    print("Generating comparison diagram...")
    fig = create_comparison_figure(viz)
    plt.savefig('rrt_vs_search_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Done!")


def demo_narrow_passage():
    """窄通道场景（展示RRT的挑战）"""
    print("\n" + "=" * 60)
    print("Example 5: Narrow Passage Challenge")
    print("=" * 60)
    
    viz = RRTVisualizer(width=14, height=10, step_size=0.4)
    
    # 创建窄通道
    viz.add_obstacles_rect(5, 0, 1, 4)
    viz.add_obstacles_rect(5, 6, 1, 4)
    # 中间留一个窄通道（在y=4到y=6之间）
    
    viz.add_obstacles_rect(9, 0, 1, 3)
    viz.add_obstacles_rect(9, 7, 1, 3)
    
    viz.set_start(1.0, 5.0)
    viz.set_goal(12.0, 5.0)
    
    # 创建动画
    print("Generating narrow passage animation...")
    print("(This may take longer due to difficult scenario)")
    viz.create_rrt_animation('rrt_narrow_passage.gif', fps=5, max_iterations=800)
    print("✓ Done!")


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("🌳 RRT Path Planning Algorithm Visualization")
    print("   (Sample-Based Motion Planning)")
    print("=" * 60)
    print()
    
    # 示例1：简单场景
    demo_simple_rrt()
    
    # 示例2：迷宫场景
    demo_maze_rrt()
    
    # 示例3：复杂场景
    demo_complex_rrt()
    
    # 示例4：对比演示
    demo_rrt_vs_search()
    
    # 示例5：窄通道挑战
    demo_narrow_passage()
    
    print("\n" + "=" * 60)
    print("✓ All RRT demonstrations completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  📊 rrt_steps_explanation.png - RRT static explanation")
    print("  📊 rrt_complex_steps.png - Complex scenario steps")
    print("  📊 rrt_vs_search_comparison.png - Algorithm comparison")
    print("  🎬 rrt_simple.gif - Simple scenario animation")
    print("  🎬 rrt_maze.gif - Maze scenario animation")
    print("  🎬 rrt_complex.gif - Complex scenario animation")
    print("  🎬 rrt_narrow_passage.gif - Narrow passage challenge")
    print("\n" + "=" * 60)
    print("🔑 Key Differences Highlighted:")
    print("  • RRT: Random sampling in continuous space")
    print("  • BFS: Systematic search in discrete graph")
    print("  • RRT: Tree-based exploration")
    print("  • BFS: Layer-by-layer expansion")
    print("  • RRT: Fast but non-optimal")
    print("  • BFS: Slower but guarantees shortest path")
    print("=" * 60)
