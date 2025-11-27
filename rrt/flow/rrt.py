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
    """RRT算法可视化器（增强版：显示算法流程步骤）"""
    
    def __init__(self, width: int, height: int, step_size: float = 0.5):
        self.width = width
        self.height = height
        self.step_size = step_size
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        self.goal_radius = 0.5
        
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
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        grid_x, grid_y = int(x), int(y)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x = grid_x + dx
                check_y = grid_y + dy
                if (0 <= check_x < self.width and 
                    0 <= check_y < self.height and 
                    self.grid[check_y, check_x] == 1):
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
        """随机采样一个点"""
        if np.random.random() < 0.9:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
        else:
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
            return RRTNode(to_point[0], to_point[1], from_node)
        else:
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
        return path[::-1]
    
    def rrt_step_by_step(self, max_iterations: int = 500):
        """RRT逐步搜索，返回每一步的状态"""
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

    def create_enhanced_animation(self, filename='rrt_enhanced.gif', 
                                fps=3, max_iterations=500):
        """创建增强版RRT动画，显示算法流程步骤"""
        steps = self.rrt_step_by_step(max_iterations=max_iterations)
        
        # 选择关键帧
        key_frames = []
        key_frames.append(steps[0])  # 初始
        
        for i, step in enumerate(steps[1:], 1):
            if step['step_type'] in ['sample', 'extend', 'found', 'max_iter']:
                key_frames.append(step)
            elif i % 5 == 0:
                key_frames.append(step)
        
        if steps[-1] not in key_frames:
            key_frames.append(steps[-1])
        
        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
        
        ax_main = plt.subplot(gs[0, :])  # 主图占据第一行
        ax_flowchart = plt.subplot(gs[1, 0])  # 流程图在第二行左侧
        ax_info = plt.subplot(gs[1, 1])  # 信息在第二行右侧
        
        # 定义算法流程图步骤
        flowchart_steps = [
            "1. Initialize: G = {q_start}, k = 0",
            "2. Sample: q_rand ← Random(C)",
            "3. Find Nearest: q_near ← Nearest(G, q_rand)", 
            "4. Extend: q_new ← Steer(q_near, q_rand)",
            "5. Collision Check: Edge(q_near, q_new) free?",
            "6. Add to Tree: G ← G ∪ {q_new}",
            "7. Check Goal: q_new = q_goal?",
            "8. k = k + 1"
        ]
        
        def animate(frame_num):
            ax_main.clear()
            ax_flowchart.clear()
            ax_info.clear()
            
            step_info = key_frames[frame_num]
            tree = step_info['tree']
            sampled_point = step_info['sampled_point']
            nearest_node = step_info['nearest_node']
            new_node = step_info['new_node']
            path = step_info['path']
            step_type = step_info['step_type']
            iteration = step_info['iteration']
            
            # ===== 主图区域 =====
            # 绘制障碍物
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == 1:
                        rect = patches.Rectangle((j, i), 1, 1,
                                                linewidth=1, edgecolor='black',
                                                facecolor='#2C3E50')
                        ax_main.add_patch(rect)
            
            # 绘制RRT树的所有边
            for node in tree:
                if node.parent is not None:
                    ax_main.plot([node.parent.x, node.x], 
                           [node.parent.y, node.y],
                           'b-', linewidth=1.5, alpha=0.4, zorder=3)
            
            # 绘制RRT树的所有节点
            for node in tree:
                ax_main.plot(node.x, node.y, 'o', color='#3498DB', 
                       markersize=4, zorder=4, alpha=0.6)
            
            # 根据当前步骤类型高亮相应元素
            current_step = None
            
            if step_type == 'sample':
                current_step = 1
                if sampled_point:
                    ax_main.plot(sampled_point[0], sampled_point[1], '*', 
                       color='#F39C12', markersize=20, zorder=7)
            
            elif step_type == 'nearest':
                current_step = 2
                if sampled_point and nearest_node:
                    ax_main.plot([nearest_node.x, sampled_point[0]], 
                           [nearest_node.y, sampled_point[1]],
                           'g--', linewidth=2, alpha=0.5, zorder=6)
            
            elif step_type == 'extend':
                current_step = 3
                if new_node and nearest_node:
                    # 高亮新添加的边和节点
                    ax_main.plot([nearest_node.x, new_node.x], 
                           [nearest_node.y, new_node.y],
                           'g-', linewidth=3, alpha=0.8, zorder=8)
                    ax_main.plot(new_node.x, new_node.y, 'o', 
                       color='#2ECC71', markersize=12, zorder=9)
            
            elif step_type == 'collision':
                current_step = 4
                if new_node:
                    ax_main.plot(new_node.x, new_node.y, 'x', 
                       color='#E74C3C', markersize=12, zorder=9)
            
            elif step_type == 'found':
                current_step = 5
                if path:
                    path_x = [x for x, y in path]
                    path_y = [y for x, y in path]
                    ax_main.plot(path_x, path_y, 'r-', linewidth=5, alpha=0.8, zorder=11)
            
            # 绘制起点和终点
            sx, sy = self.start
            ax_main.plot(sx, sy, 'o', color='#2ECC71', 
                       markersize=22, zorder=10)
            ax_main.text(sx, sy, 'S', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
            
            gx, gy = self.goal
            goal_circle = plt.Circle((gx, gy), self.goal_radius, 
                                    color='#E74C3C', alpha=0.3, zorder=2)
            ax_main.add_patch(goal_circle)
            ax_main.plot(gx, gy, 's', color='#E74C3C', 
                       markersize=22, zorder=10)
            ax_main.text(gx, gy, 'G', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
            
            ax_main.set_xlim(-0.5, self.width + 0.5)
            ax_main.set_ylim(-0.5, self.height + 0.5)
            ax_main.set_aspect('equal')
            ax_main.invert_yaxis()
            ax_main.grid(True, alpha=0.2)
            ax_main.set_title(f'RRT Algorithm - Iteration {iteration}', 
                          fontsize=14, fontweight='bold')
            
            # ===== 流程图区域 =====
            ax_flowchart.set_xlim(0, 10)
            ax_flowchart.set_ylim(0, len(flowchart_steps) + 1)
        
            # 绘制算法流程图
            for i, step_text in enumerate(flowchart_steps):
                y_pos = len(flowchart_steps) - i
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightblue", 
                         alpha=0.7, edgecolor='blue')
            
            if current_step == i + 1:
                # 当前步骤用红色高亮
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor="#FF6B6B", 
                         alpha=0.9, edgecolor='red', linewidth=2)
            
            ax_flowchart.text(5, y_pos, step_text, 
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           bbox=bbox_props)
        
            ax_flowchart.set_title('RRT Algorithm Flow', fontsize=12, fontweight='bold')
            ax_flowchart.axis('off')
            
            # ===== 信息区域 =====
            ax_info.axis('off')
            
            # 当前步骤详细信息
            current_action = ""
            if step_type == 'init':
                current_action = "Initialize RRT tree with start node"
            elif step_type == 'sample':
                current_action = "Random sampling in configuration space"
            elif step_type == 'nearest':
                current_action = "Find nearest node in existing tree"
            elif step_type == 'extend':
                current_action = "Extend tree towards sampled point"
            elif step_type == 'collision':
                current_action = "Collision detected - discard this sample"
            elif step_type == 'found':
                current_action = f"Goal reached! Path length: {len(path)-1}"
            
            info_text = f"📊 Current Status:\n"
            info_text += f"{'─'*25}\n"
            info_text += f"Iteration: {iteration}\n"
            info_text += f"Tree Nodes: {len(tree)}\n"
            info_text += f"{'─'*25}\n"
            info_text += f"🔴 Current Step:\n"
            info_text += f"{current_action}\n"
            info_text += f"{'─'*25}\n"
            info_text += f"Algorithm: RRT (Sample-Based)\n"
            info_text += f"Step Size: {self.step_size}\n"
            info_text += f"Goal Radius: {self.goal_radius}\n"
            
            if sampled_point:
                info_text += f"📍 Sampled: ({sampled_point[0]:.2f}, {sampled_point[1]:.2f})"
            
            ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                       fontsize=10, verticalalignment='top', family='monospace')
            
            # 添加进度条
            progress = iteration / max_iterations
            progress_bar = patches.Rectangle((0.1, 0.1), 0.8 * progress, 0.1,
                                       facecolor='#2ECC71', alpha=0.8)
            ax_info.add_patch(progress_bar)
            ax_info.text(0.5, 0.25, f"Progress: {progress*100:.1f}%", 
                       transform=ax_info.transAxes, fontsize=11, 
                       ha='center', va='center', fontweight='bold')
            
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=len(key_frames),
                           interval=1000/fps, repeat=True)
        
        # 保存为GIF
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=120)
        plt.close()
        
        print(f"✓ Enhanced animation saved as: {filename}")
        print(f"  Total iterations: {steps[-1]['iteration']}")
        print(f"  Key frames: {len(key_frames)}")
        return anim

    def create_step_by_step_visualization(self, max_iterations=100):
        """创建分步骤的详细可视化"""
        steps = self.rrt_step_by_step(max_iterations=max_iterations)
        
        # 选择重要的步骤来显示
        important_steps = []
        for step in steps:
            if step['step_type'] in ['init', 'sample', 'extend', 'found']:
                important_steps.append(step)
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 选择6个关键步骤
        key_indices = [0, len(important_steps)//5, len(important_steps)//2, 
                           3*len(important_steps)//4, len(important_steps)-1]
        key_steps = [important_steps[i] for i in key_indices if i < len(important_steps)]
        
        for idx, step_info in enumerate(key_steps):
            ax = axes[idx//3, idx%3]
            self._draw_enhanced_step(ax, step_info)
        
        plt.tight_layout()
        plt.savefig('rrt_step_by_step_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _draw_enhanced_step(self, ax, step_info):
        """绘制增强版的单一步骤"""
        tree = step_info['tree']
        sampled_point = step_info['sampled_point']
        nearest_node = step_info['nearest_node']
        new_node = step_info['new_node']
        path = step_info['path']
        step_type = step_info['step_type']
        iteration = step_info['iteration']
        
        # 设置标题
        titles = {
            'init': "Step 1: Initialize RRT Tree",
            'sample': "Step 2: Random Sampling",
            'nearest': "Step 3: Find Nearest Node",
            'extend': "Step 4: Extend Tree",
            'collision': "Step 5: Collision Detected",
            'found': "Step 6: Path Found!",
            'max_iter': "Maximum Iterations Reached"
        }
        
        ax.set_title(titles.get(step_type, f"Iteration {iteration}"), 
                     fontsize=12, fontweight='bold')
        
        # 绘制障碍物
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((j, i), 1, 1,
                                            linewidth=1, edgecolor='black',
                                            facecolor='#2C3E50')
                    ax.add_patch(rect)
        
        # 绘制RRT树
        for node in tree:
            if node.parent is not None:
                ax.plot([node.parent.x, node.x], 
                       [node.parent.y, node.y],
                       'b-', linewidth=1.5, alpha=0.4, zorder=3)
        
        for node in tree:
            ax.plot(node.x, node.y, 'o', color='#3498DB', 
                   markersize=4, zorder=4, alpha=0.6)
        
        # 根据步骤类型高亮相应元素
        if step_type == 'sample' and sampled_point:
            ax.plot(sampled_point[0], sampled_point[1], '*', 
                   color='#F39C12', markersize=15, zorder=7)
        
        if nearest_node and step_type in ['nearest', 'extend', 'collision']:
            ax.plot(nearest_node.x, nearest_node.y, 'o', 
                   color='#9B59B6', markersize=12, zorder=8)
        
        if new_node and step_type == 'extend':
            ax.plot(new_node.x, new_node.y, 'o', 
                   color='#2ECC71', markersize=10, zorder=9)
        
        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            ax.plot(path_x, path_y, 'r-', linewidth=4, alpha=0.8, zorder=11)
        
        # 起点和终点
        sx, sy = self.start
        ax.plot(sx, sy, 'o', color='#2ECC71', markersize=18, zorder=10)
        
        gx, gy = self.goal
        goal_circle = plt.Circle((gx, gy), self.goal_radius, 
                                color='#E74C3C', alpha=0.3, zorder=2)
        ax.add_patch(goal_circle)
        ax.plot(gx, gy, 's', color='#E74C3C', markersize=18, zorder=10)
        
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

# ==================== 使用示例 ====================

def demo_enhanced_rrt():
    """增强版RRT演示"""
    print("=" * 60)
    print("Enhanced RRT Visualization with Algorithm Flow")
    print("=" * 60)
    
    viz = RRTVisualizer(width=12, height=10, step_size=0.5)
    
    # 设置场景
    viz.add_obstacles_rect(3, 2, 2, 4)
    viz.add_obstacles_rect(7, 3, 2, 5)
    viz.set_start(1.0, 2.0)
    viz.set_goal(10.0, 7.0)
    
    # 创建增强版动画
    print("Generating enhanced RRT animation with algorithm flow...")
    viz.create_enhanced_animation('rrt_enhanced_with_flow.gif', fps=4, max_iterations=300)
    
    # 创建分步骤可视化
    print("\nGenerating step-by-step visualization...")
    viz.create_step_by_step_visualization(max_iterations=150)
    
    print("✓ Enhanced visualization completed!")

def demo_complex_enhanced():
    """复杂场景的增强版演示"""
    print("\n" + "=" * 60)
    print("Complex Scenario Enhanced Visualization")
    print("=" * 60)
    
    viz = RRTVisualizer(width=15, height=12, step_size=0.6)
    
    # 复杂障碍物
    viz.add_obstacles_rect(3, 2, 3, 2)
    viz.add_obstacles_rect(8, 3, 2, 5)
    viz.add_obstacles_circle(6.5, 4.5, 1.2)
    viz.add_obstacles_rect(11, 1, 2, 4)
    viz.set_start(1.0, 1.0)
    viz.set_goal(13.5, 10.5)
    
    print("Generating complex enhanced animation...")
    viz.create_enhanced_animation('rrt_complex_enhanced.gif', fps=3, max_iterations=400)
    
    print("✓ Complex enhanced visualization completed!")

if __name__ == "__main__":
    print("🎯 Enhanced RRT Visualization with Algorithm Flow")
    print("=" * 60)
    print("Features:")
    print("  • Real-time algorithm step highlighting")
    print("  • Visual flowchart showing current step")
    print("  • Detailed step-by-step explanation")
    print("=" * 60)
    
    # 运行增强版演示
    demo_enhanced_rrt()
    demo_complex_enhanced()
    
    print("\n" + "=" * 60)
    print("✓ All enhanced visualizations completed!")
    print("Generated files:")
    print("  🎬 rrt_enhanced_with_flow.gif - Main enhanced animation")
    print("  🎬 rrt_complex_enhanced.gif - Complex scenario")
    print("  📊 rrt_step_by_step_enhanced.png - Step-by-step diagrams")
    print("=" * 60)
