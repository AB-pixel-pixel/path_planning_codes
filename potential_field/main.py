import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ------------------------------------
# Potential Field Functions (改进版)
# ------------------------------------

def U_attractive(pos, goal, k_att=1.0):
    return 0.5 * k_att * np.linalg.norm(pos - goal)**2

def U_repulsive(pos, obstacle, influence_dist=1.5, k_rep=2.0):  # 增大影响距离
    d = np.linalg.norm(pos - obstacle)
    if d > influence_dist:
        return 0.0
    return 0.5 * k_rep * (1.0/d - 1.0/influence_dist)**2

def attractive_force(pos, goal, k_att=1):  # 增大吸引力系数
    return -k_att * (pos - goal)

def repulsive_force(pos, obstacle, influence_dist=1.5, k_rep=2.0):
    d = np.linalg.norm(pos - obstacle)
    if d > influence_dist:
        return np.array([0.0, 0.0])
    return (
        k_rep * (1.0/d - 1.0/influence_dist) * (1.0/(d**2)) *
        (pos - obstacle) / d
    )

# ------------------------------------
# Environment settings
# ------------------------------------

goal = np.array([4.0, 4.0])
start = np.array([0.5, 0.5])
obstacles = [np.array([2.0, 2.0]), np.array([3.2, 1.6])]

step_size = 0.08  # 增大步长
max_iters = 3000
convergence_threshold = 0.05  # 到达目标的阈值
stuck_threshold = 1e-4  # 判断是否卡住的阈值

# ------------------------------------
# Plan path (改进版 - 检测卡住情况)
# ------------------------------------

pos = start.copy()
path = [pos.copy()]
stuck_counter = 0

for i in range(max_iters):
    f_att = attractive_force(pos, goal)
    f_rep = np.sum([repulsive_force(pos, obs) for obs in obstacles], axis=0)
    f_total = f_att + f_rep
    
    # 添加随机扰动避免卡住
    if np.linalg.norm(f_total) < stuck_threshold:
        stuck_counter += 1
        if stuck_counter > 10:
            # 添加随机扰动
            f_total += np.random.randn(2) * 0.5
            stuck_counter = 0
    else:
        stuck_counter = 0
    
    # 归一化力并应用步长
    force_magnitude = np.linalg.norm(f_total)
    if force_magnitude > 0:
        pos = pos + step_size * f_total / force_magnitude
    
    path.append(pos.copy())
    
    # 检查是否到达目标
    if np.linalg.norm(pos - goal) < convergence_threshold:
        print(f"✓ 成功到达目标! 迭代次数: {i+1}")
        break
else:
    print(f"✗ 未到达目标。最终距离: {np.linalg.norm(pos - goal):.3f}")

path = np.array(path)

# ------------------------------------
# 诊断信息
# ------------------------------------
print(f"路径点数: {len(path)}")
print(f"起点: {start}")
print(f"终点: {path[-1]}")
print(f"目标: {goal}")
print(f"与目标距离: {np.linalg.norm(path[-1] - goal):.4f}")

# ------------------------------------
# Create 3D potential field grid
# ------------------------------------

x_vals = np.linspace(0, 5, 80)
y_vals = np.linspace(0, 5, 80)
X, Y = np.meshgrid(x_vals, y_vals)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos_ij = np.array([X[i, j], Y[i, j]])
        U = U_attractive(pos_ij, goal)
        for obs in obstacles:
            U += U_repulsive(pos_ij, obs)
        Z[i, j] = U

Z_clipped = np.clip(Z, 0, np.percentile(Z, 95))

# ------------------------------------
# 3D Plot
# ------------------------------------

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z_clipped, cmap=cm.viridis, 
                       alpha=0.7, linewidth=0, antialiased=True)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Potential Energy')

# Path with gradient color
path_z = np.array([U_attractive(p, goal) + sum(U_repulsive(p, obs) for obs in obstacles)
                   for p in path])
path_z_clipped = np.clip(path_z, 0, np.percentile(Z, 95))

colors = plt.cm.plasma(np.linspace(0, 1, len(path)))
for i in range(len(path)-1):
    ax.plot(path[i:i+2, 0], path[i:i+2, 1], path_z_clipped[i:i+2], 
            color=colors[i], linewidth=3, alpha=0.9)

# Mark start, goal, and final position
ax.scatter(start[0], start[1], 0, marker='o', s=300, c='green', 
           edgecolors='darkgreen', linewidths=2, label='Start', zorder=5)
ax.scatter(goal[0], goal[1], 0, marker='*', s=500, c='gold', 
           edgecolors='orange', linewidths=2, label='Goal', zorder=5)
ax.scatter(path[-1, 0], path[-1, 1], 0, marker='X', s=300, c='blue', 
           edgecolors='darkblue', linewidths=2, label='Final Position', zorder=5)

# Mark obstacles
for obs in obstacles:
    ax.scatter(obs[0], obs[1], 0, s=400, c='red', marker='o', 
               edgecolors='darkred', linewidths=2, alpha=0.8, zorder=5)
    z_max = Z_clipped.max() * 0.5
    ax.plot([obs[0], obs[0]], [obs[1], obs[1]], [0, z_max], 
            'r--', linewidth=2, alpha=0.5)

ax.view_init(elev=25, azim=45)
ax.set_title("3D Potential Field Path Planning", 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Potential U", fontsize=12)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

# ------------------------------------
# 2D Analysis Plot
# ------------------------------------

fig2, ax2 = plt.subplots(figsize=(10, 10))

contour = ax2.contourf(X, Y, Z_clipped, levels=30, cmap=cm.viridis, alpha=0.8)
ax2.contour(X, Y, Z_clipped, levels=15, colors='white', alpha=0.3, linewidths=0.5)
fig2.colorbar(contour, ax=ax2, label='Potential Energy')

# Plot path
ax2.plot(path[:, 0], path[:, 1], 'r-', linewidth=3, label='Planned Path', zorder=3)

# Show direction with arrows
arrow_skip = max(len(path) // 10, 1)
for i in range(0, len(path)-1, arrow_skip):
    dx = path[i+1, 0] - path[i, 0]
    dy = path[i+1, 1] - path[i, 1]
    ax2.arrow(path[i, 0], path[i, 1], dx, dy, 
              head_width=0.1, head_length=0.1, fc='yellow', ec='orange', zorder=4)

# Mark points
ax2.scatter(start[0], start[1], marker='o', s=200, c='green', 
            edgecolors='darkgreen', linewidths=2, label='Start', zorder=5)
ax2.scatter(goal[0], goal[1], marker='*', s=300, c='gold', 
            edgecolors='orange', linewidths=2, label='Goal', zorder=5)
ax2.scatter(path[-1, 0], path[-1, 1], marker='X', s=200, c='blue', 
            edgecolors='darkblue', linewidths=2, label='Final Position', zorder=5)

for obs in obstacles:
    circle = plt.Circle(obs, 0.2, color='red', alpha=0.8, zorder=4)
    ax2.add_patch(circle)
    influence = plt.Circle(obs, 1.5, color='red', fill=False, 
                          linestyle='--', alpha=0.3, linewidth=2, zorder=2)
    ax2.add_patch(influence)

ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_title('Path Analysis - Did it reach the goal?', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()
