import numpy as np
import matplotlib.pyplot as plt

# Function to create a random grid map
def generate_grid_map(grid_size, obstacle_density=0.2):
    """
    Generates a random grid map with obstacles and free spaces.
    
    Parameters:
    grid_size (tuple): Size of the grid (rows, columns).
    obstacle_density (float): The probability of a cell being an obstacle (0 to 1).
    
    Returns:
    np.ndarray: Grid map with 0s (free) and 1s (obstacle).
    """
    # Randomly generate the grid with obstacle density
    grid_map = np.random.rand(grid_size[0], grid_size[1]) < obstacle_density
    
    # Convert boolean to 1 for obstacles and 0 for free space
    grid_map = grid_map.astype(int)
    return grid_map

# Function to plot the grid map
def plot_grid_map(grid_map):
    """
    Plots the grid map using Matplotlib.
    
    Parameters:
    grid_map (np.ndarray): The grid map to be plotted.
    """
    fig, ax = plt.subplots()
    ax.imshow(grid_map, cmap='gray', origin='upper')
    
    # Add gridlines for visualization
    ax.set_xticks(np.arange(0, grid_map.shape[1], 1))
    ax.set_yticks(np.arange(0, grid_map.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='both', color='black', linestyle='-', linewidth=2)
    
    plt.title("Grid Map with Obstacles")
    plt.show()

# Define grid size and obstacle density
grid_size = (10, 10)  # 10x10 grid
obstacle_density = 0.3  # 30% of the grid will be obstacles

# Generate the grid map
grid_map = generate_grid_map(grid_size, obstacle_density)

# Plot the grid map
plot_grid_map(grid_map)
