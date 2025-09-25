"""
Visualization module for the autonomous delivery agent.
Provides matplotlib-based visualization of paths, environments, and performance.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Dict, Optional

from environment import GridEnvironment, CellType
from algorithms import SearchResult


def visualize_environment(env: GridEnvironment, title: str = "Grid Environment") -> None:
    """Visualize the grid environment with obstacles and terrain costs."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color map
    grid_colors = np.ones((env.height, env.width, 3))  # Start with white
    
    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid[y][x]
            
            if cell.cell_type == CellType.OBSTACLE:
                grid_colors[y, x] = [0.2, 0.2, 0.2]  # Dark gray
            elif cell.cell_type == CellType.START:
                grid_colors[y, x] = [0.0, 1.0, 0.0]  # Green
            elif cell.cell_type == CellType.GOAL:
                grid_colors[y, x] = [1.0, 0.0, 0.0]  # Red
            elif cell.cell_type == CellType.PACKAGE:
                grid_colors[y, x] = [1.0, 1.0, 0.0]  # Yellow
            else:
                # Color based on terrain cost
                cost_factor = min(cell.terrain_cost / 10.0, 1.0)
                grid_colors[y, x] = [1.0, 1.0 - cost_factor * 0.5, 1.0 - cost_factor * 0.5]
    
    # Display grid
    ax.imshow(grid_colors, aspect='equal')
    
    # Add grid lines
    for x in range(env.width + 1):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5)
    for y in range(env.height + 1):
        ax.axhline(y - 0.5, color='gray', linewidth=0.5)
    
    # Add moving obstacles if any
    for obstacle in env.moving_obstacles:
        pos = obstacle.current_pos
        circle = patches.Circle((pos[0], pos[1]), 0.3, color='purple', alpha=0.7)
        ax.add_patch(circle)
    
    # Labels and formatting
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Legend
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor='green', label='Start'),
        patches.Rectangle((0, 0), 1, 1, facecolor='red', label='Goal'),
        patches.Rectangle((0, 0), 1, 1, facecolor='yellow', label='Package'),
        patches.Rectangle((0, 0), 1, 1, facecolor='gray', label='Obstacle'),
        patches.Circle((0, 0), 1, facecolor='purple', alpha=0.7, label='Moving Obstacle')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()


def visualize_path(env: GridEnvironment, path: List[Tuple[int, int]], 
                  title: str = "Path Visualization", agent_path: bool = False,
                  show_obstacles_at_time: int = None) -> None:
    """Visualize a path through the environment."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create base environment visualization
    grid_colors = np.ones((env.height, env.width, 3))
    
    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid[y][x]
            
            if cell.cell_type == CellType.OBSTACLE:
                grid_colors[y, x] = [0.3, 0.3, 0.3]
            elif cell.cell_type == CellType.START:
                grid_colors[y, x] = [0.0, 1.0, 0.0]
            elif cell.cell_type == CellType.GOAL:
                grid_colors[y, x] = [1.0, 0.0, 0.0]
            elif cell.cell_type == CellType.PACKAGE:
                grid_colors[y, x] = [1.0, 1.0, 0.0]
            else:
                # Terrain cost coloring
                cost_factor = min(cell.terrain_cost / 8.0, 1.0)
                grid_colors[y, x] = [1.0, 1.0 - cost_factor * 0.4, 1.0 - cost_factor * 0.4]
    
    ax.imshow(grid_colors, aspect='equal')
    
    # Draw path
    if path and len(path) > 1:
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        
        # Draw path line
        ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7, label='Path')
        ax.plot(path_x, path_y, 'bo', markersize=4, alpha=0.8)
        
        # Mark start and end of path
        ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')
        
        # Add step numbers if path is short enough
        if len(path) <= 20:
            for i, (x, y) in enumerate(path):
                ax.annotate(str(i), (x, y), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8, color='blue')
    
    # Show moving obstacles at specific time
    if show_obstacles_at_time is not None:
        for obstacle in env.moving_obstacles:
            obs_pos = env.get_obstacle_position_at_time(obstacle, show_obstacles_at_time)
            circle = patches.Circle((obs_pos[0], obs_pos[1]), 0.3, 
                                  color='purple', alpha=0.8)
            ax.add_patch(circle)
    
    # Grid lines
    for x in range(env.width + 1):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    for y in range(env.height + 1):
        ax.axhline(y - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def create_performance_comparison(results: Dict[str, SearchResult], 
                                title: str = "Algorithm Performance Comparison") -> None:
    """Create performance comparison charts for different algorithms."""
    # Prepare data
    algorithms = list(results.keys())
    found_solutions = [results[alg].found_solution for alg in algorithms]
    costs = [results[alg].cost if results[alg].found_solution else 0 for alg in algorithms]
    nodes_expanded = [results[alg].nodes_expanded for alg in algorithms]
    times = [results[alg].time_taken for alg in algorithms]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Success rate
    success_colors = ['green' if found else 'red' for found in found_solutions]
    ax1.bar(algorithms, [1 if found else 0 for found in found_solutions], 
           color=success_colors, alpha=0.7)
    ax1.set_title('Solution Found')
    ax1.set_ylabel('Success (1) / Failure (0)')
    ax1.set_ylim(0, 1.2)
    for i, found in enumerate(found_solutions):
        ax1.text(i, 0.5, 'Yes' if found else 'No', ha='center', va='center', fontweight='bold')
    
    # Path costs (only for successful solutions)
    successful_algs = [alg for alg in algorithms if results[alg].found_solution]
    successful_costs = [results[alg].cost for alg in successful_algs]
    
    if successful_costs:
        ax2.bar(successful_algs, successful_costs, alpha=0.7, color='skyblue')
        ax2.set_title('Path Cost (Successful Solutions Only)')
        ax2.set_ylabel('Total Cost')
        # Add value labels on bars
        for i, cost in enumerate(successful_costs):
            ax2.text(i, cost + max(successful_costs) * 0.01, f'{cost:.1f}', 
                    ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No successful solutions', ha='center', va='center', 
                transform=ax2.transAxes)
        ax2.set_title('Path Cost')
    
    # Nodes expanded
    ax3.bar(algorithms, nodes_expanded, alpha=0.7, color='lightcoral')
    ax3.set_title('Nodes Expanded')
    ax3.set_ylabel('Number of Nodes')
    # Add value labels
    for i, nodes in enumerate(nodes_expanded):
        ax3.text(i, nodes + max(nodes_expanded) * 0.01, str(nodes), 
                ha='center', va='bottom')
    
    # Execution time
    ax4.bar(algorithms, times, alpha=0.7, color='lightgreen')
    ax4.set_title('Execution Time')
    ax4.set_ylabel('Time (seconds)')
    # Add value labels
    for i, time_val in enumerate(times):
        ax4.text(i, time_val + max(times) * 0.01, f'{time_val:.4f}', 
                ha='center', va='bottom')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def create_delivery_summary_chart(delivery_results: List[Dict], 
                                title: str = "Delivery Task Summary") -> None:
    """Create summary chart for delivery task results."""
    if not delivery_results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Extract data
    task_names = [f"Task {i+1}" for i in range(len(delivery_results))]
    successful = [result['successful_deliveries'] for result in delivery_results]
    failed = [result['failed_deliveries'] for result in delivery_results]
    costs = [result['total_cost'] for result in delivery_results]
    fuel_used = [result['fuel_consumed'] for result in delivery_results]
    
    # Success vs failure stacked bar
    ax1.bar(task_names, successful, label='Successful', color='green', alpha=0.7)
    ax1.bar(task_names, failed, bottom=successful, label='Failed', color='red', alpha=0.7)
    ax1.set_title('Delivery Success Rate')
    ax1.set_ylabel('Number of Deliveries')
    ax1.legend()
    
    # Total costs
    ax2.bar(task_names, costs, alpha=0.7, color='skyblue')
    ax2.set_title('Total Path Cost')
    ax2.set_ylabel('Cost')
    
    # Fuel consumption
    ax3.bar(task_names, fuel_used, alpha=0.7, color='orange')
    ax3.set_title('Fuel Consumption')
    ax3.set_ylabel('Fuel Units')
    
    # Efficiency (successful deliveries per fuel unit)
    efficiency = [s/max(f, 1) for s, f in zip(successful, fuel_used)]
    ax4.bar(task_names, efficiency, alpha=0.7, color='purple')
    ax4.set_title('Delivery Efficiency')
    ax4.set_ylabel('Deliveries per Fuel Unit')
    
    plt.tight_layout()
    plt.show()


def animate_path_execution(env: GridEnvironment, path: List[Tuple[int, int]], 
                          title: str = "Path Animation", delay: float = 0.5) -> None:
    """Create an animated visualization of path execution."""
    import time as time_module
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for step, pos in enumerate(path):
        ax.clear()
        
        # Draw environment
        grid_colors = np.ones((env.height, env.width, 3))
        
        for y in range(env.height):
            for x in range(env.width):
                cell = env.grid[y][x]
                
                if cell.cell_type == CellType.OBSTACLE:
                    grid_colors[y, x] = [0.3, 0.3, 0.3]
                elif cell.cell_type == CellType.START:
                    grid_colors[y, x] = [0.0, 1.0, 0.0]
                elif cell.cell_type == CellType.GOAL:
                    grid_colors[y, x] = [1.0, 0.0, 0.0]
                elif cell.cell_type == CellType.PACKAGE:
                    grid_colors[y, x] = [1.0, 1.0, 0.0]
        
        ax.imshow(grid_colors, aspect='equal')
        
        # Draw path up to current position
        if step > 0:
            path_x = [p[0] for p in path[:step+1]]
            path_y = [p[1] for p in path[:step+1]]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.5)
            ax.plot(path_x[:-1], path_y[:-1], 'bo', markersize=4, alpha=0.5)
        
        # Draw current agent position
        ax.plot(pos[0], pos[1], 'ro', markersize=10, label=f'Agent (Step {step})')
        
        # Grid lines
        for x in range(env.width + 1):
            ax.axvline(x - 0.5, color='gray', linewidth=0.5)
        for y in range(env.height + 1):
            ax.axhline(y - 0.5, color='gray', linewidth=0.5)
        
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(-0.5, env.height - 0.5)
        ax.set_title(f"{title} - Step {step}/{len(path)-1}")
        ax.legend()
        
        plt.pause(delay)
    
    plt.show()