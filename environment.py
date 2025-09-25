"""
Environment module for the autonomous delivery agent.
Handles the 2D grid world, obstacles, terrain costs, and dynamic elements.
"""

import json
import random
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class CellType(Enum):
    """Types of cells in the grid environment."""
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    AGENT = 4
    PACKAGE = 5


@dataclass
class Cell:
    """Represents a single cell in the grid."""
    x: int
    y: int
    cell_type: CellType
    terrain_cost: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.x == other.x and self.y == other.y
        return False


@dataclass
class MovingObstacle:
    """Represents a dynamic moving obstacle."""
    current_pos: Tuple[int, int]
    path: List[Tuple[int, int]]
    speed: int  # moves every N time steps
    current_step: int = 0


class GridEnvironment:
    """2D Grid environment for the delivery agent."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[Cell(x, y, CellType.EMPTY, 1) for x in range(width)] for y in range(height)]
        self.start_pos = None
        self.goal_pos = None
        self.packages = []  # List of package positions
        self.moving_obstacles = []
        self.time_step = 0
        
    def set_cell_type(self, x: int, y: int, cell_type: CellType, terrain_cost: int = 1):
        """Set the type and terrain cost of a cell."""
        if self.is_valid_position(x, y):
            self.grid[y][x].cell_type = cell_type
            self.grid[y][x].terrain_cost = terrain_cost
            
            if cell_type == CellType.START:
                self.start_pos = (x, y)
            elif cell_type == CellType.GOAL:
                self.goal_pos = (x, y)
            elif cell_type == CellType.PACKAGE:
                self.packages.append((x, y))
    
    def set_terrain_cost(self, x: int, y: int, cost: int):
        """Set the terrain cost for a specific cell."""
        if self.is_valid_position(x, y):
            self.grid[y][x].terrain_cost = max(1, cost)
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_obstacle(self, x: int, y: int, time_step: Optional[int] = None) -> bool:
        """Check if position is blocked by static or dynamic obstacle."""
        if not self.is_valid_position(x, y):
            return True
            
        # Check static obstacle
        if self.grid[y][x].cell_type == CellType.OBSTACLE:
            return True
            
        # Check dynamic obstacles
        if time_step is not None:
            for obstacle in self.moving_obstacles:
                obstacle_pos = self.get_obstacle_position_at_time(obstacle, time_step)
                if obstacle_pos == (x, y):
                    return True
                    
        return False
    
    def get_obstacle_position_at_time(self, obstacle: MovingObstacle, time_step: int) -> Tuple[int, int]:
        """Get position of moving obstacle at specific time step."""
        if not obstacle.path:
            return obstacle.current_pos
            
        # Calculate position based on speed and path
        moves_made = time_step // obstacle.speed
        path_index = moves_made % len(obstacle.path)
        return obstacle.path[path_index]
    
    def get_terrain_cost(self, x: int, y: int) -> int:
        """Get terrain cost for a position."""
        if self.is_valid_position(x, y):
            return self.grid[y][x].terrain_cost
        return 999999
    
    def get_neighbors(self, x: int, y: int, time_step: Optional[int] = None) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny) and not self.is_obstacle(nx, ny, time_step):
                neighbors.append((nx, ny))
                
        return neighbors
    
    def add_moving_obstacle(self, start_pos: Tuple[int, int], path: List[Tuple[int, int]], speed: int = 1):
        """Add a moving obstacle to the environment."""
        obstacle = MovingObstacle(start_pos, path, speed)
        self.moving_obstacles.append(obstacle)
    
    def update_time_step(self):
        """Advance time step and update moving obstacles."""
        self.time_step += 1
        
        for obstacle in self.moving_obstacles:
            obstacle.current_step += 1
            if obstacle.current_step >= obstacle.speed and obstacle.path:
                # Move to next position in path
                path_index = (obstacle.current_step // obstacle.speed) % len(obstacle.path)
                obstacle.current_pos = obstacle.path[path_index]
                obstacle.current_step = 0
    
    def load_from_file(self, filename: str):
        """Load environment from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            self.width = data['width']
            self.height = data['height']
            self.grid = [[Cell(x, y, CellType.EMPTY, 1) for x in range(self.width)] for y in range(self.height)]
            
            # Set obstacles and terrain
            for obstacle in data.get('obstacles', []):
                self.set_cell_type(obstacle['x'], obstacle['y'], CellType.OBSTACLE)
                
            for terrain in data.get('terrain_costs', []):
                self.set_terrain_cost(terrain['x'], terrain['y'], terrain['cost'])
                
            # Set start and goal
            if 'start' in data:
                self.set_cell_type(data['start']['x'], data['start']['y'], CellType.START)
                
            if 'goal' in data:
                self.set_cell_type(data['goal']['x'], data['goal']['y'], CellType.GOAL)
                
            # Set packages
            for package in data.get('packages', []):
                self.set_cell_type(package['x'], package['y'], CellType.PACKAGE)
                
            # Add moving obstacles
            for mov_obs in data.get('moving_obstacles', []):
                self.add_moving_obstacle(
                    (mov_obs['start_x'], mov_obs['start_y']),
                    [(p['x'], p['y']) for p in mov_obs['path']],
                    mov_obs.get('speed', 1)
                )
                
        except FileNotFoundError:
            print(f"Map file {filename} not found!")
        except Exception as e:
            print(f"Error loading map: {e}")
    
    def save_to_file(self, filename: str):
        """Save environment to JSON file."""
        data = {
            'width': self.width,
            'height': self.height,
            'obstacles': [],
            'terrain_costs': [],
            'packages': [],
            'moving_obstacles': []
        }
        
        # Collect obstacles and terrain costs
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.OBSTACLE:
                    data['obstacles'].append({'x': x, 'y': y})
                elif cell.cell_type == CellType.START:
                    data['start'] = {'x': x, 'y': y}
                elif cell.cell_type == CellType.GOAL:
                    data['goal'] = {'x': x, 'y': y}
                elif cell.cell_type == CellType.PACKAGE:
                    data['packages'].append({'x': x, 'y': y})
                    
                if cell.terrain_cost > 1:
                    data['terrain_costs'].append({'x': x, 'y': y, 'cost': cell.terrain_cost})
        
        # Add moving obstacles
        for obstacle in self.moving_obstacles:
            data['moving_obstacles'].append({
                'start_x': obstacle.current_pos[0],
                'start_y': obstacle.current_pos[1],
                'path': [{'x': p[0], 'y': p[1]} for p in obstacle.path],
                'speed': obstacle.speed
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_grid(self, agent_pos: Optional[Tuple[int, int]] = None):
        """Print visual representation of the grid."""
        symbols = {
            CellType.EMPTY: '.',
            CellType.OBSTACLE: '#',
            CellType.START: 'S',
            CellType.GOAL: 'G',
            CellType.PACKAGE: 'P',
            CellType.AGENT: 'A'
        }
        
        print(f"Grid ({self.width}x{self.height}) - Time step: {self.time_step}")
        print("  " + "".join([str(i % 10) for i in range(self.width)]))
        
        for y in range(self.height):
            row = f"{y:2d}"
            for x in range(self.width):
                if agent_pos and (x, y) == agent_pos:
                    row += 'A'
                elif self.is_obstacle(x, y, self.time_step):
                    # Check if it's a moving obstacle
                    is_moving = any(
                        self.get_obstacle_position_at_time(obs, self.time_step) == (x, y)
                        for obs in self.moving_obstacles
                    )
                    row += 'M' if is_moving else '#'
                else:
                    row += symbols.get(self.grid[y][x].cell_type, '.')
            print(row)
        print()