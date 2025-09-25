"""
Test maps for the autonomous delivery agent.
Creates different map scenarios: small, medium, large, and dynamic obstacles.
"""

import json
import os
from typing import List, Tuple
from environment import GridEnvironment, CellType


def create_small_map() -> GridEnvironment:
    """Create a small 8x8 test map."""
    env = GridEnvironment(8, 8)
    
    # Set start and goal
    env.set_cell_type(1, 1, CellType.START)
    env.set_cell_type(6, 6, CellType.GOAL)
    
    # Add some obstacles
    obstacles = [(3, 2), (3, 3), (3, 4), (5, 1), (5, 2), (2, 5), (4, 5)]
    for x, y in obstacles:
        env.set_cell_type(x, y, CellType.OBSTACLE)
    
    # Add package
    env.set_cell_type(2, 2, CellType.PACKAGE)
    
    # Add varied terrain costs
    high_cost_terrain = [(1, 4), (2, 4), (4, 1), (6, 3), (6, 4)]
    for x, y in high_cost_terrain:
        env.set_terrain_cost(x, y, 3)
    
    return env


def create_medium_map() -> GridEnvironment:
    """Create a medium 15x15 test map."""
    env = GridEnvironment(15, 15)
    
    # Set start and goal
    env.set_cell_type(1, 1, CellType.START)
    env.set_cell_type(13, 13, CellType.GOAL)
    
    # Create maze-like obstacles
    obstacles = [
        # Vertical walls
        (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
        (9, 7), (9, 8), (9, 9), (9, 10), (9, 11),
        # Horizontal walls
        (2, 8), (3, 8), (4, 8), (6, 8), (7, 8),
        (10, 4), (11, 4), (12, 4),
        # Scattered obstacles
        (3, 11), (4, 11), (11, 9), (12, 9), (7, 2), (8, 2)
    ]
    for x, y in obstacles:
        env.set_cell_type(x, y, CellType.OBSTACLE)
    
    # Add multiple packages
    packages = [(3, 3), (7, 7), (11, 2), (2, 12)]
    for x, y in packages:
        env.set_cell_type(x, y, CellType.PACKAGE)
    
    # Add varied terrain costs (muddy areas)
    muddy_areas = [
        (1, 5), (2, 5), (3, 5),
        (6, 10), (7, 10), (8, 10), (9, 10),
        (12, 6), (13, 6), (12, 7), (13, 7)
    ]
    for x, y in muddy_areas:
        env.set_terrain_cost(x, y, 4)
    
    # Rocky areas (even higher cost)
    rocky_areas = [(6, 3), (7, 3), (10, 11), (11, 11)]
    for x, y in rocky_areas:
        env.set_terrain_cost(x, y, 6)
    
    return env


def create_large_map() -> GridEnvironment:
    """Create a large 25x25 test map."""
    env = GridEnvironment(25, 25)
    
    # Set start and goal
    env.set_cell_type(2, 2, CellType.START)
    env.set_cell_type(22, 22, CellType.GOAL)
    
    # Create complex obstacle patterns
    obstacles = []
    
    # Large building blocks
    building1 = [(x, y) for x in range(5, 9) for y in range(5, 9)]
    building2 = [(x, y) for x in range(15, 19) for y in range(8, 12)]
    building3 = [(x, y) for x in range(8, 12) for y in range(15, 19)]
    
    obstacles.extend(building1)
    obstacles.extend(building2)
    obstacles.extend(building3)
    
    # Corridor walls
    corridor_walls = [
        # Vertical corridors
        (12, y) for y in range(3, 8)
    ] + [
        (12, y) for y in range(20, 24)
    ] + [
        # Horizontal corridors
        (x, 12) for x in range(3, 8)
    ] + [
        (x, 12) for x in range(20, 24)
    ]
    
    obstacles.extend(corridor_walls)
    
    # Remove duplicates and set obstacles
    obstacles = list(set(obstacles))
    for x, y in obstacles:
        if env.is_valid_position(x, y):
            env.set_cell_type(x, y, CellType.OBSTACLE)
    
    # Add multiple packages scattered around
    packages = [(3, 10), (10, 3), (20, 5), (5, 20), (15, 15), (23, 10)]
    for x, y in packages:
        env.set_cell_type(x, y, CellType.PACKAGE)
    
    # Create varied terrain zones
    # River area (high cost)
    river_path = [(x, 6) for x in range(0, 25) if x not in [5, 6, 7, 8]]  # Bridge
    for x, y in river_path:
        env.set_terrain_cost(x, y, 5)
    
    # Forest areas (medium cost)
    forest_areas = [
        (x, y) for x in range(19, 25) for y in range(15, 21)
        if (x, y) not in obstacles
    ]
    for x, y in forest_areas:
        env.set_terrain_cost(x, y, 3)
    
    # Mountain areas (very high cost)
    mountain_areas = [(x, y) for x in range(0, 5) for y in range(15, 20)]
    for x, y in mountain_areas:
        env.set_terrain_cost(x, y, 8)
    
    return env


def create_dynamic_obstacles_map() -> GridEnvironment:
    """Create a map with dynamic moving obstacles."""
    env = GridEnvironment(12, 12)
    
    # Set start and goal
    env.set_cell_type(1, 1, CellType.START)
    env.set_cell_type(10, 10, CellType.GOAL)
    
    # Static obstacles
    static_obstacles = [(4, 2), (4, 3), (4, 4), (7, 6), (7, 7), (7, 8)]
    for x, y in static_obstacles:
        env.set_cell_type(x, y, CellType.OBSTACLE)
    
    # Add packages
    packages = [(2, 5), (8, 3)]
    for x, y in packages:
        env.set_cell_type(x, y, CellType.PACKAGE)
    
    # Add moving obstacles (vehicles)
    # Vehicle 1: horizontal movement
    vehicle1_path = [(2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (5, 6), (4, 6), (3, 6)]
    env.add_moving_obstacle((2, 6), vehicle1_path, speed=2)
    
    # Vehicle 2: vertical movement
    vehicle2_path = [(9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 5), (9, 4), (9, 3)]
    env.add_moving_obstacle((9, 2), vehicle2_path, speed=3)
    
    # Vehicle 3: diagonal-like movement
    vehicle3_path = [(5, 9), (6, 8), (7, 9), (8, 8), (7, 9), (6, 8)]
    env.add_moving_obstacle((5, 9), vehicle3_path, speed=1)
    
    # Add varied terrain
    muddy_areas = [(1, 8), (2, 8), (3, 8), (8, 1), (9, 1), (10, 1)]
    for x, y in muddy_areas:
        env.set_terrain_cost(x, y, 4)
    
    return env


def save_test_maps():
    """Save all test maps to JSON files."""
    maps_dir = "test_maps"
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    
    # Create and save small map
    small_map = create_small_map()
    small_map.save_to_file(os.path.join(maps_dir, "small_map.json"))
    
    # Create and save medium map
    medium_map = create_medium_map()
    medium_map.save_to_file(os.path.join(maps_dir, "medium_map.json"))
    
    # Create and save large map
    large_map = create_large_map()
    large_map.save_to_file(os.path.join(maps_dir, "large_map.json"))
    
    # Create and save dynamic obstacles map
    dynamic_map = create_dynamic_obstacles_map()
    dynamic_map.save_to_file(os.path.join(maps_dir, "dynamic_map.json"))
    
    print("Test maps created successfully!")
    print("Available maps:")
    print("- small_map.json (8x8 with basic obstacles)")
    print("- medium_map.json (15x15 with maze-like structure)")
    print("- large_map.json (25x25 complex city layout)")
    print("- dynamic_map.json (12x12 with moving vehicles)")


def load_test_map(map_name: str) -> GridEnvironment:
    """Load a test map by name."""
    maps_dir = "test_maps"
    map_file = os.path.join(maps_dir, f"{map_name}.json")
    
    if map_name == "small":
        return create_small_map()
    elif map_name == "medium":
        return create_medium_map()
    elif map_name == "large":
        return create_large_map()
    elif map_name == "dynamic":
        return create_dynamic_obstacles_map()
    else:
        # Try to load from file
        if os.path.exists(map_file):
            env = GridEnvironment(1, 1)  # Temporary size
            env.load_from_file(map_file)
            return env
        else:
            raise ValueError(f"Unknown map: {map_name}")


if __name__ == "__main__":
    save_test_maps()