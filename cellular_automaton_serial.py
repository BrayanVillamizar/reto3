#!/usr/bin/env python3
"""
Serial implementation of Conway's Game of Life Cellular Automaton.
"""

import argparse
import numpy as np
import time
from typing import Tuple


def initialize_grid(size: int, seed: int = 42) -> np.ndarray:
    """
    Initialize a random grid of cells.
    
    Args:
        size: Grid dimensions (size x size)
        seed: Random seed for reproducibility
        
    Returns:
        2D numpy array with random binary values (0 or 1)
    """
    np.random.seed(seed)
    grid = np.random.randint(0, 2, size=(size, size), dtype=np.int8)
    return grid


def count_neighbors(grid: np.ndarray, row: int, col: int) -> int:
    """
    Count the number of alive neighbors for a cell at (row, col).
    
    Args:
        grid: Current state of the grid
        row: Row index
        col: Column index
        
    Returns:
        Number of alive neighbors (0-8)
    """
    size = grid.shape[0]
    count = 0
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            
            neighbor_row = (row + i) % size
            neighbor_col = (col + j) % size
            count += grid[neighbor_row, neighbor_col]
    
    return count


def update_cell(current_state: int, neighbors: int) -> int:
    """
    Apply Game of Life rules to determine next state of a cell.
    
    Rules:
    - Live cell with 2-3 neighbors survives
    - Dead cell with exactly 3 neighbors becomes alive
    - Otherwise, cell dies or stays dead
    
    Args:
        current_state: Current state of the cell (0 or 1)
        neighbors: Number of alive neighbors
        
    Returns:
        Next state of the cell (0 or 1)
    """
    if current_state == 1:
        if neighbors == 2 or neighbors == 3:
            return 1
        return 0
    else:
        if neighbors == 3:
            return 1
        return 0


def evolve_grid(grid: np.ndarray) -> np.ndarray:
    """
    Evolve the grid one generation according to Game of Life rules.
    Uses vectorized operations for better performance.
    
    Args:
        grid: Current state of the grid
        
    Returns:
        New state of the grid after one generation
    """
    size = grid.shape[0]
    
    # Calculate neighbor counts using vectorized operations
    neighbor_count = np.zeros_like(grid, dtype=np.int8)
    
    # Sum all 8 neighbors using periodic boundaries
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            
            # Roll grid to get neighbors with periodic boundary
            rolled = np.roll(np.roll(grid, di, axis=0), dj, axis=1)
            neighbor_count += rolled
    
    # Apply Game of Life rules vectorized
    new_grid = np.where(
        (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3)),
        1,
        np.where(
            (grid == 0) & (neighbor_count == 3),
            1,
            0
        )
    ).astype(np.int8)
    
    return new_grid


def run_simulation(size: int, iterations: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Run the complete simulation.
    
    Args:
        size: Grid dimensions
        iterations: Number of generations to simulate
        seed: Random seed for initialization
        
    Returns:
        Tuple of (final grid, execution time in seconds)
    """
    grid = initialize_grid(size, seed)
    
    start_time = time.time()
    
    for _ in range(iterations):
        grid = evolve_grid(grid)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return grid, execution_time


def main():
    parser = argparse.ArgumentParser(description='Serial Cellular Automaton Simulation')
    parser.add_argument('--size', type=int, default=1000, help='Grid size (default: 1000)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, help='Output file for final grid (optional)')
    
    args = parser.parse_args()
    
    print(f"Running serial simulation: {args.size}x{args.size} grid, {args.iterations} iterations")
    
    final_grid, execution_time = run_simulation(args.size, args.iterations, args.seed)
    
    alive_cells = np.sum(final_grid)
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Final alive cells: {alive_cells}")
    
    if args.output:
        np.save(args.output, final_grid)
        print(f"Final grid saved to {args.output}")


if __name__ == '__main__':
    main()

