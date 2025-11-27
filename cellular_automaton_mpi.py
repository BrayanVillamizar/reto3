#!/usr/bin/env python3
"""
Parallel implementation of Conway's Game of Life Cellular Automaton using MPI.
"""

import argparse
import numpy as np
import time
from typing import Tuple
from mpi4py import MPI


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


def evolve_local_grid_vectorized(local_grid: np.ndarray, ghost_top: np.ndarray, ghost_bottom: np.ndarray) -> np.ndarray:
    """
    Evolve the local portion of the grid using optimized operations.
    
    Args:
        local_grid: Local portion of the grid assigned to this process
        ghost_top: Ghost row from the top neighbor
        ghost_bottom: Ghost row from the bottom neighbor
        
    Returns:
        New state of the local grid after one generation
    """
    local_rows, cols = local_grid.shape
    
    # Create extended grid with ghost rows
    extended_grid = np.zeros((local_rows + 2, cols), dtype=np.int8)
    extended_grid[0, :] = ghost_top
    extended_grid[1:-1, :] = local_grid
    extended_grid[-1, :] = ghost_bottom
    
    # Calculate neighbor counts using vectorized operations
    neighbor_count = np.zeros((local_rows, cols), dtype=np.int8)
    
    # Sum all 8 neighbors using slicing
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            
            # Row indices in extended_grid (offset by 1 due to ghost row)
            row_slice = slice(1 + di, local_rows + 1 + di)
            
            # Column indices with periodic boundary
            if dj == -1:
                col_slice = np.concatenate([[cols - 1], np.arange(cols - 1)])
            elif dj == 1:
                col_slice = np.concatenate([np.arange(1, cols), [0]])
            else:
                col_slice = np.arange(cols)
            
            neighbor_count += extended_grid[row_slice, col_slice]
    
    # Apply Game of Life rules vectorized
    new_local_grid = np.where(
        (local_grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3)),
        1,
        np.where(
            (local_grid == 0) & (neighbor_count == 3),
            1,
            0
        )
    ).astype(np.int8)
    
    return new_local_grid


def run_simulation_mpi(size: int, iterations: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Run the complete simulation using MPI.
    
    Args:
        size: Grid dimensions
        iterations: Number of generations to simulate
        seed: Random seed for initialization
        
    Returns:
        Tuple of (final grid, execution time in seconds)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processes = comm.Get_size()
    
    # Initialize full grid only on rank 0
    if rank == 0:
        full_grid = initialize_grid(size, seed)
    else:
        full_grid = None
    
    # Distribute grid rows among processes
    rows_per_process = size // num_processes
    remainder = size % num_processes
    
    # Calculate local row range for this process
    if rank < remainder:
        local_start = rank * (rows_per_process + 1)
        local_rows = rows_per_process + 1
    else:
        local_start = rank * rows_per_process + remainder
        local_rows = rows_per_process
    
    local_end = local_start + local_rows
    
    # Scatter rows to each process
    local_grid = np.zeros((local_rows, size), dtype=np.int8)
    sendcounts = np.zeros(num_processes, dtype=np.int32)
    displs = np.zeros(num_processes, dtype=np.int32)
    
    for i in range(num_processes):
        if i < remainder:
            sendcounts[i] = (rows_per_process + 1) * size
        else:
            sendcounts[i] = rows_per_process * size
        
        if i == 0:
            displs[i] = 0
        else:
            if i - 1 < remainder:
                displs[i] = displs[i-1] + (rows_per_process + 1) * size
            else:
                displs[i] = displs[i-1] + rows_per_process * size
    
    comm.Scatterv([full_grid, sendcounts, displs, MPI.BYTE], local_grid, root=0)
    
    # Synchronize before timing
    comm.Barrier()
    start_time = time.time()
    
    # Main simulation loop
    for iteration in range(iterations):
        # Determine neighbors
        top_neighbor = (rank - 1) % num_processes
        bottom_neighbor = (rank + 1) % num_processes
        
        # Exchange ghost rows
        ghost_top = np.zeros(size, dtype=np.int8)
        ghost_bottom = np.zeros(size, dtype=np.int8)
        
        # Exchange ghost rows for periodic boundaries
        if num_processes > 1:
            # For periodic boundaries: process 0 needs last row of last process as top ghost
            # and last process needs first row of process 0 as bottom ghost
            if rank == 0:
                # Send first row to last process, receive from last process
                comm.Sendrecv(local_grid[0, :], dest=num_processes - 1, sendtag=1,
                            recvbuf=ghost_top, source=num_processes - 1, recvtag=1)
                # Send last row to next process, receive from next process
                comm.Sendrecv(local_grid[-1, :], dest=bottom_neighbor, sendtag=0,
                            recvbuf=ghost_bottom, source=bottom_neighbor, recvtag=0)
            elif rank == num_processes - 1:
                # Send last row to first process, receive from first process
                comm.Sendrecv(local_grid[-1, :], dest=0, sendtag=1,
                            recvbuf=ghost_bottom, source=0, recvtag=1)
                # Send first row to previous process, receive from previous process
                comm.Sendrecv(local_grid[0, :], dest=top_neighbor, sendtag=0,
                            recvbuf=ghost_top, source=top_neighbor, recvtag=0)
            else:
                # Middle processes: normal communication
                comm.Sendrecv(local_grid[0, :], dest=top_neighbor, sendtag=0,
                            recvbuf=ghost_top, source=top_neighbor, recvtag=0)
                comm.Sendrecv(local_grid[-1, :], dest=bottom_neighbor, sendtag=0,
                            recvbuf=ghost_bottom, source=bottom_neighbor, recvtag=0)
        else:
            # Single process case: use periodic boundary within local grid
            ghost_top = local_grid[-1, :].copy()
            ghost_bottom = local_grid[0, :].copy()
        
        # Evolve local grid
        local_grid = evolve_local_grid_vectorized(local_grid, ghost_top, ghost_bottom)
    
    # Synchronize after computation
    comm.Barrier()
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Gather results back to rank 0
    if rank == 0:
        final_grid = np.zeros((size, size), dtype=np.int8)
    else:
        final_grid = None
    
    comm.Gatherv(local_grid, [final_grid, sendcounts, displs, MPI.BYTE], root=0)
    
    return final_grid, execution_time


def main():
    parser = argparse.ArgumentParser(description='Parallel Cellular Automaton Simulation with MPI')
    parser.add_argument('--size', type=int, default=1000, help='Grid size (default: 1000)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, help='Output file for final grid (optional)')
    
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processes = comm.Get_size()
    
    if rank == 0:
        print(f"Running parallel simulation: {args.size}x{args.size} grid, {args.iterations} iterations, {num_processes} processes")
    
    final_grid, execution_time = run_simulation_mpi(args.size, args.iterations, args.seed)
    
    if rank == 0:
        alive_cells = np.sum(final_grid)
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"Final alive cells: {alive_cells}")
        
        if args.output:
            np.save(args.output, final_grid)
            print(f"Final grid saved to {args.output}")


if __name__ == '__main__':
    main()

