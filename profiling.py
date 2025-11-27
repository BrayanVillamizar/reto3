#!/usr/bin/env python3
"""
Profiling tools for CPU and memory analysis of Cellular Automaton implementations.
"""

import argparse
import subprocess
import sys
import time
from typing import Dict, Any


def profile_serial(size: int, iterations: int, seed: int = 42) -> Dict[str, Any]:
    """
    Profile the serial implementation using line_profiler and memory_profiler.
    
    Args:
        size: Grid size
        iterations: Number of iterations
        seed: Random seed
        
    Returns:
        Dictionary with profiling results
    """
    print(f"Profiling serial implementation: {size}x{size}, {iterations} iterations")
    
    # Memory profiling
    print("\n=== Memory Profiling ===")
    memory_cmd = [
        'python', '-m', 'memory_profiler',
        'cellular_automaton_serial.py',
        '--size', str(size),
        '--iterations', str(iterations),
        '--seed', str(seed)
    ]
    
    start_time = time.time()
    result = subprocess.run(memory_cmd, capture_output=True, text=True)
    memory_time = time.time() - start_time
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Line profiling (requires @profile decorator in code)
    print("\n=== Line Profiling ===")
    print("Note: Line profiling requires @profile decorator in source code")
    print("Run: kernprof -l -v cellular_automaton_serial.py --size {} --iterations {}".format(size, iterations))
    
    return {
        'mode': 'serial',
        'size': size,
        'iterations': iterations,
        'memory_time': memory_time
    }


def profile_mpi(size: int, iterations: int, processes: int, seed: int = 42) -> Dict[str, Any]:
    """
    Profile the MPI implementation.
    
    Args:
        size: Grid size
        iterations: Number of iterations
        processes: Number of MPI processes
        seed: Random seed
        
    Returns:
        Dictionary with profiling results
    """
    print(f"Profiling MPI implementation: {size}x{size}, {iterations} iterations, {processes} processes")
    
    # Memory profiling with MPI
    print("\n=== Memory Profiling (MPI) ===")
    memory_cmd = [
        'mpirun', '-np', str(processes),
        'python', '-m', 'memory_profiler',
        'cellular_automaton_mpi.py',
        '--size', str(size),
        '--iterations', str(iterations),
        '--seed', str(seed)
    ]
    
    start_time = time.time()
    result = subprocess.run(memory_cmd, capture_output=True, text=True)
    memory_time = time.time() - start_time
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return {
        'mode': 'mpi',
        'size': size,
        'iterations': iterations,
        'processes': processes,
        'memory_time': memory_time
    }


def profile_cpu_usage(size: int, iterations: int, mode: str, processes: int = 1) -> None:
    """
    Monitor CPU usage during execution.
    
    Args:
        size: Grid size
        iterations: Number of iterations
        mode: 'serial' or 'mpi'
        processes: Number of processes (for MPI mode)
    """
    print(f"\n=== CPU Usage Monitoring ===")
    print("Run this command in another terminal to monitor CPU:")
    
    if mode == 'serial':
        print(f"  watch -n 1 'ps aux | grep cellular_automaton_serial | grep -v grep'")
        print(f"  Or use: top -p $(pgrep -f cellular_automaton_serial)")
    else:
        print(f"  watch -n 1 'ps aux | grep mpirun | grep -v grep'")
        print(f"  Or use: htop and filter for python processes")


def main():
    parser = argparse.ArgumentParser(description='Profile Cellular Automaton implementations')
    parser.add_argument('--mode', type=str, choices=['serial', 'mpi'], required=True,
                       help='Profiling mode: serial or mpi')
    parser.add_argument('--size', type=int, default=1000, help='Grid size (default: 1000)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations (default: 100)')
    parser.add_argument('--processes', type=int, default=4, help='Number of MPI processes (default: 4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.mode == 'serial':
        profile_serial(args.size, args.iterations, args.seed)
        profile_cpu_usage(args.size, args.iterations, 'serial')
    else:
        profile_mpi(args.size, args.iterations, args.processes, args.seed)
        profile_cpu_usage(args.size, args.iterations, 'mpi', args.processes)


if __name__ == '__main__':
    main()

