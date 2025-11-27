#!/usr/bin/env python3
"""
Benchmark script to compare serial and MPI implementations.
"""

import argparse
import subprocess
import json
import time
from typing import List, Dict, Any
import numpy as np


def run_serial(size: int, iterations: int, seed: int = 42) -> float:
    """
    Run serial implementation and return execution time.
    
    Args:
        size: Grid size
        iterations: Number of iterations
        seed: Random seed
        
    Returns:
        Execution time in seconds
    """
    cmd = [
        'python', 'cellular_automaton_serial.py',
        '--size', str(size),
        '--iterations', str(iterations),
        '--seed', str(seed)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"Error running serial: {result.stderr}")
        return -1.0
    
    # Extract time from output
    for line in result.stdout.split('\n'):
        if 'Execution time:' in line:
            try:
                time_str = line.split(':')[1].strip().split()[0]
                return float(time_str)
            except (ValueError, IndexError):
                pass
    
    return end_time - start_time


def run_mpi(size: int, iterations: int, processes: int, seed: int = 42) -> float:
    """
    Run MPI implementation and return execution time.
    
    Args:
        size: Grid size
        iterations: Number of iterations
        processes: Number of MPI processes
        seed: Random seed
        
    Returns:
        Execution time in seconds
    """
    cmd = [
        'mpirun', '-np', str(processes),
        'python', 'cellular_automaton_mpi.py',
        '--size', str(size),
        '--iterations', str(iterations),
        '--seed', str(seed)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"Error running MPI: {result.stderr}")
        return -1.0
    
    # Extract time from output
    for line in result.stdout.split('\n'):
        if 'Execution time:' in line:
            try:
                time_str = line.split(':')[1].strip().split()[0]
                return float(time_str)
            except (ValueError, IndexError):
                pass
    
    return end_time - start_time


def run_benchmark(sizes: List[int], iterations: int, processes_list: List[int], seed: int = 42, runs: int = 3) -> Dict[str, Any]:
    """
    Run comprehensive benchmark comparing serial and MPI implementations.
    
    Args:
        sizes: List of grid sizes to test
        iterations: Number of iterations
        processes_list: List of number of processes to test
        seed: Random seed
        runs: Number of runs per configuration (for averaging)
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'serial': {},
        'mpi': {},
        'speedup': {},
        'efficiency': {}
    }
    
    print("=" * 60)
    print("BENCHMARK: Cellular Automaton Performance Analysis")
    print("=" * 60)
    
    # Benchmark serial implementation
    print("\n--- Serial Implementation ---")
    for size in sizes:
        print(f"\nTesting size: {size}x{size}")
        times = []
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}...", end=' ', flush=True)
            exec_time = run_serial(size, iterations, seed)
            if exec_time > 0:
                times.append(exec_time)
                print(f"{exec_time:.4f}s")
            else:
                print("FAILED")
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            results['serial'][size] = {
                'mean': avg_time,
                'std': std_time,
                'runs': times
            }
            print(f"  Average: {avg_time:.4f}s ± {std_time:.4f}s")
    
    # Benchmark MPI implementation
    print("\n--- MPI Implementation ---")
    for size in sizes:
        results['mpi'][size] = {}
        results['speedup'][size] = {}
        results['efficiency'][size] = {}
        
        for num_procs in processes_list:
            print(f"\nTesting size: {size}x{size}, processes: {num_procs}")
            times = []
            for run in range(runs):
                print(f"  Run {run + 1}/{runs}...", end=' ', flush=True)
                exec_time = run_mpi(size, iterations, num_procs, seed)
                if exec_time > 0:
                    times.append(exec_time)
                    print(f"{exec_time:.4f}s")
                else:
                    print("FAILED")
            
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                results['mpi'][size][num_procs] = {
                    'mean': avg_time,
                    'std': std_time,
                    'runs': times
                }
                
                # Calculate speedup and efficiency
                serial_time = results['serial'][size]['mean']
                speedup = serial_time / avg_time
                efficiency = speedup / num_procs
                
                results['speedup'][size][num_procs] = speedup
                results['efficiency'][size][num_procs] = efficiency
                
                print(f"  Average: {avg_time:.4f}s ± {std_time:.4f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Efficiency: {efficiency:.2%}")
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nSerial Performance:")
    print(f"{'Size':<10} {'Time (s)':<15} {'Std Dev':<15}")
    print("-" * 40)
    for size in sorted(results['serial'].keys()):
        data = results['serial'][size]
        print(f"{size:<10} {data['mean']:<15.4f} {data['std']:<15.4f}")
    
    print("\nMPI Performance:")
    sizes = sorted(results['mpi'].keys())
    processes_list = sorted(set(p for size_data in results['mpi'].values() for p in size_data.keys()))
    
    print(f"\n{'Size':<10} {'Processes':<12} {'Time (s)':<15} {'Speedup':<12} {'Efficiency':<12}")
    print("-" * 70)
    for size in sizes:
        for num_procs in processes_list:
            if num_procs in results['mpi'][size]:
                mpi_data = results['mpi'][size][num_procs]
                speedup = results['speedup'][size][num_procs]
                efficiency = results['efficiency'][size][num_procs]
                print(f"{size:<10} {num_procs:<12} {mpi_data['mean']:<15.4f} {speedup:<12.2f} {efficiency:<12.2%}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Cellular Automaton implementations')
    parser.add_argument('--sizes', type=str, default='500,1000,2000',
                       help='Comma-separated list of grid sizes (default: 500,1000,2000)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations (default: 100)')
    parser.add_argument('--processes', type=str, default='2,4,8',
                       help='Comma-separated list of MPI process counts (default: 2,4,8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration (default: 3)')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results (optional)')
    
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    processes_list = [int(p.strip()) for p in args.processes.split(',')]
    
    results = run_benchmark(sizes, args.iterations, processes_list, args.seed, args.runs)
    
    print_summary(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

