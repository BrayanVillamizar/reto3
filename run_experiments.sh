#!/bin/bash
# Script to run all experiments for Cellular Automaton analysis

set -e

echo "=========================================="
echo "Cellular Automaton - Experiment Suite"
echo "=========================================="
echo ""

# Check if MPI is available
if ! command -v mpirun &> /dev/null; then
    echo "Warning: mpirun not found. MPI experiments will be skipped."
    MPI_AVAILABLE=false
else
    MPI_AVAILABLE=true
    echo "MPI detected: $(mpirun --version | head -n 1)"
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import numpy, mpi4py" 2>/dev/null || {
    echo "Error: Required Python packages not installed."
    echo "Run: pip install -r requirements.txt"
    exit 1
}

echo ""
echo "=========================================="
echo "1. Serial Implementation Test"
echo "=========================================="
python3 cellular_automaton_serial.py --size 100 --iterations 10 --seed 42
echo ""

if [ "$MPI_AVAILABLE" = true ]; then
    echo "=========================================="
    echo "2. MPI Implementation Test"
    echo "=========================================="
    mpirun -np 2 python3 cellular_automaton_mpi.py --size 100 --iterations 10 --seed 42
    echo ""
fi

echo "=========================================="
echo "3. Quick Benchmark (Small Grid)"
echo "=========================================="
python3 benchmark.py --sizes 200,500 --iterations 50 --processes 2,4 --runs 2 --seed 42
echo ""

if [ "$MPI_AVAILABLE" = true ]; then
    echo "=========================================="
    echo "4. Full Benchmark (Medium Grid)"
    echo "=========================================="
    echo "This may take several minutes..."
    python3 benchmark.py --sizes 500,1000 --iterations 100 --processes 2,4,8 --runs 3 --seed 42 --output benchmark_results.json
    echo ""
    
    echo "=========================================="
    echo "5. Profiling Examples"
    echo "=========================================="
    echo "Serial profiling:"
    echo "  python3 profiling.py --mode serial --size 500 --iterations 50"
    echo ""
    echo "MPI profiling:"
    echo "  python3 profiling.py --mode mpi --size 500 --iterations 50 --processes 4"
    echo ""
fi

echo "=========================================="
echo "Experiments completed!"
echo "=========================================="

