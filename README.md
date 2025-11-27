# Análisis de Cellular Automaton con MPI

Este proyecto implementa el análisis de Cellular Automaton (Game of Life de Conway) en versiones serial y paralela utilizando MPI, con profiling y optimización de CPU y memoria.

## Estructura del Proyecto

- `cellular_automaton_serial.py`: Implementación serial del algoritmo
- `cellular_automaton_mpi.py`: Implementación paralela con MPI
- `profiling.py`: Scripts para profiling de CPU y memoria
- `benchmark.py`: Script para análisis comparativo de desempeño
- `requirements.txt`: Dependencias del proyecto
- `run_experiments.sh`: Script para ejecutar todos los experimentos

## Requisitos

- Python 3.8+
- MPI (OpenMPI o MPICH)
- mpi4py
- numpy
- memory_profiler
- line_profiler

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Versión Serial

```bash
python cellular_automaton_serial.py --size 1000 --iterations 100
```

### Versión Paralela con MPI

```bash
mpirun -np 4 python cellular_automaton_mpi.py --size 1000 --iterations 100
```

### Profiling

```bash
python profiling.py --mode serial --size 1000 --iterations 100
python profiling.py --mode mpi --size 1000 --iterations 100 --processes 4
```

### Benchmark Comparativo

```bash
python benchmark.py --sizes 500,1000,2000 --iterations 100 --processes 2,4,8
```

## Análisis de Paralelización

El análisis de opciones de paralelización y resultados de desempeño se encuentra en `ANALISIS_PARALELIZACION.md`.

