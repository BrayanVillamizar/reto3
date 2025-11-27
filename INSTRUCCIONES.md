# Instrucciones de Uso - Análisis de Cellular Automaton con MPI

## Instalación

### 1. Instalar Dependencias de Python

```bash
pip install -r requirements.txt
```

### 2. Verificar Instalación de MPI

```bash
# Verificar que MPI está instalado
mpirun --version

# Si no está instalado, en Ubuntu/Debian:
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# O para MPICH:
sudo apt-get install mpich
```

### 3. Verificar mpi4py

```bash
python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

## Ejecución Básica

### Versión Serial

```bash
python3 cellular_automaton_serial.py --size 1000 --iterations 100
```

### Versión Paralela con MPI

```bash
# Con 4 procesos
mpirun -np 4 python3 cellular_automaton_mpi.py --size 1000 --iterations 100

# Con 8 procesos
mpirun -np 8 python3 cellular_automaton_mpi.py --size 1000 --iterations 100
```

## Profiling

### Profiling de Memoria

#### Serial
```bash
python3 -m memory_profiler cellular_automaton_serial.py --size 500 --iterations 50
```

#### MPI
```bash
mpirun -np 4 python3 -m memory_profiler cellular_automaton_mpi.py --size 500 --iterations 50
```

### Profiling de Líneas de Código (Line Profiler)

Primero, agregar el decorador `@profile` a las funciones que se quieren perfilar:

```python
@profile
def evolve_grid(grid: np.ndarray) -> np.ndarray:
    ...
```

Luego ejecutar:

```bash
# Serial
kernprof -l -v cellular_automaton_serial.py --size 500 --iterations 50

# MPI (más complejo, requiere configuración adicional)
```

### Profiling de CPU con herramientas del sistema

#### Usando `time`
```bash
time python3 cellular_automaton_serial.py --size 1000 --iterations 100
```

#### Usando `perf` (Linux)
```bash
perf record python3 cellular_automaton_serial.py --size 1000 --iterations 100
perf report
```

#### Monitoreo en tiempo real
```bash
# En una terminal, ejecutar el programa
python3 cellular_automaton_serial.py --size 1000 --iterations 100 &

# En otra terminal, monitorear
watch -n 1 'ps aux | grep cellular_automaton | grep -v grep'
```

## Benchmark Comparativo

### Benchmark Básico

```bash
python3 benchmark.py --sizes 500,1000 --iterations 100 --processes 2,4 --runs 3
```

### Benchmark Completo

```bash
python3 benchmark.py \
    --sizes 500,1000,2000 \
    --iterations 100 \
    --processes 2,4,8 \
    --runs 5 \
    --output benchmark_results.json
```

## Ejecutar Todos los Experimentos

```bash
./run_experiments.sh
```

## Análisis de Resultados

### Interpretar Resultados del Benchmark

El script `benchmark.py` genera:
- **Tiempo de ejecución**: Tiempo promedio y desviación estándar
- **Speedup**: Tiempo_serial / Tiempo_paralelo
- **Eficiencia**: Speedup / Número_de_procesos

### Métricas Esperadas

- **Speedup ideal**: Igual al número de procesos
- **Eficiencia ideal**: 100% (1.0)
- **Eficiencia real**: 
  - Grillas grandes (>2000×2000): 60-80%
  - Grillas medianas (1000×1000): 40-60%
  - Grillas pequeñas (<500×500): <40%

## Troubleshooting

### Error: "mpirun: command not found"

Instalar OpenMPI o MPICH:
```bash
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

### Error: "No module named 'mpi4py'"

Instalar mpi4py:
```bash
pip install mpi4py
```

### Error: "Cannot find MPI library"

Asegurarse de que MPI está instalado antes de instalar mpi4py:
```bash
# Verificar MPI
mpicc --version

# Reinstalar mpi4py
pip uninstall mpi4py
pip install mpi4py
```

### Rendimiento Pobre con MPI

1. Verificar que hay suficientes recursos (CPU, memoria)
2. Para grillas pequeñas, el overhead de comunicación puede ser mayor que el beneficio
3. Probar con diferentes números de procesos
4. Verificar que la red entre procesos es eficiente (si están en diferentes nodos)

## Optimizaciones Adicionales

### Variables de Entorno para MPI

```bash
# Limitar número de threads por proceso
export OMP_NUM_THREADS=1

# Habilitar binding de procesos a CPUs
export OMPI_MCA_rmaps_base_mapping_policy=core
```

### Ejecutar en Múltiples Nodos

```bash
# Crear archivo de hosts (hostfile)
echo "node1 slots=4" > hostfile
echo "node2 slots=4" >> hostfile

# Ejecutar
mpirun -np 8 --hostfile hostfile python3 cellular_automaton_mpi.py --size 2000 --iterations 100
```

## Generar Reportes

Los resultados del benchmark se pueden guardar en JSON:

```bash
python3 benchmark.py --output results.json --sizes 1000,2000 --iterations 100 --processes 2,4,8
```

Luego se pueden analizar con scripts personalizados o herramientas de visualización.

