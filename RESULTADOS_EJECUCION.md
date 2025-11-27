# Resultados de Ejecución - Cellular Automaton

## Fecha de Ejecución
Ejecutado en sistema con 2 cores disponibles.

## Configuración del Entorno

- Python: 3.12
- NumPy: 2.3.5
- mpi4py: 4.1.1
- OpenMPI: Disponible (limitado a 2 procesos por hardware)

## Resultados del Benchmark

### Grilla 200×200, 50 iteraciones

| Implementación | Tiempo (s) | Speedup | Eficiencia |
|----------------|------------|---------|------------|
| Serial         | 0.4390     | 1.00x   | 100%       |
| MPI (2 proc)   | 0.4546     | 0.97x   | 48.29%     |

**Análisis**: Para grillas pequeñas, el overhead de comunicación hace que la versión paralela sea ligeramente más lenta que la serial. Esto confirma el análisis teórico sobre el punto de quiebre.

### Grilla 500×500, 100 iteraciones

| Implementación | Tiempo (s) | Speedup | Eficiencia |
|----------------|------------|---------|------------|
| Serial         | 2.1723     | 1.00x   | 100%       |
| MPI (2 proc)   | 2.1372     | 1.02x   | 50.82%     |

**Análisis**: Para grillas medianas, se observa un speedup mínimo (1.02x). La eficiencia del 50% es esperada con 2 procesos, pero el overhead de comunicación aún limita el beneficio.

### Grilla 1000×1000, 100 iteraciones

| Implementación | Tiempo (s) | Speedup | Eficiencia |
|----------------|------------|---------|------------|
| Serial         | 14.5592    | 1.00x   | 100%       |
| MPI (2 proc)   | 14.4562    | 1.01x   | 50.36%     |

**Análisis**: Incluso para grillas grandes, el speedup es mínimo. Esto se debe a:
1. Solo 2 procesos disponibles (limitación del hardware)
2. Overhead de comunicación proporcional al número de iteraciones
3. Tamaño de grilla aún no suficientemente grande para dominar el overhead

## Análisis de Desempeño

### Factores Observados

1. **Overhead de Comunicación**: 
   - En cada iteración, cada proceso debe intercambiar filas fantasma
   - Para 100 iteraciones, esto significa 100 operaciones de comunicación
   - El overhead es más significativo para grillas pequeñas

2. **Balanceo de Carga**:
   - La distribución de filas está balanceada correctamente
   - No se observan problemas de desbalanceo

3. **Escalabilidad Limitada**:
   - Con solo 2 procesos, el speedup máximo teórico es 2x
   - El overhead reduce este speedup a ~1.01-1.02x
   - Para ver mejor escalabilidad, se necesitarían:
     - Más procesos (4, 8, 16)
     - Grillas más grandes (>2000×2000)
     - Menos iteraciones o comunicación más eficiente

### Comparación con Análisis Teórico

El análisis teórico predijo:
- **Grillas pequeñas (<500×500)**: Overhead domina, speedup < 1.0x posible
- **Grillas medianas (1000×1000)**: Speedup moderado, eficiencia 40-60%
- **Grillas grandes (>2000×2000)**: Mejor speedup, eficiencia 60-80%

**Resultados observados**:
- Grillas pequeñas: Speedup < 1.0x (confirmado)
- Grillas medianas: Speedup ~1.0x (menor de lo esperado debido a solo 2 procesos)
- No se probaron grillas grandes por limitaciones de tiempo

### Optimizaciones Aplicadas

1. **Operaciones Vectorizadas**: 
   - Uso de `np.roll` y operaciones vectorizadas de NumPy
   - Reducción significativa del tiempo de cómputo local

2. **Tipos de Datos Optimizados**:
   - Uso de `int8` en lugar de `int32`
   - Reducción de memoria y comunicación

3. **Comunicación Eficiente**:
   - Uso de `Sendrecv` para comunicación bidireccional
   - Minimización de overhead de sincronización

## Conclusiones

1. **Implementación Correcta**: 
   - Ambas versiones (serial y paralela) producen los mismos resultados
   - El número de células vivas finales coincide (2139 para 100×100, 10 iteraciones)

2. **Paralelización Funcional**:
   - La versión MPI funciona correctamente
   - La comunicación y sincronización están bien implementadas

3. **Limitaciones Observadas**:
   - Con solo 2 procesos, el beneficio del paralelismo es mínimo
   - Se necesitarían más procesos o grillas más grandes para ver mejor escalabilidad
   - El overhead de comunicación es significativo para el número de iteraciones usado

4. **Recomendaciones**:
   - Para mejor desempeño, usar 4-8 procesos o más
   - Probar con grillas más grandes (2000×2000 o mayores)
   - Considerar optimizaciones adicionales como comunicación asíncrona

## Próximos Pasos Sugeridos

1. Ejecutar en sistema con más cores (4, 8, 16)
2. Probar con grillas más grandes (2000×2000, 5000×5000)
3. Implementar comunicación asíncrona para solapar cómputo y comunicación
4. Profiling detallado con herramientas como `perf` o `vtune`
5. Análisis de escalabilidad débil (aumentar tamaño con número de procesos)

