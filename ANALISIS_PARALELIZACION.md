# Análisis de Paralelización - Cellular Automaton

## 1. Descripción del Algoritmo

El algoritmo implementado es el **Game of Life de Conway**, un autómata celular que evoluciona según reglas simples:

- **Regla 1**: Una célula viva con 2 o 3 vecinos vivos sobrevive
- **Regla 2**: Una célula muerta con exactamente 3 vecinos vivos nace
- **Regla 3**: En cualquier otro caso, la célula muere o permanece muerta

### Complejidad Computacional

- **Tiempo**: O(n² × k) donde n es el tamaño de la grilla y k el número de iteraciones
- **Espacio**: O(n²) para almacenar la grilla

## 2. Análisis de Paralelización

### 2.1. Opciones de Paralelización Identificadas

#### Opción 1: Descomposición por Dominio (Domain Decomposition) - **IMPLEMENTADA**

**Descripción**: Dividir la grilla en subdominios horizontales, asignando filas contiguas a cada proceso.

**Ventajas**:
- Comunicación mínima: solo se necesitan intercambiar filas "fantasma" (ghost rows)
- Balanceo de carga relativamente simple
- Escalabilidad buena para grillas grandes

**Desventajas**:
- Requiere comunicación entre procesos vecinos en cada iteración
- Overhead de comunicación puede ser significativo para grillas pequeñas

**Complejidad de Comunicación**: O(n × k × p) donde p es el número de procesos

#### Opción 2: Descomposición por Tareas (Task Decomposition)

**Descripción**: Cada proceso calcula una porción de células independientemente.

**Ventajas**:
- Paralelismo fino
- Menor overhead de comunicación

**Desventajas**:
- Requiere sincronización más compleja
- Dificultad para mantener consistencia de datos

**No implementada**: Menos eficiente para este problema específico.

#### Opción 3: Paralelización Híbrida (MPI + OpenMP)

**Descripción**: Combinar MPI para comunicación entre nodos y OpenMP para paralelismo dentro de cada nodo.

**Ventajas**:
- Mejor aprovechamiento de recursos en sistemas multi-core
- Reduce comunicación entre nodos

**Desventajas**:
- Mayor complejidad de implementación
- Requiere optimización adicional

**No implementada**: Requiere compilación con OpenMP.

### 2.2. Estrategia de Paralelización Implementada

Se implementó la **Descomposición por Dominio** con las siguientes características:

1. **Distribución de Datos**:
   - La grilla completa se divide en bandas horizontales
   - Cada proceso recibe un conjunto contiguo de filas
   - Se maneja el caso de división no uniforme (remainder)

2. **Comunicación**:
   - Uso de filas "fantasma" (ghost rows) para mantener condiciones de frontera
   - Comunicación punto-a-punto con `Sendrecv` para minimizar overhead
   - Condiciones de frontera periódicas (toroidal)

3. **Sincronización**:
   - Barreras MPI antes y después del cálculo principal
   - Sincronización implícita en las operaciones de comunicación

## 3. Análisis de Desempeño

### 3.1. Factores que Afectan el Desempeño

#### A. Overhead de Comunicación

**Causa**: En cada iteración, cada proceso debe:
- Enviar su fila superior al proceso vecino superior
- Enviar su fila inferior al proceso vecino inferior
- Recibir las filas correspondientes

**Impacto**: 
- Para grillas pequeñas: el overhead de comunicación puede superar el beneficio del paralelismo
- Para grillas grandes: el tiempo de comunicación es pequeño comparado con el cómputo

**Fórmula aproximada**:
```
T_comm ≈ 2 × (latency + bandwidth × n)
```

#### B. Balanceo de Carga

**Causa**: Cuando el número de procesos no divide exactamente el tamaño de la grilla, algunos procesos reciben una fila adicional.

**Impacto**: 
- Procesos con más filas tardan más en completar
- Todos los procesos esperan al más lento (sincronización)

**Mitigación**: La implementación distribuye el remainder de manera uniforme entre los primeros procesos.

#### C. Escalabilidad

**Escalabilidad Fuerte**: Para problemas de tamaño fijo, el speedup debería aumentar con más procesos hasta cierto punto.

**Escalabilidad Débil**: Para problemas que crecen con el número de procesos, el tiempo de ejecución debería mantenerse constante.

**Límites**:
- **Ley de Amdahl**: La parte serial limita el speedup máximo
- **Overhead de comunicación**: Aumenta con el número de procesos
- **Memoria**: Cada proceso necesita almacenar su porción de la grilla

### 3.2. Métricas de Desempeño

#### Speedup

```
Speedup(p) = T_serial / T_parallel(p)
```

**Speedup ideal**: Igual al número de procesos p.

**Speedup real**: Generalmente menor debido a:
- Overhead de comunicación
- Desbalanceo de carga
- Parte serial del código

#### Eficiencia

```
Efficiency(p) = Speedup(p) / p
```

**Eficiencia ideal**: 100% (1.0)

**Eficiencia esperada**: 
- Para grillas grandes (>2000×2000): 60-80%
- Para grillas medianas (1000×1000): 40-60%
- Para grillas pequeñas (<500×500): <40% (puede ser <1.0, es decir, más lento que serial)

### 3.3. Análisis de Bottlenecks

#### Bottleneck 1: Comunicación

**Evidencia**: 
- El tiempo de comunicación es proporcional al número de iteraciones
- Para pocos procesos, el overhead es bajo
- Para muchos procesos, el overhead aumenta

**Optimización posible**:
- Reducir el número de iteraciones (si es posible)
- Usar comunicación asíncrona (non-blocking)
- Agrupar múltiples operaciones de comunicación

#### Bottleneck 2: Memoria

**Evidencia**:
- Cada proceso mantiene una copia local de su porción
- Operaciones de gather/scatter requieren memoria adicional

**Optimización posible**:
- Usar tipos de datos más pequeños (int8 en lugar de int32)
- Optimizar el uso de memoria en operaciones de comunicación

#### Bottleneck 3: Cálculo Local

**Evidencia**:
- El cálculo de vecinos es O(1) por célula
- No hay dependencias entre células en la misma iteración

**Optimización posible**:
- Vectorización con NumPy (ya implementado parcialmente)
- Optimización del compilador
- Uso de SIMD instructions

## 4. Resultados Esperados

### 4.1. Escenarios de Desempeño

#### Escenario 1: Grilla Pequeña (500×500)

- **Serial**: ~0.5-1.0 segundos
- **2 procesos**: Speedup ~1.5x, Eficiencia ~75%
- **4 procesos**: Speedup ~2.0x, Eficiencia ~50%
- **8 procesos**: Speedup ~2.5x, Eficiencia ~31%

**Conclusión**: Overhead de comunicación domina, paralelización no muy efectiva.

#### Escenario 2: Grilla Mediana (1000×1000)

- **Serial**: ~2-4 segundos
- **2 procesos**: Speedup ~1.8x, Eficiencia ~90%
- **4 procesos**: Speedup ~3.0x, Eficiencia ~75%
- **8 procesos**: Speedup ~4.5x, Eficiencia ~56%

**Conclusión**: Buen balance entre cómputo y comunicación.

#### Escenario 3: Grilla Grande (2000×2000)

- **Serial**: ~8-16 segundos
- **2 procesos**: Speedup ~1.9x, Eficiencia ~95%
- **4 procesos**: Speedup ~3.5x, Eficiencia ~88%
- **8 procesos**: Speedup ~6.0x, Eficiencia ~75%

**Conclusión**: Paralelización muy efectiva, comunicación es pequeña comparada con cómputo.

### 4.2. Punto de Quiebre (Break-even Point)

El punto donde la paralelización comienza a ser beneficiosa depende de:
- Tamaño de la grilla
- Número de procesos
- Latencia y ancho de banda de la red
- Overhead de MPI

**Estimación**: Para grillas menores a 500×500 con más de 4 procesos, el overhead puede hacer que la versión paralela sea más lenta que la serial.

## 5. Optimizaciones Implementadas

1. **Tipos de datos optimizados**: Uso de `int8` en lugar de `int32` para reducir memoria y comunicación
2. **Comunicación eficiente**: Uso de `Sendrecv` para comunicación bidireccional
3. **Distribución balanceada**: Manejo del remainder para balancear carga
4. **Operaciones vectorizadas**: Uso de NumPy para operaciones eficientes

## 6. Optimizaciones Futuras

1. **Comunicación asíncrona**: Usar `Isend`/`Irecv` para solapar comunicación y cómputo
2. **Paralelización híbrida**: Combinar MPI con OpenMP para paralelismo multi-nivel
3. **Optimización de memoria**: Reducir copias innecesarias
4. **Vectorización avanzada**: Usar operaciones NumPy más eficientes para el cálculo de vecinos

## 7. Conclusiones

- La paralelización con MPI es efectiva para grillas grandes (>1000×1000)
- El overhead de comunicación limita la escalabilidad para grillas pequeñas
- La eficiencia mejora con el tamaño del problema (escalabilidad débil)
- La estrategia de descomposición por dominio es adecuada para este problema
- Se requiere un análisis empírico para determinar el número óptimo de procesos según el tamaño del problema

