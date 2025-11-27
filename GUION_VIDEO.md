# Guion de Video - Análisis de Cellular Automaton con MPI
## Duración: 5 minutos | Formato: Diálogo entre 2 personas

---

## [0:00 - 0:30] INTRODUCCIÓN

**Persona 1 (P1)**: Hola, hoy vamos a explicar la implementación de un Cellular Automaton usando paralelismo con MPI.

**Persona 2 (P2)**: Exacto. Hemos trabajado en el análisis del Game of Life de Conway, implementándolo tanto en versión serial como paralela, y analizando su desempeño.

**P1**: El objetivo es entender cómo se puede paralelizar este tipo de algoritmos y qué factores afectan el rendimiento cuando usamos múltiples procesos.

**P2**: Vamos a ver la implementación, las opciones de paralelización que consideramos, y los resultados que obtuvimos.

---

## [0:30 - 1:30] EXPLICACIÓN DEL ALGORITMO

**P1**: Primero, ¿qué es el Game of Life? Es un autómata celular donde cada célula puede estar viva o muerta, y evoluciona según reglas simples.

**P2**: Las reglas son tres: una célula viva con 2 o 3 vecinos sobrevive; una célula muerta con exactamente 3 vecinos nace; y en cualquier otro caso, la célula muere o permanece muerta.

**P1**: [Mostrando código/gráfico] En nuestra implementación, representamos la grilla como una matriz de números, donde 1 es vivo y 0 es muerto.

**P2**: Para cada célula, contamos sus 8 vecinos, considerando condiciones de frontera periódicas, es decir, la grilla se "envuelve" como un toro.

**P1**: La complejidad es O(n² × k), donde n es el tamaño de la grilla y k el número de iteraciones. Para grillas grandes, esto puede ser computacionalmente costoso.

---

## [1:30 - 2:30] VERSIÓN SERIAL

**P2**: Empezamos con la versión serial. [Mostrando código] Usamos NumPy para operaciones vectorizadas, lo que mejora significativamente el rendimiento.

**P1**: La función `evolve_grid` calcula el número de vecinos usando `np.roll`, que es muy eficiente, y luego aplica las reglas del juego de forma vectorizada.

**P2**: Esto nos permite procesar toda la grilla de una vez, en lugar de célula por célula, aprovechando las optimizaciones de NumPy.

**P1**: Para una grilla de 1000×1000 con 100 iteraciones, la versión serial tarda aproximadamente 14.5 segundos en nuestro sistema.

**P2**: Pero podemos hacerlo más rápido usando paralelismo. Ahí es donde entra MPI.

---

## [2:30 - 3:45] VERSIÓN PARALELA CON MPI

**P1**: Para paralelizar, usamos descomposición por dominio. Dividimos la grilla en bandas horizontales, asignando filas contiguas a cada proceso.

**P2**: [Mostrando diagrama] Cada proceso trabaja con su porción de la grilla. Pero hay un problema: las células en los bordes necesitan información de las filas vecinas.

**P1**: Por eso usamos "filas fantasma" o ghost rows. Cada proceso mantiene una copia de la primera y última fila de sus vecinos.

**P2**: En cada iteración, los procesos intercambian estas filas fantasma usando comunicación MPI. Usamos `Sendrecv` para comunicación bidireccional eficiente.

**P1**: [Mostrando código] La función `evolve_local_grid_vectorized` evoluciona solo la porción local, usando las filas fantasma para calcular correctamente los vecinos en los bordes.

**P2**: Al final, reunimos todos los resultados usando `Gatherv`, que combina las porciones locales en la grilla completa.

**P1**: Esta estrategia minimiza la comunicación, ya que solo intercambiamos filas en lugar de toda la grilla.

---

## [3:45 - 4:30] ANÁLISIS DE RESULTADOS

**P2**: Ahora, los resultados. Para grillas pequeñas, como 200×200, el overhead de comunicación hace que la versión paralela sea incluso más lenta que la serial.

**P1**: Esto es esperado. El tiempo de comunicación es proporcional al número de iteraciones, y para grillas pequeñas, este overhead domina.

**P2**: Para grillas medianas, como 1000×1000, vemos un speedup mínimo de aproximadamente 1.01x con 2 procesos.

**P1**: La eficiencia es del 50%, lo cual es razonable, pero el overhead aún limita el beneficio. Con más procesos y grillas más grandes, esperaríamos mejor escalabilidad.

**P2**: Los factores clave que afectan el desempeño son: el tamaño de la grilla, el número de procesos, el número de iteraciones, y la latencia de comunicación.

**P1**: Para grillas muy grandes, como 2000×2000 o más, el tiempo de cómputo domina sobre la comunicación, y ahí es donde la paralelización realmente brilla.

---

## [4:30 - 5:00] CONCLUSIONES

**P2**: En resumen, implementamos exitosamente el Game of Life tanto en versión serial como paralela con MPI.

**P1**: La estrategia de descomposición por dominio funciona bien, pero requiere un balance cuidadoso entre el tamaño del problema y el número de procesos.

**P2**: Para problemas pequeños, el overhead de comunicación puede hacer que la paralelización no valga la pena. Pero para problemas grandes, la escalabilidad mejora significativamente.

**P1**: También aplicamos optimizaciones como operaciones vectorizadas y tipos de datos eficientes, que mejoran el rendimiento en ambas versiones.

**P2**: Y eso es todo. Gracias por ver el video. Si tienen preguntas, pueden revisar el código y la documentación del proyecto.

**P1**: ¡Hasta la próxima!

---

## NOTAS PARA LA GRABACIÓN

### Elementos Visuales Sugeridos:

1. **Pantalla compartida con código**: Mostrar las funciones clave durante la explicación
2. **Diagramas**:
   - Grilla dividida entre procesos (descomposición por dominio)
   - Comunicación de filas fantasma entre procesos
   - Gráfico de speedup vs tamaño de grilla
3. **Tabla de resultados**: Mostrar tiempos de ejecución y métricas
4. **Animación del Game of Life**: Mostrar la evolución de la grilla

### Tono y Estilo:

- Conversacional y natural
- P1 puede ser más técnico (código), P2 más conceptual (explicaciones)
- Pausas naturales para mostrar código/diagramas
- Entusiasmo moderado, profesional pero accesible

### Timing:

- 0:00-0:30: Introducción (30s)
- 0:30-1:30: Algoritmo (60s)
- 1:30-2:30: Versión Serial (60s)
- 2:30-3:45: Versión Paralela (75s)
- 3:45-4:30: Resultados (45s)
- 4:30-5:00: Conclusiones (30s)

**Total: ~5 minutos**

### Puntos Clave a Destacar:

1. ✅ Implementación funcional de ambas versiones
2. ✅ Misma corrección: resultados idénticos (2139 células)
3. ✅ Overhead de comunicación es crítico para grillas pequeñas
4. ✅ Paralelización efectiva requiere problemas grandes o muchos procesos
5. ✅ Optimizaciones (vectorización) mejoran ambas versiones

