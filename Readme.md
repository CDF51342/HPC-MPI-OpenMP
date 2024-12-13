# Paralelización de Código con MPI y OpenMP

Este proyecto implementa una solución de mejora de contraste de imágenes utilizando técnicas de paralelización con **MPI**, **OpenMP** y una versión híbrida que combina ambas herramientas. El objetivo principal es optimizar el rendimiento computacional del procesamiento de imágenes mediante la ecualización de histograma.

---

## Estructura del Proyecto

El proyecto está organizado en los siguientes directorios principales:

```plaintext
Root/
├── Sequential/         # Versión secuencial
├── OpenMP/             # Versión paralela con OpenMP
├── MPI/                # Versión paralela con MPI
├── MPI+OpenMP/         # Versión híbrida MPI + OpenMP
├── data/               # Datos recopilados de las diferentes versiones
├── results_assets/     # Tablas y gráficas generadas
├── generateInputFiles.sh # Script para convertir imágenes
├── obtainData.sh       # Script para obtener datos de ejecución
├── runTest.sh          # Script para verificar consistencia de resultados
├── graphics.ipynb      # Notebook para generar tablas y gráficas
└── Makefile            # Compilación de todas las versiones
```

Cada directorio incluye los archivos fuente (`contrast.cpp`, `contrast-enhancement.cpp`, `histogram-equalization.cpp`) y los correspondientes `CMakeLists.txt` para la compilación.

---

## Instalación y Ejecución

### Compilación
El proyecto utiliza un archivo `Makefile` para compilar todas las versiones:

1. Clonar el repositorio.
2. Navegar a la raíz del proyecto y ejecutar:
   ```bash
   make
   ```

Esto generará los ejecutables en la raíz del proyecto:
- `contrast_seq` (Versión secuencial)
- `contrast_omp` (Versión OpenMP)
- `contrast_mpi` (Versión MPI)
- `contrast_mpi_omp` (Versión híbrida MPI + OpenMP)

### Ejecución
Para ejecutar las versiones compiladas en avignon:

1. Añadir las imágenes de prueba `in.pgm` e `in.ppm` a la raíz del proyecto.
2. Ejecutar la versión deseada. Por ejemplo:
   ```bash
   srun -p gpus -N <Nodos> -n <procesos> ./contrast_omp
   ```

### Uso de Scripts
- `generateInputFiles.sh`: Convierte imágenes en formato `.jpg` a `.pgm` y `.ppm`.
- `obtainData.sh`: Obtiene los datos de ejecución en formato CSV para todas las versiones.
- `runTests.sh`: Verifica la consistencia de los resultados entre las versiones paralelas y la secuencial.

---

## Configuración de Parámetros

### OpenMP
- Configurar el número de hilos:
  ```bash
  export OMP_NUM_THREADS=<número_de_hilos>
  ```
- Configurar la planificación y tamaño de bloque:
  ```bash
  export C_OMP_SCHEDULE=<static|dynamic|guided>
  export C_OMP_CHUNK_SIZE=<tamaño_de_bloque>
  ```

### MPI
- Establecer el número de procesos y nodos en la ejecución:
  ```bash
  mpirun -np <número_de_procesos> ./contrast_mpi
  ```

### Versión Híbrida
Combina las configuraciones de OpenMP y MPI. Ejemplo:
```bash
mpirun -np <número_de_procesos> ./contrast_mpi_omp
```

---

## Resultados

Se han evaluado las siguientes métricas:
- Tiempo de ejecución total.
- Speedup relativo a la versión secuencial.
- Eficiencia en la distribución de tareas.


## Autores
- Raúl Ágreda García 100451269
- Diego Calvo Engelmo 100451091
- Carlos Díez Fenoy 100451342
- Jose Francisco Olivert Iserte 100532427
---