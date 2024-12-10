#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin, int full_img_size) {
    // `img_out`: Puntero a la imagen de salida (ecualizada)
    // `img_in`: Puntero a la imagen de entrada
    // `hist_in`: Histograma de la imagen de entrada
    // `img_size`: Tamaño de la imagen local (procesada por este proceso)
    // `nbr_bin`: Número de niveles en el histograma (generalmente 256 para imágenes en escala de grises)
    // `full_img_size`: Tamaño total de la imagen (toda la imagen, incluyendo la parte procesada por otros procesos)

    int *lut = (int *)malloc(sizeof(int) * nbr_bin); // Crear la tabla de búsqueda (LUT) para mapear intensidades
    int i, cdf, min, d; // Variables auxiliares
    /* 
     * `cdf`: Acumulador para la función de distribución acumulativa (CDF).
     * `min`: Mínima frecuencia no nula en el histograma.
     * `d`: Diferencia entre el tamaño total de la imagen y el mínimo (usado para normalización).
     */

    // Inicialización de variables
    cdf = 0; // Inicializar el acumulador CDF
    min = 0; // Buscar la primera frecuencia no nula en el histograma
    i = 0;

    // Encontrar el primer valor del histograma distinto de cero (mínimo no nulo)
    while (min == 0) {
        min = hist_in[i++];
    }

    // Calcular el denominador para normalizar la LUT
    d = full_img_size - min;

    // Construir la tabla de búsqueda (LUT) calculando el CDF
    for (i = 0; i < nbr_bin; i++) {
        cdf += hist_in[i]; // Incrementar el CDF con la frecuencia actual del histograma

        // Mapear el CDF al rango de intensidad de salida [0, 255]
        lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5); // Normalización al rango [0, 255]

        // Asegurarse de que los valores de la LUT no sean negativos
        if (lut[i] < 0) {
            lut[i] = 0;
        }
    }

    /* Generar la imagen de salida usando la LUT */
    #pragma omp parallel for schedule(runtime) // Usar OpenMP para paralelizar el bucle
    for (i = 0; i < img_size; i++) {
        // Asignar el valor de la LUT a cada píxel de la imagen de salida
        if (lut[img_in[i]] > 255) {
            img_out[i] = 255; // Limitar valores a 255 si exceden el rango
        } else {
            img_out[i] = (unsigned char)lut[img_in[i]]; // Asignar el valor mapeado
        }
    }

    // Liberar la memoria reservada para la LUT
    free(lut);
}



