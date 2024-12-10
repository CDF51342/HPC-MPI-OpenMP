#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <omp.h>


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
                            int * hist_in, int img_size, int nbr_bin){
    // Reservamos memoria para la tabla de búsqueda (LUT - Look-Up Table)
    int *lut = (int *)malloc(sizeof(int) * nbr_bin);

    // Variables auxiliares
    int i, cdf, min, d;

    /* Construir la LUT calculando la CDF (Función de Distribución Acumulada) */

    cdf = 0;   // Inicializamos la CDF
    min = 0;   // Almacena el valor mínimo del histograma (excluyendo ceros)
    i = 0;

    // Determinamos el primer valor no nulo en el histograma
    while(min == 0) {
        min = hist_in[i++]; 
    }

    // Calculamos el denominador para la ecualización: diferencia entre el tamaño
    // total de la imagen (número de píxeles) y el valor mínimo del histograma
    d = img_size - min;

    // Calculamos la LUT (tabla de transformación) basada en el histograma de entrada
    for(i = 0; i < nbr_bin; i++) {
        cdf += hist_in[i]; // Acumulamos el valor del histograma actual

        // Aplicamos la fórmula de ecualización de histograma a cada valor
        lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5); 

        // Aseguramos que los valores de la LUT sean válidos (no negativos)
        if(lut[i] < 0) {
            lut[i] = 0;
        } 
    }

    /* Generamos la imagen de salida usando la LUT */

    // Paralelizamos este bucle para mejorar el rendimiento usando OpenMP
    #pragma omp parallel for schedule(runtime)
    for(i = 0; i < img_size; i++) {
        // Si el valor en la LUT excede 255, limitamos a 255
        if(lut[img_in[i]] > 255) {
            img_out[i] = 255;
        } 
        // De lo contrario, asignamos el valor de la LUT al píxel de salida
        else {
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
    }

    // Liberamos la memoria asignada a la LUT
    free(lut);
}
