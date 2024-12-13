#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <mpi.h>

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist_local[256];
    int global_hist[256];

    result.w = img_in.w;
    result.h = img_in.h;

    // Inicializamos MPI
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Extraemos las dimensiones que necesitamos
    int local_width = img_in.w;
    int local_height = img_in.h / size;
    int remainder = img_in.h % size; // Resto de divisiones desiguales
    int local_size = local_width * local_height;

    // Ajustamos el tamaño para el último proceso
    if (rank == size - 1) {
        local_height += remainder;
        local_size = local_width * local_height;
    }

    // Asignamos la memoria para las partes locales de la imagen
    unsigned char *img_local = (unsigned char *)malloc(local_size * sizeof(unsigned char));

    // Dividimos la imagen entre procesos
    // Utilizamos MPI_Scatterv debido a que no tenemos tamaños iguales
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (img_in.h / size) * img_in.w;
        if (i == size - 1) {
            sendcounts[i] += remainder * img_in.w;
        }
        displs[i] = i * (img_in.h / size) * img_in.w;
    }

    MPI_Scatterv(img_in.img, sendcounts, displs, MPI_UNSIGNED_CHAR, img_local, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Calculamos el histograma local
    histogram(hist_local, img_local, local_size, 256);

    // Combinamos los histogramas de todos los procesos en un histograma global
    MPI_Allreduce(hist_local, global_hist, 256, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Aplicamos la ecualización del histograma localmente
    unsigned char *img_local_out = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    histogram_equalization(img_local_out, img_local, global_hist, local_size, 256, img_in.w * img_in.h);

    // Solo el proceso 0 tiene la imagen final
    if (rank == 0) {
        result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    // Recolectamos los datos procesados de todos los procesos
    MPI_Gatherv(img_local_out, local_size, MPI_UNSIGNED_CHAR, result.img, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Liberamos memoria
    free(img_local);
    free(img_local_out);
    free(sendcounts);
    free(displs);

    return result;
}

PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in) {
    // Estructuras para manejar imágenes
    YUV_IMG local_yuv_med;  // Imagen en espacio de color YUV (local)
    PPM_IMG local_result;   // Resultado del procesamiento local
    PPM_IMG result;         // Resultado final que será reunido por el proceso maestro
    PPM_IMG local_img_in;   // Imagen local procesada en cada proceso

    // Variables para histogramas y cálculos
    unsigned char *y_equ;   // Canal de luminancia ajustado (igualado)
    int localHist[256];     // Histograma local
    int globalHist[256];    // Histograma global (combinado)

    // Información sobre los procesos MPI
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rango del proceso actual

    // Dividir la imagen en segmentos para los procesos
    int local_width = img_in.w;            // Ancho de la imagen (igual para todos los procesos)
    int local_height = img_in.h / size;    // Altura dividida entre procesos
    int remainder = img_in.h % size;       // Resto de la división
    int local_size = local_width * local_height; // Tamaño del segmento local

    // Ajustar la altura para el último proceso (manejo del resto)
    if (rank == size - 1) {
        local_height += remainder;        // Añadir el resto a la altura local
        local_size = local_width * local_height;
    }

    // Asignar memoria para las partes locales de la imagen
    local_img_in.img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.w = local_width;
    local_img_in.h = local_height;

    // Preparar las estructuras para Scatterv (distribuir partes de la imagen)
    int *sendcounts = (int *)malloc(size * sizeof(int)); // Tamaños de los datos enviados
    int *displs = (int *)malloc(size * sizeof(int));     // Desplazamientos en el buffer de entrada
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (img_in.h / size) * img_in.w;    // Tamaño estándar para cada proceso
        if (i == size - 1) {
            sendcounts[i] += remainder * img_in.w;      // Ajustar para el último proceso
        }
        displs[i] = i * (img_in.h / size) * img_in.w;   // Desplazamiento para cada proceso
    }

    // Distribuir los canales R, G y B de la imagen de entrada
    MPI_Scatterv(img_in.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Convertir la imagen local de RGB a YUV
    local_yuv_med = rgb2yuv(local_img_in);

    // Asignar memoria para la luminancia ajustada
    y_equ = (unsigned char *)malloc(local_yuv_med.h * local_yuv_med.w * sizeof(unsigned char));

    // Calcular el histograma local para el canal de luminancia (Y)
    histogram(localHist, local_yuv_med.img_y, local_yuv_med.h * local_yuv_med.w, 256);

    // Reducir (sumar) los histogramas locales en un histograma global
    MPI_Allreduce(localHist, globalHist, 256, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Realizar la igualación del histograma en el canal de luminancia
    histogram_equalization(y_equ, local_yuv_med.img_y, globalHist, local_yuv_med.h * local_yuv_med.w, 256, img_in.h * img_in.w);

    // Reemplazar el canal de luminancia en la imagen YUV con la versión igualada
    free(local_yuv_med.img_y);
    local_yuv_med.img_y = y_equ;

    // Convertir la imagen YUV de nuevo a RGB
    local_result = yuv2rgb(local_yuv_med);

    // Liberar los canales U, V e Y de la imagen YUV local
    free(local_yuv_med.img_u);
    free(local_yuv_med.img_v);
    free(local_yuv_med.img_y);

    // Asignar memoria para la imagen final en el proceso maestro (rank 0)
    if (rank == 0) {
        result.w = img_in.w;
        result.h = img_in.h;
        result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    // Recolectar las partes procesadas de cada proceso en la imagen final
    MPI_Gatherv(local_result.img_r, local_size, MPI_UNSIGNED_CHAR, result.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_g, local_size, MPI_UNSIGNED_CHAR, result.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_b, local_size, MPI_UNSIGNED_CHAR, result.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Liberar memoria utilizada en cada proceso
    free(local_result.img_r);
    free(local_result.img_g);
    free(local_result.img_b);
    free(local_img_in.img_r);
    free(local_img_in.img_g);
    free(local_img_in.img_b);
    free(sendcounts);
    free(displs);

    // Devolver el resultado final (solo el proceso maestro tendrá los datos completos)
    return result;
}

PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in) {
    // Estructuras para manejar imágenes
    HSL_IMG local_hsl_med; // Imagen en espacio de color HSL (local)
    PPM_IMG local_result;  // Resultado del procesamiento local
    PPM_IMG result;        // Resultado final que será reunido por el proceso maestro
    PPM_IMG local_img_in;  // Imagen local procesada en cada proceso

    // Variables para histogramas y otros cálculos
    unsigned char *l_equ;   // Canal de luminancia ajustado (igualado)
    int localHist[256];     // Histograma local
    int globalHist[256];    // Histograma global

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rango del proceso actual

    // Dimensiones de la imagen local
    int local_width = img_in.w;
    int local_height = img_in.h / size; // Dividir la altura entre los procesos
    int remainder = img_in.h % size;    // Resto de la división para el último proceso
    int local_size = local_width * local_height;

    // Ajustar la altura local para el último proceso (manejo del resto)
    if (rank == size - 1) {
        local_height += remainder;
        local_size = local_width * local_height;
    }

    // Asignar memoria para las partes locales de la imagen
    local_img_in.img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.w = local_width;
    local_img_in.h = local_height;

    // Preparar las estructuras para Scatterv (distribuir partes de la imagen entre procesos)
    int *sendcounts = (int *)malloc(size * sizeof(int)); // Tamaño de los datos enviados a cada proceso
    int *displs = (int *)malloc(size * sizeof(int));     // Desplazamientos en el buffer de origen
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (img_in.h / size) * img_in.w; // Tamaño estándar
        if (i == size - 1) {
            sendcounts[i] += remainder * img_in.w; // Agregar el resto al último proceso
        }
        displs[i] = i * (img_in.h / size) * img_in.w; // Desplazamiento de cada segmento
    }

    // Distribuir los canales R, G y B de la imagen de entrada entre los procesos
    MPI_Scatterv(img_in.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Convertir la imagen local de RGB a HSL
    local_hsl_med = rgb2hsl(local_img_in);

    // Asignar memoria para la luminancia ajustada
    l_equ = (unsigned char *)malloc(local_hsl_med.height * local_hsl_med.width * sizeof(unsigned char));

    // Calcular el histograma local de la luminancia
    histogram(localHist, local_hsl_med.l, local_hsl_med.height * local_hsl_med.width, 256);

    // Reducir (sumar) los histogramas locales en un histograma global
    MPI_Allreduce(localHist, globalHist, 256, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Realizar la igualación del histograma en el canal de luminancia
    histogram_equalization(l_equ, local_hsl_med.l, globalHist, local_hsl_med.height * local_hsl_med.width, 256, img_in.h * img_in.w);

    // Reemplazar el canal de luminancia en la imagen HSL con la versión igualada
    free(local_hsl_med.l);
    local_hsl_med.l = l_equ;

    // Convertir la imagen HSL de nuevo a RGB
    local_result = hsl2rgb(local_hsl_med);

    // Liberar los canales de HSL ya no necesarios
    free(local_hsl_med.h);
    free(local_hsl_med.s);
    free(local_hsl_med.l);

    // Asignar memoria para la imagen final en el proceso maestro (rank 0)
    if (rank == 0) {
        result.w = img_in.w;
        result.h = img_in.h;
        result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    // Recolectar las partes procesadas de cada proceso en la imagen final
    MPI_Gatherv(local_result.img_r, local_size, MPI_UNSIGNED_CHAR, result.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_g, local_size, MPI_UNSIGNED_CHAR, result.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_b, local_size, MPI_UNSIGNED_CHAR, result.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Liberar memoria local utilizada en cada proceso
    free(local_result.img_r);
    free(local_result.img_g);
    free(local_result.img_b);
    free(local_img_in.img_r);
    free(local_img_in.img_g);
    free(local_img_in.img_b);
    free(sendcounts);
    free(displs);

    // Devolver el resultado final al proceso maestro
    return result;
}

//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG rgb2hsl(PPM_IMG img_in)
{
    int i;
    float H, S, L;
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    // Inicia la paralelización del bucle for utilizando OpenMP. 
    // 1. `private(H, S, L)`: Cada hilo tendrá sus propias copias de las variables H, S y L, evitando conflictos entre hilos.
    // 2. `schedule(runtime)`: El esquema de distribución de las iteraciones se puede configurar en tiempo de ejecución 
    //    mediante la variable de entorno OMP_SCHEDULE (por ejemplo, static, dynamic, etc.).
    #pragma omp parallel for private(H, S, L) schedule(runtime)
    for(i = 0; i < img_in.w*img_in.h; i ++){
        
        float var_r = ( (float)img_in.img_r[i]/255 );//Convertimos RGB a [0,1]
        float var_g = ( (float)img_in.img_g[i]/255 );
        float var_b = ( (float)img_in.img_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //mínimo de RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //máximo de RGB
        float del_max = var_max - var_min;               //Valor Delta de RGB
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )
        // Si no hay diferencia entre el máximo y mínimo, significa que el color es un gris puro.
        {
            H = 0;         
            S = 0;    
        }
        else   
        // Si hay diferencia, calculamos Saturación (S) y Tono (H).                                 
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            // Calculamos las diferencias relativas de cada componente RGB.
            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }

        // Ajustamos el rango de H para que esté en [0,1].

        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        // Asignamos los valores calculados a la estructura de salida.

        img_out.h[i] = H;
        img_out.s[i] = S;
        img_out.l[i] = (unsigned char)(L*255);
    }
    
    return img_out;
}

float Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
PPM_IMG hsl2rgb(HSL_IMG img_in)
{
    int i;
    PPM_IMG result;
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    

    // Este pragma paraleliza el bucle for con OpenMP. Cada iteración es independiente,
    // por lo que se puede ejecutar en paralelo para mejorar el rendimiento.
    // `schedule(runtime)` permite ajustar dinámicamente cómo se distribuyen las iteraciones.
    #pragma omp parallel for schedule(runtime)
    for(i = 0; i < img_in.width*img_in.height; i ++){
        float H = img_in.h[i];
        float S = img_in.s[i];
        float L = img_in.l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        // Si la saturación es 0, el color es un gris puro.
        // En este caso, todos los canales RGB tienen el mismo valor que L.
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            // Convertimos el tono (H) y las variables intermedias a valores RGB usando Hue_2_RGB.
            // `Hue_2_RGB` es una función auxiliar que calcula los valores RGB
            // a partir de las variables `var_1`, `var_2` y el tono modificado.
            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * Hue_2_RGB( var_1, var_2, H );
            b = 255 * Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        result.img_r[i] = r;
        result.img_g[i] = g;
        result.img_b[i] = b;
    }

    return result;
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);


    // Paralelizamos el bucle con OpenMP:
    // - `private(r, g, b, y, cb, cr)`: Cada hilo tiene su propia copia de estas variables temporales,
    //   evitando conflictos durante las operaciones.
    // - `schedule(runtime)`: Permite configurar el esquema de planificación en tiempo de ejecución
    //   (por ejemplo, estático o dinámico) mediante la variable OMP_SCHEDULE.
    #pragma omp parallel for private(r, g, b, y, cb, cr) schedule(runtime)
    for(i = 0; i < img_out.w*img_out.h; i ++){
        // Leemos los valores RGB del píxel actual.
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        // Convertimos de RGB a YUV utilizando las fórmulas estándar.
        // Y: Luminancia (representa el brillo del píxel)
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        // U (Cb): Componente de crominancia azul
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        // V (Cr): Componente de crominancia roja.
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }
    
    return img_out;
}

unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int  rt,gt,bt;
    int y, cb, cr;
    
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // Paralelizamos el bucle con OpenMP:
    // - `private(y, cb, cr, rt, gt, bt)`: Cada hilo tiene su propia copia de estas variables temporales.
    // - `schedule(runtime)`: Permite configurar el esquema de planificación en tiempo de ejecución.
    #pragma omp parallel for private(y, cb, cr, rt, gt, bt) schedule(runtime)
    for(i = 0; i < img_out.w*img_out.h; i ++){
        // Leemos los valores Y, U (Cb) y V (Cr) del píxel actual.
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;
        
        // Convertimos de YUV a RGB utilizando las fórmulas estándar.
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        // Limitamos los valores RGB al rango válido [0, 255] usando `clip_rgb`.
        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }
    
    return img_out;
}
