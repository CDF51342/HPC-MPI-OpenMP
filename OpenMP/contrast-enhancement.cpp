#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result; // Imagen de salida
    int hist[256];  // Histograma para la imagen en escala de grises (256 niveles)

    // Inicializar las dimensiones de la imagen de salida
    result.w = img_in.w;
    result.h = img_in.h;

    // Reservar memoria para la imagen de salida
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    // Calcular el histograma de la imagen de entrada
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);

    // Aplicar ecualización del histograma a la imagen de entrada
    histogram_equalization(result.img, img_in.img, hist, result.w * result.h, 256);

    // Retornar la imagen con contraste mejorado
    return result;
}

PPM_IMG contrast_enhancement_c_rgb(PPM_IMG img_in)
{
    PPM_IMG result; // Imagen de salida
    int hist[256];  // Histograma para cada canal de color (R, G, B)

    // Inicializar las dimensiones de la imagen de salida
    result.w = img_in.w;
    result.h = img_in.h;

    // Reservar memoria para los canales de color de la imagen de salida
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    // Calcular el histograma y aplicar ecualización del histograma para el canal rojo
    histogram(hist, img_in.img_r, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_r, img_in.img_r, hist, result.w * result.h, 256);

    // Repetir el proceso para el canal verde
    histogram(hist, img_in.img_g, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_g, img_in.img_g, hist, result.w * result.h, 256);

    // Repetir el proceso para el canal azul
    histogram(hist, img_in.img_b, img_in.h * img_in.w, 256);
    histogram_equalization(result.img_b, img_in.img_b, hist, result.w * result.h, 256);

    // Retornar la imagen con contraste mejorado para cada canal
    return result;
}


PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;       // Imagen intermedia en espacio de color YUV
    PPM_IMG result;        // Imagen de salida
    unsigned char *y_equ;  // Canal Y (luminancia) ecualizado
    int hist[256];         // Histograma para el canal Y

    // Convertir la imagen de entrada de RGB a YUV
    yuv_med = rgb2yuv(img_in);

    // Reservar memoria para el canal Y ecualizado
    y_equ = (unsigned char *)malloc(yuv_med.h * yuv_med.w * sizeof(unsigned char));

    // Calcular el histograma del canal Y
    histogram(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);

    // Aplicar ecualización del histograma al canal Y
    histogram_equalization(y_equ, yuv_med.img_y, hist, yuv_med.h * yuv_med.w, 256);

    // Reemplazar el canal Y original por el ecualizado
    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;

    // Convertir la imagen intermedia de YUV de regreso a RGB
    result = yuv2rgb(yuv_med);

    // Liberar memoria utilizada por los canales Y, U, y V
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);

    // Retornar la imagen con contraste mejorado
    return result;
}

PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;       // Imagen intermedia en espacio de color HSL
    PPM_IMG result;        // Imagen de salida
    unsigned char *l_equ;  // Canal L (luminancia) ecualizado
    int hist[256];         // Histograma para el canal L

    // Convertir la imagen de entrada de RGB a HSL
    hsl_med = rgb2hsl(img_in);

    // Reservar memoria para el canal L ecualizado
    l_equ = (unsigned char *)malloc(hsl_med.height * hsl_med.width * sizeof(unsigned char));

    // Calcular el histograma del canal L
    histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);

    // Aplicar ecualización del histograma al canal L
    histogram_equalization(l_equ, hsl_med.l, hist, hsl_med.width * hsl_med.height, 256);

    // Reemplazar el canal L original por el ecualizado
    free(hsl_med.l);
    hsl_med.l = l_equ;

    // Convertir la imagen intermedia de HSL de regreso a RGB
    result = hsl2rgb(hsl_med);

    // Liberar memoria utilizada por los canales H, S, y L
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);

    // Retornar la imagen con contraste mejorado
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
        // Si hay diferencia, calculamos Saturación (S) y Tono (H).                                    //Chromatic data...
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
        {
            // Si la saturación es 0, el color es un gris puro.
            // En este caso, todos los canales RGB tienen el mismo valor que L.
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
