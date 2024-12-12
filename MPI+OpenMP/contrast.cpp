#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <omp.h>
#include <mpi.h>

void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);

void set_schedule_openmp();

struct Times {
    double ReadTimeGray;
    double ReadTimeColor;
    double GrayTime;
    double HslTime;
    double YuvTime;
    double WriteTimeGray;
    double WriteTimeHsl;
    double WriteTimeYuv;
    double TotalTime;
};

Times times;

int main(int argc, char *argv[]) {
    PGM_IMG img_ibuf_g; // Estructura para almacenar una imagen en escala de grises
    PPM_IMG img_ibuf_c; // Estructura para almacenar una imagen en color

    // Inicializar MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el rango del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtener el número total de procesos
    set_schedule_openmp(); // Configurar el programador de OpenMP según las variables de entorno

    // Medir el tiempo total de ejecución
    times.TotalTime = MPI_Wtime();

    // Leer la imagen en escala de grises y medir el tiempo necesario
    times.ReadTimeGray = MPI_Wtime();
    img_ibuf_g = read_pgm("in.pgm"); // Leer archivo PGM
    times.ReadTimeGray = MPI_Wtime() - times.ReadTimeGray;

    // Procesar la imagen en escala de grises
    run_cpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g); // Liberar memoria de la imagen en escala de grises

    // Leer la imagen a color y medir el tiempo necesario
    times.ReadTimeColor = MPI_Wtime();
    img_ibuf_c = read_ppm("in.ppm"); // Leer archivo PPM
    times.ReadTimeColor = MPI_Wtime() - times.ReadTimeColor;

    // Procesar la imagen a color
    run_cpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c); // Liberar memoria de la imagen a color

    // Calcular el tiempo total de ejecución
    times.TotalTime = MPI_Wtime() - times.TotalTime;

    // Imprimir estadísticas en el proceso maestro
    if (rank == 0) {
        printf("Processes,Num Threads,ReadGray(s),ReadColor(s),Gray(s),Hsl(s),Yuv(s),WriteGray(s),WriteHsl(s),WriteYuv(s),Total(s)\n");
        printf("%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", size, omp_get_max_threads(), 
               times.ReadTimeGray, times.ReadTimeColor, times.GrayTime, 
               times.HslTime, times.YuvTime, times.WriteTimeGray, 
               times.WriteTimeHsl, times.WriteTimeYuv, times.TotalTime);
    }

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}

void run_cpu_color_test(PPM_IMG img_in) {
    PPM_IMG img_obuf_hsl, img_obuf_yuv; // Imágenes de salida en formato HSL y YUV
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el rango del proceso actual
    
    // Procesar la imagen en espacio de color HSL y medir el tiempo necesario
    times.HslTime = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in); // Mejora de contraste en HSL
    times.HslTime = MPI_Wtime() - times.HslTime;

    // Escribir la imagen HSL procesada en el disco si es el proceso maestro
    if (rank == 0) {
        times.WriteTimeHsl = MPI_Wtime();
        write_ppm(img_obuf_hsl, "out_hsl.ppm"); // Guardar imagen en archivo PPM
        times.WriteTimeHsl = MPI_Wtime() - times.WriteTimeHsl;
        free_ppm(img_obuf_hsl); // Liberar memoria
    }

    // Procesar la imagen en espacio de color YUV y medir el tiempo necesario
    times.YuvTime = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in); // Mejora de contraste en YUV
    times.YuvTime = MPI_Wtime() - times.YuvTime;

    // Escribir la imagen YUV procesada en el disco si es el proceso maestro
    if (rank == 0) {
        times.WriteTimeYuv = MPI_Wtime();
        write_ppm(img_obuf_yuv, "out_yuv.ppm"); // Guardar imagen en archivo PPM
        times.WriteTimeYuv = MPI_Wtime() - times.WriteTimeYuv;
        free_ppm(img_obuf_yuv); // Liberar memoria
    }
}

void get_custom_schedule(const char *schedule_str, const char* chunk_str, omp_sched_t *out_schedule_type, int *out_chunk_size) {
    // Configurar el tipo de programador según el valor proporcionado
    if (strcmp(schedule_str, "static") == 0) {
        *out_schedule_type = omp_sched_static;
    } else if (strcmp(schedule_str, "dynamic") == 0) {
        *out_schedule_type = omp_sched_dynamic;
    } else if (strcmp(schedule_str, "guided") == 0) {
        *out_schedule_type = omp_sched_guided;
    } else if (strcmp(schedule_str, "auto") == 0) {
        *out_schedule_type = omp_sched_auto;
    } else {
        *out_schedule_type = omp_sched_static; // Valor por defecto
    }

    // Convertir el tamaño del chunk de cadena a entero
    *out_chunk_size = atoi(chunk_str);
}

void set_schedule_openmp() {
    // Variables para configurar el programador
    omp_sched_t schedule_type;
    int chunk_size;

    // Obtener valores de las variables de entorno
    const char *schedule_str = getenv("C_OMP_SCHEDULE");
    const char *chunk_str = getenv("C_OMP_CHUNK_SIZE");

    // Usar valores por defecto si las variables de entorno no están definidas
    if (schedule_str == NULL || chunk_str == NULL) {
        schedule_str = "auto";
        chunk_str = "0";
    }

    // Configurar el programador y el tamaño del chunk
    get_custom_schedule(schedule_str, chunk_str, &schedule_type, &chunk_size);
    omp_set_schedule(schedule_type, chunk_size); // Establecer el programador en OpenMP
}

void run_cpu_gray_test(PGM_IMG img_in) {
    PGM_IMG img_obuf; // Imagen de salida para escala de grises
    
    // Procesar la imagen en escala de grises y medir el tiempo necesario
    times.GrayTime = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in); // Mejora de contraste
    times.GrayTime = MPI_Wtime() - times.GrayTime;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el rango del proceso actual

    // Escribir la imagen procesada en el disco si es el proceso maestro
    if (rank == 0) {
        times.WriteTimeGray = MPI_Wtime();
        write_pgm(img_obuf, "out.pgm"); // Guardar imagen en archivo PGM
        times.WriteTimeGray = MPI_Wtime() - times.WriteTimeGray;
        free_pgm(img_obuf); // Liberar memoria
    }
}



PPM_IMG read_ppm(const char * path){
    // Aplicamos paralelización mediante omp en la lectura en imágenes de color
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    // Leemos todos los datos de la imagen desde el archivo.
    // Los datos están organizados como una secuencia de valores RGB intercalados.
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    // Paralelizamos la separación de los canales R, G y B utilizando OpenMP.
    #pragma omp parallel for schedule(runtime)
    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0]; //Extraemos el componente rojo
        result.img_g[i] = ibuf[3*i + 1]; //Extraemos el componente verde
        result.img_b[i] = ibuf[3*i + 2]; //Extraemos el componente azul
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    // Se paraleliza la organización de los datos de los canales R, G y B en un solo buffer intercalado.
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    // Paralelizamos la construcción del buffer intercalado utilizando OpenMP.
    #pragma omp parallel for schedule(runtime)
    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i]; // Canal rojo
        obuf[3*i + 1] = img.img_g[i]; // Canal verde
        obuf[3*i + 2] = img.img_b[i]; // Canal azul
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}
