#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <mpi.h>
#include <omp.h>


typedef struct {
    double time_test;
    double time_write;
} timeGray;

typedef struct {
    double time_hsl;
    double time_yuv;
    double time_write_hsl;
    double time_write_yuv;
} timeColor;

timeColor run_cpu_color_test(PPM_IMG img_in);
timeGray run_cpu_gray_test(PGM_IMG img_in);

const char *obtain_schedule_string(omp_sched_t schedule_type);
void save_data_csv(const char *planning, const char *process, const char *type, double time, double TotalTime);
void get_custom_schedule(const char *schedule_str, const char* chunk_str, omp_sched_t *out_schedule_type, int *out_chunk_size);


int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;

    // Inicializar MPI para medir el tiempo total y otros aspectos paralelos
    MPI_Init(&argc, &argv);

    // Tomar el tiempo de inicio general
    double tstart = MPI_Wtime();

    // Obtener el número de núcleos disponibles en el sistema
    int cores = omp_get_num_procs();
    printf("Number of cores: %d\n", cores);

    // Procesar imágenes en escala de grises
    printf("Running contrast enhancement for gray-scale images.\n");
    double tstart_read_pgm = MPI_Wtime(); // Tiempo de inicio de lectura PGM
    img_ibuf_g = read_pgm("./TestFiles/in.pgm"); // Leer archivo PGM
    double tend_read_pgm = MPI_Wtime(); // Tiempo al finalizar lectura

    // Ejecutar la mejora de contraste en imágenes en escala de grises
    timeGray t_gray = run_cpu_gray_test(img_ibuf_g);

    // Liberar memoria de la imagen en escala de grises
    free_pgm(img_ibuf_g);
    
    // Procesar imágenes a color
    printf("Running contrast enhancement for color images.\n");
    double tstart_read_ppm = MPI_Wtime(); // Tiempo de inicio de lectura PPM
    img_ibuf_c = read_ppm("./TestFiles/in.ppm"); // Leer archivo PPM
    double tend_read_ppm = MPI_Wtime(); // Tiempo al finalizar lectura

    // Ejecutar la mejora de contraste en imágenes a color
    timeColor time_c = run_cpu_color_test(img_ibuf_c);

    // Liberar memoria de la imagen a color
    free_ppm(img_ibuf_c);
    
    // Tomar el tiempo al finalizar todo el proceso
    double tfinish = MPI_Wtime();
    double TotalTime = tfinish - tstart;
    printf("Total time: %f\n", TotalTime);

    // Finalizar MPI
    MPI_Finalize();

    // Guardar datos de tiempo en un archivo CSV
    save_data_csv("OpenMP", "gray", "read-pgm", tend_read_pgm - tstart_read_pgm, TotalTime);
    save_data_csv("OpenMP", "gray", "G", t_gray.time_test, TotalTime);
    save_data_csv("OpenMP", "gray", "write-pgm", t_gray.time_write, TotalTime);

    save_data_csv("OpenMP", "color", "read-ppm", tend_read_ppm - tstart_read_ppm, TotalTime);
    save_data_csv("OpenMP", "color", "HSL", time_c.time_hsl, TotalTime);
    save_data_csv("OpenMP", "color", "write-HSL", time_c.time_write_hsl, TotalTime);
    save_data_csv("OpenMP", "color", "YUV", time_c.time_yuv, TotalTime);
    save_data_csv("OpenMP", "color", "write-YUV", time_c.time_write_yuv, TotalTime);

    return 0;
}

void get_custom_schedule(const char *schedule_str, const char* chunk_str, omp_sched_t *out_schedule_type, int *out_chunk_size)
{
    // Configura el tipo de planificación (schedule) y el tamaño de chunk para OpenMP
    if (strcmp(schedule_str, "static") == 0) {
        *out_schedule_type = omp_sched_static;
    } else if (strcmp(schedule_str, "dynamic") == 0) {
        *out_schedule_type = omp_sched_dynamic;
    } else if (strcmp(schedule_str, "guided") == 0) {
        *out_schedule_type = omp_sched_guided;
    } else if (strcmp(schedule_str, "auto") == 0) {
        *out_schedule_type = omp_sched_auto;
    } else {
        *out_schedule_type = omp_sched_static;
    }

    *out_chunk_size = atoi(chunk_str);
}

void save_data_csv(const char *planning, const char *process, const char *type, double time, double TotalTime) {
    char line[256], path_csv[256];
    FILE *f_csv;

    // Construir el nombre del archivo CSV
    sprintf(path_csv, "data/%s/%s/time_%s.csv", planning, process, type);

    // Abrir el archivo en modo lectura para verificar su existencia
    f_csv = fopen(path_csv, "r");
    if (f_csv == NULL) {
        // Si no existe, lo abrimos en modo escritura y escribimos la cabecera
        f_csv = fopen(path_csv, "w");
        fprintf(f_csv, "Threads,Schedule,ChunkSize,Time (s),TotalTime\n");
    } else {
        // Si existe, lo cerramos y volvemos a abrir en modo append
        fclose(f_csv);
        f_csv = fopen(path_csv, "a");
    }

    // Obtener información del schedule
    omp_sched_t schedule_type;
    int chunk_size;
    omp_get_schedule(&schedule_type, &chunk_size);
    const char *schedule_name = obtain_schedule_string(schedule_type);

    // Crear la línea de datos y escribirla en el archivo
    sprintf(line, "%d,%s,%d,%f,%f\n", omp_get_max_threads(), schedule_name, chunk_size, time, TotalTime);
    fprintf(f_csv, "%s", line);

    // Cerrar el archivo
    fclose(f_csv);
}

const char *obtain_schedule_string(omp_sched_t schedule_type) {
    // Devuelve el nombre del tipo de planificación en formato de cadena
    switch (schedule_type) {
        case omp_sched_static:
            return "static";
        case omp_sched_dynamic:
            return "dynamic";
        case omp_sched_guided:
            return "guided";
        case omp_sched_auto:
            return "auto";
        default:
            return "unknown"; // Tipo desconocido
    }
}

timeColor run_cpu_color_test(PPM_IMG img_in) {
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    timeColor times;
    
    printf("Starting CPU processing...\n");
    
    // Procesar imagen en espacio de color HSL
    double tstart = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    double tfinish = MPI_Wtime();
    printf("HSL processing time: %f (s)\n", tfinish - tstart);
    times.time_hsl = tfinish - tstart;

    // Guardar imagen procesada en HSL
    tstart = MPI_Wtime();
    write_ppm(img_obuf_hsl, "out_hsl.ppm");
    tfinish = MPI_Wtime();
    times.time_write_hsl = tfinish - tstart;

    // Procesar imagen en espacio de color YUV
    tstart = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    tfinish = MPI_Wtime();
    printf("YUV processing time: %f (s)\n", tfinish - tstart);
    times.time_yuv = tfinish - tstart;

    // Guardar imagen procesada en YUV
    tstart = MPI_Wtime();
    write_ppm(img_obuf_yuv, "out_yuv.ppm");
    tfinish = MPI_Wtime();
    times.time_write_yuv = tfinish - tstart;

    // Liberar memoria de las imágenes procesadas
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);

    return times;
}


void set_schedule_openmp(int size) {
    // Configurar la planificación de OpenMP basada en variables de entorno
    omp_sched_t schedule_type;
    int chunk_size;

    const char *schedule_str = getenv("C_OMP_SCHEDULE");
    const char *chunk_str = getenv("C_OMP_CHUNK_SIZE");

    if (schedule_str == NULL || chunk_str == NULL) {
        schedule_str = "auto";
        chunk_str = "0";
    }

    // Se comprueba que tipo de schedule para optimizar el chunk size en función al tamaño de la imagen
    // if (strcmp("static", schedule_str) == 0) {
    //     const char *k = getenv("C_OMP_K");
    //     if (k == NULL)
    //         k = "2";
    //     chunk_size = size / (atoi(k) * omp_get_max_threads());
    //     omp_set_schedule(omp_sched_static, chunk_size);
    // }
    // else if (strcmp("dynamic", schedule_str) == 0) {
    //     chunk_size = size / (10 * omp_get_max_threads());
    //     omp_set_schedule(omp_sched_dynamic, chunk_size);
    // }
    // else if (strcmp("guided", schedule_str) == 0) {
    //     chunk_size = size / (10 * omp_get_max_threads());
    //     omp_set_schedule(omp_sched_guided, chunk_size);
    // }
    // else {
    //     const char *chunk_str = getenv("C_OMP_CHUNK_SIZE");
    //     if (chunk_str == NULL) {
    //         chunk_str = "0";
    //     }

    //     get_custom_schedule(schedule_str, chunk_str, &schedule_type, &chunk_size);
    //     omp_set_schedule(schedule_type, chunk_size);
    // }

    get_custom_schedule(schedule_str, chunk_str, &schedule_type, &chunk_size);
    omp_set_schedule(schedule_type, chunk_size);
}

timeGray run_cpu_gray_test(PGM_IMG img_in) {
    PGM_IMG img_obuf;  
    timeGray t_gray;

    // Configurar planificación para OpenMP basada en la imagen
    set_schedule_openmp(img_in.w * img_in.h);

    printf("Starting CPU processing...\n");
    
    // Procesar imagen en escala de grises
    double tstart = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in);
    double tfinish = MPI_Wtime();
    t_gray.time_test = tfinish - tstart;

    printf("Processing time: %f (s)\n", t_gray.time_test);

    // Guardar imagen procesada
    tstart = MPI_Wtime();
    write_pgm(img_obuf, "out.pgm");
    tfinish = MPI_Wtime();
    t_gray.time_write = tfinish - tstart;

    free_pgm(img_obuf); // Liberar memoria de la imagen procesada

    return t_gray;
}



PPM_IMG read_ppm(const char * path){
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
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    #pragma omp parallel for schedule(runtime)
    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    #pragma omp parallel for schedule(runtime)
    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
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
    printf("Image size: %d x %d\n", result.w, result.h);
    

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

