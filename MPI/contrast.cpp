#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <mpi.h>

void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);

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

    // Inicializar el entorno de MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el identificador del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtener el número total de procesos

    // Iniciar el cronómetro general
    times.TotalTime = MPI_Wtime();

    // Leer la imagen en escala de grises y medir el tiempo que toma
    times.ReadTimeGray = MPI_Wtime();
    img_ibuf_g = read_pgm("in.pgm");
    times.ReadTimeGray = MPI_Wtime() - times.ReadTimeGray;

    // Realizar el procesamiento en escala de grises
    run_cpu_gray_test(img_ibuf_g);
    free_pgm(img_ibuf_g); // Liberar memoria utilizada por la imagen en escala de grises

    // Leer la imagen en color y medir el tiempo que toma
    times.ReadTimeColor = MPI_Wtime();
    img_ibuf_c = read_ppm("in.ppm");
    times.ReadTimeColor = MPI_Wtime() - times.ReadTimeColor;

    // Realizar el procesamiento en color
    run_cpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c); // Liberar memoria utilizada por la imagen en color

    // Finalizar el cronómetro general
    times.TotalTime = MPI_Wtime() - times.TotalTime;

    // El proceso con rank 0 escribe los resultados en la consola
    if (rank == 0) {
        printf("Processes,ReadGray(s),ReadColor(s),Gray(s),Hsl(s),Yuv(s),WriteGray(s),WriteHsl(s),WriteYuv(s),Total(s)\n");
        printf("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", size, times.ReadTimeGray, times.ReadTimeColor, times.GrayTime, times.HslTime, times.YuvTime, times.WriteTimeGray, times.WriteTimeHsl, times.WriteTimeYuv, times.TotalTime);
    }

    // Finalizar el entorno de MPI
    MPI_Finalize();
    return 0;
}

// Función para procesar imágenes en color
void run_cpu_color_test(PPM_IMG img_in) {
    PPM_IMG img_obuf_hsl, img_obuf_yuv; // Buffers para las imágenes procesadas en los espacios de color HSL y YUV
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el identificador del proceso actual

    // Procesar la imagen en el espacio de color HSL y medir el tiempo que toma
    times.HslTime = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    times.HslTime = MPI_Wtime() - times.HslTime;

    // Si el proceso actual es el maestro (rank 0), escribir la imagen procesada en HSL a un archivo
    if (rank == 0) {
        times.WriteTimeHsl = MPI_Wtime();
        write_ppm(img_obuf_hsl, "out_hsl.ppm");
        times.WriteTimeHsl = MPI_Wtime() - times.WriteTimeHsl;
        free_ppm(img_obuf_hsl); // Liberar memoria utilizada por la imagen procesada
    }

    // Procesar la imagen en el espacio de color YUV y medir el tiempo que toma
    times.YuvTime = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    times.YuvTime = MPI_Wtime() - times.YuvTime;

    // Si el proceso actual es el maestro (rank 0), escribir la imagen procesada en YUV a un archivo
    if (rank == 0) {
        times.WriteTimeYuv = MPI_Wtime();
        write_ppm(img_obuf_yuv, "out_yuv.ppm");
        times.WriteTimeYuv = MPI_Wtime() - times.WriteTimeYuv;
        free_ppm(img_obuf_yuv); // Liberar memoria utilizada por la imagen procesada
    }
}

// Función para procesar imágenes en escala de grises
void run_cpu_gray_test(PGM_IMG img_in) {
    PGM_IMG img_obuf; // Buffer para la imagen procesada

    // Procesar la imagen en escala de grises y medir el tiempo que toma
    times.GrayTime = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in);
    times.GrayTime = MPI_Wtime() - times.GrayTime;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtener el identificador del proceso actual

    // Si el proceso actual es el maestro (rank 0), escribir la imagen procesada a un archivo
    if (rank == 0) {
        times.WriteTimeGray = MPI_Wtime();
        write_pgm(img_obuf, "out.pgm");
        times.WriteTimeGray = MPI_Wtime() - times.WriteTimeGray;
        free_pgm(img_obuf); // Liberar memoria utilizada por la imagen procesada
    }
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
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

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

