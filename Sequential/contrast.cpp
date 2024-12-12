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

void save_data_csv(const char *planning, const char *process, const char *type, double time, double TotalTime);


int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;

    //Initialize MPI
    MPI_Init(&argc, &argv);

    double tstart = MPI_Wtime();

    printf("Running contrast enhancement for gray-scale images.\n");
    double tstart_read_pgm = MPI_Wtime();
    img_ibuf_g = read_pgm("in.pgm");
    double tend_read_pgm = MPI_Wtime();

    timeGray t_gray = run_cpu_gray_test(img_ibuf_g);

    free_pgm(img_ibuf_g);
    
    printf("Running contrast enhancement for color images.\n");
    double tstart_read_ppm = MPI_Wtime();
    img_ibuf_c = read_ppm("in.ppm");
    double tend_read_ppm = MPI_Wtime();

    timeColor time_c = run_cpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    double tfinish = MPI_Wtime();
    double TotalTime = tfinish - tstart;
    printf("Total time: %f\n", TotalTime);

    //Finalize MPI
    MPI_Finalize();

    // Save data time in csv
    save_data_csv("Sequential", "gray", "read-pgm", tend_read_pgm - tstart_read_pgm, TotalTime);
    save_data_csv("Sequential", "gray", "G", t_gray.time_test, TotalTime);
    save_data_csv("Sequential", "gray", "write-pgm", t_gray.time_write, TotalTime);

    save_data_csv("Sequential", "color", "read-ppm", tend_read_ppm - tstart_read_ppm, TotalTime);
    save_data_csv("Sequential", "color", "HSL", time_c.time_hsl, TotalTime);
    save_data_csv("Sequential", "color", "write-HSL", time_c.time_write_hsl, TotalTime);
    save_data_csv("Sequential", "color", "YUV", time_c.time_yuv, TotalTime);
    save_data_csv("Sequential", "color", "write-YUV", time_c.time_write_yuv, TotalTime);

    return 0;
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
        fprintf(f_csv, "Time (s),TotalTime\n");
    } else {
        // Si existe, lo cerramos y volvemos a abrir en modo append
        fclose(f_csv);
        f_csv = fopen(path_csv, "a");
    }

    // Crear la línea de datos y escribirla en el archivo
    sprintf(line, "%f,%f\n", time, TotalTime);
    fprintf(f_csv, "%s", line);

    // Cerrar el archivo
    fclose(f_csv);
}

timeColor run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    timeColor times;
    
    printf("Starting CPU processing...\n");
    
    double tstart = MPI_Wtime();
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    double tfinish = MPI_Wtime();
    printf("HSL processing time: %f (s)\n", tfinish - tstart);
    times.time_hsl = tfinish - tstart;
    
    tstart = MPI_Wtime();
    write_ppm(img_obuf_hsl, "out_hsl.ppm");
    tfinish = MPI_Wtime();
    times.time_write_hsl = tfinish - tstart;

    tstart = MPI_Wtime();
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    tfinish = MPI_Wtime();
    printf("YUV processing time: %f (s)\n", tfinish - tstart);
    times.time_yuv = tfinish - tstart;
    
    tstart = MPI_Wtime();
    write_ppm(img_obuf_yuv, "out_yuv.ppm");
    tfinish = MPI_Wtime();
    times.time_write_yuv = tfinish - tstart;
    
    free_ppm(img_obuf_hsl);
    free_ppm(img_obuf_yuv);

    return times;
}




timeGray run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;  
    timeGray t_gray;

    printf("Starting CPU processing...\n");
    
    double tstart = MPI_Wtime();
    img_obuf = contrast_enhancement_g(img_in);
    double tfinish = MPI_Wtime();
    t_gray.time_test = tfinish - tstart;

    printf("Processing time: %f (s)\n", t_gray.time_test);
    
    tstart = MPI_Wtime();
    write_pgm(img_obuf, "out.pgm");
    tfinish = MPI_Wtime();
    t_gray.time_write = tfinish - tstart;
    free_pgm(img_obuf);

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

