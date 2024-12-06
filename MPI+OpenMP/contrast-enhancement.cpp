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
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned char *img_local = (unsigned char *)malloc(result.w * result.h / size * sizeof(unsigned char));
    MPI_Scatter(img_in.img, result.w * result.h / size, MPI_UNSIGNED_CHAR, img_local, result.w * result.h / size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    histogram(hist_local, img_local, result.w * result.h / size, 256);
    MPI_Allreduce(hist_local, global_hist, 256, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    unsigned char *img_local_out = (unsigned char *)malloc(result.w * result.h / size * sizeof(unsigned char));
    histogram_equalization(img_local_out, img_local, global_hist, result.w * result.h / size, 256, result.w * result.h);

    // Solo el proceso 0 tiene la imagen final
    if (rank == 0)
        result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    MPI_Gather(img_local_out, result.w * result.h / size, MPI_UNSIGNED_CHAR, result.img, result.w * result.h / size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    free(img_local);
    free(img_local_out);
    return result;
}

PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    YUV_IMG local_yuv_med;
    PPM_IMG local_result;
    PPM_IMG result;
    PPM_IMG local_img_in;
    
    unsigned char * y_equ;
    int localHist[256];
    int globalHist[256];

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_width = img_in.w;
    int local_height = img_in.h / size;
    int remainder = img_in.h % size;
    int local_size = local_width * local_height;

    // Adjust the local height for the last process to handle the remainder
    if (rank == size - 1) {
        local_height += remainder;
        local_size = local_width * local_height;
    }

    local_img_in.img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.w = local_width;
    local_img_in.h = local_height;

    // Scatter the image data
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (img_in.h / size) * img_in.w;
        if (i == size - 1) {
            sendcounts[i] += remainder * img_in.w;
        }
        displs[i] = i * (img_in.h / size) * img_in.w;
    }

    MPI_Scatterv(img_in.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    local_yuv_med = rgb2yuv(local_img_in);
    y_equ = (unsigned char *)malloc(local_yuv_med.h * local_yuv_med.w * sizeof(unsigned char));

    histogram(localHist, local_yuv_med.img_y, local_yuv_med.h * local_yuv_med.w, 256);
    MPI_Allreduce(localHist, globalHist, 256, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    histogram_equalization(y_equ, local_yuv_med.img_y, globalHist, local_yuv_med.h * local_yuv_med.w, 256, img_in.h * img_in.w);
    
    free(local_yuv_med.img_y);
    local_yuv_med.img_y = y_equ;

    local_result = yuv2rgb(local_yuv_med);
    free(local_yuv_med.img_u);
    free(local_yuv_med.img_v);
    free(local_yuv_med.img_y);

    if (rank == 0) {
        result.w = img_in.w;
        result.h = img_in.h;
        result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    MPI_Gatherv(local_result.img_r, local_size, MPI_UNSIGNED_CHAR, result.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_g, local_size, MPI_UNSIGNED_CHAR, result.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_b, local_size, MPI_UNSIGNED_CHAR, result.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    free(local_result.img_r);
    free(local_result.img_g);
    free(local_result.img_b);
    free(local_img_in.img_r);
    free(local_img_in.img_g);
    free(local_img_in.img_b);
    free(sendcounts);
    free(displs);

    return result;
}

PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    HSL_IMG local_hsl_med;
    PPM_IMG local_result;
    PPM_IMG result;
    PPM_IMG local_img_in;
    
    unsigned char * l_equ;
    int localHist[256];
    int globalHist[256];

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_width = img_in.w;
    int local_height = img_in.h / size;
    int remainder = img_in.h % size;
    int local_size = local_width * local_height;

    // Adjust the local height for the last process to handle the remainder
    if (rank == size - 1) {
        local_height += remainder;
        local_size = local_width * local_height;
    }

    local_img_in.img_r = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_g = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.img_b = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_img_in.w = local_width;
    local_img_in.h = local_height;

    // Scatter the image data
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (img_in.h / size) * img_in.w;
        if (i == size - 1) {
            sendcounts[i] += remainder * img_in.w;
        }
        displs[i] = i * (img_in.h / size) * img_in.w;
    }

    MPI_Scatterv(img_in.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_r, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_g, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatterv(img_in.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, local_img_in.img_b, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    local_hsl_med = rgb2hsl(local_img_in);
    l_equ = (unsigned char *)malloc(local_hsl_med.height * local_hsl_med.width * sizeof(unsigned char));

    histogram(localHist, local_hsl_med.l, local_hsl_med.height * local_hsl_med.width, 256);
    MPI_Allreduce(localHist, globalHist, 256, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    histogram_equalization(l_equ, local_hsl_med.l, globalHist, local_hsl_med.height * local_hsl_med.width, 256, img_in.h * img_in.w);
    
    free(local_hsl_med.l);
    local_hsl_med.l = l_equ;

    local_result = hsl2rgb(local_hsl_med);
    free(local_hsl_med.h);
    free(local_hsl_med.s);
    free(local_hsl_med.l);

    if (rank == 0) {
        result.w = img_in.w;
        result.h = img_in.h;
        result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
        result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    }

    MPI_Gatherv(local_result.img_r, local_size, MPI_UNSIGNED_CHAR, result.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_g, local_size, MPI_UNSIGNED_CHAR, result.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_result.img_b, local_size, MPI_UNSIGNED_CHAR, result.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    free(local_result.img_r);
    free(local_result.img_g);
    free(local_result.img_b);
    free(local_img_in.img_r);
    free(local_img_in.img_g);
    free(local_img_in.img_b);
    free(sendcounts);
    free(displs);

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
    
    #pragma omp parallel for private(H, S, L) schedule(runtime)
    for(i = 0; i < img_in.w*img_in.h; i ++){
        
        float var_r = ( (float)img_in.img_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)img_in.img_g[i]/255 );
        float var_b = ( (float)img_in.img_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

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
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

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
    
    #pragma omp parallel for schedule(runtime)
    for(i = 0; i < img_in.width*img_in.height; i ++){
        float H = img_in.h[i];
        float S = img_in.s[i];
        float L = img_in.l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
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

    #pragma omp parallel for private(r, g, b, y, cb, cr) schedule(runtime)
    for(i = 0; i < img_out.w*img_out.h; i ++){
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
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

    #pragma omp parallel for private(y, cb, cr, rt, gt, bt) schedule(runtime)
    for(i = 0; i < img_out.w*img_out.h; i ++){
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }
    
    return img_out;
}
