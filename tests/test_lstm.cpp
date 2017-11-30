 /**
  * @file      test_lstm.cpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-11-26 14:55:26
  * @brief
  **/

#include <mkl.h>
#include <stdio.h>
#include "lstm.h"
const int T = 2;    //time_step
const int N = 2;    //batch_size
const int D = 2;    //input_dim
const int H = 2;    //hidden_dim
void random_fill(float *parray, int len)
{
    int i;
    for(i = 0; i < len; ++i)
    {   
        parray[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }   
}

void add(float *parray, float bias, int len)
{
    int i;
    for(i = 0; i < len; ++i)
    {   
        parray[i] += bias;
    }   
}

int main(int argc, char *argv[])
{
    void* buf = mkl_calloc(T*N*H*9, sizeof(float), 64);
    float *c_0 = (float*)mkl_calloc(N*H, sizeof(float), 64);
    float *h_0 = (float*)mkl_calloc(N*H, sizeof(float), 64);
    float *x = (float*)mkl_calloc(T*N*D, sizeof(float), 64);

    float *wx = (float*)mkl_calloc(D*4*H, sizeof(float), 64);
    float *wh = (float*)mkl_calloc(H*4*H, sizeof(float), 64);
    float *bias = (float*)mkl_calloc(4*H, sizeof(float), 64);

    float *h_out = (float*)mkl_calloc(T*N*H, sizeof(float), 64);
    float *c_out = (float*)mkl_calloc(T*N*H, sizeof(float), 64);

    float *dc = (float*)mkl_calloc(N*H, sizeof(float), 64);
    float *dh = (float*)mkl_calloc(N*H, sizeof(float), 64);
    float *dx = (float*)mkl_calloc(T*N*D, sizeof(float), 64);
    float *dwx = (float*)mkl_calloc(D*4*H, sizeof(float), 64);
    float *dwh = (float*)mkl_calloc(H*4*H, sizeof(float), 64);
    float *db = (float*)mkl_calloc(4*H, sizeof(float), 64);

    float *grad = (float*)mkl_calloc(N*H, sizeof(float), 64);

    //init
    random_fill(x, T*N*D);
    random_fill(wx, D*H*4);
    random_fill(wh, D*H*4);

    add(grad, 1, N*H);

    lstm_xw_forward(buf, N, T, D, H, x, c_0, h_0, wx, wh, bias, c_out, h_out);
    printf("h_out:\n");
    print(h_out, T, N, H);
    printf("c_out:\n");
    print(c_out, T, N, H);
    
    lstm_xw_backward(buf, N, T, D, H, x, c_0, h_0, wx, wh, bias, c_out, h_out, grad, dwx, dwh, db, dx, dc, dh);
   
    //check grad dh  
    float *h_check = (float*)mkl_calloc(T*N*H, sizeof(float), 64);
    float *c_check = (float*)mkl_calloc(T*N*H, sizeof(float), 64);
    

    mkl_free(buf);
    mkl_free(h_0);
    mkl_free(c_0);
    mkl_free(x);
    mkl_free(wx);
    mkl_free(wh);
    mkl_free(bias);
    mkl_free(h_out);
    mkl_free(c_out);
}
