 /**
  * @file      test_lstm.cpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-11-26 14:55:26
  * @brief
  **/

#include <lstm.h>
#include <mkl.h>
#include <stdio.h>
#include <string.h>
extern "C" {
int T = 2;    //time_step
int N = 2;    //batch_size
int D = 2;    //input_dim
int H = 2;    //hidden_dim
void print(const float *array, int time_step, int row, int col)
{
    int i, j, k;
    for(i = 0; i < time_step; ++i)
    {
        printf("timestep: %d\n", i);
        for(j = 0; j < row; ++j)
        {
            for(k = 0; k < col; ++k)
            {
                printf("%f ", array[i * row * col + j * col + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}

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
    if (argc != 5)
    {
        printf("Error args. You must input T, N, D, H.\n");
        return -1;
    }
    int T = atoi(argv[1]);    //time_step
    int N = atoi(argv[2]);    //batch_size
    int D = atoi(argv[3]);    //input_dim
    int H = atoi(argv[4]);    //hidden_dim
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
//    printf("h_out:\n");
//    print(h_out, T, N, H);
//    printf("c_out:\n");
//    print(c_out, T, N, H);
    
    lstm_xw_backward(buf, N, T, D, H, x, c_0, h_0, wx, wh, bias, c_out, h_out, grad, dwx, dwh, db, dx, dc, dh);
    double begin, end, dura;
    int count = 100;
    begin = dsecnd();
    for(int i = 0; i < count; ++i) {
        lstm_xw_forward(buf, N, T, D, H, x, c_0, h_0, wx, wh, bias, c_out, h_out);
    }
    end = dsecnd();
    dura = (end-begin)/count;
    printf("T=%d, N=%d, D=%d, H=%d\n",  T, N, D, H);
    printf("Forward: dur=%.4f, SPS=%.4f\n", dura, N/dura);
    begin = dsecnd();
    for(int i = 0; i < count; ++i) {
        lstm_xw_forward(buf, N, T, D, H, x, c_0, h_0, wx, wh, bias, c_out, h_out);
        lstm_xw_backward(buf, N, T, D, H, x, c_0, h_0, wx, wh, bias, c_out, h_out, grad, dwx, dwh, db, dx, dc, dh);
    }
    end = dsecnd();
    dura = (end-begin)/count;
    printf("Forward+Backward: dur=%.4f, SPS=%.4f\n", dura, N/dura);
   
    mkl_free(buf);
    mkl_free(h_0);
    mkl_free(c_0);
    mkl_free(x);
    mkl_free(wx);
    mkl_free(wh);
    mkl_free(bias);
    mkl_free(h_out);
    mkl_free(c_out);
    mkl_free(grad);
    mkl_free(dwx);
    mkl_free(dwh);
    mkl_free(db);
    mkl_free(dx);
    mkl_free(dc);
    mkl_free(dh);
}
}
