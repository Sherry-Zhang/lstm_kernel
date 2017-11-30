 /**
  * @file           lstm.cpp
  * @author         zhangshu(shu.zhang@intel.com)
  * @date           2017-11-25 10:50:08
  * @brief          lstm forward computation
  *
  * @formula list:  [----------------------forward-----------------------]
  *                 i_t = sigmoid(w_xi * x_t + b_xi + w_hi * h_t-1 + b_hi)
  *                 g_t =    tanh(w_xg * x_t + b_xg + w_hg * h_t-1 + b_hg)
  *                 f_t = sigmoid(w_xf * x_t + b_xf + w_hf * h_t-1 + b_hf)
  *                 o_t = sigmoid(w_xo * x_t + b_xo + w_ho * h_t-1 + b_ho)
  *                 c_t = f_t · c_t-1 + i_t · g_t
  *                 h_t = o_t · tanh(c_t) 
  *                 [----------------------backward----------------------]
  *                 dc_t = dh_t · o_t * (1 - tanh(c_t)**2) + dc_t+1 · f_t+1
  *                 di = dc_t · g_t · i_t ·(1 - i_t)
  *                 dg = dc_t · i_t · (1 - g_t**2)
  *                 df = dc_t · c_t-1 · f_t ·(1 - f_t)
  *                 do = dh_t · tanh(c_t) · o_t ·(1 - o_t) 
  *                 dx_t = w_xi.T · di + w_xg.T · dg + w_xf.T · df + w_xo.T · do 
  *                 dh_t-1 = w_hi.T · di + w_hg.T · dg + w_hf.T · df + w_ho.T · do 
  *
  *                 dw_xi = di * x_t.T
  *                 dw_xg = dg * x_t.T
  *                 dw_xf = df * x_t.T
  *                 dw_xo = do * x_t.T
  *
  *                 dw_hi = di * h_t-1.T
  *                 dw_hg = dg * h_t-1.T
  *                 dw_hf = df * h_t-1.T
  *                 dw_ho = do * h_t-1.T
  *
  *                 db_xi = db_hi = di * [1,1,1...1].T
  *                 db_xg = db_hg = dg * [1,1,1...1].T 
  *                 db_xf = db_hf = df * [1,1,1...1].T 
  *                 db_xo = db_ho = do * [1,1,1...1].T 
  *
  **/
#include <math.h>
#include <mkl.h>
#include <stdio.h>
#include <string.h>
inline float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}
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
int get_workspace_size(const int T, const int N, const int D, const int H) {
 //   int forward_size = T * N * H * 4 + N * H;
    int backward_size = T * N * H * 9;
    return backward_size;
}
int lstm_xw_forward(void* buf,
                    const int N,    //batch_size
                    const int T,    //time_step(seq_len)
                    const int D,    //input_dim
                    const int H,    //hidden_dim
                    const float* x, //T*N*D
                    const float* c_0,     //N*H
                    const float* h_0,     //N*H
                    const float* wx,      //D*4H      
                    const float* wh,      //H*4H      
                    const float* bias,    //4H, because bx == bh, so just need one
                    float* c_out,   //T*N*H
                    float* h_out    //T*N*H
                    ) {
    //x*wx
//    printf("x\n");
//    print((float*)x, T, N, D);
//    printf("wx\n");
//    print(wx, 1, D, 4*H);
//    printf("wh\n");
//    print(wh, 1, H, 4*H);

    MKL_INT gemm_m = T*N, gemm_n = 4*H, gemm_k = D;
    MKL_INT lda = gemm_k, ldb = gemm_n, ldc = gemm_n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, gemm_m, gemm_n, gemm_k, 1.0f, x, lda, wx, ldb, 0.0f, (float*)buf, ldc);
    float (*ix)[gemm_n] = (float(*)[gemm_n])buf;
    float (*gx)[gemm_n] = (float(*)[gemm_n])((float*)ix + H);
    float (*fx)[gemm_n] = (float(*)[gemm_n])((float*)gx + H);
    float (*ox)[gemm_n] = (float(*)[gemm_n])((float*)fx + H);
    float *h_buf = (float*)buf + gemm_m * gemm_n;
    float (*ih)[gemm_n] = ix + gemm_m;
    float (*gh)[gemm_n] = gx + gemm_m; 
    float (*fh)[gemm_n] = fx + gemm_m;
    float (*oh)[gemm_n] = ox + gemm_m;
    
    const float *bi = bias;
    const float *bg = bi + H;
    const float *bf = bg + H;
    const float *bo = bf + H;

    const float *h_pre = h_0;
    const float (*c_pre)[H] = (const float(*)[H])c_0;
    float (*c)[H] = (float(*)[H])c_out;
    float (*h)[H] = (float(*)[H])h_out;
    gemm_k = H;
    lda = H;
    int i = 0, j = 0, k = 0;
    for (i = 0; i < gemm_m; i += N) {
        //h_t-1 * wh
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, gemm_n, gemm_k, 1.0f, h_pre, lda, wh, ldb, 0.0f, h_buf, ldc);
        for (j = 0; j < N; ++j) {
            for (k = 0; k < H; ++k) {
                ix[i+j][k] = sigmoid(ix[i+j][k] + ih[j][k] + bi[k]);
                gx[i+j][k] =    tanh(gx[i+j][k] + gh[j][k] + bg[k]);
                fx[i+j][k] = sigmoid(fx[i+j][k] + fh[j][k] + bf[k]);
                ox[i+j][k] = sigmoid(ox[i+j][k] + oh[j][k] + bo[k]);
                c[i+j][k] = c_pre[j][k] * fx[i+j][k] + ix[i+j][k] * gx[i+j][k];
                h[i+j][k] = ox[i+j][k] * tanh(c[i+j][k]); 
            }
        }
        c_pre = c + i;
        h_pre = (float*)(h + i);
    } 
 //   printf("buf\n");
 //   print((float*)buf, 2*T, N, gemm_n);
 //   printf("h_out\n");
 //   print(h_out, T, N, H);
}

int lstm_xw_backward(void* buf,
                     const int N,    //batch_size
                     const int T,    //time_step(seq_len)
                     const int D,    //input_dim
                     const int H,    //hidden_dim
                     const float* x, //T*N*D
                     const float* c_0,     //N*H
                     const float* h_0,     //N*H
                     const float* wx,      //D*4H      
                     const float* wh,      //H*4H      
                     const float* bias,    //8H
                     const float* c_out,   //T*N*H
                     const float* h_out,   //T*N*H
                     const float* grad_loss,   //N*H, next_layer's grad
                     float* dwx,     //D*4H
                     float* dwh,     //H*4H
                     float* db,      //4H
                     float* dx, 
                     float* dc0,     //N*H
                     float* dh0      //N*H
                     ) {
    MKL_INT gemm_m = T * N, gemm_n = 4 * H;
    float (*it)[gemm_n] = (float(*)[gemm_n])buf;
    float (*gt)[gemm_n] = (float(*)[gemm_n])((float*)it + H);
    float (*ft)[gemm_n] = (float(*)[gemm_n])((float*)gt + H);
    float (*ot)[gemm_n] = (float(*)[gemm_n])((float*)ft + H);
    
    const float (*ct)[H] = (float(*)[H])c_out;
    const float (*ht)[H] = (float(*)[H])h_out;
    int i = 0, j = 0, k = 0;
    //[di|dc|dg|do] : size=[T*N, 4H]
    float (*deta_i)[gemm_n] = it + gemm_m; 
    float (*deta_g)[gemm_n] = gt + gemm_m; 
    float (*deta_f)[gemm_n] = ft + gemm_m; 
    float (*deta_o)[gemm_n] = ot + gemm_m; 
    //dc:[T*N, H] 
    float (*deta_c)[H] = (float(*)[H])(deta_i + gemm_m);
    float (*deta_h)[H] = (float(*)[H])dh0;
//    printf("back_buf\n");
//    print((float*)buf, 2*T, N, gemm_n);
    
    float tc = 0.0f;
    memset(dwh, 0, sizeof(float) * H * gemm_n);
    memset(db, 0, sizeof(float) * gemm_n);
    float *gemm_a = NULL;
    for (i = gemm_m - N; i >= 0; i -= N) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < H; ++k) {
                tc = tanh(ct[i+j][k]);
                if (i == gemm_m - N) {
                    deta_h[j][k] = grad_loss[j*H+k];
                    deta_c[i+j][k] = deta_h[j][k] * ot[i+j][k] * (1 - tc * tc);
                }
                else {
                    deta_c[i+j][k] = deta_h[j][k] * ot[i+j][k] * (1 - tc * tc) + deta_c[i+j+N][k] * ft[i+j+N][k];
                }
                deta_i[i+j][k] = deta_c[i+j][k] * gt[i+j][k] * it[i+j][k] * (1 - it[i+j][k]);
                deta_g[i+j][k] = deta_c[i+j][k] * it[i+j][k] * (1 - gt[i+j][k] * gt[i+j][k]);
                deta_o[i+j][k] = deta_h[j][k] * tc * ot[i+j][k] * (1 - ot[i+j][k]);
                if (i != 0) {
                    deta_f[i+j][k] = deta_c[i+j][k] * ct[i+j-N][k] * ft[i+j][k] * (1 - ft[i+j][k]);
                }
                else {
                    dc0[j*H + k] = deta_c[j][k] * ft[j][k];
                    deta_f[j][k] = deta_c[j][k] * c_0[j*H+k] * ft[j][k] * (1 - ft[j][k]);
                }
            }
        }
        if (i != 0) {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, gemm_n, N, 1.0f, (float*)(ht+i-N), H, (float*)(deta_i+i), gemm_n, 1.0f, dwh, gemm_n);
        }
        else {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, gemm_n, N, 1.0f, h_0, H, (float*)(deta_i+i), gemm_n, 1.0f, dwh, gemm_n);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, gemm_n, 1.0f, (float*)(deta_i+i), gemm_n, wh, gemm_n, 0.0f, dh0, H);

    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, gemm_m, D, gemm_n, 1.0f, (float*)(deta_i), gemm_n, wx, gemm_n, 0.0f, dx, D);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, gemm_n, gemm_m, 1.0f, x, D, (float*)(deta_i), gemm_n, 0.0f, dwx, gemm_n);
    for(i = 0; i < gemm_m; ++i) {
        for (j = 0; j < gemm_n; ++j) {
            db[j] += deta_i[i][j];
        }
    }
//    printf("dc\n");
//    print((float*)deta_c, T, N, H);
//    printf("dicfo\n");
//    print((float*)deta_i, T, N, gemm_n);

//    printf("dwx\n");
//    print(dwx, 1, D, 4*H);
//    printf("dwh\n");
//    print(dwh, 1, H, 4*H);
//    printf("db\n");
//    print(db, 1, 1, 4*H);
//    printf("dx\n");
//    print(dx, T, N, D);

    return 0;
}

