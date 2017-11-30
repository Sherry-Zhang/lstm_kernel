 /**
  * @file      lstm.h
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-11-26 14:54:20
  * @brief
  **/
void print(const float *array, int time_step, int row, int col);
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
                    );
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
                     );
int get_workspace_size(const int T, const int N, const int D, const int H);
