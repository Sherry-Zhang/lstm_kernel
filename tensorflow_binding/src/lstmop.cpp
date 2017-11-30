 /**
  * @file      lstmop.cpp
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-11-13 00:21:29
  * @brief
  **/
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <lstm.h>

namespace tf = tensorflow;

REGISTER_OP("LstmForward")
    .Input("workspace: float")
    .Input("x: float32")
    .Input("c_0: float32")
    .Input("h_0: float32")
    .Input("w_x: float32")
    .Input("w_h: float32")
    .Input("b: float32")
    .Output("h_out: float32")
    .Output("c_out: float32");
class LstmForwardOp : public tf::OpKernel {
public:
    explicit LstmForwardOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx){}
    void Compute(tf::OpKernelContext* ctx) override {
        //workspace
        tf::Tensor& workspace_tensor = const_cast<tf::Tensor&>(ctx->input(0));
        auto workspace = workspace_tensor.flat<float>();
        //x
        tf::Tensor& x_tensor = const_cast<tf::Tensor&>(ctx->input(1));
        auto x = x_tensor.flat<float>();
        //c_0
        tf::Tensor& c_0_tensor = const_cast<tf::Tensor&>(ctx->input(2));
        auto c_0 = c_0_tensor.flat<float>();
        //h_0
        tf::Tensor& h_0_tensor = const_cast<tf::Tensor&>(ctx->input(3));
        auto h_0 = h_0_tensor.flat<float>();
        //w_x
        tf::Tensor& w_x_tensor = const_cast<tf::Tensor&>(ctx->input(4));
        auto w_x = w_x_tensor.flat<float>();
        //w_h
        tf::Tensor& w_h_tensor = const_cast<tf::Tensor&>(ctx->input(5));
        auto w_h = w_h_tensor.flat<float>();
        //b
        tf::Tensor& b_tensor = const_cast<tf::Tensor&>(ctx->input(6));
        auto b = b_tensor.flat<float>();

        auto T = x_tensor.shape().dim_size(0);
        auto N = x_tensor.shape().dim_size(1);
        auto D = x_tensor.shape().dim_size(2);
        auto H = h_0_tensor.shape().dim_size(1);
        //h_out
        tf::Tensor* h_out_tensor = nullptr;
        ctx->allocate_output(0, tf::TensorShape({T, N, H}), &h_out_tensor);
        auto h_out = h_out_tensor->flat<float>();
        //c_out
        tf::Tensor* c_out_tensor = nullptr;
        ctx->allocate_output(1, tf::TensorShape({T, N, H}), &c_out_tensor);
        auto c_out = c_out_tensor->flat<float>();
        lstm_xw_forward(workspace.data(), 
                        N,
                        T,
                        D,
                        H,
                        x.data(), 
                        c_0.data(), 
                        h_0.data(), 
                        w_x.data(), 
                        w_h.data(), 
                        b.data(), 
                        c_out.data(), 
                        h_out.data()); 
    }
};
REGISTER_KERNEL_BUILDER(Name("LstmForward").Device(tf::DEVICE_CPU), LstmForwardOp);

REGISTER_OP("LstmBackward")
    .Input("workspace: float")
    .Input("x: float32")
    .Input("c_0: float32")
    .Input("h_0: float32")
    .Input("w_x: float32")
    .Input("w_h: float32")
    .Input("b: float32")
    .Input("c_out: float32")
    .Input("h_out: float32")
    .Input("dh: float32")
    .Output("dwx: float32")
    .Output("dwh: float32")
    .Output("db: float32")
    .Output("dx: float32")
    .Output("dh0: float32")
    .Output("dc0: float32");
class LstmBackwardOp : public tf::OpKernel {
public:
    explicit LstmBackwardOp(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx){}
    void Compute(tf::OpKernelContext* ctx) override {
        //workspace
        tf::Tensor& workspace_tensor = const_cast<tf::Tensor&>(ctx->input(0));
        auto workspace = workspace_tensor.flat<float>();
        //x
        tf::Tensor& x_tensor = const_cast<tf::Tensor&>(ctx->input(1));
        auto x = x_tensor.flat<float>();
        //c_0
        tf::Tensor& c_0_tensor = const_cast<tf::Tensor&>(ctx->input(2));
        auto c_0 = c_0_tensor.flat<float>();
        //h_0
        tf::Tensor& h_0_tensor = const_cast<tf::Tensor&>(ctx->input(3));
        auto h_0 = h_0_tensor.flat<float>();
        //w_x
        tf::Tensor& w_x_tensor = const_cast<tf::Tensor&>(ctx->input(4));
        auto w_x = w_x_tensor.flat<float>();
        //w_h
        tf::Tensor& w_h_tensor = const_cast<tf::Tensor&>(ctx->input(5));
        auto w_h = w_h_tensor.flat<float>();
        //b
        tf::Tensor& b_tensor = const_cast<tf::Tensor&>(ctx->input(6));
        auto b = b_tensor.flat<float>();
        //c_out
        tf::Tensor& c_out_tensor = const_cast<tf::Tensor&>(ctx->input(7));
        auto c_out = c_out_tensor.flat<float>();
        //h_out
        tf::Tensor& h_out_tensor = const_cast<tf::Tensor&>(ctx->input(8));
        auto h_out = h_out_tensor.flat<float>();
        //dh
        tf::Tensor& dh_tensor = const_cast<tf::Tensor&>(ctx->input(9));
        auto dh = dh_tensor.flat<float>();

        auto T = x_tensor.shape().dim_size(0);
        auto N = x_tensor.shape().dim_size(1);
        auto D = x_tensor.shape().dim_size(2);
        auto H = h_0_tensor.shape().dim_size(1);
        //dwx
        tf::Tensor* dwx_tensor = nullptr;
        ctx->allocate_output(0, tf::TensorShape({D, 4*H}), &dwx_tensor);
        auto dwx = dwx_tensor->flat<float>();
        //dwh
        tf::Tensor* dwh_tensor = nullptr;
        ctx->allocate_output(1, tf::TensorShape({H, 4*H}), &dwh_tensor);
        auto dwh = dwh_tensor->flat<float>();
        //db
        tf::Tensor* db_tensor = nullptr;
        ctx->allocate_output(2, tf::TensorShape({4*H}), &db_tensor);
        auto db = db_tensor->flat<float>();
        //dx
        tf::Tensor* dx_tensor = nullptr;
        ctx->allocate_output(3, tf::TensorShape({T, N, D}), &dx_tensor);
        auto dx = dx_tensor->flat<float>();
        //dh0
        tf::Tensor* dh0_tensor = nullptr;
        ctx->allocate_output(4, tf::TensorShape({N*H}), &dh0_tensor);
        auto dh0 = dh0_tensor->flat<float>();
        //dc0
        tf::Tensor* dc0_tensor = nullptr;
        ctx->allocate_output(5, tf::TensorShape({N*H}), &dc0_tensor);
        auto dc0 = dc0_tensor->flat<float>();
        lstm_xw_backward(workspace.data(),
                         N,
                         T,
                         D,
                         H,
                         x.data(), 
                         c_0.data(), 
                         h_0.data(), 
                         w_x.data(), 
                         w_h.data(), 
                         b.data(), 
                         c_out.data(), 
                         h_out.data(),
                         dh.data(),
                         dwx.data(),
                         dwh.data(),
                         db.data(),
                         dx.data(),
                         dc0.data(),
                         dh0.data()
                         ); 
    }
};
REGISTER_KERNEL_BUILDER(Name("LstmBackward").Device(tf::DEVICE_CPU), LstmBackwardOp);
