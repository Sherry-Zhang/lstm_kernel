import imp
import tensorflow as tf
from tensorflow.python.framework import ops
lib_file = imp.find_module('kernels', __path__)[1]
lstm_lib = tf.load_op_library(lib_file)
class LSTM(object):
    def __init__(self, input_size, hidden_size):
        self.max_seq_length = 1
        self.max_batch_size = 1
        self.w_x = tf.Variable(tf.random_uniform(shape=[input_size, 4*hidden_size],name="wx"))
        self.w_h = tf.Variable(tf.random_uniform(shape=[hidden_size, 4*hidden_size],name="wh"))
        self.bias = tf.Variable(tf.zeros(shape=[4*hidden_size],name="intelb")) 
        self.update_workspace = True
        self.workspace = None
    def inference(self, x, h0, c0):
        T, N, D = x.shape
        H = h0.shape[1]
        if T > self.max_seq_length :
            self.max_seq_length = T
            self.update_workspace = True 
        if N > self.max_batch_size:
            self.max_batch_size = N
            self.update_workspace = True 
        if self.update_workspace:
            size = T*N*H*9 
            self.workspace = tf.zeros(shape=[size])
            self.update_workspace = False
        return lstm_lib.lstm_forward(self.workspace, x, c0, h0, self.w_x, self.w_h, self.bias)



@ops.RegisterGradient("LstmForward")
def _LstmBackward(op, grad_loss, _):
    T, N, D = op.inputs[1].get_shape()
    _, H = op.inputs[2].get_shape()
    dwx, dwh, db, dx, dh0, dc0 = lstm_lib.lstm_backward(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6], 
                                                op.outputs[1], op.outputs[0], grad_loss[-1]);
    return [None, dx, dc0, dh0, dwx, dwh, db] 

@ops.RegisterShape("LstmForward")                                                                                                                            
def _LstmForwardShape(op):
    T, N, D = op.inputs[1].get_shape()
    _, H = op.inputs[2].get_shape()
    return [[T, N, H], None]

@ops.RegisterShape("LstmBackward")
def _LstmBackwardShape(op):
    wx_shape = op.inputs[4].get_shape()
    wh_shape = op.inputs[5].get_shape()
    b_shape = op.inputs[6].get_shape()
    x_shape = op.inputs[1].get_shape()
    hc_shape = op.inputs[2].get_shape()
    return [wx_shape, wh_shape, b_shape, x_shape, hc_shape, hc_shape]
