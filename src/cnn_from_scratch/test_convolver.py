import numpy as np


def test_conv_for_loop(dbg, convolver_obj, epsilon):
    """ Compares shape an dvalue of conv_for_loop to ground truth from debug data"""
    
    X_ims, Fs = dbg.load_X_Fs()
    conv_out = convolver_obj.conv_for_loop(X_ims, Fs)
    conv_out_gt = dbg.load_convolv_gt()
    assert conv_out.shape ==  conv_out_gt.shape
    assert np.sum(np.abs(conv_out -  conv_out_gt)) <= epsilon
    
def test_conv_mat_mul(dbg, convolver_obj, cnn_obj, epsilon):
    X_ims, Fs = dbg.load_X_Fs()
    MX = cnn_obj._construct_MX(X_ims, Fs) # Required by conv_mat_mul
    conv_output_fl = convolver_obj.conv_for_loop(X_ims, Fs)
    
    conv_output_matmul = convolver_obj.conv_mat_mul(MX, Fs)
    
    # Change shape of for-loop ground truth to enable comparison
    conv_output_fl_flat = conv_output_fl.reshape((conv_output_matmul.shape), order='C') # ERROR TODO
    
    assert conv_output_fl_flat.shape==conv_output_matmul.shape
    assert np.sum(np.abs(conv_output_fl_flat-conv_output_matmul)) <= epsilon