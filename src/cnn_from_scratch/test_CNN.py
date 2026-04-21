import cnn
import debug_data
import convolver
import numpy as np

def test_forward_pass():
    # Load debug data
    init_net = {}
    dbg = debug_data.Debug_data()
    init_net['W'], init_net['b'] = dbg.load_W_b()
    X_tr, init_net['Fs'] = dbg.load_X_Fs()
    
    nh = init_net['W'][0].shape[0]
    
    
    # Initialize objects
    convolver_obj = convolver.Convolver()
    cnn_obj = cnn.CNN(convolver_obj)
    
    # Do forward pass
    h, X1, P = cnn_obj._forward_pass(X_tr, init_net)
    h_gt, X1_gt, P_gt = dbg.load_fp_output()
    X1_gt = X1_gt.squeeze(0) # Wrong shape in debug data values
    
    # Test correct shape
    assert h.shape == h_gt.shape
    assert X1.shape == X1_gt.shape
    assert P.shape == P_gt.shape
    
    # Test correct values
    assert np.sum(np.abs(h-h_gt)) == 0
    assert np.sum(np.abs(X1-X1_gt)) == 0
    assert np.sum(np.abs(P-P_gt)) == 0
    
    
    