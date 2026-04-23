import cnn
import debug_data
import convolver
import numpy as np


def test_forward_pass():
    """Tests if forward pass produces correct shape and values of output for debug data"""
    # Load debug data
    init_net = {}
    dbg = debug_data.Debug_data()
    init_net['W'], init_net['b'] = dbg.load_W_b()
    X_tr, init_net['Fs'] = dbg.load_X_Fs()
    
    nh = init_net['W'][0].shape[0]
    
    
    # Initialize objects
    convolver_obj = convolver.Convolver()
    cnn_obj = cnn.CNN(convolver_obj)
    
    # Get MX - not very isolated testing.. TODO
    MX = cnn_obj._construct_MX(X_tr, init_net['Fs'])
    
    # Do forward pass
    h, X1, P = cnn_obj._forward_pass(MX, init_net)
    h_gt, X1_gt, P_gt = dbg.load_fp_output()
    
    # Test correct shape
    assert h.shape == h_gt.shape
    assert X1.shape == X1_gt.shape
    assert P.shape == P_gt.shape
    
    # Test correct values
    assert np.sum(np.abs(h-h_gt)) == 0
    assert np.sum(np.abs(X1-X1_gt)) == 0
    assert np.sum(np.abs(P-P_gt)) == 0
    
def test_backward_pass():
    """Tests if backward pass produces produces grads for Ws and bs of correct shape (values not tested)
    and grads for fs_flat of correct shape and value for debug data
    
    Some shapes:
        init_net['W'][0] (10,128)
        init_net['W'][1] (10,10)
        init_net['Fs']  (4, 4, 3, 2)
        h_gt: (128,5)
        X1_gt; (10,5) 
        p_gt (10,5)
        Y: (10,5)
        
    """

    # Load debug data, gt=groud truth
    init_net = {}
    dbg = debug_data.Debug_data()
    init_net['W'], init_net['b'] = dbg.load_W_b()
    X_tr, init_net['Fs'] = dbg.load_X_Fs()
    h_gt, X1_gt, p_gt = dbg.load_fp_output()
    Y_tr, grads_fs_flat_gt = dbg.load_backprop_data()
    #print("fs flat shape", grads_fs_flat_gt.shape) #(48, 2)
    
    # Initialize objects
    convolver_obj = convolver.Convolver()
    cnn_obj = cnn.CNN(convolver_obj)
    
    # Get MX - remove dependence on other functions in this test.. TODO
    MX = cnn_obj._construct_MX(X_tr, init_net['Fs'])
    
    # Get n_b and n_f for backward pass (needed for unflattening h)
    n_f = init_net['Fs'].shape[3]
    n_p = int(h_gt.shape[0] / n_f)
    
    
    # Do backward pass
    grads = cnn_obj._backward_pass(MX, Y_tr, h_gt, X1_gt, p_gt, init_net, n_f, n_p)
    
    assert grads['fs_flat'].shape == grads_fs_flat_gt.shape
    assert np.sum(np.abs(grads['fs_flat']-grads_fs_flat_gt)) <= 10**-6 # Some error margin
    
    assert grads['W'][0].shape == init_net['W'][0].shape
    assert grads['W'][1].shape == init_net['W'][1].shape
    assert grads['b'][0].shape == init_net['b'][0].shape
    assert grads['b'][1].shape == init_net['b'][1].shape
    