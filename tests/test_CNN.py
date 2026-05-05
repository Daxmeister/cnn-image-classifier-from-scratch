import numpy as np
import cnn_from_scratch.data_handling as data_handling 
import tests.utils.torch_grads as torch_grads


def test_forward_pass(dbg, cnn_obj, epsilon):
    """Tests if forward pass produces correct shape and values of output for debug data"""
    # Load debug data
    init_net = {}

    init_net['W'], init_net['b'] = dbg.load_W_b()
    X_tr, init_net['Fs'] = dbg.load_X_Fs()
    
    nh = init_net['W'][0].shape[0]
    

    
    # Get MX - not very isolated testing.. TODO
    MX = cnn_obj._construct_MX(X_tr, init_net['Fs'])
    
    # Do forward pass
    h, X1, P = cnn_obj._forward_pass(MX, init_net, use_bias=False)
    h_gt, X1_gt, P_gt = dbg.load_fp_output()
    
    # Test correct shape
    assert h.shape == h_gt.shape
    assert X1.shape == X1_gt.shape
    assert P.shape == P_gt.shape
    
    # Test correct values
    assert np.sum(np.abs(h-h_gt)) == 0
    assert np.sum(np.abs(X1-X1_gt)) == 0
    assert np.sum(np.abs(P-P_gt)) == 0
    
def test_backward_pass(dbg, cnn_obj, epsilon):
    """Tests if backward pass produces  grads for Ws and bs of correct shape (values not tested)
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
   
    init_net['W'], init_net['b'] = dbg.load_W_b()
    X_tr, init_net['Fs'] = dbg.load_X_Fs()
    h_gt, X1_gt, p_gt = dbg.load_fp_output()
    Y_tr, grads_fs_flat_gt = dbg.load_backprop_data()
    #print("fs flat shape", grads_fs_flat_gt.shape) #(48, 2)


    
    # Get MX - remove dependence on other functions in this test.. TODO
    MX = cnn_obj._construct_MX(X_tr, init_net['Fs'])
    
    # Get n_b and n_f for backward pass (needed for unflattening h)
    n_f = init_net['Fs'].shape[3]
    n_p = int(h_gt.shape[0] / n_f)
    
    
    # Do backward pass
    grads = cnn_obj._backward_pass(MX, Y_tr, h_gt, X1_gt, p_gt, init_net, n_f, n_p)
    
    assert grads['fs_flat'].shape == grads_fs_flat_gt.shape
    assert np.sum(np.abs(grads['fs_flat']-grads_fs_flat_gt)) <= epsilon 
    
    assert grads['W'][0].shape == init_net['W'][0].shape
    assert grads['W'][1].shape == init_net['W'][1].shape
    assert grads['b'][0].shape == init_net['b'][0].shape
    assert grads['b'][1].shape == init_net['b'][1].shape

def test_init_network(dbg, cnn_obj, epsilon):
    """ Tests that the shapes of the init_network parameters are correct """
    
    network_gt = {}

    network_gt['W'], network_gt['b'] = dbg.load_W_b()
    X_tr, network_gt['Fs'] = dbg.load_X_Fs()
    
    nh = network_gt['W'][0].shape[0]
    nf = network_gt['Fs'].shape[3]
    f = network_gt['Fs'].shape[0]
    channels = network_gt['Fs'].shape[2]
    L = len(network_gt['W'])
    K =  network_gt['W'][1].shape[0]   


    
    network = cnn_obj._init_network(nh, nf, f, channels, L, K)
    
    assert network['W'][0].shape == network_gt['W'][0].shape
    assert network['W'][1].shape == network_gt['W'][1].shape
    assert network['b'][0].shape == network_gt['b'][0].shape
    assert network['b'][1].shape == network_gt['b'][1].shape
    assert network['Fs'].shape == network_gt['Fs'].shape

def test_compare_analytical_and_numerical_grads(dbg, cnn_obj, epsilon):
    nh = 5
    f=4
    nf=2
    network = cnn_obj._init_network(nh, nf=nf, f=f, channels=3, L=2, K=10)
    
    data  = data_handling.get_MX_data(f=f, d=3, val_size=50, total_samples=500, small_data=True)
    h, X1, p = cnn_obj._forward_pass(data['MX_tr'], network)
    analytical_grads = cnn_obj._backward_pass(data['MX_tr'], data['Y_tr'], h, X1, p, network, n_f=nf, n_p=int((32/f)**2))
    numerical_grads = torch_grads.compute_grads_with_torch(data['X_ims'], data['y_tr'], network)
    
    # Compare weight gradients
    for i in range(2):
        num = np.linalg.norm((analytical_grads["W"][i]-numerical_grads["W"][i]))
        added_norms = np.linalg.norm(analytical_grads["W"][i]) + np.linalg.norm(numerical_grads["W"][i])
        eps = 10**(-6)
        den = max(eps, added_norms)
        assert num/den <= eps
    
    # Compare bias gradients
    for i in range(2):
        num = np.linalg.norm((analytical_grads["b"][i]-numerical_grads["b"][i]))
        added_norms = np.linalg.norm(analytical_grads["b"][i]) + np.linalg.norm(numerical_grads["b"][i])
        eps = 10**(-6)
        den = max(eps, added_norms)
        assert num/den <= eps

    # Compare filter gradianets
    # Flatten num gradients to enable comparison 
    Fs_analytical_flat = analytical_grads["fs_flat"]
    Fs_num_flat = numerical_grads["Fs"].reshape((Fs_analytical_flat.shape), order='C') 

    added_norms = np.linalg.norm(Fs_analytical_flat) + np.linalg.norm(Fs_num_flat)
    eps = 10**(-6)
    den = max(eps, added_norms)
    assert num/den <= eps
    
    # Compare convolution layer bias gradients
    num = np.linalg.norm((analytical_grads["Fs_b"]-numerical_grads["Fs_b"]))
    added_norms = np.linalg.norm(analytical_grads["Fs_b"]) + np.linalg.norm(numerical_grads["Fs_b"])
    eps = 10**(-6)
    den = max(eps, added_norms)
    assert num/den <= eps


def test_compare_analytical_and_numerical_grads_w_cost(dbg, cnn_obj, epsilon):
    nh = 5
    f=4
    nf=2
    lam = 0.1
    network = cnn_obj._init_network(nh, nf=nf, f=f, channels=3, L=2, K=10)
    
    data  = data_handling.get_MX_data(f=f, d=3, val_size=50, total_samples=500, small_data=True)
    h, X1, p = cnn_obj._forward_pass(data['MX_tr'], network)
    analytical_grads = cnn_obj._backward_pass(data['MX_tr'], data['Y_tr'], h, X1, p, network, n_f=nf, n_p=int((32/f)**2), lam=lam)
    numerical_grads = torch_grads.compute_grads_with_torch(data['X_ims'], data['y_tr'], network, lam=lam)
    
    # Compare weight gradients
    for i in range(2):
        num = np.linalg.norm((analytical_grads["W"][i]-numerical_grads["W"][i]))
        added_norms = np.linalg.norm(analytical_grads["W"][i]) + np.linalg.norm(numerical_grads["W"][i])
        eps = 10**(-6)
        den = max(eps, added_norms)
        assert num/den <= eps
    
    # Compare bias gradients
    for i in range(2):
        num = np.linalg.norm((analytical_grads["b"][i]-numerical_grads["b"][i]))
        added_norms = np.linalg.norm(analytical_grads["b"][i]) + np.linalg.norm(numerical_grads["b"][i])
        eps = 10**(-6)
        den = max(eps, added_norms)
        assert num/den <= eps

    # Compare filter gradianets
    # Flatten num gradients to enable comparison 
    Fs_analytical_flat = analytical_grads["fs_flat"]
    Fs_num_flat = numerical_grads["Fs"].reshape((Fs_analytical_flat.shape), order='C') 

    added_norms = np.linalg.norm(Fs_analytical_flat) + np.linalg.norm(Fs_num_flat)
    eps = 10**(-6)
    den = max(eps, added_norms)
    assert num/den <= eps
    
    # Compare convolution layer bias gradients
    num = np.linalg.norm((analytical_grads["Fs_b"]-numerical_grads["Fs_b"]))
    added_norms = np.linalg.norm(analytical_grads["Fs_b"]) + np.linalg.norm(numerical_grads["Fs_b"])
    eps = 10**(-6)
    den = max(eps, added_norms)
    assert num/den <= eps
    

    
    
    