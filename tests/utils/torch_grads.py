import torch
import numpy as np

def compute_grads_with_torch(X, y, network, lam=0):
    """
    Computes grads numerically using pytorch. Used for checking analytical gradients.
    Note that it takes images X, not MX
    
    Args:
        X : nd array (32,32,3,n) of images
        y: n length list of true labels
        network: dict with
            Fs: ndarray (f, f, 3, nf) Filters of layer 1
            W: List of weights for layer 2 and 3 
                [0]: W1 ndarray (nh, n_p * nf) nh= #nodes in layer 2 or 3?
                [1]: W2 ndarray (K, nh)
            b: List of biases for layer 2 and 3 
                [0]: b1 ndarray (nh, 1)
                [1]: b2 ndarray (K, 1)
    """
    
    Xt = torch.from_numpy(X)

    L = len(network['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network['W'])):
        W[i] = torch.tensor(network['W'][i], requires_grad=True)
        b[i] = torch.tensor(network['b'][i], requires_grad=True)
    
    # Filter gradients TODO
    Fs = torch.tensor(network['Fs'], requires_grad=True)        
    Fs_b = torch.tensor(network['Fs_b'], requires_grad=True) 
    
    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### Implement forward pass
    conv_out = torch_conv_for_loop(Xt, Fs)
    conv_out = conv_out + Fs_b.reshape(1, network['Fs_b'].shape[0], 1)
    conv_out= apply_relu(conv_out)
    
    #npnf = network['W'][0].shape[1]
    #n = X.shape[3]
    #h = np.fmax(conv_out.reshape((npnf, n), order='C'), 0)
    
    n = Xt.shape[3]
    h = conv_out.reshape(-1, n)
    
    # Layer 2
    x1 = torch.mm(W[0], h) + b[0]
    h1 = apply_relu(x1) # ReLu
    
    # Layer 3
    scores = W[1]@ h1 + b[1]
     
    
    #s1=torch.mm(W[0], Xt) + b[0]
    #h = apply_relu(s1)
    #scores = torch.mm(W[1], h) + b[1]

    ####            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    y_t = torch.tensor(y, dtype=torch.long) # Turn y into a torch recognized object
    loss = torch.mean(-torch.log(P[y_t, torch.arange(n)]))
    
    # Compute cost
    l2 = torch.sum(Fs*Fs)
    for i in range(L):
        l2 = l2 + torch.sum(W[i] * W[i])
    
    loss = loss + lam * l2
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()
    grads['Fs'] = Fs.grad.numpy()
    grads['Fs_b'] = Fs_b.grad.numpy()

    return grads

def torch_conv_for_loop(X, Fs):
    f = Fs.shape[0]
    n = int(X.shape[3])
    nf = int(Fs.shape[3])
    
    patches_per_row = X.shape[0]//f
    patches_per_col = X.shape[1]//f
    
    conv_output = torch.zeros(patches_per_row, patches_per_col, nf,n,dtype=X.dtype, device=X.device)
    
    for i in range(n):
        for j in range(0,X.shape[0], f):
            pj = j // f # Patch index
            
            for k in range(0,X.shape[1], f):
                pk = k // f # Patch index
                
                # Create subpatch of size (f,f,d)
                sub_patch = X[j:j+f,
                                    k:k+f,
                                    :,
                                    i]
                
                for l in range(nf):
                    dot_prod = torch.sum(torch.multiply(sub_patch, Fs[:, :, :, l]))
                    conv_output[pj, pk, l, i] = dot_prod
    return conv_output
