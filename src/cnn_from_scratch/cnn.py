import numpy as np
from cnn_from_scratch import helper_func


class CNN():
    
    def __init__(self, convolver, plotter = None):
        self.network = {}
        self.convolver = convolver
        self.rng = np.random.default_rng(42)
        self.plotter = plotter
    
    def train(self, X_train, Y_train, y_train, debug_network=None):
        """
        Trains the CNN network, in effect updating self.network
        
        Args: 
            X_train
            Y_train
            y_train
        """
        
        """self.X_tr = X_train
        self.Y_tr = Y_train
        self.y_tr = y_train"""
        if debug_network != None: # For testing
            self.network = debug_network
        else:
            pass
            # Initiate network TODO
        
        MX = _construct_MX(self, X_train, self.network['Fs'])
        
        _forward_pass(self, MX, self.network)
        # TODO n_f = 
        # TODO n_p = 
        # 1. Load parameters
        # 2. Forward pass
        # 3. Backward pass
        
    def _construct_MX(self, X, Fs):
        """
        Create Mx for all images in X, see equations in documentation for more details TODO
        Args:
            X = ndarray (h, w, 3, n) a matrix with n images
            Fs = ndarray (f, f, 3, nf) amtrix with nf square filters
        Returns:
            Mx: ndarray (n_p, f*f*3, n) matrix representation of X for convolution, 
            n_p is number of patches
        """
        f = Fs.shape[0]
        if X.shape[0] % f != 0 or X.shape[1] % f != 0:
            raise ValueError(
                f"Image size ({X.shape[0]}x{X.shape[1]}) "
                f"must be divisible by filter size f={f}"
            )
        
        
        stride_row=Fs.shape[0] # Stride = f for patchify
        stride_col=Fs.shape[1]
        d = Fs.shape[2]
        n = int(X.shape[3])
        patches_per_row = int(X.shape[0]/stride_row)
        patches_per_col = int(X.shape[1]/stride_col)
        n_p = patches_per_row*patches_per_col # number of patches
    
        MX = np.zeros((n_p, stride_row*stride_col*d, n))
        for i in range(n):
            subregion = 0
            for j in range(0,X.shape[0], stride_row):
                for k in range(0,X.shape[1], stride_col):
                    X_patch = X[j:j+stride_row,k:k+stride_col,:, i]
                    
                    MX[subregion, :, i] = X_patch.reshape((1, stride_row*stride_col*d), order='C')
                    subregion += 1 
        return MX    
    
    def _forward_pass(self, MX, network, use_bias=True):
        """
        Forward pass of back-propagation algorithm.
        
        Args:
            Mx: ndarray (n_p, f*f*3, n) matrix representation of X for convolution
            
            network dict with 
                Fs: ndarray (f, f, 3, nf) Filters of layer 1
                Fs_b: ndarray (nf,1) Convolution layer bias
                W: List of weights for layer 2 and 3 
                    [0]: W1 ndarray (nh, n_p * nf) nh= #nodes in layer 2 
                    [1]: W2 ndarray (K, nh)
                b: List of biases for layer 2 and 3 
                    [0]: b1 ndarray (nh, 1)
                    [1]: b2 ndarray (K, 1)
                
            self.convolver object that represents convolution layer
        
        Returns:
            h      ndarray (n_p * nf, n)  l1 output 
            X1     ndarray (n_h, n)       l2 output
            p      ndarray (10, n)        (l3) final class probabilities
            
        """
        
        
        
        
        conv_out = self.convolver.conv_mat_mul(MX, network['Fs'])
        if use_bias:
            conv_out += network['Fs_b'].reshape(1, network['Fs_b'].shape[0], 1) # Apply bias
        conv_out[conv_out<0] = 0 # ReLu
        npnf = network['W'][0].shape[1]
        n = MX.shape[2]
        h = np.fmax(conv_out.reshape((npnf, n), order='C'), 0)
       
        # Layer 2
        x1 = network["W"][0]@ h + network['b'][0]
        x1[x1<0] = 0 # ReLu
        
        # Layer 3
        s = network["W"][1]@ x1 + network['b'][1]
        p = self._soft_max(s)
   
        return h, x1, p
        
    def _soft_max(self, s):
        """SoftMax implemented with shifting to prevent overflow"""
        s_shift = s - np.max(s, axis=0, keepdims=True) # Shift to prevent overflow
        s_exp = np.exp(s_shift)
        P = s_exp / np.sum(s_exp, axis=0, keepdims=True) # We broadcast the columnwise sums to get P
        return P
        
        
    def _backward_pass(self, MX, Y, h, X1, p, network, n_f, n_p, use_bias=True, lam=0):
        """
        Performs the backward pass in network training
        
        Args:
            Mx: ndarray (n_p, f*f*3, nb) matrix representation of X for convolution
            Y: Kxn
            h      ndarray (n_p * nf, nb)  l1 output 
            X1     ndarray (n_h, nb)       l2 output
            p      ndarray (K, nb)        (l3) final class probabilities
            network: dict with 
                Fs: ndarray (f, f, 3, nf) Filters of layer 1
                Fs_b: ndarray (nf,1) Convolution layer bias
                W: List of weights for layer 2 and 3 
                    [0]: W1 ndarray (nh, n_p * nf) nh= #nodes in layer 2 or 3?
                    [1]: W2 ndarray (K, nh)
                b: List of biases for layer 2 and 3 
                    [0]: b1 ndarray (nh, 1)
                    [1]: b2 ndarray (K, 1)
            use_bias: Boolean to suggest if bias for convolutionlayer should be used (Fs_b)

        lam: float TODO
        
        Returns:
            grads: dict with
                Fs: ndarray (f, f, 3, nf) gradients of loss (TODO cost) relative to Fs
                Fs_b: ndarray (nf,1) gradients of loss (TODO cost) relative to conv layer bias
                W: List with gradients of loss (TODO cost) relative to W
                    [0]: W1 mxd
                    [1]: W2 Kxm
                b: List with gradients of loss (TODO cost) relative to b
                    [0]: b1 mx1
                    [1]: b2 Kx1
        """
        grads = {}
        nb = h.shape[1] # batch size
        
        # Grads of fully connected layers wrt loss (TODO cost)
        grads["W"] = [None]*2
        grads["b"] = [None]*2

        # 2. Grads of z1/s or s TODO wrt lossfunction TODO layer 3
        G = -(Y-p) # (K,nb) # dl/dp
        
        grads["W"][1] = 1/nb * G @ X1.T + 2*lam*network["W"][1] # (K,nh) = (K,nb)@(nb, nh)
        grads["b"][1] = 1/nb * G @ np.ones(nb).reshape(nb,1) 
       
        # 3. Propagate gradient to X1, then do ReLu from X1 to z/s of layer 2
        G = network["W"][1].T @ G # (nh, nb) = (nh, K) @ (K,nb) nh is num nodes in layer 2
        G = G * np.sign(X1) # (nh,nb), (nh,nb) 
        
         # 4. Grads of lossfunction TODO layer 2
        grads["W"][0] = 1/nb * G @ h.T + 2*lam*network["W"][0] # (nh, np*nf) = (nh,nb) @ (nb, np*nf)
        grads["b"][0] = 1/nb * G @ np.ones(nb).reshape(nb,1)  
        
        # 5. backprop to h node
        G_batch = network["W"][0].T @ G # (np*nf, nb) = (np*nf, nh) @ (nh, nb)
        
        # ReLu in L1 (reverse order from forward pass, since we stored h but not conv_out)
        G_batch = G_batch * np.sign(h) # (nh,nb), (nh,nb) 
        
        # Unflatten, GG is the grad of conv out in forward pass
        GG = G_batch.reshape((n_p, n_f, nb), order='C') #(n_p, nf, nb) (64, 2, 5) for debug
        
        if use_bias:
            # Get grads for convolution layer bias
            GG_filters = np.transpose(GG, (1, 0, 2))
            GG_filters = GG_filters.reshape(n_f, n_p * nb)

            grads["Fs_b"] = (
                1 / nb
                * GG_filters
                @ np.ones(n_p * nb).reshape(n_p * nb, 1)
            )

        # Einsum below does what the commented out code does
        """grads_fs_flat = np.zeros((MX.shape[1], GG.shape[1])) # (f*f*3, nf)Mx.T(i) @ GG(i) is ( f*f*3, n_p) @ (n_p, nf)
        for i in range(nb):
            grads_fs_flat += MX[:, :, i].T @ GG[:, :, i]
        grads["fs_flat"] = grads_fs_flat * 1/nb"""
        
        MXt = np.transpose(MX, (1, 0, 2))
        flat_grads = np.einsum('ijn, jln ->il', MXt, GG, optimize=True) * 1/nb
        
        Fs_flat = network["Fs"].reshape(MX.shape[1], n_f, order="C")
        flat_grads += 2*lam*Fs_flat
        grads["fs_flat"] = flat_grads # used for some tests
        grads["Fs"] = flat_grads.reshape(network["Fs"].shape, order="C")
        return grads
        
        
        
        
    
    def init_network(self,  nh, nf=2, f=2, channels=3, L=2, K=10):
        """
        Initializes the network with parameters. Uses He-initialization
        
        Args:
            nh: int         # nodes in layer 2 or 3? TODO
            n_p: int        # sub_patches in convolution layer
            nf: int         # filters applied in convolution layer
            f: int          height and width of filters
            channels: int   # channels in images
            L: int          # fully connected layers
            K: int          # classes 
            
            self.rng: random generator
        
        Returns:
            init_net: dict with 
                Fs: ndarray (f, f, channels, nf) Filters of convolution layer
                Fs_b: ndarray (nf,1) Convolution layer bias
                W: List of weights for layer 2 and 3 
                    [0]: W1 ndarray (nh, n_p * nf)
                    [1]: W2 ndarray (K, nh)
                b: List of biases for layer 2 and 3 
                    [0]: b1 ndarray (nh, 1)
                    [1]: b2 ndarray (K, 1)
        """
   
        n_p = int((32/f)**2) # Assumes 32x32 images and fxf filters
        
        init_net = {}
        init_net['W'] = [None]*L
        init_net['W'][0] = np.sqrt(2/n_p*nf)*self.rng.standard_normal(size = (nh, n_p*nf)) # He initialization
        init_net['W'][1] = np.sqrt(2/nh)*self.rng.standard_normal(size = (K, nh)) # He initialization
        
        init_net['b'] = [None]*L
        init_net['b'][0] = np.zeros((nh, 1))
        init_net['b'][1] = np.zeros((K, 1))
        
        init_net['Fs'] = np.sqrt(2/f)*self.rng.standard_normal(size = (f, f, channels, nf)) # He initialization TODO is n_in = f correct?
        init_net['Fs_b'] = np.zeros((nf, 1))
        
        self.network = init_net
        return init_net # For testing

    def training_cyclical(self, MX_train, Ytrain, y_train, MX_val, y_val,lam=0):
        """
        Uses mini-batch gradient descent with cyclical  learning rates.
        Updates self.network and stores intermediary performance on train and val in self.plotter
        
        Requires plotter != None
        Requires network ti have been initialized
        
        Args:
            MX_train: ndarray (n_p, f*f*3, n1) matrix representation of X for convolution
            MX_val: ndarray (n_p, f*f*3, n2) matrix representation of X for convolution
            init_net: dict with 
                Fs: ndarray (f, f, channels, nf) Filters of convolution layer
                Fs_b: ndarray (nf,1) Convolution layer bias
                W: List of weights for layer 2 and 3 
                    [0]: W1 ndarray (nh, n_p * nf)
                    [1]: W2 ndarray (K, nh)
                b: List of biases for layer 2 and 3 
                    [0]: b1 ndarray (nh, 1)
                    [1]: b2 ndarray (K, 1)
        
        """
        assert self.plotter != None
        GD_params = self.GD_params
        # Deep copy of network
        init_net = self.network
        
        trained_net = {}
        trained_net["W"] = init_net["W"].copy()
        trained_net["b"] = init_net["b"].copy()
        trained_net["Fs"] = init_net["Fs"].copy()
        trained_net["Fs_b"] = init_net["Fs_b"].copy()
        
        n_f = init_net["Fs_b"].shape[0]
        n_p = MX_train.shape[0]

        # For shuffling
        n = MX_train.shape[2]
        n_epochs_per_n_s = int(GD_params["n_s"]*GD_params["n_batch"]/n)

        # store initial performance on training and val
        self._save_performance(MX_train, y_train,MX_val, y_val, trained_net, lam, update_step=0)
        
        # Number of update steps
        for l in range(GD_params["half_cycles"]):
            # Every step
            is_rising = (l%2)==0

            for epoch in range(n_epochs_per_n_s):
                
                perm = np.random.permutation(n)
                X_train_shuf = MX_train[:,:, perm]
                Y_train_shuf = Ytrain[:, perm]
                batches_per_epoch = round(n/GD_params["n_batch"])
                
                for j in range(batches_per_epoch):
                    t = epoch*batches_per_epoch + j # Checked to be correct climbing from 0 to n_s
                    eta = helper_func.update_eta(GD_params, t, is_rising) # Checked to be correct
                    
                    j_start = j*GD_params["n_batch"]
                    j_end = (j+1)*GD_params["n_batch"]
                    #inds = j_start:j_end # This is not python code....
                    Xbatch = X_train_shuf[:,:, j_start:j_end]
                    Ybatch = Y_train_shuf[:, j_start:j_end]
                    
                    h, x1, p = self._forward_pass(Xbatch, trained_net)
                    
                    grads = self._backward_pass(Xbatch, Ybatch, h, x1, p, trained_net, n_f, n_p, use_bias=True, lam=lam)
                    
                    # Update W and b
                    for k in range(len(trained_net["W"])):
                        trained_net["W"][k] = trained_net["W"][k] - eta * grads["W"][k]
                        trained_net["b"][k] = trained_net["b"][k] - eta * grads["b"][k]
                    trained_net["Fs"] = trained_net["Fs"] - eta * grads["Fs"]
                    trained_net["Fs_b"] = trained_net["Fs_b"] - eta * grads["Fs_b"]
                              
                # After each epoch, store performance on training and val
                update_step = l*n_epochs_per_n_s*batches_per_epoch + epoch*batches_per_epoch + j
                self._save_performance(MX_train, y_train,MX_val, y_val, trained_net, lam, update_step) 
        
        self.network = trained_net

    
    def set_learning_parameters(self, n_batch=100, eta_min = 1e-5, eta_max = 1e-1, n_s=500, half_cycles=2 ):
        GD_params = {}
        GD_params["n_batch"] = n_batch
        GD_params["eta_min"] = eta_min 
        GD_params["eta_max"] = eta_max
        GD_params["n_s"]=n_s # n_s=k * n/n_batch for  k=[2,8]
        GD_params["half_cycles"]=half_cycles # Makes sure it is an even number so we end on lowest

        self.GD_params = GD_params
    
    def _save_performance(self, MX_train, y_train,MX_val, y_val, trained_net, lam, update_step):
        h, x1, p = self._forward_pass(MX_train, trained_net)
        loss_train = helper_func.compute_loss(p, y_train)
        cost_train = loss_train + lam* (np.sum((trained_net["W"][0]**2))+np.sum((trained_net["W"][1]**2)) + np.sum((trained_net["Fs"]**2))) 
        accuracy_train = helper_func.compute_accuracy(p, y_train)

        h, x1, p = self._forward_pass(MX_val, trained_net, use_bias=True)
        loss_val = helper_func.compute_loss(p, y_val)
        cost_val = loss_val + lam* (np.sum((trained_net["W"][0]**2))+np.sum((trained_net["W"][1]**2)) + np.sum((trained_net["Fs"]**2)))    
        accuracy_val = helper_func.compute_accuracy(p, y_val)
        
        self.plotter.add_update_step(loss_train,  cost_train, accuracy_train, loss_val, cost_val, accuracy_val, update_step)