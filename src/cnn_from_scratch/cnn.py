import numpy as np

class CNN():
    
    def __init__(self, convolver):
        self.network = {}
        self.convolver = convolver
        self.rng = np.random.default_rng(42)
    
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
        
        # 1. Load parameters
        # 2. Forward pass
        # 3. Backward pass
        
        
    
    def _forward_pass(self, X_tr, debug_network=None):
        """
        Forward pass of back-propagation algorithm
        
        Args:
            X_tr: ndarray (h, w, 3, n) training data (not MX representation)
            
            self.network dict with 
                Fs: ndarray (f, f, 3, nf) Filters of layer 1
                W: List of weights for layer 2 and 3 
                    [0]: W1 ndarray (nh, n_p * nf) nh= #nodes in layer 2 or 3?
                    [1]: W2 ndarray (10, nh)
                b: List of biases for layer 2 and 3 
                    [0]: b1 ndarray (nh, 1)
                    [1]: b2 ndarray (10, 1)
                
            self.convolver object that represents convolution layer
        
        Returns:
            h      ndarray (n_p * nf, n)  l1 output 
            X1     ndarray (n_h, n)       l2 output
            p      ndarray (10, n)        (l3) final class probabilities
            
        """
        
        
        if debug_network != None: # For testing
            self.network = debug_network
        
        conv_out = self.convolver.conv_mat_mul(X_tr, self.network['Fs'])
        conv_out[conv_out<0] = 0 # ReLu
        npnf = self.network['W'][0].shape[1]
        n = X_tr.shape[3]
        h = np.fmax(conv_out.reshape((npnf, n), order='C'), 0)
       
        # Layer 2
        x1 = self.network["W"][0]@ h + self.network['b'][0]
        x1[x1<0] = 0 # ReLu
        
        # Layer 3
        s = self.network["W"][1]@ x1 + self.network['b'][1]
        p = self._soft_max(s)
   
        return h, x1, p
        
    def _soft_max(self, s):
        """SoftMax implemented with shifting to prevent overflow"""
        s_shift = s - np.max(s, axis=0, keepdims=True) # Shift to prevent overflow
        s_exp = np.exp(s_shift)
        P = s_exp / np.sum(s_exp, axis=0, keepdims=True) # We broadcast the columnwise sums to get P
        return P
        
        
        
        
        
        
        
    
    def _init_network(self,  nh, n_p, nf=2, f=3, channels=3, L=2, K=10, Fs_debug=None):
        """
        Initializes the network with parameters
        TODO finish this later
        
        Args:
            nh: int # nodes in layer 2 or 3? TODO
            n_p: int # sub_patches in convolution layer
            nf: int # filters applied in convolution layer
            f: int height and width of filters
            channels: # channels in images
            L: int # fully connected layers
            K: int # classes 
            
            self.rng: random generator
        
        Returns:
            init_net: dict with 
                Fs: ndarray (f, f, 3, nf) Filters of layer 1
                W: List of weights for layer 2 and 3 
                    [0]: W1 ndarray (nh, n_p * nf)
                    [1]: W2 ndarray (10, nh)
                b: List of biases for layer 2 and 3 
                    [0]: b1 ndarray (nh, 1)
                    [1]: b2 ndarray (10, 1)
        """
        
        init_net = {}
        init_net['W'] = [None]*L
        init_net['W'][0] = 1/np.sqrt(n_p*nf)*self.rng.standard_normal(size = (nh, n_p*nf))
        init_net['W'][1] = 1/np.sqrt(nh)*self.rng.standard_normal(size = (K, nh))
        
        init_net['b'] = [None]*L
        init_net['b'][0] = np.zeros((nh, 1))
        init_net['b'][1] = np.zeros((K, 1))
        
        if Fs_debug != None:
            init_net['Fs'] = Fs_debug
        else:
            # TODO
            assert True == False
            
        return init_net

        
        
        