import numpy as np

class Convolver():
    
    def __init__(self):
        pass
    
    def conv_for_loop(self, X, Fs):
        """
        Performs convolution using for loops.
        Assumes that height == width of image
        
        Args:
            X = X (32, 32, 3, n)
            Fs = Fs (f, f, 3, nf)
        Returns:
            conv_outputs = ndarray (32/f, 32/f, nf, n)
        """
        stride=Fs.shape[0] # Stride = f for patcihify
        n = int(X.shape[3])
        nf = int(Fs.shape[3])
        patches_per_row = int(X.shape[0]/stride)
        patches_per_col = int(X.shape[1]/stride)
        
        conv_output = np.zeros((patches_per_row, patches_per_col, nf,n))
        
        for i in range(n):
            for j in range(0,X.shape[0], stride):
                pj = int(j/stride) # Patch index
                
                for k in range(0,X.shape[1], stride):
                    pk = int(k/stride) # Patch index
                    # Calc subpatch
                    sub_patch = X[j:j+stride,
                                      k:k+stride,
                                      :,
                                      i]
                    
                    for l in range(nf):
                        dot_prod = np.sum(np.multiply(sub_patch, Fs[:, :, :, l]))
                        conv_output[pj, pk, l, i] = dot_prod
        
        return conv_output
                        
            

        