import numpy as np

class Convolver():
    
    def __init__(self):
        pass
    
    def conv_for_loop(self, X, Fs):
        """
        Performs convolution using for loops.
        Should work, but not tested, when height != width of image
        
        Args:
            X = X (32, 32, 3, n)
            Fs = Fs (f, f, 3, nf)
        Returns:
            conv_outputs = ndarray (32/f, 32/f, nf, n)
        """
        stride_row=Fs.shape[0] # Stride = f for patchify
        stride_col=Fs.shape[1]
        n = int(X.shape[3])
        nf = int(Fs.shape[3])
        patches_per_row = int(X.shape[0]/stride_row)
        patches_per_col = int(X.shape[1]/stride_col)
        
        conv_output = np.zeros((patches_per_row, patches_per_col, nf,n))
        
        for i in range(n):
            for j in range(0,X.shape[0], stride_row):
                pj = int(j/stride_row) # Patch index
                
                for k in range(0,X.shape[1], stride_col):
                    pk = int(k/stride_col) # Patch index
                    # Calc subpatch
                    sub_patch = X[j:j+stride_row,
                                      k:k+stride_col,
                                      :,
                                      i]
                    
                    for l in range(nf):
                        dot_prod = np.sum(np.multiply(sub_patch, Fs[:, :, :, l]))
                        conv_output[pj, pk, l, i] = dot_prod
        
        return conv_output

    def conv_mat_mul(self, MX, Fs):
        """
        Performs convolution using efficient matrix multiplication.
        Assumes that height == width of image
        
        Args:
            X = ndarray (h, w, 3, n)
            Fs = ndarray (f, f, 3, nf)
        Returns:
            conv_outputs = ndarray (h/f, w/f, nf, n)
        """
        
        n_p = MX.shape[0]
        n_f = Fs.shape[3]
        n = MX.shape[2]
        
        Fs_flat = Fs.reshape((Fs.shape[0]*Fs.shape[1]*Fs.shape[2], n_f), order='C') #(fxfxd, n_f)

        # Einsum does the commented out code in a faster way
        """conv_outputs_mat = np.zeros((n_p, n_f, n))
        # Perform convolution for each image
        for i in range(n):
            conv_outputs_mat[:, :, i] = np.matmul(MX[:, :, i], Fs_flat)"""
            
        conv_outputs_mat = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)
        return conv_outputs_mat
                   
    
    
