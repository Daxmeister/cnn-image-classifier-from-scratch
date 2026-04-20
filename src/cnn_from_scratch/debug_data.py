import numpy as np

class Debug_data():
    """Class for handling debug data"""
    
    def __init__(self, debug_file='debug_info.npz'):
        self.load_data = np.load(debug_file)
        X = self.load_data['X']
        self.Fs = self.load_data['Fs']
        n = X.shape[1]
        self.X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    
    def get_images_and_filters(self):
        """ Returns images of correct shape, ready for convolution
        
            Returns:
                X_ims: ndarray of shape (32, 32, 3, 5) w/ images in correct shape 
                Fs: nd array of shape (4, 4, 3, 2) w/ Filters           
        """
        
        return self.X_ims, self.Fs

    def compare_conv_out_to_ground_truth(self, convolver):
        conv_output = convolver.conv_for_loop(self.X_ims, self.Fs)
        ground_truth= self.load_data['conv_outputs']
        print("Shape convoler is:", conv_output.shape, 
              " should be: ", ground_truth.shape, 
              "Is_same? ", conv_output.shape==ground_truth.shape)
        print(np.sum(conv_output-ground_truth))