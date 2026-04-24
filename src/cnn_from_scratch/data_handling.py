import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
from cnn_from_scratch import paths


def load_batch( filename):
    """ Loads and reshapes all data from one batch """

    with open(filename, 'rb') as fo:
        in_data = pickle.load(fo, encoding='bytes')

    X = in_data[b'data'].astype(np.float64) / 255.0 #dx
    X = X.transpose()
    y = in_data[b'labels']
    Y = np.eye(10)[y].T

    assert X.shape == (3072, 10000)
    assert Y.shape == (10, 10000)
    assert len(y) == 10000
    
    return X, Y, y

def load_all_data( val_size=5000):
    """ Loads train, val and test data. """
    

    tr_filenames = []
    tr_filenames.append(paths.RAW_DATA / "data_batch_1")
    tr_filenames.append(paths.RAW_DATA / "data_batch_2")
    tr_filenames.append(paths.RAW_DATA / "data_batch_3")
    tr_filenames.append(paths.RAW_DATA / "data_batch_4")
    tr_filenames.append(paths.RAW_DATA / "data_batch_5")

    
    filename_test = paths.RAW_DATA / "test_batch"
    X_in, Y_in, y_in = load_batch(tr_filenames[0]) # Initialize
    for i in range(1,5):
        X, Y, y = load_batch(tr_filenames[i])
        X_in = np.concatenate((X_in, X), axis=1)
        Y_in = np.concatenate((Y_in, Y), axis=1)
        y_in.extend(y)

    X_tr = X_in[:, val_size:]
    X_val = X_in[:,:val_size]
    
    Y_tr = Y_in[:, val_size:]
    Y_val = Y_in[:,:val_size]

    y_tr = y_in[val_size:]
    y_val = y_in[:val_size]

    X_te, Y_te, y_te = load_batch(filename_test)
    
    return X_tr, Y_tr, y_tr, X_val, Y_val, y_val, X_te, Y_te, y_te 

def load_some_data(val_size=1000):
    """ Loads train, val and test a small set part of data. Used for testing mainly. """
    
    tr_filenames = []
    tr_filenames.append(paths.RAW_DATA / "data_batch_1")

    filename_test = paths.RAW_DATA / "test_batch"
    X_in, Y_in, y_in = load_batch(tr_filenames[0]) # Initialize

    X_tr = X_in[:, val_size:]
    X_val = X_in[:,:val_size]
    
    Y_tr = Y_in[:, val_size:]
    Y_val = Y_in[:,:val_size]

    y_tr = y_in[val_size:]
    y_val = y_in[:val_size]

    X_te, Y_te, y_te = load_batch(filename_test)
    
    return X_tr, Y_tr, y_tr, X_val, Y_val, y_val, X_te, Y_te, y_te 

def pre_processing( X_tr, X_val, X_te):
    """ Normalizes X """
    d = X_tr.shape[0]
    assert d == X_val.shape[0] and d == X_te.shape[0]
    mean_X = np.mean(X_tr, axis=1).reshape(d,1)
    std_X = np.std(X_tr, axis=1).reshape(d,1)

    X_tr_c = (X_tr - mean_X) / std_X
    X_val_c = (X_val - mean_X) / std_X
    X_te_c = (X_te - mean_X) / std_X
    return X_tr_c, X_val_c, X_te_c

def construct_MX(X, f, d):
        """
        Create Mx for all images in X, see equations in documentation for more details TODO
        Args:
            X: ndarray (h * w * channels, n) a matrix with n images
            f int: h and w of filters
            d: int # of channels of X and filters
            
        Returns:
            Mx: ndarray (n_p, f*f*3, n) matrix representation of X for convolution, 
            n_p is number of patches
        """
        
        # Reshape data in ndarray (h, w, 3, n) a matrix with n images
        n = X.shape[1]
        X = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3)) # Hardcoded values for CIFAR dataset images
        
        n = int(X.shape[3])
        patches_per_row = int(X.shape[0]/f)
        patches_per_col = int(X.shape[1]/f)
        n_p = patches_per_row*patches_per_col # number of patches
    
        MX = np.zeros((n_p, f*f*d, n))
        for i in range(n):
            subregion = 0
            for j in range(0,X.shape[0], f):
                for k in range(0,X.shape[1], f):
                    X_patch = X[j:j+f,k:k+f,:, i]
                    
                    MX[subregion, :, i] = X_patch.reshape((1, f*f*d), order='C')
                    subregion += 1 
        return MX  

def save_MX_data(f=2, d=3, val_size=1000):
    """ Loads, splits, cleans, converts X into MX and saves data (if not already exists) 
    
    returns
        file_location: pathlib object with location of stored data
        """
    file_location = paths.PROCESSED_DATA / f"dataset_f{f}_d{d}_val{val_size}.npz"
    
    if file_location.exists():
        print("Loading cached data set")
        return file_location

    X_tr, Y_tr, y_tr, X_val, Y_val, y_val, X_te, Y_te, y_te  = load_all_data(val_size) 
    X_tr_clean, X_val_clean, X_test_clean = pre_processing(X_tr, X_val, X_te)
    
    print("Building MX for training data")
    MX_tr = construct_MX(X_tr_clean, f, d)
    
    print("Building MX for validation data")
    MX_val = construct_MX(X_val_clean, f, d)
    
    print("Building MX for test data")
    MX_test = construct_MX(X_test_clean, f, d)
    
    np.savez(
        file_location,
        MX_tr=MX_tr,
        Y_tr=Y_tr,
        y_tr=y_tr,
        MX_val=MX_val,
        Y_val=Y_val,
        y_val=y_val,
        MX_test=MX_test,
        Y_te=Y_te,
        y_te=y_te,)
    
    return file_location
    
def get_MX_data(f=2, d=3, val_size=1000):
    """ Gets data. Checks if it already exists, if not, store it first and then load it. 
    
    return
        dict[str, np.ndarray]
            Dictionary containing:
                MX_tr, Y_tr, y_tr,
                MX_val, Y_val, y_val,
                MX_test, Y_te, y_te
    """
    path = save_MX_data(f, d, val_size)
    return dict(np.load(path))