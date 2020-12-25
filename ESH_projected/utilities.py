
import numpy as np 
from numpy.linalg import inv
from scipy.spatial.distance import cdist

def one_hot_encode(x):
    """
    One hot encodes a list of sample labels. Return a one-hot encoded vector 
        for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    n_classes = len(np.unique(x))
    return np.eye(n_classes)[x]


def normalize_Z(Z):
    D = np.diag(1/np.sqrt(np.sum(Z,axis=0)))
    Z_normalized = Z@D
    return Z_normalized


# Least-square estimation
def RRC(tr_data, tr_labels, lambda_, testing=False):
    """
    tr_data: train data. Shape must be (n_samples,n_features)
    tr_labels: train labels. Shape must be (n_samples,), or (n_samples,1), or 
        (n_samples,n_classes)
    returns projection matrix
    """
    
    n_samples, n_features = tr_data.shape
    
    # projection matrix computing
    if n_samples<n_features:
        Proj_M = tr_data.T@(inv(tr_data@tr_data.T + lambda_*np.eye(n_samples)))
    else:
        Proj_M = (inv(tr_data.T@tr_data + lambda_*np.eye(n_features)))@tr_data.T
        
    # label matrix
    if len(tr_labels.shape)<2 or tr_labels.shape[1]==1:
        # One-hot coding (only use it for multi-class (single-label) classification)
        Y = one_hot_encode(tr_labels.flatten())
    else:
        Y = tr_labels
        
    W = Proj_M@Y

    return W
    

def to_Z(inputs, anchors, Z_spec):
    """ 
    This function handles data-to-anchor mapping of inputs data. In other 
    words, the inputs representation is mapped to anchor-based representation.
    Inputs:
        inputs: input samples. Shape is (N, n_features) which N is the number of
                inputs samples.
        anchors: anchor points. shape is (n_anchors, n_features).
        Z_spec: a dictionary that specifies how to compute Z. Keys should be
                'n_anchors', 's', 'sigma', and 'metric'. 
                n_anchors: Number of anchors to compute (sparse) affinity matrix
                s: number of selection for nearest anchors
                metric: Distance metric used for computing affinity matrix
                        It is used in cdist function
                sigma: bandwitch to normalize distances
                
    Output:
        Z: anchor-based represntation of inputs samples. shape is (N,n_anchors)
    """
    if len(inputs.shape)==1:
        N = 1
        inputs = inputs[None]
    else:
        N = inputs.shape[0]
    
    n_anchors = anchors.shape[0] 
    assert Z_spec['n_anchors'] == n_anchors 
    s = Z_spec['s']
    sigma = Z_spec['sigma']
    Z = np.zeros((N,n_anchors), dtype='float32') # the truncated similarities
    Dis = np.float32(cdist(inputs, anchors, metric=Z_spec['metric']))
    
    # select s nearest neighbors of each example
    min_val = np.zeros((N,s), dtype='float32')
    min_pos = np.zeros((N,s), dtype='int')
    
    for i in range(s):
        min_pos[:,i] = np.argmin(Dis, axis=1)
        min_val[:,i] = Dis[np.arange(N), min_pos[:,i]]
        Dis[np.arange(N), min_pos[:,i]] = np.inf
    
    del Dis 
    
    if sigma is None:
        sigma = np.mean(min_val[:,-1])
    
    min_val = np.exp(-((min_val/sigma)**2))
    min_val = min_val/np.sum(min_val, axis=1, keepdims=True)    
    
    # fill Z matrix
    for i in range(s):   
        Z[np.arange(N), min_pos[:,i]] = min_val[:,i]
    
    return Z