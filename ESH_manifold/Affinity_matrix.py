
import numpy as np
from utilities import to_Z
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import KMeans

def Affinity(train_data, Z_spec, random_state=42, init='random'):
    """
    inputs:
        train_data shape is (n_samples,n_features)
        Z_spec: a dictionary that specifies how to compute Z. Keys should be
                'n_anchors', 's', 'sigma', and 'metric'
        random_state: Must be integer
        init: method of anchor initialization. Can be 'random' or 'kmeans'
    outputs: 
        Z: data-to-anchor mapping of train data. shape is (n_samples, n_anchors)
        anchors: anchor points. shape is (n_anchors, n_features)
    """
    
    N = train_data.shape[0] # number of samples
    n_anchors = Z_spec['n_anchors']
    R = np.random.RandomState(random_state)
    if init.lower()=='random':
        anchors = train_data[R.choice(N, size=n_anchors, replace=False),:]
    elif init.lower()=='kmeans':
        kmeans = KMeans(n_clusters=n_anchors, max_iter=10, n_init=1, 
                        random_state=random_state).fit(train_data)
        anchors = kmeans.cluster_centers_
    else:
        raise ValueError('init must be "random" or "kmeans"')
        
    Z = to_Z(train_data, anchors, Z_spec)

    return Z, anchors
    
