
from scipy.sparse.linalg import eigsh
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import linalg as lin 
trace=lin.trace
matmul=lin.matmul
svd=lin.svd

from utilities import normalize_Z


def get_feature_affinity(train_features, Z):
    Z = normalize_Z(Z)
    L = train_features.T@Z
    Affinity = L@L.T
    return Affinity


def initialize_W(W=None, Affinity=None, K=16):
    if W is None:
        eig_val, eig_vec = eigsh(Affinity, k=K, which='LA')
        W =  tf.Variable(eig_vec, dtype=tf.float32, name="W")
    else:
        W = tf.Variable(W, dtype=tf.float32, name="W")
    return W


def cost_fn(train_features, W, Affinity, alpha=1):
    main_cost = -1*trace(matmul(tf.transpose(W), matmul(Affinity, W)))
    projected_values = matmul(train_features,W)
    #ones1 = tf.ones((train_features.shape[0], W.shape[1]), dtype=np.float32)
    reg = tf.math.square(tf.norm(tf.math.subtract(tf.math.abs(projected_values),
                                                   1), ord='fro', axis=[0,1]))                                               
    cost = (main_cost + 0.5*alpha*reg)/train_features.shape[0]
    return cost


def compute_alpha(train_features, W, Affinity):
    main_cost = -1*trace(matmul(tf.transpose(W), matmul(Affinity, W)))
    projected_values = matmul(train_features,W)
    reg = tf.math.square(tf.norm(tf.math.subtract(tf.math.abs(projected_values),
                                                   1), ord='fro', axis=[0,1]))
    alpha = tf.math.abs(2*main_cost/reg)
    return alpha


def ESH_projected_grad(train_features, Z, K=16, alpha=None, lr=0.01, 
                       maxiter=10000, W=None):
    """ Efficient Spectral hashing using projected gradient solver
    train_features: shape is (n_samples,n_features)
    Z: data-to-anchor mapping of train data. shape is (n_samples, n_anchors)
    K: Number of bits for each binary code
    alpha: the regularization coefficient for binary constraint. If None,
        compute_alpha function is used.
    lr: learning rate for gradient descent
    maxiter: maximum number of iterations used for ESH solver
    W: the initialization matrix used for hash functions. If None, eigenvectors
        of feature affinity matrix is used as initialization of W. 
    
    Output: 
        W: The learned hash functions. It has shape (n_features,K)
        cost_values: cost values until last convergence
    """ 
    # compute feature affinity
    Affinity = get_feature_affinity(train_features, Z)
    # initialize W
    W = initialize_W(W=W, Affinity=Affinity, K=K)
    # setting constants
    Affinity = tf.constant(Affinity, dtype=np.float32)
    train_features = tf.constant(train_features, dtype=np.float32)
    cost_values = np.zeros((maxiter,))
    optimizer = tf.keras.optimizers.SGD(lr=lr)
    if alpha is None:
        alpha = compute_alpha(train_features, W, Affinity) 
        print(f"alpha={alpha} is selected")
    
    # main loop
    for it in range(maxiter):
        with tf.GradientTape(persistent=True) as tape:
            # define cost
            cost = cost_fn(train_features, W, Affinity, alpha=alpha) 
            cost_values[it] = cost.numpy()                      
        # compute gradient
        grad = tape.gradient(cost, W)
        del tape
        # apply optimizer
        optimizer.apply_gradients(zip([grad], [W]))        
        # apply projection
        diag_mat, U, V = svd(W, full_matrices=False)
        W.assign(matmul(U,tf.transpose(V)), read_value=False) 
        # print cost
        if (it+1)%50==0:
            print(f'cost value after {it+1} iterations: {cost_values[it]}')
        # convergence check
        if it%10==0 and it>0:
            convg_check = (cost_values[it] - cost_values[it-10])/cost_values[it-10]
            if np.abs(convg_check)<1e-4:
                print(f"The problem converged after {it+1} iteration")
                break
    return W.numpy(), cost_values[:it+1]

