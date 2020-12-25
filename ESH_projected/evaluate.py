
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist


# evaluation based on label information
def is_multilabel(y_train):
    shape_ = y_train.shape
    if (len(shape_)==1) or shape_[1]==1:
        return False
    
#    y_train = y_train>0 
#    n_labels_per_sample = np.sum(y_train, axis=1)
#    return np.any(n_labels_per_sample>1)
    return True


def precision_recall(x_train, y_train, x_test, y_test, M_set, weights=None):
    """
    x_train and x_test are of boolean type and have shape (x, K) which 
        x is number of observations and K is the number of bits. 
    y_train and y_test have shape (x,) which x is number of observations. These
        two vectors contain labels. They can also be matrices (in one-hot or 
        multi-label case).
    M_set contains the (sorted) points in which precision and recall are 
        computed. In other words, if m be the member of M_set, precision@m and 
        recall@m are computed after retrieving m nearest neighbors to the query. 
    weights: have shape (K,)
    
    return: Precision and recall for each member of (sorted) M_set.
    
    Note: x_train and x_test must have (0,1) values not (-1,1) values.
    """
    M_set = np.sort(np.array(M_set, dtype=np.uint))
    n_test = x_test.shape[0]
    K = x_train.shape[1] # number of bits
    precision = np.zeros((n_test, len(M_set)), dtype='float32') 
    recall = np.zeros((n_test, len(M_set)), dtype='float32') 
    M = np.max(M_set)
    x_train = np.uint16(x_train) # convert boolean type to uint
    x_test = np.uint16(x_test)
    if weights is None:
        weights = 1
        normalization_term = K
    else:
        weights = weights.flatten()
        assert(len(weights) == K)
        normalization_term = np.sum(np.abs(weights))

    # big data flag
    big_flag = True
    if x_train.shape[0]*x_test.shape[0]<=0.5e9:
        big_flag = False
        if not isinstance(weights,int):
            weights = np.tile(weights[None],[n_test,1])
    
    if big_flag==False:
        hamm_dist_mat = normalization_term - (x_train@(x_test*weights).T + \
                                              (1-x_train)@((1-x_test)*weights).T)
        arg_dist_mat = np.argsort(hamm_dist_mat, axis=0)
        del hamm_dist_mat
        
    # compute precision and recall for each query
    is_multi_label = is_multilabel(y_train)
    if is_multi_label==False:
        real_pos_per_class = Counter(y_train)
        
    for i in range(n_test):
        
        if big_flag==False:
            # top (nearset) M neighbors to the query
            arg_dist = arg_dist_mat[:M,i] 
        else:
            # hamming distance between train_data and the query
            hamm_dist = normalization_term - (x_train@(x_test[i,]*weights) + \
                                              (1-x_train)@((1-x_test[i,])*weights)) 
            # top (nearset) M neighbors to the query
            arg_dist = np.argsort(hamm_dist)[:M] 
        
        if is_multi_label: # multi-label case
            ### for multi-label case, We define the true neighbors of a query 
            ### as the images sharing at least one labels with the query image.
            is_correct = (y_train@y_test[i,:].T)>0
            total_Positives = np.sum(is_correct) # used for recall computation
            is_correct = is_correct[arg_dist]
        else: # multi-class case
            q_label = y_test[i] # query label   
            is_correct = (q_label==y_train[arg_dist])
            total_Positives = real_pos_per_class[q_label]
            
        # compute precision and recall for each m
        is_correct_sum = np.cumsum(is_correct)
        TP = is_correct_sum[M_set-1] # true positives
        precision[i,:] = TP/M_set
        recall[i,:] = TP/total_Positives
        
    # compute average precision and recall
    precision = np.mean(precision, axis=0)
    recall = np.mean(recall, axis=0)
    
    return precision, recall, M_set

         
def precision_radius(x_train, y_train, x_test, y_test, Radius=2):
    """
    x_train and x_test are of boolean type and have shape (x, K) which
        x is number of observations and K is the number of bits.
    y_train and y_test have shape (x,) which x is number of observations. These
        two vectors contain labels. They can also be matrices (in one-hot or 
        multi-label case).
    Radius is the radius of hamming ball for search.
    
    return: Precision @ Radius=Radius
    
    Note: x_train and x_test must have (0,1) values not (-1,1) values.
    """
    n_test = x_test.shape[0]
    K = x_train.shape[1] # number of bits
    precision = np.zeros((n_test), dtype='float32') 
    
    x_train = np.uint16(x_train) # convert boolean type to uint
    x_test = np.uint16(x_test)
    
    normalization_term = K
        
    # compute precision and recall for each query
    is_multi_label = is_multilabel(y_train)

    for i in range(n_test):
        # hamming distance between train_data and the query
        hamm_dist = normalization_term - (x_train@(x_test[i,]) + \
                                          (1-x_train)@((1-x_test[i,]))) 
        idx_radius = hamm_dist <= Radius
        m = np.sum(idx_radius)
        if m==0:
            precision[i]=0
        else:
            y_train_radius = y_train[idx_radius]
            
            if is_multi_label: # multi-label case
                ### for multi-label case, We define the true neighbors of a query 
                ### as the images sharing at least one labels with the query image.
                is_correct = (y_train_radius@y_test[i,:].T)>0
                
            else: # multi-class case
                q_label = y_test[i] # query label   
                is_correct = (q_label==y_train_radius)
                
            # compute precision for images in radius 2
            
            TP = np.sum(is_correct) # true positives
            precision[i] = TP/m
    
        
    # compute average precision and recall
    precision = np.mean(precision, axis=0)
    
    return precision
    

def interpolated_precision_recall(precisions, recalls):
    """
    for ref see the link below:
    # https://github.com/rafaelpadilla/Object-Detection-Metrics
    """
    idx = np.argsort(recalls)
    recalls = recalls[idx]
    precisions = precisions[idx]
    precisions_intp = np.zeros((len(precisions),))
    for i,r in enumerate(recalls):
        bool_index = recalls>=r
        precisions_intp[i] = np.max(precisions[bool_index])
        
    return precisions_intp, recalls


def AP(x_train, y_train, x_test, y_test, num_return_NN=None, weights=None):
    """
    x_train and x_test are of boolean type and have shape (x,K) which
        x is number of observations and K is the number of bits.
    y_train and y_test have shape (x,) which x is number of observations. These
        two vectors contain labels. They can also be matrices (in one-hot or 
        multi-label case).
    num_return_NN: only compute mAP on returned top num_return_NN neighbours.
    weights: have shape (K,)
    
    return: average precision (AP) per query.
    
    Note: x_train and x_test must have (0,1) values not (-1,1) values.
    """
    n_test = x_test.shape[0]
    if num_return_NN is None:
        num_return_NN = x_train.shape[0] 
    K = x_train.shape[1] # number of bits
    APall = np.zeros((n_test,))
    x_train = np.uint16(x_train) # convert boolean type to uint
    x_test = np.uint16(x_test)
    if weights is None:
        weights = 1
        normalization_term = K
    else:
        weights = weights.flatten()
        assert(len(weights) == K)
        normalization_term = np.sum(np.abs(weights))
    
    # big data flag
    big_flag = True
    if x_train.shape[0]*x_test.shape[0]<=0.5e9:
        big_flag = False
        if not isinstance(weights,int):
            weights = np.tile(weights[None],[n_test,1])
        
    if big_flag==False:
        hamm_dist_mat = normalization_term - (x_train@(x_test*weights).T + \
                                              (1-x_train)@((1-x_test)*weights).T)
        arg_dist_mat = np.argsort(hamm_dist_mat, axis=0)
        del hamm_dist_mat
        
    # compute AP for each query
    is_multi_label = is_multilabel(y_train)
    for i in range(n_test):
        
        if big_flag==False:
            # top (nearset) neighbors to the query
            arg_dist = arg_dist_mat[:num_return_NN,i] 
        else:
            # hamming distance between train_data and the query
            hamm_dist = normalization_term - (x_train@(x_test[i,]*weights) + \
                                              (1-x_train)@((1-x_test[i,])*weights)) 
            # top (nearset) neighbors to the query
            arg_dist = np.argsort(hamm_dist)[:num_return_NN] 
            
        if is_multi_label: # multi-label case
            ### for multi-label case, We define the true neighbors of a query 
            ### as the images sharing at least one labels with the query image.
            is_correct = (y_train[arg_dist,:]@y_test[i,:].T)>0
        else: # multi-class case
            q_label = y_test[i] # query label   
            is_correct = (q_label==y_train[arg_dist])
        
        TP_loc = np.where(is_correct)[0]
        n_TP = len(TP_loc) # total number of true positives in num_return_NN
        if n_TP==0:
            APall[i] = 0
        else:
            precisions = (np.arange(1,n_TP+1))/(TP_loc+1) # calculate precisions 
            # only at the true positive locations
            APall[i] = np.sum(precisions)/n_TP
            
    return APall


def mAP(x_train, y_train, x_test, y_test, num_return_NN=None, weights=None):
    """
    x_train and x_test are of boolean type and have shape (x,K) which 
        x is number of observations and K is the number of bits.
    y_train and y_test have shape (x,) which x is number of observations. These
        two vectors contain labels. They can also be matrices (in one-hot or 
        multi-label case).
    num_return_NN: only compute mAP on returned top num_return_NN neighbours.
    weights: have shape (K,)
    
    return: mean average precision (mAP).
    
    Note: x_train and x_test must have (0,1) values not (-1,1) values.
    """
    ap_all = AP(x_train, y_train, x_test, y_test, num_return_NN=None, weights=None)
    return np.mean(ap_all)


def return_all_metrics(x_train, y_train, x_test, y_test, M_set, weights=None, 
                       num_return_NN=None, Radius=None):
    """
    This function is useful for large datasets as it computes hamming distance 
        between train data and query only once (instead of computing it for each
        metric seperately).
    
    It returns:
      1- Mean average precision (mAP) 
      2- Precision and recall for each member of (sorted) M_set
      3- Precision @ Radius=Radius if radius is not None
    
    x_train and x_test are of boolean type and have shape (x,K) which
        x is number of observations and K is the number of bits.
    y_train and y_test have shape (x,) which x is number of observations. These
        two vectors contain labels. They can also be matrices (in one-hot or 
        multi-label case).
    M_set contains the (sorted) points in which precision and recall are 
        computed. In other words, if m be the member of M_set, precision@m and
        recall@m are computed after retrieving m nearest neighbors to the query. 
    weights: have shape (K,)
    num_return_NN: only compute mAP on returned top num_return_NN neighbours.
    Radius is the radius of hamming ball for search.

    Note: x_train and x_test must have (0,1) values not (-1,1) values.
    """
    if (weights is not None) and (Radius is not None):
        print('You cannot specify "weights" argument when you want to compute \
              the radious-based percision')
        return None
    
    if num_return_NN is None:
        num_return_NN = x_train.shape[0]  
    
    M_set = np.sort(np.array(M_set, dtype=np.uint))
    n_test = x_test.shape[0]
    K = x_train.shape[1] # number of bits
    precision = np.zeros((n_test, len(M_set)), dtype='float32') 
    recall = np.zeros((n_test, len(M_set)), dtype='float32') 
    APall = np.zeros((n_test,))  
    precision_R = np.zeros((n_test), dtype='float32') 
    M = np.max(M_set)
    x_train = np.uint16(x_train) # convert boolean type to uint
    x_test = np.uint16(x_test)
    
    if weights is None:
        weights = 1
        normalization_term = K
    else:
        weights = weights.flatten()
        assert(len(weights) == K)
        normalization_term = np.sum(np.abs(weights))
        
    # compute precision and recall for each query
    is_multi_label = is_multilabel(y_train)
    if is_multi_label==False:
        real_pos_per_class = Counter(y_train)
        
    for i in range(n_test):
        # hamming distance between train_data and the query
        hamm_dist = normalization_term - (x_train@(x_test[i,]*weights) + \
                                          (1-x_train)@((1-x_test[i,])*weights)) 
        # top (nearset) M neighbors to the query
        arg_dist = np.argsort(hamm_dist) 
        arg_dist_PR = arg_dist[:M]
        arg_dist_mAP = arg_dist[:num_return_NN]
        idx_radius = hamm_dist <= Radius
        m_radius = np.sum(idx_radius)
        y_train_radius = y_train[idx_radius]
        
        if is_multi_label: # multi-label case
            ### for multi-label case, We define the true neighbors of a query 
            ### as the images sharing at least one labels with the query image.
            is_correct = (y_train@y_test[i,:].T)>0
            total_Positives = np.sum(is_correct) # used for recall computation
            is_correct_PR = is_correct[arg_dist_PR]
            is_correct_mAP  = is_correct[arg_dist_mAP]
            is_correct_radius = (y_train_radius@y_test[i,:].T)>0
        else: # multi-class case
            q_label = y_test[i] # query label   
            is_correct_PR = (q_label==y_train[arg_dist_PR])
            is_correct_mAP = (q_label==y_train[arg_dist_mAP])
            is_correct_radius = (q_label==y_train_radius)
            total_Positives = real_pos_per_class[q_label]
         
        # compute AP for each query
        TP_loc = np.where(is_correct_mAP)[0]
        n_TP = len(TP_loc) # total number of true positives in num_return_NN
        if n_TP!=0:
            # calculate precisions only at the true positive locations
            precisions = (np.arange(1,n_TP+1))/(TP_loc+1) 
            APall[i] = np.sum(precisions)/n_TP
        
        # compute precision and recall for each m
        is_correct_sum = np.cumsum(is_correct_PR)
        TP = is_correct_sum[M_set-1] # true positives
        precision[i,:] = TP/M_set
        recall[i,:] = TP/total_Positives
        
        # compute precision_radius
        if (m_radius!=0) and (Radius is not None):
            # compute precision for images in radius of Radius
            TP = np.sum(is_correct_radius) # true positives
            precision_R[i] = TP/m_radius
            
    # compute average precision and recall
    precision = np.mean(precision, axis=0)
    recall = np.mean(recall, axis=0)
    precision_R = np.mean(precision_R, axis=0)
    
    # compute mAP
    mAP = np.mean(APall)
    
    if Radius is None:
        return mAP, precision, recall
    else:
        return mAP, precision, recall, precision_R

    
def Macro_AP(x_train, y_train, x_test, y_test, num_return_NN=None, weights=None):
    """
    x_train and x_test are of boolean type and have shape (x,K) which 
        x is number of observations and K is the number of bits.
    y_train and y_test have shape (x,) which x is number of observations. These
        two vectors contain labels. They can also be matrices (in one-hot or 
        multi-label case).
    num_return_NN: only compute mAP on returned top num_return_NN neighbours.
    weights: have shape (K,)
    
    return: Macro average precision (MacroAP)
    
    Note: x_train and x_test must have (0,1) values not (-1,1) values.
    """
    ap_all = AP(x_train, y_train, x_test, y_test, num_return_NN=num_return_NN,
                weights=weights)
    
    is_multi_label = is_multilabel(y_train)
    if is_multi_label==False and len(y_train.shape)>1:
        y_train, y_test = y_train.flatten(), y_test.flatten()
    unique_labels = np.unique(y_train, axis=0)
    numclas = len(unique_labels)
    class_AP = np.zeros((numclas))
    
    for i in range(numclas):
        if is_multi_label:   
            index = np.all(unique_labels[i]==y_test, axis=1)
        else:
            index = (unique_labels[i]==y_test)
        if np.sum(index)==0:
            class_AP[i] = -1
            print(f"there is no test sample with label={unique_labels[i]}")
        else:            
            class_AP[i] = np.sum(ap_all[index])/np.sum(index)
        
    return np.mean(class_AP)

