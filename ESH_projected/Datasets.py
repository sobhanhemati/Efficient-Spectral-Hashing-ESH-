
import numpy as np
import os
from scipy.io import loadmat

from utilities import one_hot_encode


def cifar10_vggfc7(path, one_hot=True, **kwargs):
    """
    Inputs:
        path: The path containing cifar10_vggfc7 features
        one_hot: If True, return one hoted labels.
    Outputs:
        train features, train labels, test features, and test labels.    
    """
    train_path = os.path.join(path,'cifar10_vggfc7_train.mat')
    test_path = os.path.join(path,'cifar10_vggfc7_test.mat')
    
    train_dict = loadmat(train_path, squeeze_me=True)
    test_dict = loadmat(test_path, squeeze_me=True)
    
    train_features, train_labels = train_dict['train_features'], train_dict['train_labels']
    test_features, test_labels = test_dict['test_features'], test_dict['test_labels']
    
    if one_hot:
        train_labels, test_labels = one_hot_encode(train_labels), one_hot_encode(test_labels)
    
    return train_features, train_labels, test_features, test_labels


def labelme_vggfc7(path, one_hot=True, **kwargs):
    """
    Inputs:
        path: The path containing labelme_vggfc7 features
        one_hot: If True, return one hoted labels.
    Outputs:
        train features, train labels, test features, and test labels.    
    """
    train_path = os.path.join(path,'labelme_vggfc7_train.mat')
    test_path = os.path.join(path,'labelme_vggfc7_test.mat')
    
    train_dict = loadmat(train_path, squeeze_me=True)
    test_dict = loadmat(test_path, squeeze_me=True)
    
    train_features, train_labels = train_dict['train_features'], train_dict['train_labels']
    test_features, test_labels = test_dict['test_features'], test_dict['test_labels']
    
    if one_hot:
        train_labels, test_labels = one_hot_encode(train_labels), one_hot_encode(test_labels)
    
    return train_features, train_labels, test_features, test_labels


def nuswide_vgg(path, **kwargs):
    """
    Inputs:
        path: The path containing nuswide_vgg features (with global avg pooling)
    Outputs:
        train features, train labels, test features, and test labels.    
    """
    path = os.path.join(path,'deep_features_global_AVG_POOL.npy')
    
    data = np.load(path, allow_pickle=True).item()
    
    train_features, train_labels = data['x_train'], data['y_train']
    test_features, test_labels = data['x_test'], data['y_test']
    
    return train_features, train_labels, test_features, test_labels


def colorectal_eff(path, one_hot=True, **kwargs):
    """
    Inputs:
        path: The path containing EfficientNet features of colorectal dataset
    Outputs:
        train features, train labels, test features, and test labels.    
    """
    path = os.path.join(path,'eff_net_colorectal_deep_features_no_tuning.npy')
    
    data = np.load(path, allow_pickle=True).item()
    
    train_features, train_labels = data['x_train'], data['y_train']
    test_features, test_labels = data['x_test'], data['y_test']
    
    if one_hot:
        train_labels, test_labels = one_hot_encode(train_labels), one_hot_encode(test_labels)
    
    return train_features, train_labels, test_features, test_labels



# map dataset names to dataset loaders
Dataset_maps = {'cifar10_vggfc7':cifar10_vggfc7, 'labelme_vggfc7':labelme_vggfc7, 
                'nuswide_vgg':nuswide_vgg, 'colorectal_efficientnet':colorectal_eff}


# loader function
def load_dataset(name, path='.', **kwargs):
    """
    the name of dataset. It can be one of 'cifar10_vggfc7', 
        nuswide_vgg, labelme_vggfc7, colorectal_efficientnet
    path: the path containing dataset files (Do not include the filenames of 
        dataset)
    **kwargs are passed to the function that loads name dataset.
    """
    dataset_loader = Dataset_maps[name.lower()]
    train_features, train_labels, test_features, test_labels = dataset_loader(
        path, **kwargs)
    return train_features, train_labels, test_features, test_labels
    