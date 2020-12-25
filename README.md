#  Python implementation of "A non-alternating graph hashing algorithm for large scale image search" paper. Efficient-Spectral-Hashing (ESH) algorithm

# Dependencies :
Tensorflow 2.1.0 or later  

# How to use?
1- Download the dataset: <br />
[labelme_vggfc7](https://www.dropbox.com/s/0nc80qepzj8615f/labelme_vggfc7.rar?dl=0) <br />
[cifar10_vggfc7](https://www.dropbox.com/s/bnybq48ljtsyuit/cifar10_vggfc7.rar?dl=0) <br />
[nuswide_vgg](https://www.dropbox.com/s/6hl9t6oy78w028d/nuswide_vgg.rar?dl=0) <br />
[colorectal_EfficientNet](https://www.dropbox.com/s/wdsalhu73bnrtsg/colorectal_EfficientNet.rar?dl=0) <br />

2- Complete the parameter initialization in either demo_ESH.py or demo_ESH_manifold.py    
For example:   

method_name = 'ESH1'  # or ESH2 (manifold optimization)  
path = r'cifar10_vggfc7' # folder containing dataset  
dataset_name = 'cifar10_vggfc7'  #options: cifar10_vggfc7, labelme_vggfc7, nuswide_vgg, colorectal_efficientnet    
K = 16 # number of bits   

3- Run either demo_ESH.py for ESH1 or demo_ESH_manifold.py for ESH2.

