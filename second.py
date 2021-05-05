#this is the method of extracting information from the MNIST dataset
#had to extract all the files individually to a working directory first though
# if this doesn't work get rid of the additional folders in dataset

from mnist import MNIST

mndata = MNIST('dataset')

images, labels = mndata.load_training()
# or
images, labels = mndata.load_testing()