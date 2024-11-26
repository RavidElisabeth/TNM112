import numpy as np
from scipy import signal
import skimage
import data_generator
    
# Different activations functions
def activation(x, activation):
    
    # # TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    if activation == 'relu':
        return np.maximum(0,x)
    if activation == 'sigmoid':
        return 1/(1 + np.exp(-x)) 
    if activation == 'softmax':
        return (np.exp(x)/np.exp(x).sum())
    else:
        raise Exception("Activation function is not valid", activation) 

# 2D convolutional layer
def conv2d_layer(h,     # activations from previous layer, shape = [height, width, channels prev. layer]
                 W,     # conv. kernels, shape = [kernel height, kernel width, channels prev. layer, channels this layer]
                 b,     # bias vector
                 act    # activation function
):
    # TODO: implement the convolutional layer
    # 1. Specify the number of input and output channels
    CI = W.shape[2] # Number of input channels
    CO = W.shape[3] # Number of output channels
    
    # 2. Setup a nested loop over the number of output channels 
    #    and the number of input channels
    for i in CI:
        for j in CO:
            act(np.dot(W,h) + b)
            
    # 3. Get the kernel mapping between channels i and j
            kernel =
    # 4. Flip the kernel horizontally and vertically (since
    #    We want to perform cross-correlation, not convolution.
    #    You can, e.g., look at np.flipud and np.fliplr
    # 5. Run convolution (you can, e.g., look at the convolve2d
    #    function in the scipy.signal library)
    # 6. Sum convolutions over input channels, as described in the 
    #    equation for the convolutional layer
    # 7. Finally, add the bias and apply activation function


# 2D max pooling layer
def pool2d_layer(h):  # activations from conv layer, shape = [height, width, channels]
    # TODO: implement the pooling operation
    # 1. Specify the height and width of the output
    sy, sx = 2, 2  # Pooling window size (fixed to 2x2 as default)

    # 2. Specify array to store output

    # 3. Perform pooling for each channel.
    #    You can, e.g., look at the measure.block_reduce() function
    #    in the skimage library
    ho = np.stack([
        skimage.measure.block_reduce(h[:, :, c], block_size=(sy, sx), func=np.max)
        for c in range(h.shape[2])
    ], axis=-1)
    
    return ho

# Flattening layer
def flatten_layer(h): # activations from conv/pool layer, shape = [height, width, channels]
    # TODO: Flatten the array to a vector output.
    # You can, e.g., look at the np.ndarray.flatten() function
    return np.ndarray.flatten(h)
    
# Dense (fully-connected) layer
def dense_layer(h,   # Activations from previous layer
                W,   # Weight matrix
                b,   # Bias vector
                act):  # Activation function
    # TODO: implement the dense layer.
    # You can use the code from your implementation
    # in Lab 1. Make sure that the h vector is a [Kx1] array.
    return act(np.dot(h, W) + b)

#---------------------------------
# Our own implementation of a CNN
#---------------------------------
class CNN:
    def __init__(
        self,
        dataset,         # DataGenerator
        verbose=True     # For printing info messages
    ):
        self.verbose = verbose
        self.dataset = dataset

    # Set up the CNN from provided weights
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        lname,               # List of layer names
        activation='relu'    # Activation function of layers
    ):
        self.activation = activation
        self.lname = lname

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model
        #       (convolutional kernels, weight matrices, and bias vectors)
        self.N = 

        print('Number of model weights: ', self.N)

    # Feedforward through the CNN of one single image
    def feedforward_sample(self, h):

        # Loop over all the model layers
        for l in range(len(self.lname)):
            act = self.activation
            
            if self.lname[l] == 'conv':
                h = conv2d_layer(h, self.W[l], self.b[l], act)
            elif self.lname[l] == 'pool':
                h = pool2d_layer(h)
            elif self.lname[l] == 'flatten':
                h = flatten_layer(h)
            elif self.lname[l] == 'dense':
                if l==(len(self.lname)-1):
                    act = 'softmax'
                h = dense_layer(h, self.W[l], self.b[l], act)
        return h

    # Feedforward through the CNN of a dataset
    def feedforward(self, x):
        # Output array
        y = np.zeros((x.shape[0],self.dataset.K))

        # Go through each image
        for k in range(x.shape[0]):
            if self.verbose and np.mod(k,1000)==0:
                print('sample %d of %d'%(k,x.shape[0]))

            # Apply layers to image
            y[k,:] = self.feedforward_sample(x[k])   
            
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the CNN.
        # Assume the cross-entropy loss.
        # For the accuracy, you can use the implementation from Lab 1.

        y_predict_train = self.feedforward(self.dataset.x_train)

        train_loss = -np.sum(self.dataset.y_train * np.log(y_predict_train)) / len(y_predict_train)
        train_acc = np.mean(np.argmax(y_predict_train, 1) == self.dataset.y_train)
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the CNN

        y_predict_test = self.feedforward(self.dataset.x_test)

        test_loss = -np.sum(self.dataset.y_test * np.log(y_predict_test)) / len(y_predict_test)
        test_acc = np.mean(np.argmax(y_predict_test, 1) == self.dataset.y_test)
        print("\tTest loss:      %0.4f"%train_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
