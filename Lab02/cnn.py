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
    
    h_out = np.zeros((h.shape[0], h.shape[1], CO))  # [height, width, output_channels]

    # 2. Setup a nested loop over the number of output channels 
    #    and the number of input channels
    for j in range(CO):
        conv_sum = np.zeros_like(h[:, :, 0])
        for i in range(CI):

    # 3. Get the kernel mapping between channels i and j
            kernel = W[:, :, i, j]

    # 4. Flip the kernel horizontally and vertically (since
    #    We want to perform cross-correlation, not convolution.
    #    You can, e.g., look at np.flipud and np.fliplr
            flipped_kernel = np.flipud(np.fliplr(kernel))

    # 5. Run convolution (you can, e.g., look at the convolve2d
    #    function in the scipy.signal library)
            conv = signal.convolve2d(h[:,:,i], flipped_kernel, mode='same')

    # 6. Sum convolutions over input channels, as described in the 
    #    equation for the convolutional layer
            conv_sum += conv

    # 7. Finally, add the bias and apply activation function
        h_out[:, :, j] = conv_sum + b[j]

    return activation(h_out, act)

# 2D max pooling layer
def pool2d_layer(h):  # activations from conv layer, shape = [height, width, channels]
    # TODO: implement the pooling operation
    # 1. Specify the height and width of the output
    sy, sx = 2, 2  # Pooling window size (fixed to 2x2 as default)
    h_out_height = h.shape[0]//sy
    h_out_width = h.shape[1]//sx

    # 2. Specify array to store output
    ho = np.zeros((h_out_height, h_out_width, h.shape[2]))
    
    # 3. Perform pooling for each channel.
    #    You can, e.g., look at the measure.block_reduce() function
    #    in the skimage library

    for channel in range(h.shape[2]):
        ho[:, :, channel] = skimage.measure.block_reduce(h[:, :, channel], block_size=(sy, sx), func=np.max)
    
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
    h = h.reshape(-1, 1)
    
    return activation(np.dot(W, h) + b, act)

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
        self.N = 0  # Initialize total weights counter

        # Loop over weights
        for w in self.W:
            self.N += np.prod(np.array(w).shape)  # Total elements in each weight tensor

        # Loop over biases
        for b in self.b:
            self.N += np.prod(np.array(b).shape)  # Total elements in each bias vector

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
                h = dense_layer(h, self.W[l], self.b[l], act).flatten()
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

    def evaluate(self):
        print('Model performance:')

        # Predictions for the training set
        y_predict_train = self.feedforward(self.dataset.x_train)

        # Convert y_train to one-hot encoding
        y_train_one_hot = np.zeros_like(y_predict_train)  # Same shape as predictions
        y_train_one_hot[np.arange(self.dataset.y_train.shape[0]), self.dataset.y_train] = 1

        # Compute cross-entropy loss
        train_loss = -np.sum(y_train_one_hot * np.log(y_predict_train)) / len(y_predict_train)

        # Compute accuracy
        train_acc = np.mean(np.argmax(y_predict_train, axis=1) == self.dataset.y_train)

        print("\tTrain loss:     %0.4f" % train_loss)
        print("\tTrain accuracy: %0.2f" % train_acc)

        # Predictions for the test set
        y_predict_test = self.feedforward(self.dataset.x_test)

        # Convert y_test to one-hot encoding
        y_test_one_hot = np.zeros_like(y_predict_test)
        y_test_one_hot[np.arange(self.dataset.y_test.shape[0]), self.dataset.y_test] = 1

        # Compute test loss
        test_loss = -np.sum(y_test_one_hot * np.log(y_predict_test)) / len(y_predict_test)

        # Compute test accuracy
        test_acc = np.mean(np.argmax(y_predict_test, axis=1) == self.dataset.y_test)

        print("\tTest loss:      %0.4f" % test_loss)
        print("\tTest accuracy:  %0.2f" % test_acc)
