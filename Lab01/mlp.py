import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    # specify the different activation functions
    
    # specify the different activation functions
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

#-------------------------------
# Our own implementation of an Multi Layered Perceptron
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = (len(W) - 1) # The layers inbetween

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = 0

        # Loop over the weight matrices W
        for weight_matrix in self.W:
            self.N += weight_matrix.size  # `size` gives the total number of elements in the matrix

        # Loop over the bias vectors b
        for bias_vector in self.b:
            self.N += bias_vector.size  # `size` gives the number of elements in the bias vector

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values

        y = np.zeros((x.shape[0], self.dataset.K))

        # TODO: implement the feed-forward layer operations

        # 1. Specify a loop over all the datapoints
        for i in range(x.shape[0]):

            # 2. Specify the input layer (2x1 matrix)
            h = x[i,:]
            h = h[:, np.newaxis]

            # 3. For each hidden layer, perform the MLP operations
            for layer in range(self.hidden_layers):
                # - multiply weight matrix and output from previous layer
                # - add bias vector
                h = np.dot(self.W[layer], h) + self.b[layer]
                # - apply activation function
                h = activation(h, self.activation)  
            
            # 4. Specify the final layer, with 'softmax' activation
            h = activation(np.dot(self.W[-1], h) + self.b[-1], 'softmax')
            
            # Store the output in the y matrix for this sample
            y[i,:] = h[:,0]
        
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class

        y_predict_train = self.feedforward(self.dataset.x_train)

        train_loss = np.mean(np.square(y_predict_train - self.dataset.y_train_oh))
        train_acc = np.mean(np.argmax(y_predict_train, 1) == self.dataset.y_train)

        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        
        y_predict_test = self.feedforward(self.dataset.x_test)

        test_loss = np.mean(np.square(y_predict_test - self.dataset.y_test_oh))
        test_acc = np.mean(np.argmax(y_predict_test, 1) == self.dataset.y_test)

        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
