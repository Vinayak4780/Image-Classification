"""
Neural network implementation from scratch using NumPy
"""
import numpy as np
import time
from tqdm import tqdm

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        """Forward pass"""
        raise NotImplementedError
        
    def backward(self, output_error, learning_rate):
        """Backward pass"""
        raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        """
        Initialize a fully connected layer
        
        Args:
            input_size: Number of input features
            output_size: Number of neurons in this layer
        """
        super().__init__()
        # He initialization for ReLU networks - scale by sqrt(2/input_size)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        # For gradient accumulation and momentum
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)
        self.momentum = 0.9  # Momentum coefficient
        
    def forward(self, input_data):
        """
        Forward pass for this layer
        
        Args:
            input_data: Input data of shape (batch_size, input_size)
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
      def backward(self, output_error, learning_rate):
        """
        Backward pass for this layer
        
        Args:
            output_error: Error from the next layer
            learning_rate: Learning rate for gradient descent
        """
        # Calculate error for the previous layer
        input_error = np.dot(output_error, self.weights.T)
        
        # Calculate gradients
        batch_size = output_error.shape[0]
        weights_error = np.dot(self.input.T, output_error) / batch_size  # Normalize by batch size
        bias_error = np.sum(output_error, axis=0, keepdims=True) / batch_size
        
        # Update parameters using momentum
        self.weight_momentum = self.momentum * self.weight_momentum - learning_rate * weights_error
        self.bias_momentum = self.momentum * self.bias_momentum - learning_rate * bias_error
        
        self.weights += self.weight_momentum
        self.bias += self.bias_momentum
        
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        """
        Initialize an activation layer
        
        Args:
            activation: Activation function
            activation_derivative: Derivative of activation function
        """
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward(self, input_data):
        """
        Forward pass for this layer
        
        Args:
            input_data: Input data
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_error, learning_rate):
        """
        Backward pass for this layer
        
        Args:
            output_error: Error from the next layer
            learning_rate: Learning rate (not used in this layer)
        """
        return self.activation_derivative(self.input) * output_error

class DropoutLayer(Layer):
    def __init__(self, dropout_rate):
        """
        Initialize a dropout layer
        
        Args:
            dropout_rate: Probability of setting a neuron to zero (between 0 and 1)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
        
    def set_training_mode(self, training):
        """Set layer to training or evaluation mode"""
        self.training = training
        
    def forward(self, input_data):
        """
        Forward pass for dropout layer
        
        Args:
            input_data: Input data
        """
        self.input = input_data
        
        if self.training:
            # Generate dropout mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input_data.shape) / (1 - self.dropout_rate)
            # Apply mask
            self.output = input_data * self.mask
        else:
            self.output = input_data  # No dropout during evaluation
            
        return self.output
    
    def backward(self, output_error, learning_rate):
        """
        Backward pass for dropout layer
        
        Args:
            output_error: Error from the next layer
            learning_rate: Not used in dropout layer
        """
        if self.training:
            # Apply the same mask during backward pass
            return output_error * self.mask
        else:
            return output_error

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip to prevent overflow

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    # Subtract max for numerical stability
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def softmax_derivative(x):
    # For cross-entropy loss with softmax, this derivative is handled 
    # directly in the loss function's derivative
    return np.ones_like(x)

# Loss functions and their derivatives
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy(y_true, y_pred):
    # Add small epsilon for numerical stability
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cross_entropy_derivative(y_true, y_pred):
    # With softmax + cross-entropy, the derivative simplifies
    return y_pred - y_true

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None
        self.is_training = True
        
    def add(self, layer):
        """
        Add a layer to the network
        """
        self.layers.append(layer)
    
    def set_loss(self, loss, loss_derivative):
        """
        Set the loss function and its derivative
        """
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    def predict(self, input_data, training=None):
        """
        Make predictions with the network
        
        Args:
            input_data: Input data
            training: Boolean indicating whether in training mode (for dropout)
        """
        # Set training mode if specified
        training_mode = self.is_training if training is None else training
        
        # Update dropout layers to the proper mode
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.set_training_mode(training_mode)
        
        # Forward pass
        result = input_data
        for layer in self.layers:
            result = layer.forward(result)
        return result
    
    def train_mode(self):
        """Set network to training mode"""
        self.is_training = True
        
    def eval_mode(self):
        """Set network to evaluation mode"""
        self.is_training = False
    
    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate):
        """
        Train the network
        
        Args:
            x_train: Training data
            y_train: Training labels (one-hot encoded)
            x_val: Validation data
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        n_samples = x_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            for batch in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}', leave=False):
                # Get batch data
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = self.predict(x_batch)
                
                # Compute loss for this batch
                batch_loss = self.loss(y_batch, output)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Backward pass
                error = self.loss_derivative(y_batch, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
              # Compute training accuracy (in training mode)
            self.train_mode()
            train_preds = np.argmax(self.predict(x_train), axis=1)
            train_true = np.argmax(y_train, axis=1)
            train_accuracy = np.mean(train_preds == train_true)
            
            # Compute validation loss and accuracy (in evaluation mode)
            self.eval_mode()
            val_output = self.predict(x_val)
            val_loss = self.loss(y_val, val_output)
            val_preds = np.argmax(val_output, axis=1)
            val_true = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(val_preds == val_true)
            
            # Set back to training mode for next epoch
            self.train_mode()
            
            # Record history
            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
            
        return history
      def evaluate(self, x_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            x_test: Test data
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary with test loss and accuracy
        """
        # Set to evaluation mode for testing
        self.eval_mode()
        
        test_output = self.predict(x_test)
        test_loss = self.loss(y_test, test_output)
        
        test_preds = np.argmax(test_output, axis=1)
        test_true = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_preds == test_true)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': test_preds,
            'true_labels': test_true
        }
