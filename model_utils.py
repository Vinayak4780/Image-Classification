"""
Neural network model serialization and deserialization utilities
"""
import numpy as np
import pickle
import os

def save_model(model, filepath):
    """
    Save a neural network model to a file
    
    Args:
        model: NeuralNetwork instance to save
        filepath: Path where the model will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Extract weights and biases from each layer
    model_data = {
        'layer_types': [],
        'layer_params': [],
        'is_training': model.is_training
    }
    
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        model_data['layer_types'].append(layer_type)
        
        # Extract parameters based on layer type
        if layer_type == 'DenseLayer':
            model_data['layer_params'].append({
                'weights': layer.weights,
                'bias': layer.bias,
                'weight_momentum': layer.weight_momentum,
                'bias_momentum': layer.bias_momentum,
                'momentum': layer.momentum
            })
        elif layer_type == 'BatchNormLayer':
            model_data['layer_params'].append({
                'gamma': layer.gamma,
                'beta': layer.beta,
                'running_mean': layer.running_mean,
                'running_var': layer.running_var,
                'epsilon': layer.epsilon,
                'momentum': layer.momentum
            })
        elif layer_type == 'DropoutLayer':
            model_data['layer_params'].append({
                'dropout_rate': layer.dropout_rate
            })
        elif layer_type == 'ActivationLayer':
            # For activation layers, we need to store the activation function name
            if layer.activation.__name__ == 'relu':
                act_name = 'relu'
            elif layer.activation.__name__ == 'leaky_relu':
                act_name = 'leaky_relu'
            elif layer.activation.__name__ == 'softmax':
                act_name = 'softmax'
            elif layer.activation.__name__ == 'sigmoid':
                act_name = 'sigmoid'
            else:
                act_name = 'unknown'
                
            model_data['layer_params'].append({
                'activation_name': act_name
            })
        else:
            model_data['layer_params'].append({})
    
    # Save the model data
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    """
    Load a neural network model from a file
    
    Args:
        model: Empty NeuralNetwork instance to load into
        filepath: Path to the saved model file
        
    Returns:
        Loaded NeuralNetwork instance
    """
    from neural_network_improved import (
        DenseLayer, BatchNormLayer, ActivationLayer, 
        DropoutLayer, relu, relu_derivative, leaky_relu, 
        leaky_relu_derivative, sigmoid, sigmoid_derivative,
        softmax, softmax_derivative
    )
    
    # Load the model data
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Restore model state
    model.is_training = model_data['is_training']
    
    # Clear existing layers (if any)
    model.layers = []
    
    # Restore each layer with its parameters
    for layer_type, layer_params in zip(model_data['layer_types'], model_data['layer_params']):
        if layer_type == 'DenseLayer':
            weights = layer_params['weights']
            input_size = weights.shape[0]
            output_size = weights.shape[1]
            
            layer = DenseLayer(input_size, output_size)
            layer.weights = weights
            layer.bias = layer_params['bias']
            layer.weight_momentum = layer_params['weight_momentum']
            layer.bias_momentum = layer_params['bias_momentum']
            layer.momentum = layer_params['momentum']
            
            model.add(layer)
            
        elif layer_type == 'BatchNormLayer':
            input_size = layer_params['gamma'].shape[1]
            
            layer = BatchNormLayer(input_size)
            layer.gamma = layer_params['gamma']
            layer.beta = layer_params['beta']
            layer.running_mean = layer_params['running_mean']
            layer.running_var = layer_params['running_var']
            layer.epsilon = layer_params['epsilon']
            layer.momentum = layer_params['momentum']
            
            model.add(layer)
            
        elif layer_type == 'DropoutLayer':
            layer = DropoutLayer(layer_params['dropout_rate'])
            model.add(layer)
            
        elif layer_type == 'ActivationLayer':
            act_name = layer_params['activation_name']
            
            if act_name == 'relu':
                layer = ActivationLayer(relu, relu_derivative)
            elif act_name == 'leaky_relu':
                layer = ActivationLayer(leaky_relu, lambda x: leaky_relu_derivative(x))
            elif act_name == 'sigmoid':
                layer = ActivationLayer(sigmoid, sigmoid_derivative)
            elif act_name == 'softmax':
                layer = ActivationLayer(softmax, softmax_derivative)
            else:
                raise ValueError(f"Unknown activation function: {act_name}")
                
            model.add(layer)
    
    print(f"Model loaded from {filepath}")
    return model
