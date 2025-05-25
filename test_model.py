"""
Script to load a saved model and test it
"""
import numpy as np
import os
from neural_network_improved import NeuralNetwork, cross_entropy, cross_entropy_derivative
from data_loader import load_cifar10_data
from evaluation import one_hot_encode
from evaluation_enhanced import evaluate_model_performance
from model_utils import load_model

def test_saved_model(model_path='./models/cifar10_model_30epochs.pkl', 
                    norm_params_path='./models/data_normalization_params.npz'):
    """
    Load a saved model and test it on the test set
    
    Args:
        model_path: Path to the saved model
        norm_params_path: Path to the normalization parameters
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    # Check if the normalization parameters file exists
    if not os.path.exists(norm_params_path):
        print(f"Normalization parameters file not found: {norm_params_path}")
        return
    
    # Load normalization parameters
    norm_params = np.load(norm_params_path)
    mean = norm_params['mean']
    std = norm_params['std']
    
    # Load test data
    print("Loading CIFAR-10 test data...")
    selected_classes = [0, 1, 2]  # Airplane, Automobile, Bird
    data = load_cifar10_data(selected_classes=selected_classes)
    
    test_data = data['test_data']
    test_labels = data['test_labels']
    class_names = data['class_names']
    num_classes = len(class_names)
    
    # Normalize test data using saved parameters
    test_data = (test_data - mean) / std
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = NeuralNetwork()
    model.set_loss(cross_entropy, cross_entropy_derivative)  # Set the loss function
    model = load_model(model, model_path)
    
    # Set model to evaluation mode
    model.eval_mode()
    
    # One-hot encode labels for loss computation
    test_labels_one_hot = one_hot_encode(test_labels, num_classes)
    
    # Perform comprehensive evaluation
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nEvaluating loaded model...")
    evaluation_results = evaluate_model_performance(
        model, test_data, test_labels, class_names, results_dir)
    
    return evaluation_results

if __name__ == "__main__":
    test_saved_model()
