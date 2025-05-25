"""
Script to train the neural network for 30 epochs and save the model
"""
import numpy as np
import os
import time
from data_loader import load_cifar10_data
from neural_network import (
    NeuralNetwork, 
    DenseLayer, 
    ActivationLayer, 
    DropoutLayer, 
    BatchNormLayer,
    relu, 
    leaky_relu, 
    leaky_relu_derivative,
    softmax, 
    cross_entropy, 
    cross_entropy_derivative
)
from data_augmentation import augment_batch
from evaluation import one_hot_encode, plot_loss_accuracy
from evaluation import visualize_samples
from model_utils import save_model

def train_for_30_epochs():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create results directory if it doesn't exist
    models_dir = './models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading CIFAR-10 dataset...")
    # Load data for 3 classes from CIFAR-10
    selected_classes = [0, 1, 2]  # Airplane, Automobile, Bird
    data = load_cifar10_data(selected_classes=selected_classes)
    
    train_data = data['train_data']
    train_labels = data['train_labels']
    test_data = data['test_data']
    test_labels = data['test_labels']
    class_names = data['class_names']
    num_classes = len(class_names)
    
    print(f"Loaded {len(class_names)} classes: {', '.join(class_names)}")
    
    # Normalize data to zero mean and unit variance for better convergence
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    # Split training data into training and validation sets (80-20 split)
    indices = np.random.permutation(len(train_data))
    split_idx = int(len(train_data) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    x_train = train_data[train_indices]
    y_train = train_labels[train_indices]
    x_val = train_data[val_indices]
    y_val = train_labels[val_indices]
    
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_val_one_hot = one_hot_encode(y_val, num_classes)
    y_test_one_hot = one_hot_encode(test_labels, num_classes)
    
    # Input dimension (3072 = 32x32x3) for flattened images
    input_dim = train_data.shape[1]
    
    # Create and configure neural network with improved architecture
    model = NeuralNetwork()
    
    # First block with batch normalization
    model.add(DenseLayer(input_dim, 512))  # First hidden layer with 512 neurons
    model.add(BatchNormLayer(512))  # Add batch normalization
    model.add(ActivationLayer(leaky_relu, lambda x: leaky_relu_derivative(x)))
    model.add(DropoutLayer(0.2))  # Light dropout after first layer
    
    # Second block
    model.add(DenseLayer(512, 256))  # Second hidden layer with 256 neurons
    model.add(BatchNormLayer(256))  # Add batch normalization
    model.add(ActivationLayer(leaky_relu, lambda x: leaky_relu_derivative(x)))
    model.add(DropoutLayer(0.3))  # Moderate dropout
    
    # Third block
    model.add(DenseLayer(256, 128))  # Third hidden layer with 128 neurons
    model.add(BatchNormLayer(128))  # Add batch normalization
    model.add(ActivationLayer(leaky_relu, lambda x: leaky_relu_derivative(x)))
    model.add(DropoutLayer(0.3))  # Moderate dropout
    
    # Output layer
    model.add(DenseLayer(128, num_classes))  # Output layer with neurons for each class
    
    # Set loss function to cross-entropy with softmax (for classification)
    model.set_loss(cross_entropy, cross_entropy_derivative)
    
    # Training parameters
    epochs = 30  # Exactly 30 epochs as requested
    batch_size = 128
    learning_rate = 0.003
    lr_decay = 0.95
    decay_every = 5
    
    print(f"\nTraining neural network for {epochs} epochs...")
    start_time = time.time()
    
    # Train the model
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }
    
    n_samples = x_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    current_lr = learning_rate
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Apply learning rate decay
        if epoch > 0 and epoch % decay_every == 0:
            current_lr *= lr_decay
            print(f"Learning rate decreased to {current_lr:.6f}")
        
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train_one_hot[indices]
        
        epoch_loss = 0
        for batch in range(n_batches):
            # Get batch data
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Apply data augmentation to this batch
            if epoch > 0:  # Start augmentation after the first epoch
                x_batch = augment_batch(x_batch, aug_factor=0.5)
            
            # Forward pass
            output = model.predict(x_batch)
            
            # Compute loss for this batch
            batch_loss = model.loss(y_batch, output)
            batch_weight = (end_idx - start_idx) / n_samples
            epoch_loss += batch_loss * batch_weight
            
            # Backward pass
            error = model.loss_derivative(y_batch, output)
            for layer in reversed(model.layers):
                error = layer.backward(error, current_lr)
        
        # Compute training accuracy (in training mode)
        model.train_mode()
        train_preds = np.argmax(model.predict(x_train), axis=1)
        train_true = y_train
        train_accuracy = np.mean(train_preds == train_true)
        
        # Compute validation loss and accuracy (in evaluation mode)
        model.eval_mode()
        val_output = model.predict(x_val)
        val_loss = model.loss(y_val_one_hot, val_output)
        val_preds = np.argmax(val_output, axis=1)
        val_true = y_val
        val_accuracy = np.mean(val_preds == val_true)
        
        # Set back to training mode for next epoch
        model.train_mode()
        
        # Record history
        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot training history
    plot_loss_accuracy(history)
    
    # Save the model
    model_path = os.path.join(models_dir, 'cifar10_model_30epochs.pkl')
    save_model(model, model_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval_mode()
    test_output = model.predict(test_data)
    test_loss = model.loss(y_test_one_hot, test_output)
    test_preds = np.argmax(test_output, axis=1)
    test_true = test_labels
    test_accuracy = np.mean(test_preds == test_true)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save training data normalization parameters for future use
    np.savez(
        os.path.join(models_dir, 'data_normalization_params.npz'),
        mean=mean,
        std=std
    )
    
    print(f"\nModel saved to {model_path}")
    print("Normalization parameters saved for future use.")
    
    return {
        'model': model,
        'accuracy': test_accuracy,
        'history': history
    }

if __name__ == "__main__":
    train_for_30_epochs()
