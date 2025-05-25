"""
Download and prepare CIFAR-10 dataset
"""
import os
import pickle
import numpy as np
import urllib.request
import tarfile
from tqdm import tqdm

def download_cifar10():
    """
    Download CIFAR-10 dataset if not already downloaded
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for CIFAR-10 dataset
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    
    # Download only if the directory doesn't exist
    if not os.path.exists(extracted_dir):
        print(f"Downloading CIFAR-10 dataset...")
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded to {filename}")
        
        print("Extracting files...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print(f"Extracted to {extracted_dir}")
    else:
        print(f"CIFAR-10 dataset already exists at {extracted_dir}")
    
    return extracted_dir

def unpickle(file):
    """
    Load a pickled file from CIFAR-10
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(selected_classes=None):
    """
    Load CIFAR-10 data for selected classes
    
    Args:
        selected_classes: List of class indices to load. If None, all classes are loaded.
        
    Returns:
        Dictionary containing train_data, train_labels, test_data, test_labels, class_names
    """
    extracted_dir = download_cifar10()
    
    # Load test batch
    test_batch = unpickle(os.path.join(extracted_dir, 'test_batch'))
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
    
    # Load training batches
    train_data = []
    train_labels = []
    for i in range(1, 6):  # 5 training batches
        batch = unpickle(os.path.join(extracted_dir, f'data_batch_{i}'))
        if i == 1:
            train_data = batch[b'data']
            train_labels = batch[b'labels']
        else:
            train_data = np.vstack((train_data, batch[b'data']))
            train_labels = np.concatenate((train_labels, batch[b'labels']))
    train_labels = np.array(train_labels)
    
    # Load class names
    meta_data = unpickle(os.path.join(extracted_dir, 'batches.meta'))
    class_names = [class_name.decode('utf-8') for class_name in meta_data[b'label_names']]
    
    # Filter by selected classes if specified
    if selected_classes is not None:
        train_indices = np.isin(train_labels, selected_classes)
        test_indices = np.isin(test_labels, selected_classes)
        
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
        
        # Remap labels to be consecutive starting from 0
        label_map = {old: new for new, old in enumerate(selected_classes)}
        train_labels = np.array([label_map[label] for label in train_labels])
        test_labels = np.array([label_map[label] for label in test_labels])
        
        # Also update class names
        class_names = [class_names[i] for i in selected_classes]
    
    # Reshape and normalize data
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    
    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'class_names': class_names
    }

def preprocess_data(data):
    """
    Preprocess CIFAR-10 data: reshape and normalize
    """
    # Reshape from (N, 3072) to (N, 32, 32, 3) and normalize
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    data = data / 255.0  # Normalize to [0, 1]
    
    # Flatten images for our simple neural network
    data = data.reshape(data.shape[0], -1)
    
    return data

if __name__ == "__main__":
    # Test by loading three classes: airplane (0), automobile (1), and bird (2)
    data = load_cifar10_data(selected_classes=[0, 1, 2])
    print("Train data shape:", data['train_data'].shape)
    print("Test data shape:", data['test_data'].shape)
    print("Classes:", data['class_names'])
