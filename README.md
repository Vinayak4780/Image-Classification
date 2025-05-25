# CIFAR-10 Image Classification from Scratch

This project implements a neural network from scratch (without using pre-trained models or high-level frameworks like Keras/PyTorch's nn.Module) to classify images from the CIFAR-10 dataset. The implementation achieves at least 60% accuracy on three classes (airplane, automobile, and bird) using only NumPy for core operations.

## Project Structure

```
.
├── data/                    # Data directory (will be created during execution)
├── results/                 # Directory for saving results and visualizations
├── data_loader.py           # Functions for downloading and loading CIFAR-10 dataset
├── data_augmentation.py     # Data augmentation functions
├── neural_network.py        # Base neural network implementation
├── neural_network_improved.py # Enhanced neural network with batch normalization and more
├── evaluation.py            # Basic evaluation metrics and visualization
├── evaluation_enhanced.py   # Enhanced metrics, plots, and visualization functions
├── train.py                 # Basic training script
├── train_improved.py        # Enhanced training with augmentation and scheduling
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- seaborn
- opencv-python

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cifar10-from-scratch.git
   cd cifar10-from-scratch
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Neural Network
Run the basic training script:

```
python train.py
```

### Enhanced Neural Network with Optimizations
Run the improved training script for better performance:

```
python train_improved.py
```

Either script will:
1. Download the CIFAR-10 dataset (if not already downloaded)
2. Preprocess the data for 3 selected classes
3. Train a neural network from scratch
4. Evaluate the model and display metrics
5. Save visualization plots and results

## Implementation Details

### Base Neural Network Architecture

The basic neural network is implemented from scratch using only NumPy:

- **Input Layer**: Flattened 32x32x3 images (3072 neurons)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation
- **Hidden Layer 3**: 64 neurons with ReLU activation
- **Output Layer**: 3 neurons (one for each class) with softmax activation

### Enhanced Neural Network Architecture

The improved implementation includes:

- **Input Layer**: Flattened 32x32x3 images (3072 neurons)
- **Hidden Layer 1**: 512 neurons with BatchNorm + Leaky ReLU + Dropout
- **Hidden Layer 2**: 256 neurons with BatchNorm + Leaky ReLU + Dropout
- **Hidden Layer 3**: 128 neurons with BatchNorm + Leaky ReLU + Dropout
- **Output Layer**: 3 neurons (one for each class) with softmax activation

### Key Features

- **Forward Propagation**: Manually implemented matrix operations
- **Backpropagation**: Gradient computation with momentum
- **Optimization**: Mini-batch gradient descent with learning rate scheduling
- **Loss Function**: Cross-entropy loss
- **Regularization**: Dropout and early stopping
- **Normalization**: Batch normalization
- **Initialization**: He weight initialization for better gradient flow
- **Data Augmentation**: Random flips and brightness/contrast adjustments

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (raw counts and normalized)
- Loss and accuracy curves
- Per-class performance visualization
- Misclassified samples analysis

## Results

The enhanced model achieves over 60% accuracy on the test set for the three selected classes from CIFAR-10 (airplane, automobile, and bird). The implementation includes:

1. **Base Model Performance**:
   - ~33% accuracy with basic implementation
   
2. **Enhanced Model Performance**:
   - 60%+ accuracy with optimizations
   - Improved training stability
   - Better generalization to unseen data
   
3. **Visualizations**:
   - Loss and accuracy curves
   - Confusion matrices
   - Class performance metrics
   - Sample visualizations
   - Misclassified examples analysis

All metrics and visualizations are automatically generated and saved to the results directory.

## Comparison with High-level Frameworks

This implementation achieves reasonable performance without using high-level deep learning frameworks, showing how neural networks function at their core. Key differences from framework-based implementations:

1. **Manual gradient computation**: We explicitly calculate gradients for each layer
2. **Custom layer implementations**: Each layer type is implemented from scratch
3. **Memory management**: All intermediate values are stored and managed manually
4. **Numerical stability**: Special care is taken for operations like softmax and log
5. **Debugging complexity**: Much more visibility into the network's inner workings

## Architecture Compatibility

The solution is compatible with both x86_64 and ARM architectures as it relies only on platform-independent Python libraries (NumPy, Matplotlib, scikit-learn).

