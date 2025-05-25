"""
Enhanced evaluation metrics and visualization for CIFAR-10 classification
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

def one_hot_encode(labels, num_classes):
    """
    Convert integer labels to one-hot encoded vectors
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels
    """
    return np.eye(num_classes)[labels]

def plot_loss_accuracy(history, save_path='./results/loss_accuracy_curves.png'):
    """
    Plot training and validation loss and accuracy curves
    
    Args:
        history: Dictionary with training history from model.fit()
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss curve
    ax1.plot(history['loss'], label='Training Loss', color='#2C7BB6', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='#D7191C', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Set y-axis to start at 0
    ax1.set_ylim(bottom=0)
    
    # Accuracy curve
    ax2.plot(history['accuracy'], label='Training Accuracy', color='#2C7BB6', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='#D7191C', linewidth=2, linestyle='--')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Set y-axis limits for accuracy
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='./results/confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row (true labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def compute_metrics(y_true, y_pred, class_names):
    """
    Compute precision, recall, and F1-score for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary with precision, recall, and F1-score for each class
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    # Also compute average metrics
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['average'] = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'support': sum(support)
    }
    
    # Calculate overall accuracy
    accuracy = np.mean(y_true == y_pred)
    metrics['accuracy'] = accuracy
    
    return metrics

def display_metrics_table(metrics):
    """
    Display metrics as a formatted table
    
    Args:
        metrics: Dictionary with metrics from compute_metrics()
    """
    print("\n" + "="*75)
    print(f"{'Class':15} {'Precision':12} {'Recall':12} {'F1-Score':12} {'Support':12}")
    print("="*75)
    
    for class_name, values in metrics.items():
        if class_name != 'accuracy':  # Skip the overall accuracy in this loop
            print(f"{class_name:15} {values['precision']:.4f}        {values['recall']:.4f}        {values['f1_score']:.4f}        {values['support']}")
        
    print("-"*75)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print("="*75)

def plot_metrics(metrics, class_names, save_path='./results/metrics_comparison.png'):
    """
    Plot metrics comparison across classes
    
    Args:
        metrics: Dictionary with metrics from compute_metrics()
        class_names: List of class names
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract metrics for each class (exclude 'average' and 'accuracy')
    classes = [name for name in class_names]
    precisions = [metrics[name]['precision'] for name in classes]
    recalls = [metrics[name]['recall'] for name in classes]
    f1_scores = [metrics[name]['f1_score'] for name in classes]
    
    x = np.arange(len(classes))  # Label positions
    width = 0.25  # Width of bars
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, precisions, width, label='Precision', color='#3274A1')
    ax.bar(x, recalls, width, label='Recall', color='#E1812C')
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='#3A923A')
    
    # Add labels, title and axis ticks
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics by Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Set the y-axis to go from 0 to 1
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Add a horizontal line for the average F1-score
    avg_f1 = metrics['average']['f1_score']
    ax.axhline(y=avg_f1, linestyle='--', color='gray', alpha=0.7, label=f'Avg F1: {avg_f1:.2%}')
    
    # Add text on top of each bar showing the value
    for i, v in enumerate(precisions):
        ax.text(i - width, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontsize=8, rotation=0)
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontsize=8, rotation=0)
    for i, v in enumerate(f1_scores):
        ax.text(i + width, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_samples(data, labels, class_names, num_samples=10, save_path='./results/sample_images.png'):
    """
    Visualize random samples from the dataset
    
    Args:
        data: Image data with shape (N, 3072)
        labels: Labels for the images
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Reshape data back to images
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # For visualization, clip values to [0, 1] range
    images = np.clip(images, 0, 1)
    
    # Select random indices, stratified by class
    indices = []
    samples_per_class = max(2, num_samples // len(class_names))
    
    for class_idx in range(len(class_names)):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) > 0:  # Make sure there are samples for this class
            selected = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
            indices.extend(selected)
    
    # If we need more samples, select randomly
    if len(indices) < num_samples:
        remaining = num_samples - len(indices)
        all_indices = np.arange(len(images))
        mask = np.ones(len(images), dtype=bool)
        mask[indices] = False
        remaining_indices = all_indices[mask]
        if len(remaining_indices) > 0:
            selected = np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False)
            indices.extend(selected)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if num_samples == 10 else \
               plt.subplots(int(np.ceil(num_samples/5)), 5, figsize=(15, 3*np.ceil(num_samples/5)))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices[:num_samples]):
        # Display image
        axes[i].imshow(images[idx])
        axes[i].set_title(f"Class: {class_names[labels[idx]]}")
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_misclassified(data, true_labels, pred_labels, class_names, num_samples=10, save_path='./results/misclassified_samples.png'):
    """
    Visualize misclassified samples
    
    Args:
        data: Image data
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Find misclassified samples
    misclassified = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified) == 0:
        print("No misclassified samples found!")
        return
    
    # Select random misclassified samples
    indices = np.random.choice(misclassified, min(num_samples, len(misclassified)), replace=False)
    
    # Reshape data back to images
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    images = np.clip(images, 0, 1)  # Clip values for visualization
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if num_samples == 10 else \
               plt.subplots(int(np.ceil(num_samples/5)), 5, figsize=(15, 3*np.ceil(num_samples/5)))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Display image
        axes[i].imshow(images[idx])
        axes[i].set_title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}")
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_performance(model, test_data, test_labels, class_names, results_dir='./results'):
    """
    Comprehensive evaluation of model performance
    
    Args:
        model: Trained neural network model
        test_data: Test data
        test_labels: Test labels (integers, not one-hot)
        class_names: List of class names
        results_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert labels to one-hot encoding for the model
    num_classes = len(class_names)
    test_labels_one_hot = one_hot_encode(test_labels, num_classes)
    
    # Get model predictions
    model.eval_mode()  # Set model to evaluation mode
    test_output = model.predict(test_data)
    test_preds = np.argmax(test_output, axis=1)
    
    # Calculate test loss
    test_loss = model.loss(test_labels_one_hot, test_output)
    
    # Calculate accuracy
    test_accuracy = np.mean(test_preds == test_labels)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(test_labels, test_preds, class_names, 
                              save_path=os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Compute and display metrics
    metrics = compute_metrics(test_labels, test_preds, class_names)
    display_metrics_table(metrics)
    
    # Plot metrics comparison
    plot_metrics(metrics, class_names, save_path=os.path.join(results_dir, 'metrics_comparison.png'))
    
    # Visualize misclassified samples
    visualize_misclassified(test_data, test_labels, test_preds, class_names, 
                           save_path=os.path.join(results_dir, 'misclassified_samples.png'))
    
    # Save metrics to file
    np.save(os.path.join(results_dir, 'metrics.npy'), metrics)
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'predictions': test_preds,
        'metrics': metrics
    }
