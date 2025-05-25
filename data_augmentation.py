"""
Data augmentation functions for enhancing the dataset
"""
import numpy as np

def random_flip_horizontal(images, prob=0.5):
    """
    Randomly flip images horizontally
    
    Args:
        images: Array of images with shape (N, 3072) (flattened 32x32x3 images)
        prob: Probability of flipping each image
    
    Returns:
        Augmented images
    """
    # Reshape to (N, 32, 32, 3)
    images_reshaped = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    result = images_reshaped.copy()
    
    # Randomly select images to flip
    flip_mask = np.random.random(size=len(images)) < prob
    
    # Flip selected images
    result[flip_mask] = result[flip_mask, :, ::-1, :]
    
    # Flatten back to original shape
    return result.transpose(0, 3, 1, 2).reshape(-1, 3072)

def random_brightness_contrast(images, brightness_range=(-0.1, 0.1), contrast_range=(0.8, 1.2)):
    """
    Randomly adjust brightness and contrast of images
    
    Args:
        images: Array of images with shape (N, 3072) (flattened 32x32x3 images)
        brightness_range: Tuple with min and max brightness adjustment
        contrast_range: Tuple with min and max contrast adjustment
    
    Returns:
        Augmented images
    """
    # Reshape to (N, 32, 32, 3)
    images_reshaped = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    result = images_reshaped.copy()
    
    # Generate random brightness and contrast factors
    brightness = np.random.uniform(brightness_range[0], brightness_range[1], size=(len(images), 1, 1, 1))
    contrast = np.random.uniform(contrast_range[0], contrast_range[1], size=(len(images), 1, 1, 1))
    
    # Apply brightness and contrast adjustment
    # Adjust brightness by adding a value
    # Adjust contrast by multiplying by a factor
    result = result * contrast + brightness
    
    # Clip values to be in [0, 1] range
    result = np.clip(result, 0, 1)
    
    # Flatten back to original shape
    return result.transpose(0, 3, 1, 2).reshape(-1, 3072)

def random_rotate(images, max_angle=15):
    """
    Randomly rotate images
    
    Note: This is a simple implementation that doesn't handle boundaries well.
    For production use, consider using a library like scikit-image or scipy.
    
    Args:
        images: Array of images with shape (N, 3072) (flattened 32x32x3 images)
        max_angle: Maximum rotation angle in degrees
    
    Returns:
        Augmented images
    """
    # This is a placeholder, as rotation requires more advanced image processing
    # that's hard to implement efficiently in raw NumPy.
    # For now, we'll just return the original images.
    return images

def random_crop_and_resize(images, crop_factor_range=(0.8, 0.95)):
    """
    Randomly crop images and resize them back to original size
    
    Note: This is a simple implementation. For production use, consider using a library.
    
    Args:
        images: Array of images with shape (N, 3072) (flattened 32x32x3 images)
        crop_factor_range: Range for the crop size as a fraction of the original
    
    Returns:
        Augmented images
    """
    # This is a placeholder, as proper crop and resize requires more advanced image processing
    # that's hard to implement efficiently in raw NumPy.
    # For now, we'll just return the original images.
    return images

def augment_batch(images, aug_factor=1.0):
    """
    Apply multiple augmentations to a batch of images
    
    Args:
        images: Array of images with shape (N, 3072)
        aug_factor: Fraction of images to augment (1.0 means all images)
    
    Returns:
        Augmented batch of images
    """
    num_to_augment = int(len(images) * aug_factor)
    if num_to_augment == 0:
        return images
    
    indices = np.random.choice(len(images), num_to_augment, replace=False)
    augmented_images = images.copy()
    
    # Apply a chain of augmentations
    augmented_subset = images[indices]
    augmented_subset = random_flip_horizontal(augmented_subset, prob=0.5)
    augmented_subset = random_brightness_contrast(augmented_subset)
    
    # Replace original images with augmented versions
    augmented_images[indices] = augmented_subset
    
    return augmented_images
