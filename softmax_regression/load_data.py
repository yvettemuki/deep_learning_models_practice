import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_fashion_mnist(batch_size=256, data_dir='./data'):
    """
    Load Fashion-MNIST dataset
    
    Args:
        batch_size: Number of samples per batch
        data_dir: Directory to store/load the dataset
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    
    # Fashion-MNIST images are 28x28 grayscale images
    # We'll flatten them to 784-dimensional vectors for our model
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 1D: (784,)
    ])
    
    # Load training dataset
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test dataset
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def check_batch_shapes(train_loader):
    """
    Check and display the shapes of a single batch
    
    Args:
        train_loader: DataLoader for training data
    """
    
    # Get a single batch
    images, labels = next(iter(train_loader))
    
    print("=== Batch Shape Information ===")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"\nImage tensor details:")
    print(f"  - Batch size: {images.shape[0]}")
    print(f"  - Feature dimension (flattened): {images.shape[1]}")
    print(f"  - Expected: (batch_size, 784) since 28×28 = 784")
    print(f"\nLabel tensor details:")
    print(f"  - Batch size: {labels.shape[0]}")
    print(f"  - Label values range: {labels.min().item()} to {labels.max().item()}")
    print(f"  - Number of classes: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)")
    print(f"\nSample labels from batch: {labels[:10].tolist()}")
    
    return images, labels


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load the dataset
    print("Loading Fashion-MNIST dataset...")
    train_loader, test_loader = load_fashion_mnist(batch_size=256)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of batches (train): {len(train_loader)}\n")
    
    # Check batch shapes
    images, labels = check_batch_shapes(train_loader)
    
    print(f"\n✓ Data loading complete! Ready for model implementation.")

