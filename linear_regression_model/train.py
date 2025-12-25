from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import LinearRegressionModel
from utils import visualize_predictions, evaluate_model_metrics, plot_training_history

class SyntheticDataset(Dataset):

    def __init__(self, n_samples=1000, input_dim=1, noise=0.1, seed=42):
        """
        Generate a synthetic data: y = 2x + 1 + noise
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        # generate input features
        self.x = torch.randn(n_samples, input_dim)

        # generate target values with linear relationship: y = 2x + 1 + noise
        # The noise term typically falls between ~-0.3 and 0.3 (for noise=0.1) since torch.randn samples from N(0,1) (so ~99.7% values in [-3, 3]), but actual values can be any real number.
        self.y = 2 * self.x + 1 + torch.randn(n_samples, input_dim) * noise

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Return the sample at the given index.
        """
        return self.x[idx], self.y[idx]

def train_model(model, train_loader, num_epochs=100, learning_rate=0.01, device="cpu"):
    """
    Train the linear regression model
    
    Args:
        model: LinearRegressionModel instance
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cpu' or 'cuda')
    """
    
    # move model to device (move model parameters to the specified device)
    model = model.to(device)

    # loss function: Mean Squared Error (MSE)
    criterion = nn.MSELoss()

    # optimizer: Stochastic Gradient Descent (SGD)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            # move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # forward pass: compute predictions
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # backward pass: compute gradients
            optimizer.zero_grad()  # clear gradients from previous iteration
            loss.backward()  # compute gradients
            optimizer.step()  # update weights

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model, loss_history


def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate model on test data
    
    Args:
        model: Trained LinearRegressionModel
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    """

    model.eval() # set model to evaluation mode, disables dropout and batch normalization
    criterion = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): # disable gradient computation for evalution
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Test Loss (MSE): {avg_loss:.4f}")
    return avg_loss

def main():
    # Hyperparameters
    input_dim = 1
    output_dim = 1
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.01

    # device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create dataset
    train_dataset = SyntheticDataset(n_samples=1000, input_dim=input_dim, noise=0.1)
    test_dataset = SyntheticDataset(n_samples=200, input_dim=input_dim, noise=0.1, seed=123)

    # create datas loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = LinearRegressionModel(input_size=input_dim, output_size=output_dim)
    print(f'\nInitial weights: {model.weights.data.item():.4f}')
    print(f'Initial bias: {model.bias.data.item():.4f}')
    print(f'Expected: weight ≈ 2.0, bias ≈ 1.0\n')

    # train model
    print("Training...")
    trained_model, loss_history = train_model(
        model, 
        train_loader, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        device=device
    )

    # final weights
    print(f'\nFinal weights: {trained_model.weights.data.item():.4f}')
    print(f'Final bias: {trained_model.bias.data.item():.4f}')

    # evaluate on test set
    evaluate_model(trained_model, test_loader, device)

     # plot training history
    plot_training_history(loss_history)

    # visualize predictions
    x_test, y_test = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)))
    visualize_predictions(trained_model, x_test, y_test, title="Linear Regression: Test Set", device=device)

    evaluate_model_metrics(trained_model, x_test, y_test, device)

if __name__ == "__main__":
    main()