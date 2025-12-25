import torch
import matplotlib.pyplot as plt
import numpy as np
from model import LinearRegressionModel


def visualize_predictions(model, x, y_true, title="Linear Regression Predictions", device="cpu"):
    """
    Visualize model predictions vs true values
    
    Args:
        model: Trained LinearRegressionModel
        x: Input features
        y_true: True target values
        title: Plot title
        device: Device to run inference on
    """

    model.eval()
    model = model.to(device)
    x = x.to(device)
    y_true = y_true.to(device)

    with torch.no_grad():
        y_pred = model(x)

    # Always move tensors to CPU before converting to NumPy
    x_np = x.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # sort for better visualization
    sort_idx = np.argsort(x_np.flatten())
    x_sorted = x_np[sort_idx]
    y_true_sorted = y_true_np[sort_idx]
    y_pred_sorted = y_pred_np[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_sorted, y_true_sorted, label="True values", s=20)
    plt.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label="Predictions")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_model_metrics(model, x_test, y_test, device="cpu"):
    """
    Evaluate model with multiple metrics
    
    Args:
        model: Trained LinearRegressionModel
        x_test: Test input features
        y_test: Test target values
        device: Device to run evaluation on
    
    Returns:
        Dictionary with metrics
    """

    model.eval()
    model = model.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        y_pred = model(x_test)

        # Mean Squared Error (MSE)
        mse = torch.mean((y_pred - y_test) ** 2)

        # Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(y_pred - y_test))

        # Root Mean Squared Error (RMSE)
        rmse = torch.sqrt(mse)

        # R-squared (R²- coefficient of determination)
        # R-squared=1, perfect fit; R-squared=0, no linear relationship; R-squared<0, model worse than just using the mean of y
        ss_res = torch.sum((y_test - y_pred) ** 2)
        ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    metrics = {
        'MSE': mse.item(),
        'MAE': mae.item(),
        'RMSE': rmse.item(),
        'R2': r2.item()
    }
    
    print("\n=== Model Evaluation Metrics ===")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.6f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.6f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.6f}")
    print(f"R-squared (R²): {metrics['R2']:.6f}")

    return metrics

def plot_training_history(losses):
    """
    Plot training loss over epochs
    
    Args:
        losses: List of loss values per epoch
    """

    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()