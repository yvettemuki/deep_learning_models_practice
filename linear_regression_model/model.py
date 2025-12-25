import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    """
    Linear Regression model from scratch using PyTorch.
    y = Wx + b
    """

    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        
        # input size == feature size
        # output size == number of output classes (number of target variables)
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        """
        Forward pass: y = xW + b
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        
        # x: (batch_size, input_size) - each row is a sample, each column is a feature
        # self.weights: (input_size, output_size) - each column represents weights for one output target
        # self.bias: (output_size,) - one bias per output
        # output: (batch_size, output_size) - each row is the predicted targets for a sample
        # Matrix multiplication: (batch_size, input_size) @ (input_size, output_size) + (output_size,)

        # each row of the output is the predicted targets for one sample in the batch.
        # Matrix multiplication: x @ weights + bias
        return torch.matmul(x, self.weights) + self.bias
        