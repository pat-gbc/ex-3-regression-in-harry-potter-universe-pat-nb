import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    learning_rate = 0.001 # Pick a better learning rate
    num_epochs = 10000 # Pick a better number of epochs
    input_features = X.shape[1] # extract the number of features from the input `shape` of X
    output_features = y.shape[1] # extract the number of features from the output `shape` of y
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss() # Use mean squared error loss, like in class

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")
        if abs(previous_loss - loss.item()) < 1e-5:
            print("Stopping early, loss is not changing much.")
            break
        previous_loss = loss.item()
        # This is a good place to print the loss every 1000 epochs.
    return model, loss

