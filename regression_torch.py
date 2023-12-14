import numpy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim



def regression_train(X_train: numpy.ndarray, y_train: numpy.ndarray):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    class RegressionModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegressionModel, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    input_size = 4
    hidden_size = 64
    output_size = 1

    num_epochs = 100
    batch_size = 32

    model = RegressionModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    regression_file = 'regression_model.py'
    with open(regression_file,'wb') as file:
        pickle.dump(model, file)


def regression_test(X_test: numpy.ndarray, y_test: numpy.ndarray) -> numpy.ndarray:
    criterion = nn.MSELoss()
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    regression_file = 'regression_model.py'
    with open(regression_file, 'rb') as file:
        model = pickle.load(file)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)
        rmse = torch.sqrt(test_loss)
        print(f'RMSE: {rmse.item()}')
        test_predictions_np = test_predictions.numpy()
        y_test_np = y_test_tensor.numpy()
    return y_test_np

