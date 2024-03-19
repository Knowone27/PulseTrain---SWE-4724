import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Load the datasets
known_data = pd.read_csv("Stagger.csv")
unknown_data = pd.read_csv("unknown_cycler.csv")

# Aggregate the data from each CSV into a single feature vector
# For example, using mean and standard deviation
known_features = known_data.describe().iloc[1:3, :].values.flatten()  # Mean and std
unknown_features = unknown_data.describe().iloc[1:3, :].values.flatten()  # Mean and std

# Combine and scale the features
X = np.array([known_features, unknown_features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(
    [1, 0], dtype=torch.float32
)  # Labels (1 for known, 0 for unknown)

# Create PyTorch datasets and data loaders
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(
    dataset, batch_size=1
)  # Batch size of 1 since we have only 2 samples

# Neural network
model = nn.Sequential(
    nn.Linear(X_scaled.shape[1], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(100):
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), targets)
        loss.backward()
        optimizer.step()

# Evaluate the model (In this case, we have only 2 samples)
with torch.no_grad():
    predictions = model(X_tensor)
    print("Predictions:", predictions)
