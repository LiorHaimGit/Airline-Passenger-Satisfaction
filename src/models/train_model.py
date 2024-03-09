# src/models/train_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return (correct / total) * 100

def train_model(model, data, labels, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Convert data and labels to PyTorch tensors
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(data_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the starting time
    start_time = time.time()

    # Initialize the total number of steps
    total_steps = num_epochs * len(dataloader)
    current_step = 0

    # Train the model
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Start time for current step
            step_start_time = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Increment current step
            current_step += 1

            # Calculate time taken for current step
            step_time = time.time() - step_start_time

            # Calculate average time per step
            avg_time_per_step = (time.time() - start_time) / current_step

            # Calculate remaining time estimation
            remaining_steps = total_steps - current_step
            estimated_remaining_time = remaining_steps * avg_time_per_step


            # Print progress information every 10 epochs
            if current_step % 500 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {calculate_accuracy(model, dataloader):.2f}%, '
                      f'Elapsed Time: {time.time() - start_time:.2f}s, '
                      f'Estimated Time Remaining: {estimated_remaining_time:.2f}s')

    # Return the trained model
    return model