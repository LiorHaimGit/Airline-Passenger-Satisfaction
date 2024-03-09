# train_and_save_model.py
import torch
import datetime
from .train_model import train_model

def train_and_save_model(data,model):

    # Assuming you have the labels and num_epochs defined somewhere
    labels = data['target']
    num_epochs = 10  # Or any other suitable value

    # Train the model
    model = train_model(model, data.drop('target', axis=1), labels, num_epochs)

    # Save the model with the current date and time appended to the file name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_path_and_name = f'trained-models/ml_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path_and_name)

