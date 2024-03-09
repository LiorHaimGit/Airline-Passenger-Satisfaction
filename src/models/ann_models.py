# ann_model.py

import torch.nn as nn

class AnnModel1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AnnModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class AnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AnnModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)  # Adding ReLU activation for the second hidden layer
        out = self.fc3(out)
        out = self.relu(out)  # Adding ReLU activation for the third hidden layer
        out = self.fc4(out)
        return out