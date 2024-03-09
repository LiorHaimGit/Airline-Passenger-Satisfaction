from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import torch
from src.data import make_dataset
import seaborn as sns
import os
from src.models.ann_models import AnnModel

# Load the data
train_data, test_data = make_dataset.load_data('../../data/train.csv','../../data/test.csv')

# Load the model
# Get the directory path of the current script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Specify the path to the trained model relative to the current directory
model_path_and_name = os.path.join(current_directory, '../../trained-models/ml_model_2024-03-09_09-23-05.pth')
input_size = len(train_data.columns) - 1
hidden_size = 32
num_classes = 2
model = AnnModel(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(model_path_and_name))
model.eval()

# Convert the DataFrame to a Tensor
train_data_tensor = torch.Tensor(train_data.drop('target', axis=1).values)
test_data_tensor = torch.Tensor(test_data.drop('target', axis=1).values)

# Make predictions on training and testing data
train_predictions = model(train_data_tensor)
test_predictions = model(test_data_tensor)

# Convert probabilities to class labels
train_predictions = torch.argmax(train_predictions, dim=1)
test_predictions = torch.argmax(test_predictions, dim=1)

# Convert tensors to numpy arrays
train_predictions = train_predictions.detach().numpy()
test_predictions = test_predictions.detach().numpy()

# Plot confusion matrix
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
train_cm = confusion_matrix(train_data['target'], train_predictions)
plt.title('Training Data Confusion Matrix')
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues')

plt.subplot(1, 2, 2)
test_cm = confusion_matrix(test_data['target'], test_predictions)
plt.title('Testing Data Confusion Matrix')
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')

plt.tight_layout()
plt.show()

# Plot ROC curve
train_roc_auc = roc_auc_score(train_data['target'], train_predictions)
test_roc_auc = roc_auc_score(test_data['target'], test_predictions)

plt.figure()
fpr, tpr, _ = roc_curve(train_data['target'], train_predictions)
plt.plot(fpr, tpr, label=f'Training ROC Curve (AUC = {train_roc_auc:.2f})')

fpr, tpr, _ = roc_curve(test_data['target'], test_predictions)
plt.plot(fpr, tpr, label=f'Testing ROC Curve (AUC = {test_roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Plot Precision-Recall curve
plt.figure()
precision, recall, _ = precision_recall_curve(train_data['target'], train_predictions)
plt.plot(recall, precision, label='Training Precision-Recall Curve')

precision, recall, _ = precision_recall_curve(test_data['target'], test_predictions)
plt.plot(recall, precision, label='Testing Precision-Recall Curve')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Print classification report
print('Training Data Classification Report:')
print(classification_report(train_data['target'], train_predictions))

print('Testing Data Classification Report:')
print(classification_report(test_data['target'], test_predictions))