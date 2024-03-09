# src/data/make_dataset.py
from src.features.feature_importance import feature_importance
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_data(train_file='data/train.csv', test_file='data/test.csv'):
    # Load the data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Preprocess the data
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Calculate feature importance and drop less important features
    train_data = feature_importance(train_data)
    test_data = feature_importance(test_data)

    return train_data, test_data

def preprocess_data(data):
    # Create a label encoder
    le = LabelEncoder()

    # Fit the encoder and transform the required columns
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Customer Type'] = le.fit_transform(data['Customer Type'])
    data['Type of Travel'] = le.fit_transform(data['Type of Travel'])
    data['Class'] = le.fit_transform(data['Class'])
    data['satisfaction'] = le.fit_transform(data['satisfaction'])

    # Drop unnecessary columns
    data = data.drop(columns=['number', 'id'])

    # Rename the target column
    data = data.rename(columns={'satisfaction': 'target'})

    return data
